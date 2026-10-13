"""Joint-space trapezoidal references for native MuJoCo Control-panel commands."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco

from master_controller import MasterController, actuator_gear
from robot_master_configuration import TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2, TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S


EPSILON = 1e-9


def trapezoidal_duration(
    distance_rad: float,
    max_velocity_rad_s: float = TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S,
    max_acceleration_rad_s2: float = TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2,
) -> float:
    """Return the minimum rest-to-rest time for a scalar displacement."""
    distance = abs(float(distance_rad))
    if distance <= EPSILON:
        return 0.0
    if max_velocity_rad_s <= 0.0 or max_acceleration_rad_s2 <= 0.0:
        raise ValueError("velocity and acceleration limits must be greater than zero")
    acceleration_time = max_velocity_rad_s / max_acceleration_rad_s2
    acceleration_distance = 0.5 * max_acceleration_rad_s2 * acceleration_time**2
    if distance <= 2.0 * acceleration_distance:
        return 2.0 * math.sqrt(distance / max_acceleration_rad_s2)
    return 2.0 * acceleration_time + (distance - 2.0 * acceleration_distance) / max_velocity_rad_s


def trapezoidal_position(
    initial_rad: float,
    target_rad: float,
    elapsed_s: float,
    max_velocity_rad_s: float = TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S,
    max_acceleration_rad_s2: float = TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2,
) -> float:
    """Return a rest-to-rest, velocity- and acceleration-limited position."""
    displacement = float(target_rad) - float(initial_rad)
    distance = abs(displacement)
    if distance <= EPSILON or elapsed_s <= 0.0:
        return float(initial_rad)
    direction = math.copysign(1.0, displacement)
    acceleration_time = max_velocity_rad_s / max_acceleration_rad_s2
    acceleration_distance = 0.5 * max_acceleration_rad_s2 * acceleration_time**2
    if distance <= 2.0 * acceleration_distance:
        peak_velocity = math.sqrt(distance * max_acceleration_rad_s2)
        peak_time = peak_velocity / max_acceleration_rad_s2
        duration = 2.0 * peak_time
        if elapsed_s >= duration:
            progress = distance
        elif elapsed_s <= peak_time:
            progress = 0.5 * max_acceleration_rad_s2 * elapsed_s**2
        else:
            progress = distance - 0.5 * max_acceleration_rad_s2 * (duration - elapsed_s) ** 2
    else:
        cruise_time = (distance - 2.0 * acceleration_distance) / max_velocity_rad_s
        duration = 2.0 * acceleration_time + cruise_time
        if elapsed_s >= duration:
            progress = distance
        elif elapsed_s <= acceleration_time:
            progress = 0.5 * max_acceleration_rad_s2 * elapsed_s**2
        elif elapsed_s <= acceleration_time + cruise_time:
            progress = acceleration_distance + max_velocity_rad_s * (elapsed_s - acceleration_time)
        else:
            progress = distance - 0.5 * max_acceleration_rad_s2 * (duration - elapsed_s) ** 2
    return float(initial_rad) + direction * progress


@dataclass
class TrapezoidalReference:
    """One velocity- and acceleration-limited joint-space reference."""

    actuator_id: int
    gear: float
    position_rad: float
    velocity_rad_s: float
    requested_position_rad: float
    last_applied_control: float


class TrapezoidalReferenceLimiter:
    """Convert Control-panel jumps into 1-D trapezoidal joint references.

    The native viewer and this limiter share ``data.ctrl``.  A new user slider
    value is detected when it differs from the previous filtered value.  The
    limiter then writes its smooth reference back to ``data.ctrl`` before the
    position actuator computes force.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        controller: MasterController,
        max_velocity_rad_s: float = TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S,
        max_acceleration_rad_s2: float = TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2,
    ) -> None:
        if not math.isfinite(max_velocity_rad_s) or max_velocity_rad_s <= 0.0:
            raise ValueError("max_velocity_rad_s must be finite and greater than zero")
        if not math.isfinite(max_acceleration_rad_s2) or max_acceleration_rad_s2 <= 0.0:
            raise ValueError("max_acceleration_rad_s2 must be finite and greater than zero")
        self.max_velocity_rad_s = max_velocity_rad_s
        self.max_acceleration_rad_s2 = max_acceleration_rad_s2
        self.references: dict[str, TrapezoidalReference] = {}
        for schedule in controller.schedules:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, schedule.actuator)
            if actuator_id < 0:
                raise ValueError(f"scheduled actuator {schedule.actuator!r} is absent from this MJCF")
            self.references[schedule.actuator] = TrapezoidalReference(actuator_id, actuator_gear(model, actuator_id), 0.0, 0.0, 0.0, 0.0)
        self.last_time: float | None = None

    def reset(self, data: mujoco.MjData) -> None:
        """Synchronize reference state with the current Control-panel values."""
        for reference in self.references.values():
            control = float(data.ctrl[reference.actuator_id])
            position = control / reference.gear
            reference.position_rad = position
            reference.velocity_rad_s = 0.0
            reference.requested_position_rad = position
            reference.last_applied_control = control
        self.last_time = float(data.time)

    def _advance(self, reference: TrapezoidalReference, dt: float) -> None:
        error = reference.requested_position_rad - reference.position_rad
        if abs(error) <= EPSILON and abs(reference.velocity_rad_s) <= EPSILON:
            reference.position_rad = reference.requested_position_rad
            reference.velocity_rad_s = 0.0
            return
        direction = 1.0 if error >= 0.0 else -1.0
        # Reverse first if moving away from the requested position; otherwise
        # brake when stopping distance reaches the remaining distance.
        if reference.velocity_rad_s * direction < -EPSILON:
            acceleration = direction * self.max_acceleration_rad_s2
        elif abs(error) <= reference.velocity_rad_s**2 / (2.0 * self.max_acceleration_rad_s2):
            acceleration = -math.copysign(self.max_acceleration_rad_s2, reference.velocity_rad_s)
        else:
            acceleration = direction * self.max_acceleration_rad_s2
        next_velocity = max(
            -self.max_velocity_rad_s,
            min(self.max_velocity_rad_s, reference.velocity_rad_s + acceleration * dt),
        )
        next_position = reference.position_rad + next_velocity * dt
        if (reference.requested_position_rad - reference.position_rad) * (
            reference.requested_position_rad - next_position
        ) <= 0.0:
            reference.position_rad = reference.requested_position_rad
            reference.velocity_rad_s = 0.0
        else:
            reference.position_rad = next_position
            reference.velocity_rad_s = next_velocity

    def update(self, data: mujoco.MjData) -> None:
        """Capture user controls, advance once per simulation time step, write references."""
        current_time = float(data.time)
        if self.last_time is None:
            self.reset(data)
        for reference in self.references.values():
            incoming_control = float(data.ctrl[reference.actuator_id])
            # The viewer only changes ctrl on an actual slider edit. Values
            # equal to our previous filtered output are our own feedback.
            if abs(incoming_control - reference.last_applied_control) > 1e-7:
                reference.requested_position_rad = incoming_control / reference.gear
        if current_time > self.last_time + EPSILON:
            dt = min(current_time - self.last_time, 0.05)
            for reference in self.references.values():
                self._advance(reference, dt)
            self.last_time = current_time
        for reference in self.references.values():
            control = reference.gear * reference.position_rad
            data.ctrl[reference.actuator_id] = control
            reference.last_applied_control = control

    def set_requested_position(self, actuator: str, position_rad: float) -> None:
        """Set a joint-output request without relying on a viewer slider edit."""
        if actuator not in self.references:
            raise KeyError(f"unknown scheduled actuator {actuator!r}")
        if not math.isfinite(position_rad):
            raise ValueError("requested position must be finite")
        self.references[actuator].requested_position_rad = float(position_rad)

    def advance(self, data: mujoco.MjData) -> None:
        """Advance previously set requests, without reading a new control value."""
        current_time = float(data.time)
        if self.last_time is None:
            self.reset(data)
        if current_time > self.last_time + EPSILON:
            dt = min(current_time - self.last_time, 0.05)
            for reference in self.references.values():
                self._advance(reference, dt)
            self.last_time = current_time
        for reference in self.references.values():
            control = reference.gear * reference.position_rad
            data.ctrl[reference.actuator_id] = control
            reference.last_applied_control = control

    def requested_position(self, actuator: str) -> float:
        return self.references[actuator].requested_position_rad

    def reference_position(self, actuator: str) -> float:
        return self.references[actuator].position_rad

    def reference_velocity(self, actuator: str) -> float:
        return self.references[actuator].velocity_rad_s
