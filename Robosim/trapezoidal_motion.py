"""Joint-space motion references for native MuJoCo Control-panel commands.

Provides both legacy trapezoidal standalone helpers (used by training scripts)
and a cubic-spline reference limiter for real-time smooth trajectory generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import mujoco

from master_controller import MasterController, actuator_gear
from robot_master_configuration import (
    AK40_MAX_MOTOR_VELOCITY_RAD_S,
    TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2,
    TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S,
)


EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Legacy trapezoidal standalone helpers (used by train_multidof_master_controller)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cubic spline reference limiter
# ---------------------------------------------------------------------------

@dataclass
class CubicSplineReference:
    """One cubic-spline smoothed joint-space reference."""

    actuator_id: int
    gear: float
    position_rad: float
    velocity_rad_s: float
    requested_position_rad: float
    last_applied_control: float
    max_velocity_rad_s: float
    max_acceleration_rad_s2: float


def _cubic_duration(
    d: float,
    v0: float,
    max_vel: float,
    max_acc: float,
) -> float:
    """Compute minimum duration for a cubic from (p0, v0) to (p0+d, 0).

    The cubic p(t) = p0 + v0*t + a2*t^2 + a3*t^3 satisfies:
        p(0) = p0, p'(0) = v0, p(T) = p0+d, p'(T) = 0
    Duration T is chosen so peak velocity <= max_vel and peak |accel| <= max_acc.
    """
    dist = abs(d)
    if dist < EPSILON and abs(v0) < EPSILON:
        return 0.0

    if dist < EPSILON:
        T = 2.0 * abs(v0) / max_acc
    else:
        T = max(1.5 * dist / max_vel, math.sqrt(6.0 * dist / max_acc))

    if abs(v0) > EPSILON:
        T = max(T, 2.0 * abs(v0) / max_acc)

    T = max(T, 1e-6)

    for _ in range(8):
        T2 = T * T
        T3 = T2 * T
        a2 = (3.0 * d - 2.0 * v0 * T) / T2
        a3 = (v0 * T - 2.0 * d) / T3

        peak_acc = max(abs(2.0 * a2), abs(2.0 * a2 + 6.0 * a3 * T))

        peak_vel = abs(v0)
        if abs(a3) > EPSILON:
            t_crit = -a2 / (3.0 * a3)
            if 0.0 < t_crit < T:
                v_crit = abs(v0 + 2.0 * a2 * t_crit + 3.0 * a3 * t_crit * t_crit)
                peak_vel = max(peak_vel, v_crit)

        scale = 1.0
        if peak_vel > max_vel * 1.001:
            scale = max(scale, peak_vel / max_vel)
        if peak_acc > max_acc * 1.001:
            scale = max(scale, math.sqrt(peak_acc / max_acc))
        if scale <= 1.001:
            break
        T *= scale

    return T


class CubicSplineReferenceLimiter:
    """Convert Control-panel jumps into cubic-spline smoothed joint references.

    Each simulation step, the limiter plans a cubic polynomial from the current
    (position, velocity) to the target (requested_position, 0) and evaluates one
    time step forward.  This gives C1-continuous motion with smooth acceleration,
    unlike the trapezoidal profile which has discontinuous acceleration.
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
        self.references: dict[str, CubicSplineReference] = {}
        for schedule in controller.schedules:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, schedule.actuator)
            if actuator_id < 0:
                raise ValueError(f"scheduled actuator {schedule.actuator!r} is absent from this MJCF")
            gear = actuator_gear(model, actuator_id)
            reference_max_velocity = self.max_velocity_rad_s
            if schedule.actuator == "ak40_revolute_4":
                reference_max_velocity = AK40_MAX_MOTOR_VELOCITY_RAD_S / abs(gear)
            self.references[schedule.actuator] = CubicSplineReference(
                actuator_id,
                gear,
                0.0,
                0.0,
                0.0,
                0.0,
                reference_max_velocity,
                self.max_acceleration_rad_s2,
            )
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

    def _advance(self, reference: CubicSplineReference, dt: float) -> None:
        d = reference.requested_position_rad - reference.position_rad
        v0 = reference.velocity_rad_s
        if abs(d) <= EPSILON and abs(v0) <= EPSILON:
            reference.position_rad = reference.requested_position_rad
            reference.velocity_rad_s = 0.0
            return

        T = _cubic_duration(d, v0, reference.max_velocity_rad_s, reference.max_acceleration_rad_s2)

        if T <= dt:
            reference.position_rad = reference.requested_position_rad
            reference.velocity_rad_s = 0.0
            return

        T2 = T * T
        T3 = T2 * T
        a2 = (3.0 * d - 2.0 * v0 * T) / T2
        a3 = (v0 * T - 2.0 * d) / T3

        dt2 = dt * dt
        reference.position_rad += v0 * dt + a2 * dt2 + a3 * dt2 * dt
        reference.velocity_rad_s = v0 + 2.0 * a2 * dt + 3.0 * a3 * dt2

    def update(self, data: mujoco.MjData) -> None:
        """Capture user controls, advance once per simulation time step, write references."""
        current_time = float(data.time)
        if self.last_time is None:
            self.reset(data)
        for reference in self.references.values():
            incoming_control = float(data.ctrl[reference.actuator_id])
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


# Backwards-compatible alias
TrapezoidalReferenceLimiter = CubicSplineReferenceLimiter
