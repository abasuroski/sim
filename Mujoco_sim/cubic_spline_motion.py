"""Joint-space cubic-spline references for native MuJoCo Control-panel commands.

Each new request is planned as one cubic

    p(t) = p0 + v0*t + a2*t^2 + a3*t^3,   p(0) = p0, p'(0) = v0, p(T) = target, p'(T) = 0

starting from the reference's current position and velocity, with the
shortest duration T for which |velocity| and |acceleration| stay within the
same limits the trapezoidal limiter uses.  The cubic is then followed in time
until the request changes again.  Position and velocity are continuous across
replans (the trapezoid's velocity is too, but its acceleration switches
between +a_max, 0 and -a_max); acceleration ramps linearly inside a segment
and only steps at its start and end.

For the same limits a rest-to-rest cubic is slower than a trapezoid:
T = max(1.5 d / v_max, sqrt(6 d / a_max)) against d / v_max + v_max / a_max
(1.73 s vs 1.5 s for 1 rad at 1 rad/s, 2 rad/s^2).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco

from master_controller import MasterController
from robot_master_configuration import TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2, TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S
from trapezoidal_motion import EPSILON, TrapezoidalReference, TrapezoidalReferenceLimiter


# Relative slack on the limit check, so a duration found by bisection is not
# rejected for rounding.
_LIMIT_TOLERANCE = 1e-9


def cubic_duration(
    distance_rad: float,
    max_velocity_rad_s: float = TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S,
    max_acceleration_rad_s2: float = TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2,
) -> float:
    """Return the minimum rest-to-rest time for a scalar displacement.

    A rest-to-rest cubic peaks at 1.5 d/T velocity (mid-move) and 6 d/T^2
    acceleration (at both ends)."""
    distance = abs(float(distance_rad))
    if distance <= EPSILON:
        return 0.0
    if max_velocity_rad_s <= 0.0 or max_acceleration_rad_s2 <= 0.0:
        raise ValueError("velocity and acceleration limits must be greater than zero")
    return max(1.5 * distance / max_velocity_rad_s, math.sqrt(6.0 * distance / max_acceleration_rad_s2))


def cubic_position(
    initial_rad: float,
    target_rad: float,
    elapsed_s: float,
    max_velocity_rad_s: float = TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S,
    max_acceleration_rad_s2: float = TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2,
) -> float:
    """Return a rest-to-rest, velocity- and acceleration-limited cubic position."""
    displacement = float(target_rad) - float(initial_rad)
    duration = cubic_duration(displacement, max_velocity_rad_s, max_acceleration_rad_s2)
    if duration <= 0.0 or elapsed_s <= 0.0:
        return float(initial_rad)
    s = min(elapsed_s / duration, 1.0)
    return float(initial_rad) + displacement * s * s * (3.0 - 2.0 * s)


def _coefficients(distance_rad: float, velocity_rad_s: float, duration_s: float) -> tuple[float, float]:
    """a2, a3 of the cubic from (0, v0) to (distance, 0) in duration_s."""
    t2 = duration_s * duration_s
    a2 = (3.0 * distance_rad - 2.0 * velocity_rad_s * duration_s) / t2
    a3 = (velocity_rad_s * duration_s - 2.0 * distance_rad) / (t2 * duration_s)
    return a2, a3


def _within_limits(
    distance_rad: float,
    velocity_rad_s: float,
    duration_s: float,
    max_velocity_rad_s: float,
    max_acceleration_rad_s2: float,
) -> bool:
    a2, a3 = _coefficients(distance_rad, velocity_rad_s, duration_s)
    # Acceleration is linear in t: its peak is at an end.
    peak_acceleration = max(abs(2.0 * a2), abs(2.0 * a2 + 6.0 * a3 * duration_s))
    if peak_acceleration > max_acceleration_rad_s2 * (1.0 + _LIMIT_TOLERANCE):
        return False
    # Velocity is quadratic in t: its peak is at an end (|v0|, 0) or the vertex.
    peak_velocity = abs(velocity_rad_s)
    if abs(a3) > EPSILON:
        t_vertex = -a2 / (3.0 * a3)
        if 0.0 < t_vertex < duration_s:
            peak_velocity = max(
                peak_velocity, abs(velocity_rad_s + 2.0 * a2 * t_vertex + 3.0 * a3 * t_vertex * t_vertex)
            )
    return peak_velocity <= max_velocity_rad_s * (1.0 + _LIMIT_TOLERANCE)


def minimum_cubic_duration(
    distance_rad: float,
    velocity_rad_s: float,
    max_velocity_rad_s: float,
    max_acceleration_rad_s2: float,
) -> float:
    """Shortest duration (to bisection precision) of a cubic from (0, v0) to
    (distance, 0) within both limits.  Always returns a duration that is
    within the limits when |v0| <= max_velocity: as T grows the acceleration
    tends to 0 and the velocity to v0 * (1 - 4s + 3s^2), which stays within |v0|."""
    if abs(velocity_rad_s) <= EPSILON:
        return cubic_duration(distance_rad, max_velocity_rad_s, max_acceleration_rad_s2)
    # Both are necessary: stop from v0, and cover the distance at <= v_max.
    low = max(abs(velocity_rad_s) / max_acceleration_rad_s2, abs(distance_rad) / max_velocity_rad_s, 1e-4)
    if _within_limits(distance_rad, velocity_rad_s, low, max_velocity_rad_s, max_acceleration_rad_s2):
        return low
    high = low
    for _ in range(80):
        high *= 1.25
        if _within_limits(distance_rad, velocity_rad_s, high, max_velocity_rad_s, max_acceleration_rad_s2):
            break
        low = high
    else:
        return high  # |v0| above the limit; give the gentlest cubic tried.
    for _ in range(40):
        middle = 0.5 * (low + high)
        if _within_limits(distance_rad, velocity_rad_s, middle, max_velocity_rad_s, max_acceleration_rad_s2):
            high = middle
        else:
            low = middle
    return high


@dataclass
class CubicSegment:
    """The cubic being followed by one reference (joint side)."""

    start_position_rad: float
    start_velocity_rad_s: float
    target_rad: float
    duration_s: float
    a2: float
    a3: float
    elapsed_s: float = 0.0


class CubicSplineReferenceLimiter(TrapezoidalReferenceLimiter):
    """Convert Control-panel jumps into cubic-spline joint references.

    Same interface and ``data.ctrl`` sharing as ``TrapezoidalReferenceLimiter``;
    only how the reference advances differs.  A request change (slider edit,
    goal, home) replans from the current position and velocity.  Requests and
    references are clamped to the actuator's ctrlrange where it is limited, so
    a replan that has to overshoot (stopping from speed close to the new
    target) cannot carry the reference past M2's or M4's hard stop.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        controller: MasterController,
        max_velocity_rad_s: float = TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S,
        max_acceleration_rad_s2: float = TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2,
    ) -> None:
        super().__init__(model, controller, max_velocity_rad_s, max_acceleration_rad_s2)
        self.segments: dict[int, CubicSegment | None] = {}
        # Joint-side position bounds per actuator (ctrl = gear * joint).
        self.bounds_rad: dict[int, tuple[float, float]] = {}
        for reference in self.references.values():
            self.segments[reference.actuator_id] = None
            if model.actuator_ctrllimited[reference.actuator_id]:
                low, high = (float(v) / reference.gear for v in model.actuator_ctrlrange[reference.actuator_id])
                self.bounds_rad[reference.actuator_id] = (min(low, high), max(low, high))
            else:
                self.bounds_rad[reference.actuator_id] = (-math.inf, math.inf)

    def reset(self, data: mujoco.MjData) -> None:
        super().reset(data)
        for actuator_id in self.segments:
            self.segments[actuator_id] = None

    def _plan(self, reference: TrapezoidalReference, target_rad: float) -> CubicSegment:
        distance = target_rad - reference.position_rad
        velocity = reference.velocity_rad_s
        duration = minimum_cubic_duration(distance, velocity, self.max_velocity_rad_s, self.max_acceleration_rad_s2)
        a2, a3 = _coefficients(distance, velocity, duration)
        return CubicSegment(reference.position_rad, velocity, target_rad, duration, a2, a3)

    def _advance(self, reference: TrapezoidalReference, dt: float) -> None:
        low, high = self.bounds_rad[reference.actuator_id]
        target = min(max(reference.requested_position_rad, low), high)
        segment = self.segments[reference.actuator_id]
        if segment is None or abs(segment.target_rad - target) > EPSILON:
            if abs(target - reference.position_rad) <= EPSILON and abs(reference.velocity_rad_s) <= EPSILON:
                reference.position_rad = target
                reference.velocity_rad_s = 0.0
                self.segments[reference.actuator_id] = None
                return
            segment = self._plan(reference, target)
            self.segments[reference.actuator_id] = segment
        segment.elapsed_s += dt
        if segment.elapsed_s >= segment.duration_s:
            reference.position_rad = segment.target_rad
            reference.velocity_rad_s = 0.0
            self.segments[reference.actuator_id] = None
            return
        t = segment.elapsed_s
        position = segment.start_position_rad + t * (segment.start_velocity_rad_s + t * (segment.a2 + t * segment.a3))
        velocity = segment.start_velocity_rad_s + t * (2.0 * segment.a2 + 3.0 * segment.a3 * t)
        if position < low or position > high:
            # Only an overshooting replan reaches here; stop at the bound.
            position = min(max(position, low), high)
            velocity = 0.0
            self.segments[reference.actuator_id] = None
        reference.position_rad = position
        reference.velocity_rad_s = velocity
