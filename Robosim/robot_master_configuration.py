"""Shared, human-editable poses and limits for the robot_master model."""

from __future__ import annotations


# The named MuJoCo keyframe used when the real arm is at its repeatable zero
# configuration. These are actuator-control coordinates: AK40's value is on
# the motor side of its 1.25:1 reduction, while the other three are joint-side.
# Edit this one block to change the common simulation/hardware starting pose.
HARDWARE_ZERO_KEYFRAME_NAME = "hardware_zero"
HARDWARE_ZERO_ACTUATOR_CTRL_RAD = {
    "ak60_revolute_1": 1.45,
    "ak70_revolute_2": 0.113,
    "ak70_revolute_3": -1.29,
    "ak40_revolute_4": 1.94,
}

# STM32 feedback readings at the same home configuration. M4 reports the
# motor-side coordinate, matching its actuator control above.
HARDWARE_ZERO_ENCODER_RAD = {
    "revolute_1": HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak60_revolute_1"],
    "revolute_2": HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak70_revolute_2"],
    "revolute_3": HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak70_revolute_3"],
    "revolute_4": HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak40_revolute_4"],
}

# Scheduled PD safety limits.  ``SCHEDULED_PD_POSITION_ERROR_FOR_FULL_TORQUE_RAD``
# converts each actuator's output-torque limit into a maximum usable position
# stiffness: kp_max = torque_limit / position_error.  This prevents a large
# configuration-dependent inertia estimate from producing impractically high
# stiffness requests (for example, 200+ N m/rad on the 9 N m AK60).
# A gain is a torque-per-angle value, so comparing it directly to a motor's
# torque limit is not dimensionally meaningful.  By selecting one radian here,
# each position-gain cap numerically equals that actuator's output torque limit
# in N m/rad.  The MuJoCo force range remains the non-negotiable torque cap.
SCHEDULED_PD_POSITION_ERROR_FOR_FULL_TORQUE_RAD = 1.0
SCHEDULED_PD_VELOCITY_ERROR_FOR_FULL_TORQUE_RAD_S = 1.0

# Native MuJoCo Control-panel commands pass through a joint-space trapezoidal
# reference generator before reaching the position actuators.  These values
# are deliberately easy to tune without changing controller source code.
TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S = 1.0
TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2 = 2.0

# Permit the requested 1.94 rad AK40 motor reference: 1.94 / 1.25 = 1.552 rad
# at the joint. The coupled follower gets the matching lower bound.
JOINT_RANGE_OVERRIDES_RAD = {
    "revolute_4": (-0.00381777, HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak40_revolute_4"] / 1.25),
    "revolute_5": (-HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak40_revolute_4"] / 1.25, 0.0156052),
}

def hardware_zero_joint_positions(joint_names: list[str]) -> list[float]:
    """Return the physically consistent zero-qpos keyframe in joint-ID order."""
    return [0.0 for _ in joint_names]
