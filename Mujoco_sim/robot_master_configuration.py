"""Shared, human-editable poses and limits for the robot_master model."""

from __future__ import annotations


# The named MuJoCo keyframe used when the real arm is at its repeatable zero
# configuration. These are actuator-control coordinates: AK40's value is on
# the motor side of its 1.25:1 reduction, while the other three are joint-side.
# Edit this one block to change the common simulation/hardware starting pose.
HARDWARE_ZERO_KEYFRAME_NAME = "hardware_zero"

# revolute_2 (shoulder) uses a reversed joint coordinate: the URDF hard stop at
# +0.153038 rad becomes 0 and travel away from the stop is positive, so the
# viewer slider starts at the stop.  revolute_2 = REF - urdf_revolute_2, which
# the MJCF expresses as a flipped hinge axis with ref=REF (identical poses).
REVOLUTE_2_REVERSED_REF_RAD = 0.153038

HARDWARE_ZERO_ACTUATOR_CTRL_RAD = {
    "ak60_revolute_1": 1.45,
    "ak70_revolute_2": REVOLUTE_2_REVERSED_REF_RAD - 0.113,  # 0.113 rad in the URDF frame
    "ak70_revolute_3": -1.29,
    "ak40_revolute_4": 1.94,
}

# STM32 feedback readings at the same home configuration. M4 reports the
# motor-side coordinate, matching its actuator control above.  M2's encoder is
# unaffected by the reversed sim coordinate, so it keeps its original value.
HARDWARE_ZERO_ENCODER_RAD = {
    "revolute_1": HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak60_revolute_1"],
    "revolute_2": 0.113,
    "revolute_3": HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak70_revolute_3"],
    "revolute_4": HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak40_revolute_4"],
}

# M4 startup is expressed in raw motor-encoder coordinates. The controller
# treats the live M4 reading at control activation as software zero, then uses
# its normal trapezoidal profile to move by this positive encoder delta.
M4_STARTUP_ENCODER_DELTA_RAD = 0.5

# M1 uses the same startup convention. Its 1:1 transmission means this is
# also a +10 degree joint move.
#M1_STARTUP_ENCODER_DELTA_RAD = 0.17453292519943295

# AK40 motor-side torque safety limit. This is the torque coordinate used by
# the serial MIT command and by the MJCF actuator force range. Its 1.25:1
# reduction makes the corresponding joint-output limit 2.5 N m.
AK40_MOTOR_TORQUE_LIMIT_NM = 1.0

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

# Per-actuator overrides of the position error above (joint side).  At 1 rad
# the AK40's cap was 1.25 N m/rad joint (0.8 motor side), far below the 4.41
# its schedule requests, and with ZERO_TFF the ~0.2 N m motor-side linkage load
# plus friction held M4 still (run 2026-10-27 21-31-36: a 0.136 rad motor-side
# command moved it 0.0015 rad).  At 0.25 rad the cap is 5.0 joint, so the
# scheduled gain (~2.8 N m/rad motor side) is used; the motor reaches its
# 1.0 N m limit at about 0.35 rad motor-side error.
SCHEDULED_PD_POSITION_ERROR_FOR_FULL_TORQUE_OVERRIDES_RAD = {
    "ak40_revolute_4": 0.25,
}

# Native MuJoCo Control-panel commands pass through a joint-space trapezoidal
# reference generator before reaching the position actuators.  These values
# are deliberately easy to tune without changing controller source code.
TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S = 1.0
TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2 = 2.0

# AK40 is specified at the motor encoder rather than joint output. With its
# 1.25:1 reduction, the limiter converts this to 0.4 rad/s joint-side.
AK40_MAX_MOTOR_VELOCITY_RAD_S = 0.5

# Permit the requested 1.94 rad AK40 motor reference: 1.94 / 1.25 = 1.552 rad
# at the joint. The coupled follower gets the matching lower bound.
JOINT_RANGE_OVERRIDES_RAD = {
    "revolute_4": (-0.00381777, HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak40_revolute_4"] / 1.25),
    "revolute_5": (-HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak40_revolute_4"] / 1.25, 0.0156052),
}

def hardware_zero_joint_positions(joint_names: list[str]) -> list[float]:
    """Return the physically consistent zero-qpos keyframe in joint-ID order."""
    return [0.0 for _ in joint_names]
