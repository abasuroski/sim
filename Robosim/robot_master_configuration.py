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

# Permit the requested 1.94 rad AK40 motor reference: 1.94 / 1.25 = 1.552 rad
# at the joint. The coupled follower gets the matching lower bound.
JOINT_RANGE_OVERRIDES_RAD = {
    "revolute_4": (-0.00381777, HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak40_revolute_4"] / 1.25),
    "revolute_5": (-HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak40_revolute_4"] / 1.25, 0.0156052),
}

def hardware_zero_joint_positions(joint_names: list[str]) -> list[float]:
    """Return the physically consistent zero-qpos keyframe in joint-ID order."""
    return [0.0 for _ in joint_names]
