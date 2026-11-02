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

# MJCF position actuators for the MG995 end-effector servos (hand-added to
# robot_master.xml).  They are hobby PWM servos driven by the STM32, not
# CubeMars motors, so gain-schedule training and the MIT controller skip them.
END_EFFECTOR_SERVO_ACTUATORS = ("mg995_revolute_12", "mg995_gripper_pinch")

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

# Reference shape within those limits: "cubic" (cubic_spline_motion.py, one
# cubic per request, acceleration ramps instead of switching) or
# "trapezoidal" (trapezoidal_motion.py).  The hardware controller's limiter and
# the IK path-clearance check both follow this, so they agree on the path.
MOTION_PROFILE = "cubic"

# AK40 is specified at the motor encoder rather than joint output. With its
# 1.25:1 reduction, the limiter converts this to 0.4 rad/s joint-side.
AK40_MAX_MOTOR_VELOCITY_RAD_S = 0.5

# Permit the requested 1.94 rad AK40 motor reference: 1.94 / 1.25 = 1.552 rad
# at the joint. The coupled follower gets the matching lower bound.
JOINT_RANGE_OVERRIDES_RAD = {
    "revolute_4": (-0.00381777, HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak40_revolute_4"] / 1.25),
    "revolute_5": (-HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak40_revolute_4"] / 1.25, 0.0156052),
}

# Inverse kinematics (inverse_kinematics.py, terminal command "ik x y z").
# Base frame: the frame of the fixed base housing body.  Its origin is on the
# revolute_1 (base yaw) axis at the housing's underside, 7.0 mm above the
# table top (the top of the 7 mm base plate); its axes are the table/world
# axes, +z up.  At hardware_zero the tool point is at about
# (+0.023, +0.267, +0.279) m in this frame.
IK_BASE_BODY = "part_1_11"
# The table plate; its top face is the clearance reference.
IK_TABLE_BODY = "part_1_13"
# Tool point: midway between the two soft-finger tips (the finger mesh
# vertices farthest along the gripper's pointing axis, local -z), in the frame
# of the gripper body, which moves rigidly with the fingers.
IK_TOOL_BODY = "part_2_6"
IK_TOOL_POINT_IN_BODY_M = (-0.00587, -0.00500, -0.09676)
# A solution is accepted only if the tool point lands within this distance of
# the target (model kinematics; tracking error on the real arm comes on top).
IK_ACCEPT_ERROR_M = 1e-4
# Every moving mesh vertex must stay at least this far above the table top,
# at the target and along the joint-space path to it.
IK_TABLE_CLEARANCE_M = 0.01
# Bodies left out of that check.  proximal_limb__1_ is a stray M3x10 screw
# from the CAD export: its URDF visual origin (0.2526, -0.1491, 0.1277) is an
# absolute assembly position applied relative to the shoulder link, so it
# floats ~22 cm from any other part and swings through the table plane
# whenever the shoulder moves.  The real arm has nothing there.
IK_CLEARANCE_IGNORE_BODIES = ("proximal_limb__1_",)
# Largest change of any one joint (joint side) that a single "ik" command may
# make.  The model has no self-collision geometry and revolute_1/revolute_3
# are unlimited, so a large reconfiguration (e.g. lowering the tool from
# hardware_zero, where the shoulder is on its hard stop, gives a ~180 degree
# elbow flip) cannot be checked for the arm hitting itself.  Moves at and above
# the hardware_zero height within +-10 cm need at most ~41 degrees.
IK_MAX_JOINT_CHANGE_RAD = 1.0

def hardware_zero_joint_positions(joint_names: list[str]) -> list[float]:
    """Return the physically consistent zero-qpos keyframe in joint-ID order."""
    return [0.0 for _ in joint_names]
