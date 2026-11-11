#!/usr/bin/env python3
"""
hardware_controller.py — Closed-loop control of the physical robot arm.

Combines:
  - Serial encoder feedback from the STM32 (same format as sync_hardware.py)
  - Position-scheduled PD gains from master_controller.py
  - Cubic-spline (or trapezoidal) motion references, chosen by MOTION_PROFILE
    in robot_master_configuration.py
  - Gravity-compensation feedforward via mujoco.mj_inverse
  - MIT control mode command sender (P, V, K, D, T over serial)

Control loop (runs at UPDATE_HZ):
  1. Read latest encoder feedback (pos, vel) from serial thread
  2. Write encoder state into MuJoCo data.qpos / data.qvel
  3. Run mj_forward to update kinematics and mass matrix
  4. MasterController.update() schedules kp/kd from inertia + position
  5. The reference limiter's update() steps the reference trajectory
  6. mj_inverse computes gravity-compensation feedforward torques
  7. Pack and send MIT control frame per motor: P, V, Kp, Kd, tau_ff

Usage:
    python3 hardware_integrated_main_controller.py [serial_port] [--enable-motors] [--gui]

    --gui opens a control panel (control_panel.py): enable / start / pause /
    hold / disable buttons, and tool-point targets that are solved and
    previewed in the MuJoCo viewer (orange ghost) before they are run.

Runaway stop (RUNAWAY_STOP): a motor that leaves its command, moves too fast
or stops reporting triggers 0X and returns to IDLE; type 'a' then 's' again.
With TX_ECHO firmware, a CAN command echo that disagrees with what the PC
sent (ECHO_CHECK) trips the same stop, and so does a motor reporting a
nonzero fault code (MOTOR_FAULT_STOP).

Startup workflow:
    1. Motors must already be enabled (use motor_gui.py or type 'e' here).
    2. Viewer opens showing current sim state.
    3. Type  a  and press Enter to arm — enables all body motors.
    4. Type  s  and press Enter to start the control loop.
    5. Use  g <j> <rad>  to command a joint to a position, e.g.  g 1 0.5
    6. Type  p  to pause (stop sending commands), r  to resume, q  to quit.

Commands (type in terminal, press Enter):
    a            — enable all body motors
    s            — start control loop
    g <j> <rad>  — go: move joint j (1-4) to position in radians
    ik <x> <y> <z> — move the gripper tool point to a base-frame position (m,
                   or mm/cm suffixes), or to the nearest location found if it
                   is out of reach; see inverse_kinematics.py
    w            — print the tool point position (base frame)
    state [gripping|pushing] — print or change the end-effector state
    h            — go to hardware_zero pose
    p            — pause control
    r            — resume control
    z            — print current encoder readings
    q            — quit

Serial protocol (to STM32, see serial_interface.py):
    <n>M<pos>,<vel>,<kp>,<kd>,<tff>*HH\\n  — full MIT frame, applied atomically
    <n>E*HH\\n                             — enable motor

Requires: mujoco, pyserial, numpy
"""

from __future__ import annotations

import atexit
import csv
import enum
import json
import math
import platform
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from robot_master_configuration import (
    IK_TABLE_CLEARANCE_M,
    PUSHING_TOOL_JOINT,
    PUSHING_WRIST_SERVO_DEG,
    HARDWARE_ZERO_KEYFRAME_NAME,
    HARDWARE_ZERO_ACTUATOR_CTRL_RAD,
    MOTION_PROFILE,
)
from control_panel import ControlPanel, MotorRow, PanelStatus, ServoRow
from cubic_spline_motion import CubicSplineReferenceLimiter, cubic_duration, cubic_position
from inverse_kinematics import ArmIK, IKResult
from master_controller import MasterController, PositionSchedule
from trapezoidal_motion import TrapezoidalReferenceLimiter, trapezoidal_duration, trapezoidal_position
from plot_run_log import plot_run_log
import serial_interface as si

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "robot_master_urdf_table" / "mjcf" / "robot_master.xml"
SCHEDULE_PATH = Path(__file__).parent / "master_controller_schedule.json"

UPDATE_HZ  = 84   # control loop rate (Hz)
SEND_EVERY = 1    # send commands every N control ticks (1 = every tick)

# Set to False to run open-loop: the sim tracks the reference trajectory
# instead of encoder feedback. Commands still go to the real motors.
# Use this first to verify commands are sensible before enabling feedback.
CLOSED_LOOP = True

# Print every MIT frame sent to the STM32 (once per second per motor).
# Toggle at runtime by typing 'd' in the terminal.
DEBUG_COMMANDS = True

# Print controller telemetry in the terminal.  The encoder is displayed both
# in its raw STM32 coordinate and, after synchronization, projected into the
# same joint coordinate as MuJoCo before calculating the position error.
# Toggle at runtime by typing 'f'.
PRINT_MOTOR_FEEDBACK = True
MOTOR_FEEDBACK_PRINT_HZ = 4.0

# STM32 lines that are not motor feedback (faults, "feedback lost", unknown
# CAN ids) are always written to the run log; at most this many per second
# are also printed as [stm32] so a chatty firmware cannot flood the terminal.
STM32_PRINT_MAX_PER_S = 5

# Set to True to send zero torque feedforward to all motors.
# Useful for initial testing before verifying TFF sign and magnitude.
ZERO_TFF = True

# Every run writes logs/<branch>_<YYYY-MM-DD>_<HH-MM-SS UTC>.csv (one row per
# control tick, flushed immediately), a matching .json with the run's
# settings and git revision, and a .svg graph (plot_run_log.py) written when
# the run ends.  The controller refuses to start without a log.
CYCLE_LOG_DIRECTORY = Path(__file__).parent / "logs"

# Live slider streaming.  A motor gets a frame only when one of its fields
# changed by more than these thresholds (during a slider move), plus a
# keepalive frame every KEEPALIVE_S.  Each frame is one checksummed
# "<n>M<pos>,<vel>,<kp>,<kd>,<tff>" line (~45 bytes) that the firmware
# applies whole or not at all: separate per-field lines could arrive in part,
# as on 2026-10-26 when an overrun dropped 3D/3T but kept 3P/3V.
STREAM_POS_EPS_RAD   = 1e-4   # resend P when it moves more than this (motor side)
STREAM_VEL_EPS_RAD_S = 1e-3   # resend V when it changes more than this
STREAM_GAIN_REL_EPS  = 0.02   # resend K/D on a >2% change
STREAM_TFF_EPS_NM    = 0.01   # resend T when it changes more than this
STREAM_KEEPALIVE_S   = 0.25   # full frame per motor at least this often

# End-effector MG995 servos on the STM32's TIM2 PWM.  Positions are servo
# angles, 0-180 deg over 500-2500 us (the firmware header's MG995 mapping);
# 90 deg is centre.  The firmware parks each servo at SERVO_PARK_DEG at boot,
# which must match its own values.  The firmware's own
# servo unit is degrees from centre at 8.333 us/deg (+/-120 = 500-2500 us), so
# angle S goes out as "<address>P<(S - 90) * 4/3>".  The servos report nothing,
# and are independent of the CubeMars motors: whenever the serial port is
# open, a slider move (control panel, or the wrist's MuJoCo Control-tab
# slider) is sent at once, with or without 'a'/'s'.  Nothing is sent until a
# slider moves, so connecting never moves a servo.
SERVOS = True
SERVO_CENTER_DEG = 90.0
# name: (lowest, highest) servo angle.  The gripper must not go below 156 deg.
# Its top end is 230 deg, 50 deg past the MG995's rated 180 deg: about 3056 us,
# outside the 500-2500 us spec, so the servo may stall against its internal
# stop up there (requested 2026-11-11).  The sim shows anything past 180 as
# fully closed.  The firmware enforces the same gripper limits
# (SERVO_GRIPPER_CMD_MIN/MAX in uart_cmd.h), so change both together.
SERVO_LIMITS_DEG = {
    "wrist":   (0.0, 180.0),
    "gripper": (156.0, 230.0),
}
# Where the firmware parks each servo at boot; the gripper's is mid-range,
# clear of both ends of its travel (SERVO_GRIPPER_CMD_PARK in uart_cmd.h).
SERVO_PARK_DEG = {"wrist": 90.0, "gripper": 168.0}
SERVO_RESEND_DEG = 0.5     # send only when the angle changes by this much
# name: (panel label, firmware address, MuJoCo actuator mirroring it or None)
SERVO_CHANNELS = {
    "wrist":   ("Wrist (A0 / PA0)",   5, "mg995_revolute_12"),
    "gripper": ("Gripper (A1 / PA1)", 7, "mg995_gripper_pinch"),
}
# The wrist servo turns revolute_12, which the sim shows at the commanded
# angle.  Servo 90 deg is the middle of the CAD joint range
# (-1.79646 .. 0.489918, so -0.65327).  A +90 deg shift tried earlier left the
# sim 90 deg out of phase with the real wrist (2026-11-11), so it was undone.
# With direction -1 the servo's 0-180 deg spans -2.22407 .. 0.91753 rad; the
# MJCF range was widened by 90 deg at each end (-3.367256 .. 2.060714) to hold it.
WRIST_SERVO_JOINT = "revolute_12"
WRIST_SERVO_CENTER_JOINT_RAD = -0.65327   # revolute_12 at servo 90 deg
# -1: the real wrist turned opposite to the sim with +1 (2026-11-02).
WRIST_SERVO_DIRECTION = -1
# The gripper servo turns the pinch gear gripper_pinch 1:1 (its meshed idler,
# gripper_pinch_follower, turns opposite); positive joint angle closes, and
# the fingers touch at 0.0586 rad.  NOT CALIBRATED: slider 180 (the servo's
# full extension) is shown as fully closed, so raising the slider closes the
# gripper in the sim.  The gripper is never part of the IK.
GRIPPER_SERVO_JOINT = "gripper_pinch"
GRIPPER_SERVO_FOLLOWER = "gripper_pinch_follower"
GRIPPER_SERVO_DIRECTION = 1
GRIPPER_SERVO_CENTER_JOINT_RAD = 0.0586 - GRIPPER_SERVO_DIRECTION * math.pi / 2   # gripper_pinch at servo 90 deg
# name: (joint, joint angle at servo 90 deg, direction, opposite-turning follower or None)
SERVO_JOINTS = {
    "wrist":   (WRIST_SERVO_JOINT, WRIST_SERVO_CENTER_JOINT_RAD, WRIST_SERVO_DIRECTION, None),
    "gripper": (GRIPPER_SERVO_JOINT, GRIPPER_SERVO_CENTER_JOINT_RAD, GRIPPER_SERVO_DIRECTION, GRIPPER_SERVO_FOLLOWER),
}
# Solve the wrist angle in IK too (panel "Move tool point", terminal "ik"):
# revolute_12 becomes a fifth driven joint, limited to what the servo reaches
# (0-180 deg), and a run moves it alongside the arm on the same MOTION_PROFILE
# path the clearance check sampled.  Dragging the wrist slider cancels it.
WRIST_IK = True

# Runaway stop.  On 2026-10-21 M1 accelerated to 21 rad/s at 19-37 A while
# its command was constant, and the controller kept streaming for 10 s.  Any
# motor that had encoder feedback at sync is checked every tick; a trip sends
# 0X (STM32 freeze), stops streaming and returns to IDLE, so 'a' then 's'
# re-synchronizes before the arm moves again.  All values are motor side.
RUNAWAY_STOP = True
# |command - encoder| above this for RUNAWAY_POS_ERROR_S trips.  Normal
# tracking lag at 1 rad/s is ~0.26 rad on M1; M4's soft gains (Kp ~0.4 N·m/rad)
# leave a larger steady-state error under load, so its limit is wider.
RUNAWAY_POS_ERROR_RAD = {1: 0.5, 2: 0.5, 3: 0.5, 4: 1.0}
RUNAWAY_POS_ERROR_S   = 0.25
# Measured speed above this on RUNAWAY_SPEED_PACKETS consecutive feedback
# packets trips.  References are limited to 1 rad/s joint (1.25 rad/s motor
# side on M4).  M1 has shown 1-2 packet glitches up to 5.6 rad/s that
# recover on their own; the 2026-10-21 runaway gave 3 fast packets in 46 ms.
RUNAWAY_SPEED_RAD_S   = 3.0
RUNAWAY_SPEED_PACKETS = 3
# No new feedback packet for this long trips: the checks above are blind.
RUNAWAY_FEEDBACK_TIMEOUT_S = 0.5

# Command echo check (needs TX_ECHO=1 firmware).  On 2026-10-22 the STM32
# latched P = 0 that the PC never sent (a damaged serial line), and the
# encoder checks above only tripped 45 ms later, after a 17.5 A slam.  Each
# "[TX Mn]" echo is compared with what the PC sent; ECHO_MISMATCH_COUNT
# consecutive mismatching echoes trip the same stop.  Two, so one damaged
# echo line on the way back cannot trip it alone.
ECHO_CHECK = True
ECHO_MISMATCH_COUNT = 2
# An echo may show any value sent within this long before it arrived (serial
# and firmware-loop latency), or the value current before that.
ECHO_WINDOW_S = 0.15
# Allowed |echo - sent| per field, in CAN counts (si.command_resolution).
# The echo is the frame as packed for CAN: the firmware truncates to a count
# (up to 1 below) and the PC rounds what it sends (well under half a count).
ECHO_TOLERANCE_COUNTS = 2.0

# A motor that reports a nonzero CubeMars fault code (si.MOTOR_ERROR_TEXT)
# while control is active trips the same stop.  Fault codes are always
# logged (m<n>_encoder_err) and every change is a run-log event.
MOTOR_FAULT_STOP = True

# After a control-panel Disable, wait this long, then check that every motor's
# command echo shows zero gains (TX_ECHO heartbeats come every 250 ms).
DISABLE_ECHO_CHECK_S = 1.0

# Motor index → joint name (driven joints only)
MOTOR_TO_JOINT = {
    1: "revolute_1",
    2: "revolute_2",
    3: "revolute_3",
    4: "revolute_4",
}

# Actuator name → motor index (for command sending)
ACTUATOR_TO_MOTOR = {
    "ak60_revolute_1": 1,
    "ak70_revolute_2": 2,
    "ak70_revolute_3": 3,
    "ak40_revolute_4": 4,
}

# Motor-side scale factors (encoder → joint angle).
# M4 has a 1.25:1 reduction; negative = direction convention.
# Keep this mapping symmetric: it applies to feedback, position/velocity
# commands, and feedforward torque. Change M4 only after a low-risk direction
# test demonstrates that the physical encoder sign is opposite.
# M2 is -1 only because the MJCF revolute_2 coordinate is reversed (hard stop
# = 0, see REVOLUTE_2_REVERSED_REF_RAD); the physical motor sign is unchanged.
MOTOR_DIRECTION = {1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0}
MOTOR_SCALE = {1: 1.0, 2: MOTOR_DIRECTION[2], 3: 1.0, 4: MOTOR_DIRECTION[4] / 1.25}

# Constrained (passive) joints driven kinematically from revolute_4.
# Format: joint_name → (source_joint, scale)
CONSTRAINED_JOINTS = {
    "revolute_5":  ("revolute_4", -1.0),
    "revolute_7":  ("revolute_4",  1.0),
    "revolute_8":  ("revolute_4", -1.0),
    "revolute_9":  ("revolute_4", -1.0),
    "revolute_6":  ("revolute_4",  1.0),
    "revolute_13": ("revolute_4",  1.0),
}

# Torque feedforward scale per motor (sign + gear ratio applied to mj_inverse output).
# mj_inverse gives joint-side torques; M4 actuator is on the motor side (÷ gear).
TORQUE_FF_SCALE = {1: 1.0, 2: MOTOR_DIRECTION[2], 3: 1.0, 4: MOTOR_DIRECTION[4] / 1.25}

# Default PD gains used if no gain_schedule.json is found.
# These are joint-output values (N·m/rad and N·m·s/rad).
DEFAULT_GAINS = {
    "ak60_revolute_1": (9.0,  1.0),
    "ak70_revolute_2": (24.8, 1.5),
    "ak70_revolute_3": (24.8, 1.5),
    "ak40_revolute_4": (3.28, 0.3),
}

class EndEffectorState(enum.Enum):
    """What the end of the arm is doing; it decides the IK tool point.

    GRIPPING: tool point between the fingers, wrist free (solved by the IK).
    PUSHING: tool point on PUSHING_TOOL_JOINT, wrist held at PUSHING_WRIST_SERVO_DEG.
    """

    GRIPPING = "Gripping"
    PUSHING = "Pushing"


# ---------------------------------------------------------------------------
# Control state flags (written by terminal thread, read by main loop)
# ---------------------------------------------------------------------------

_control_active  = False   # True once 's' is pressed
_request_start   = False
_request_pause   = False
_paused          = False
_request_goal: dict[str, float] | None = None   # actuator → target (rad, joint side)
_ee_state = EndEffectorState.GRIPPING
_request_ee_state: EndEffectorState | None = None
_arm_iks: dict[EndEffectorState, ArmIK] = {}    # the IK of each available state
_request_home    = False
_request_rezero  = False
_request_start_home = False   # reset the reference to hardware_zero before the start sync
_request_hold    = False      # send 0X and stop streaming (control panel HOLD)
_request_disable = False      # stop streaming, then send zero-gain frames (limp)

# Motor power as far as this program knows: "not enabled", "enabling",
# "enabled", "limp (disabled)" or "held (0X)".  Shown by the control panel.
_motor_power = "not enabled"
# Latest event worth showing in the control panel: (text, is_alert).
_notice: tuple[str, bool] = ("", False)

# Snapshots captured at the moment 's' is pressed.
# All subsequent encoder readings are treated as deltas from these values,
# applied on top of the sim qpos at that instant.  This eliminates any
# constant offset between encoder zero and MuJoCo joint zero.
_encoder_at_sync: dict[int, float]  = {}   # {motor: encoder_rad}
_sim_at_sync:     dict[str, float]  = {}   # {joint_name: qpos_rad}
_pos_cmd_offset:  dict[int, float]  = {}   # {motor: encoder_rad - ref_pos_motor at sync}
_encoder_feedback_at_sync: set[int] = set()  # motors with a real encoder sample at sync

# Terminal/pipe commands and controller events, drained into the next log row.
_log_events: deque[str] = deque()

# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------

def _apply_encoder_state(
    data: mujoco.MjData,
    qpos_idx: dict[str, int],
    qvel_idx: dict[str, int],
    keyframe_qpos: dict[str, float],
    pos_snap: dict[int, float | None],
    vel_snap: dict[int, float],
) -> None:
    driven_qpos: dict[str, float] = {}

    for motor, jname in MOTOR_TO_JOINT.items():
        pos = pos_snap[motor]
        if pos is None or jname not in qpos_idx or motor not in _encoder_at_sync:
            driven_qpos[jname] = keyframe_qpos.get(jname, 0.0)
            continue

        scale = MOTOR_SCALE[motor]
        delta_encoder = pos - _encoder_at_sync[motor]
        joint_pos = _sim_at_sync[jname] + delta_encoder * scale
        data.qpos[qpos_idx[jname]] = joint_pos
        data.qvel[qvel_idx[jname]] = vel_snap[motor] * scale
        driven_qpos[jname] = joint_pos

    for follower, (source, fscale) in CONSTRAINED_JOINTS.items():
        if source in driven_qpos and follower in qpos_idx:
            source_sim_at_sync   = _sim_at_sync.get(source,   keyframe_qpos.get(source,   0.0))
            follower_sim_at_sync = _sim_at_sync.get(follower, keyframe_qpos.get(follower, 0.0))
            delta = driven_qpos[source] - source_sim_at_sync
            data.qpos[qpos_idx[follower]] = follower_sim_at_sync + delta * fscale
            if follower in qvel_idx and source in qvel_idx:
                data.qvel[qvel_idx[follower]] = data.qvel[qvel_idx[source]] * fscale


def _apply_reference_state(
    data: mujoco.MjData,
    qpos_idx: dict[str, int],
    qvel_idx: dict[str, int],
    keyframe_qpos: dict[str, float],
    limiter: "TrapezoidalReferenceLimiter",
) -> None:
    driven_qpos: dict[str, float] = {}
    for actuator, motor in ACTUATOR_TO_MOTOR.items():
        jname = MOTOR_TO_JOINT[motor]
        if jname not in qpos_idx:
            continue

        ref_pos = limiter.reference_position(actuator)
        data.qpos[qpos_idx[jname]] = ref_pos
        data.qvel[qvel_idx[jname]] = limiter.reference_velocity(actuator)
        driven_qpos[jname] = ref_pos

    for follower, (source, fscale) in CONSTRAINED_JOINTS.items():
        if source in driven_qpos and follower in qpos_idx:
            kf_source   = keyframe_qpos.get(source,   0.0)
            kf_follower = keyframe_qpos.get(follower, 0.0)
            delta = driven_qpos[source] - kf_source
            data.qpos[qpos_idx[follower]] = kf_follower + delta * fscale


def _compute_tau_ff(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_idx: dict[str, int],
    qvel_idx: dict[str, int],
) -> dict[int, float]:
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)

    tau_ff: dict[int, float] = {}
    for motor, jname in MOTOR_TO_JOINT.items():
        if jname not in qvel_idx:
            tau_ff[motor] = 0.0
            continue
        dof = qvel_idx[jname]
        tau_ff[motor] = float(data.qfrc_inverse[dof]) * TORQUE_FF_SCALE[motor]
    return tau_ff

class _CommandStreamer:
    """Send a motor's full MIT frame when any field changes, plus a keepalive."""

    def __init__(self) -> None:
        self._last: dict[int, dict[str, float]] = {}
        self._last_full: dict[int, float] = {}
        # Field letters written on each motor's latest send() ("" = nothing), for the cycle log.
        self.last_sent_fields: dict[int, str] = {}
        # Values written on each motor's latest send(), for the echo check.
        self.last_sent_values: dict[int, dict[str, float]] = {}

    def reset(self) -> None:
        self._last.clear()
        self._last_full.clear()
        self.last_sent_fields.clear()
        self.last_sent_values.clear()

    def send(self, motor: int, pos: float, vel: float, kp: float, kd: float, tau_ff: float) -> bool:
        """Return True if anything was written for this motor."""
        now = time.perf_counter()
        last = self._last.get(motor)
        full = last is None or now - self._last_full.get(motor, 0.0) >= STREAM_KEEPALIVE_S

        def changed(key: str, value: float, eps: float, relative: bool = False) -> bool:
            if full:
                return True
            tol = eps * max(abs(last[key]), 1e-6) if relative else eps
            return abs(value - last[key]) > tol

        fields = {
            "pos":    pos    if changed("pos", pos, STREAM_POS_EPS_RAD) else None,
            "vel":    vel    if changed("vel", vel, STREAM_VEL_EPS_RAD_S) else None,
            "kp":     kp     if changed("kp", kp, STREAM_GAIN_REL_EPS, relative=True) else None,
            "kd":     kd     if changed("kd", kd, STREAM_GAIN_REL_EPS, relative=True) else None,
            "tau_ff": tau_ff if changed("tau_ff", tau_ff, STREAM_TFF_EPS_NM) else None,
        }
        # When a move finishes, the exact final P and V=0 must reach the motor
        # even if they differ from the last sent values by less than eps.
        move_finished = last is not None and vel == 0.0 and last["vel"] != 0.0
        if all(v is None for v in fields.values()) and not move_finished:
            self.last_sent_fields[motor] = ""
            self.last_sent_values[motor] = {}
            return False

        # One atomic frame with every field, whichever ones changed.
        values = {"pos": pos, "vel": vel, "kp": kp, "kd": kd, "tau_ff": tau_ff}
        si.send_mit_frame(motor, **values)
        self.last_sent_fields[motor] = "PVKDT"
        self.last_sent_values[motor] = values
        self._last[motor] = dict(values)
        self._last_full[motor] = now
        return True


# Rest-to-rest path of MOTION_PROFILE, for the wrist's IK moves (the limiter
# runs the same profile for M1-M4, and ArmIK samples it for clearance).
_PROFILE_DURATION, _PROFILE_POSITION = {
    "cubic": (cubic_duration, cubic_position),
    "trapezoidal": (trapezoidal_duration, trapezoidal_position),
}[MOTION_PROFILE]

# Set in main() when the wrist takes part in IK (WRIST_IK); None otherwise.
_ik_wrist: "_Servo | None" = None


def _ik_joint_label(actuator: str) -> str:
    """'M<n>' for a CubeMars motor, 'wrist' for the servo joint."""
    return f"M{ACTUATOR_TO_MOTOR[actuator]}" if actuator in ACTUATOR_TO_MOTOR else "wrist"


def _ik_references(arm: dict[str, float], wrist: str = "reference") -> dict[str, float]:
    """IK references: the given M1-M4 values, plus the wrist's current
    ('reference') or destination ('goal') angle when it takes part in IK."""
    if _ik_wrist is None:
        return arm
    q = _ik_wrist.joint_reference_rad() if wrist == "reference" else _ik_wrist.joint_goal_rad()
    return {**arm, SERVO_CHANNELS["wrist"][2]: q}


def _servo_firmware_deg(servo_deg: float) -> float:
    """Servo angle (0-180, 500-2500 us) -> firmware command (deg from centre, 8.333 us/deg)."""
    return (servo_deg - SERVO_CENTER_DEG) * 4.0 / 3.0


def _servo_joint_rad(name: str, servo_deg: float) -> float:
    """Servo angle -> its joint's angle (SERVO_JOINTS; joint side, rad)."""
    _, center, direction, _ = SERVO_JOINTS[name]
    return center + direction * math.radians(servo_deg - SERVO_CENTER_DEG)


def _servo_deg_from_joint(name: str, joint_rad: float) -> float:
    _, center, direction, _ = SERVO_JOINTS[name]
    return SERVO_CENTER_DEG + direction * math.degrees(joint_rad - center)


def _wrist_joint_rad(servo_deg: float) -> float:
    return _servo_joint_rad("wrist", servo_deg)


class _Servo:
    """One end-effector servo (SERVO_CHANNELS).  ``target_deg`` is the slider
    value and ``commanded_deg`` what the servo was last sent (in simulation, the
    target itself); both are servo angles, within SERVO_LIMITS_DEG."""

    def __init__(self, model: mujoco.MjModel, name: str, label: str, address: int, actuator: str | None) -> None:
        self.name, self.label, self.address = name, label, address
        self.actuator_id = -1 if actuator is None else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        self.min_deg, self.max_deg = SERVO_LIMITS_DEG[name]
        self.park_deg = SERVO_PARK_DEG[name]   # where the firmware leaves it at boot
        self.target_deg = self.park_deg
        self.commanded_deg = self.park_deg
        self.last_sent_deg: float | None = None
        self._last_written_ctrl: float | None = None
        self._previous_target_deg: float | None = None
        # Timed move to an IK goal (the wrist only), in joint radians:
        # (start, goal, start time on data.time, duration).  data.time only
        # advances while control runs, so a pause or hold freezes it with the arm.
        self.move: tuple[float, float, float, float] | None = None
        self.locked = False     # sliders are ignored (the wrist in the Pushing state)
        self._move_limits = (1.0, 1.0)
        self._move_elapsed_s = 0.0

    def set_target(self, servo_deg: float) -> None:
        self.target_deg = round(max(self.min_deg, min(self.max_deg, float(servo_deg))), 1)

    def set_target_by_operator(self, servo_deg: float) -> None:
        """A slider edit: it wins over any IK move in progress, unless locked."""
        if self.locked:
            return
        self.move = None
        self.set_target(servo_deg)

    def joint_reference_rad(self) -> float:
        """Where this servo's joint reference is now (joint side, rad)."""
        if self.move is not None:
            return self._move_position(self._move_elapsed_s)
        return _servo_joint_rad(self.name, self.target_deg)

    def joint_goal_rad(self) -> float:
        return self.move[1] if self.move is not None else _servo_joint_rad(self.name, self.target_deg)

    def start_move(self, goal_joint_rad: float, now_s: float, max_velocity_rad_s: float,
                   max_acceleration_rad_s2: float) -> None:
        """Move to an IK goal on the MOTION_PROFILE rest-to-rest path, the one
        ArmIK.path_references checks for clearance."""
        start = self.joint_reference_rad()
        self._move_limits = (max_velocity_rad_s, max_acceleration_rad_s2)
        duration = _PROFILE_DURATION(goal_joint_rad - start, max_velocity_rad_s, max_acceleration_rad_s2)
        self.move = (start, float(goal_joint_rad), now_s, duration)
        self._move_elapsed_s = 0.0

    def advance_move(self, now_s: float) -> None:
        if self.move is None:
            return
        _, _, t0, duration = self.move
        self._move_elapsed_s = max(0.0, now_s - t0)
        self.set_target(_servo_deg_from_joint(self.name, self._move_position(self._move_elapsed_s)))
        if self._move_elapsed_s >= duration:
            self.move = None

    def _move_position(self, elapsed_s: float) -> float:
        start, goal, _, _ = self.move
        return _PROFILE_POSITION(start, goal, elapsed_s, *self._move_limits)

    def sync_control_tab(self, data: mujoco.MjData) -> None:
        """Take a MuJoCo Control-tab edit of the mirrored actuator as the target,
        then write the target back so that slider always shows it."""
        if self.actuator_id < 0:
            return
        ctrl = float(data.ctrl[self.actuator_id])
        if self._last_written_ctrl is not None and abs(ctrl - self._last_written_ctrl) > 1e-7:
            self.set_target_by_operator(_servo_deg_from_joint(self.name, ctrl))
        self._last_written_ctrl = _servo_joint_rad(self.name, self.target_deg)
        data.ctrl[self.actuator_id] = self._last_written_ctrl

    def update(self, send: bool) -> str | None:
        """Follow the target.  With send, write "<address>P<firmware deg>" when
        it moved SERVO_RESEND_DEG, or when the slider stopped short of that;
        returns the line sent.  Without send (simulation) the sim just follows."""
        settled = self.target_deg == self._previous_target_deg
        self._previous_target_deg = self.target_deg
        if not send:
            self.commanded_deg = self.target_deg
            return None
        if self.last_sent_deg is None:
            if self.target_deg == self.park_deg:
                return None   # still where the firmware parked it; send nothing
        else:
            change = abs(self.target_deg - self.last_sent_deg)
            if change < SERVO_RESEND_DEG and not (settled and change > 0.0):
                return None
        line = f"{self.address}P{_servo_firmware_deg(self.target_deg):.1f}"
        si.send(line)
        self.last_sent_deg = self.target_deg
        self.commanded_deg = self.target_deg
        return line


def _apply_servo_state(
    data: mujoco.MjData, model: mujoco.MjModel, qpos_idx: dict[str, int], qvel_idx: dict[str, int], servo: _Servo
) -> None:
    """Show a servo's commanded angle on its joint (the servo reports nothing),
    clamped to the joint's range; a follower (the idler pinch gear) mirrors it."""
    joint, _, _, follower = SERVO_JOINTS[servo.name]
    if joint not in qpos_idx:
        return
    q = _servo_joint_rad(servo.name, servo.commanded_deg)
    for name, value in ((joint, q), (follower, -q)):
        if name is None or name not in qpos_idx:
            continue
        low, high = model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        data.qpos[qpos_idx[name]] = min(max(value, low), high)
        data.qvel[qvel_idx[name]] = 0.0


class _RunawayMonitor:
    """Detect a motor that is not following its command (see RUNAWAY_* above)."""

    def __init__(self) -> None:
        self._error_since: dict[int, float] = {}
        self._fast_packets: dict[int, int] = {}
        self._last_packet_time: dict[int, float] = {}
        self._last_packet_seen: dict[int, float] = {}

    def reset(self) -> None:
        self._error_since.clear()
        self._fast_packets.clear()
        self._last_packet_time.clear()
        self._last_packet_seen.clear()

    def check(
        self,
        motor: int,
        cmd_pos_motor_rad: float,
        encoder_pos_motor_rad: float | None,
        encoder_vel_motor_rad_s: float | None,
        packet_time: float,
        now: float,
    ) -> str | None:
        """Return a trip reason, or None. Call once per tick per monitored motor."""
        # Feedback timeout. packet_time is the PC time the last packet arrived.
        if packet_time != self._last_packet_time.get(motor):
            self._last_packet_time[motor] = packet_time
            self._last_packet_seen[motor] = now
            new_packet = True
        else:
            new_packet = False
        silent_s = now - self._last_packet_seen.setdefault(motor, now)
        if encoder_pos_motor_rad is None or silent_s > RUNAWAY_FEEDBACK_TIMEOUT_S:
            return f"M{motor} no feedback for {silent_s:.2f} s"

        # Speed: count consecutive packets, so a repeated stale value or a
        # one-packet glitch does not trip.
        if new_packet and encoder_vel_motor_rad_s is not None:
            if abs(encoder_vel_motor_rad_s) > RUNAWAY_SPEED_RAD_S:
                self._fast_packets[motor] = self._fast_packets.get(motor, 0) + 1
            else:
                self._fast_packets[motor] = 0
            if self._fast_packets[motor] >= RUNAWAY_SPEED_PACKETS:
                return (
                    f"M{motor} speed {encoder_vel_motor_rad_s:+.2f} rad/s "
                    f"> {RUNAWAY_SPEED_RAD_S} rad/s"
                )

        # Sustained position error.
        error = cmd_pos_motor_rad - encoder_pos_motor_rad
        if abs(error) > RUNAWAY_POS_ERROR_RAD[motor]:
            since = self._error_since.setdefault(motor, now)
            if now - since >= RUNAWAY_POS_ERROR_S:
                return (
                    f"M{motor} position error {error:+.3f} rad for {now - since:.2f} s "
                    f"(cmd {cmd_pos_motor_rad:+.3f}, encoder {encoder_pos_motor_rad:+.3f})"
                )
        else:
            self._error_since.pop(motor, None)
        return None


class _EchoMonitor:
    """Compare the STM32's CAN command echoes with what the PC sent (see ECHO_* above)."""

    _LETTER = {"pos": "P", "vel": "V", "kp": "K", "kd": "D", "tau_ff": "T"}

    def __init__(self) -> None:
        # motor -> field -> [(PC send time, value)], oldest first.  Pruned to
        # the window plus the one value current before it.
        self._sent: dict[int, dict[str, list[tuple[float, float]]]] = {}
        self._mismatches: dict[int, int] = {}

    def reset(self) -> None:
        self._sent.clear()
        self._mismatches.clear()

    def record_sent(self, motor: int, values: dict[str, float], now: float) -> None:
        for key, value in values.items():
            if key == "tau_ff":
                # send_mit_frame clamps T before writing it.
                limit = si.TAU_FF_LIMIT[motor]
                value = max(-limit, min(limit, value))
            # The firmware clamps every field to its CAN range before packing.
            lo, hi = si.command_range(motor, key)
            value = max(lo, min(hi, value))
            history = self._sent.setdefault(motor, {}).setdefault(key, [])
            history.append((now, value))
            while len(history) > 1 and history[1][0] < now - 2.0 * ECHO_WINDOW_S:
                history.pop(0)

    def check(self, motor: int, echo: "si.CommandEcho", received: float) -> str | None:
        """Return a trip reason, or None. Call once per echo, in arrival order."""
        mismatched = []
        judged = False
        for key in self._LETTER:
            tolerance = ECHO_TOLERANCE_COUNTS * si.command_resolution(motor, key)
            history = self._sent.get(motor, {}).get(key)
            window_start = received - ECHO_WINDOW_S
            # Inconclusive until the first sent value is older than the
            # window: the firmware may still show what it held before.
            if not history or history[0][0] > window_start:
                continue
            candidates = [v for t, v in history if window_start <= t <= received]
            older = [v for t, v in history if t < window_start]
            candidates.append(older[-1])
            judged = True
            echoed = getattr(echo, key)
            if all(abs(echoed - v) > tolerance for v in candidates):
                mismatched.append(f"{self._LETTER[key]} echo {echoed:+.3f} vs sent {candidates[-1]:+.3f}")
        if not judged:
            return None
        if not mismatched:
            self._mismatches[motor] = 0
            return None
        self._mismatches[motor] = self._mismatches.get(motor, 0) + 1
        if self._mismatches[motor] >= ECHO_MISMATCH_COUNT:
            return (
                f"M{motor} command echo mismatch on {self._mismatches[motor]} echoes: "
                + ", ".join(mismatched)
            )
        return None

# ---------------------------------------------------------------------------
# Per-run cycle log
# ---------------------------------------------------------------------------

def _git_revision() -> str:
    """Return the short HEAD hash, suffixed with -dirty for uncommitted edits."""
    repo = Path(__file__).parent
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not rev:
        return "unknown"
    return f"{rev}-dirty" if dirty else rev


def _git_branch() -> str:
    """Return the checked-out branch name, made safe for a file name."""
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        branch = ""
    if not branch:
        return "no-branch"
    if branch == "HEAD":
        return "detached"
    return "".join(ch if ch.isalnum() or ch in "-." else "-" for ch in branch)


_finished_logs: set[Path] = set()


def _finish_run_log(log_file: object, path: Path) -> None:
    """Close the run's CSV and write its graph next to it. Safe to call twice.

    Registered with atexit as well as called at the normal end, so a crash or
    Ctrl+C still leaves a graph of whatever was logged.
    """
    if path in _finished_logs:
        return
    _finished_logs.add(path)
    try:
        log_file.close()
    except OSError:
        pass
    print(f"[ctrl] Run log saved: {path}")
    try:
        print(f"[ctrl] Run graph saved: {plot_run_log(path)}")
    except Exception as exc:  # a plotting bug must never hide the CSV
        print(f"[ctrl] WARNING: could not draw the run graph: {exc}")
        print(f"[ctrl]   Retry with: python plot_run_log.py {path}")


def _open_cycle_log(
    qpos_idx: dict[str, int],
    limiter: "TrapezoidalReferenceLimiter",
    metadata: dict[str, object],
) -> tuple[object, csv.DictWriter, Path]:
    """Create this run's CSV log and its JSON settings file. Raises OSError on failure."""
    fieldnames = [
        "host_time_utc",
        "wall_time_s",
        "tick",
        "simulation_time_s",
        "mode",
        "control_state",
        "contact_count",
    ]
    for motor in MOTOR_TO_JOINT:
        fieldnames.extend(
            [
                f"m{motor}_encoder_position_rad",
                f"m{motor}_encoder_velocity_rad_s",
                f"m{motor}_encoder_current_a",
                f"m{motor}_encoder_torque_nm",
                f"m{motor}_encoder_temp_c",
                f"m{motor}_encoder_err",
                f"m{motor}_feedback_age_s",
            ]
        )
    # What the STM32 echoed back as sent on CAN (TX_ECHO firmware), motor side.
    # Compare with m<n>_cmd_* to see whether commands arrived intact.
    for motor in MOTOR_TO_JOINT:
        fieldnames.extend(
            [
                f"m{motor}_echo_position_rad",
                f"m{motor}_echo_velocity_rad_s",
                f"m{motor}_echo_kp",
                f"m{motor}_echo_kd",
                f"m{motor}_echo_tau_ff_nm",
                f"m{motor}_echo_age_s",
            ]
        )
    for motor in MOTOR_TO_JOINT:
        fieldnames.extend(
            [
                f"m{motor}_cmd_position_rad",
                f"m{motor}_cmd_velocity_rad_s",
                f"m{motor}_cmd_kp",
                f"m{motor}_cmd_kd",
                f"m{motor}_cmd_tau_ff_nm",
                f"m{motor}_cmd_fields_sent",
            ]
        )
    for joint in qpos_idx:
        fieldnames.extend([f"sim_{joint}_qpos_rad", f"sim_{joint}_qvel_rad_s"])
    for actuator in limiter.references:
        fieldnames.extend(
            [
                f"{actuator}_ctrl",
                f"{actuator}_requested_joint_rad",
                f"{actuator}_reference_joint_rad",
                f"{actuator}_reference_velocity_joint_rad_s",
            ]
        )
    fieldnames.append("events")

    CYCLE_LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # <branch>_<date>_<time> in UTC; the .json and .svg share this name.
    stamp = f"{_git_branch()}_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}"
    # Exclusive create: two runs started in the same second never overwrite each other.
    for suffix in [""] + [f"_{n}" for n in range(1, 100)]:
        path = CYCLE_LOG_DIRECTORY / f"{stamp}{suffix}.csv"
        try:
            log_file = path.open("x", newline="", encoding="utf-8", buffering=1)
            break
        except FileExistsError:
            continue
    else:
        raise OSError(f"no free log filename for {stamp} in {CYCLE_LOG_DIRECTORY}")

    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    log_file.flush()
    metadata = dict(metadata, csv_file=path.name)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return log_file, writer, path


def _write_cycle_log(
    log_file: object,
    writer: csv.DictWriter,
    start_time: float,
    tick: int,
    simulation_only: bool,
    data: mujoco.MjData,
    qpos_idx: dict[str, int],
    qvel_idx: dict[str, int],
    limiter: "TrapezoidalReferenceLimiter",
    raw_pos: dict[int, float | None],
    raw_vel: dict[int, float | None],
    feedback_age: dict[int, float | None],
    commanded: dict[int, tuple[float, float, float, float, float, str]],
    feedback_current: dict[int, float | None] | None = None,
    feedback_torque: dict[int, float | None] | None = None,
    command_echo: dict[int, "si.CommandEcho | None"] | None = None,
    command_echo_age: dict[int, float | None] | None = None,
    feedback_temp: dict[int, int | None] | None = None,
    feedback_err: dict[int, int | None] | None = None,
) -> None:
    """Append one controller-cycle record and flush it to disk."""
    status = "ACTIVE" if (_control_active and not _paused) else ("PAUSED" if _paused else "IDLE")
    events = []
    while _log_events:
        events.append(_log_events.popleft())
    row: dict[str, object] = {
        "host_time_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": time.perf_counter() - start_time,
        "tick": tick,
        "simulation_time_s": float(data.time),
        "mode": "simulation" if simulation_only else "hardware",
        "control_state": status,
        "contact_count": int(data.ncon),
        "events": " | ".join(events),
    }
    for motor in MOTOR_TO_JOINT:
        row[f"m{motor}_encoder_position_rad"] = raw_pos[motor]
        row[f"m{motor}_encoder_velocity_rad_s"] = raw_vel[motor]
        row[f"m{motor}_encoder_current_a"] = feedback_current[motor] if feedback_current else None
        row[f"m{motor}_encoder_torque_nm"] = feedback_torque[motor] if feedback_torque else None
        row[f"m{motor}_encoder_temp_c"] = feedback_temp[motor] if feedback_temp else None
        row[f"m{motor}_encoder_err"] = feedback_err[motor] if feedback_err else None
        row[f"m{motor}_feedback_age_s"] = feedback_age[motor]
        echo = command_echo.get(motor) if command_echo else None
        if echo is not None:
            row[f"m{motor}_echo_position_rad"] = echo.pos
            row[f"m{motor}_echo_velocity_rad_s"] = echo.vel
            row[f"m{motor}_echo_kp"] = echo.kp
            row[f"m{motor}_echo_kd"] = echo.kd
            row[f"m{motor}_echo_tau_ff_nm"] = echo.tau_ff
            row[f"m{motor}_echo_age_s"] = command_echo_age[motor] if command_echo_age else None
        if motor in commanded:
            pos, vel, kp, kd, tau_ff, sent = commanded[motor]
            row[f"m{motor}_cmd_position_rad"] = pos
            row[f"m{motor}_cmd_velocity_rad_s"] = vel
            row[f"m{motor}_cmd_kp"] = kp
            row[f"m{motor}_cmd_kd"] = kd
            row[f"m{motor}_cmd_tau_ff_nm"] = tau_ff
            row[f"m{motor}_cmd_fields_sent"] = sent
    for joint, address in qpos_idx.items():
        row[f"sim_{joint}_qpos_rad"] = float(data.qpos[address])
        row[f"sim_{joint}_qvel_rad_s"] = float(data.qvel[qvel_idx[joint]])
    for actuator, reference in limiter.references.items():
        row[f"{actuator}_ctrl"] = float(data.ctrl[reference.actuator_id])
        row[f"{actuator}_requested_joint_rad"] = reference.requested_position_rad
        row[f"{actuator}_reference_joint_rad"] = reference.position_rad
        row[f"{actuator}_reference_velocity_joint_rad_s"] = reference.velocity_rad_s
    writer.writerow(row)
    log_file.flush()

def _encoder_position_in_sim_coordinates(motor: int, raw_position_rad: float | None) -> float | None:
    """Project one raw encoder value into the synchronized MuJoCo joint frame."""
    joint = MOTOR_TO_JOINT[motor]
    if (
        raw_position_rad is None
        or motor not in _encoder_feedback_at_sync
        or motor not in _encoder_at_sync
        or joint not in _sim_at_sync
    ):
        return None
    return _sim_at_sync[joint] + (raw_position_rad - _encoder_at_sync[motor]) * MOTOR_SCALE[motor]


def _print_live_telemetry(
    data: mujoco.MjData,
    qpos_idx: dict[str, int],
    raw_pos: dict[int, float | None],
    command_gains: dict[int, tuple[float, float] | None],
    simulation_only: bool,
) -> None:
    """Print positions and the motor-side Kp/Kd values used for control."""
    status = "ACTIVE" if (_control_active and not _paused) else ("PAUSED" if _paused else "IDLE")
    source = "simulation; no STM32 feedback" if simulation_only else "STM32 raw encoder"
    print(f"[telemetry/{status}] {source}")
    for motor, joint in MOTOR_TO_JOINT.items():
        sim_position = float(data.qpos[qpos_idx[joint]])
        encoder_position = raw_pos[motor]
        encoder_sim_position = _encoder_position_in_sim_coordinates(motor, encoder_position)
        if encoder_position is None:
            encoder_text = "n/a"
        else:
            encoder_text = f"{encoder_position:+.4f} raw rad"
        if encoder_sim_position is None:
            comparison_text = "enc_sim=n/a  error=n/a"
        else:
            error = sim_position - encoder_sim_position
            comparison_text = (
                f"enc_sim={encoder_sim_position:+.4f} rad  "
                f"error={error:+.4f} rad"
            )
        gains = command_gains[motor]
        gain_text = "Kp=n/a  Kd=n/a"
        if gains is not None:
            gain_text = f"Kp={gains[0]:.3f} N m/rad  Kd={gains[1]:.3f} N m s/rad"
        print(
            f"    M{motor} {joint}: sim={sim_position:+.4f} rad  "
            f"enc={encoder_text}  {comparison_text}  {gain_text}"
        )


# ---------------------------------------------------------------------------
# Fallback controller (when no JSON schedule is present)
# ---------------------------------------------------------------------------

def _build_default_controller() -> MasterController:
    schedules = []
    for actuator, (kp, kd) in DEFAULT_GAINS.items():
        joint = MOTOR_TO_JOINT[ACTUATOR_TO_MOTOR[actuator]]
        omega_n = math.sqrt(max(kp, 1e-3) / 0.1)
        schedules.append(
            PositionSchedule(
                actuator=actuator,
                joint=joint,
                position_min_rad=-math.pi,
                position_max_rad=math.pi,
                natural_frequency_low_rad_s=omega_n,
                natural_frequency_high_rad_s=omega_n,
                damping_ratio=0.7,
            )
        )
    return MasterController(schedules)

# ---------------------------------------------------------------------------
# Shared command helpers (terminal and control panel)
# ---------------------------------------------------------------------------

def _start_enable() -> None:
    """Enable M1-M4 in a background thread (each enable blocks the STM32 ~1.3 s)."""
    global _motor_power
    _motor_power = "enabling"

    def run() -> None:
        global _motor_power
        si.enable_motors([1, 2, 3, 4])
        _motor_power = "enabled"
        _log_events.append("motors enabled")

    threading.Thread(target=run, daemon=True).start()


def _limiter_moving(limiter: "TrapezoidalReferenceLimiter") -> bool:
    """True while a goal is pending, any reference is not at rest on its
    request, or the wrist is on an IK move."""
    return (
        _request_goal is not None or _request_ee_state is not None
        or (_ik_wrist is not None and _ik_wrist.move is not None)
    ) or any(
        limiter.reference_velocity(a) != 0.0
        or abs(limiter.requested_position(a) - limiter.reference_position(a)) > 1e-9
        for a in ACTUATOR_TO_MOTOR
    )


def _request_end_effector_state(
    requested: EndEffectorState, limiter: "TrapezoidalReferenceLimiter", ik_lock: threading.Lock
) -> str:
    """Ask the control loop to change state; returns a message for the operator.

    Entering Pushing swings the wrist, so it needs a running control loop and
    a swing that clears the table.  Leaving it moves nothing: the wrist stays
    at PUSHING_WRIST_SERVO_DEG until the next IK move or slider edit.
    """
    global _request_ee_state
    if requested not in _arm_iks:
        return f"Refused: {requested.value} needs the wrist servo in the IK (SERVOS and WRIST_IK)."
    if requested is _ee_state:
        return f"Already {requested.value}."
    if _limiter_moving(limiter):
        return "Refused: wait for the current move to finish."
    if requested is EndEffectorState.PUSHING:
        if not _control_active:
            return "Refused: control is not active. Start control first."
        if _paused:
            return "Refused: control is paused. Resume first."
        gripping_ik = _arm_iks[EndEffectorState.GRIPPING]
        q_now = gripping_ik.to_array(_ik_references({a: limiter.reference_position(a) for a in ACTUATOR_TO_MOTOR}))
        q_pushing = q_now.copy()
        q_pushing[gripping_ik.actuators.index(SERVO_CHANNELS["wrist"][2])] = _wrist_joint_rad(PUSHING_WRIST_SERVO_DEG)
        with ik_lock:
            clearance = gripping_ik.move_clearance_m(q_now, q_pushing)
        if clearance < IK_TABLE_CLEARANCE_M:
            return (f"Refused: turning the wrist to {PUSHING_WRIST_SERVO_DEG:.0f} deg here would pass "
                    f"{1000 * clearance:.1f} mm above the table. Raise the arm first.")
    _request_ee_state = requested
    return f"Changing to {requested.value}."


def _measured_references(
    limiter: "TrapezoidalReferenceLimiter",
    actuator_gear: dict[str, float],
    simulation_only: bool,
) -> dict[str, float] | None:
    """Encoder positions in the limiter's reference coordinates (joint side).

    Inverts the P command (P = dir * ref * gear + offset), so with perfect
    tracking this equals the reference.  Motors without feedback at sync
    report their reference.  None when there is nothing to measure.
    """
    if simulation_only or not _control_active or not _encoder_feedback_at_sync:
        return None
    pos_raw, _ = si.raw_snapshot()
    measured = {}
    for actuator, motor in ACTUATOR_TO_MOTOR.items():
        if motor in _encoder_feedback_at_sync and pos_raw[motor] is not None:
            measured[actuator] = (pos_raw[motor] - _pos_cmd_offset[motor]) / (
                MOTOR_DIRECTION[motor] * actuator_gear[actuator]
            )
        else:
            measured[actuator] = limiter.reference_position(actuator)
    # The wrist servo reports nothing: use its reference.
    return _ik_references(measured)


# ---------------------------------------------------------------------------
# Control panel backend (--gui; the UI itself is control_panel.py)
# ---------------------------------------------------------------------------

GHOST_RGBA = (1.0, 0.55, 0.1, 0.35)       # preview pose
PATH_RGBA = (1.0, 0.85, 0.1, 0.9)         # preview tool-point path
TARGET_OK_RGBA = (0.1, 0.9, 0.1, 0.6)
TARGET_BAD_RGBA = (0.9, 0.1, 0.1, 0.8)
PATH_MAX_SEGMENTS = 200


@dataclass
class _Preview:
    target_base_m: tuple[float, float, float]
    start: dict[str, float]                 # references the solve started from
    result: IKResult
    qpos: np.ndarray | None = None          # full qpos of the solved pose
    path_world: list[np.ndarray] = field(default_factory=list)
    executed: bool = False
    state: EndEffectorState = EndEffectorState.GRIPPING   # the state it was solved in


class _GuiBackend:
    """Controller side of control_panel.PanelBackend.

    Called on the main thread (panel callbacks run inside ControlPanel.pump),
    except the IK solve, which runs in a worker thread under ``ik_lock``.
    Actions only set request flags; the control loop carries them out, as it
    does for the terminal commands.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        limiter: "TrapezoidalReferenceLimiter",
        ik_lock: threading.Lock,
        actuator_gear: dict[str, float],
        simulation_only: bool,
        port: str | None,
        viewer: object,
        servos: dict[str, _Servo],
    ) -> None:
        self.model = model
        self.limiter = limiter
        self.servos = servos
        self.ik_lock = ik_lock
        self.actuator_gear = actuator_gear
        self.simulation_only = simulation_only
        self.port = port
        self.viewer = viewer
        self._display_data = mujoco.MjData(model)   # main-thread FK for the status
        self._ghost_data = mujoco.MjData(model)     # preview pose for drawing
        self._opt = mujoco.MjvOption()
        self._pert = mujoco.MjvPerturb()
        self._preview_lock = threading.Lock()
        self._preview: _Preview | None = None
        self._preview_busy = False
        self._preview_version = 0
        self._drawn_version = 0

    # --- helpers ---

    @property
    def arm_ik(self) -> ArmIK:
        return _arm_iks[_ee_state]

    def _references(self) -> dict[str, float]:
        """IK references: M1-M4 from the limiter, plus the wrist (WRIST_IK)."""
        return _ik_references({a: self.limiter.reference_position(a) for a in ACTUATOR_TO_MOTOR})

    def _set_preview(self, preview: _Preview | None) -> None:
        with self._preview_lock:
            self._preview = preview
            self._preview_version += 1

    def _run_refusal(self) -> str | None:
        """Why the current preview cannot run now, or None if it can."""
        preview = self._preview
        if preview is None or not preview.result.success:
            return "No valid preview to run."
        if preview.executed:
            return "Refused: this preview was already run. Preview again."
        if preview.state is not _ee_state:
            return "Refused: the end-effector state changed since this preview. Preview again."
        if not _control_active:
            return "Refused: control is not active. Start control first."
        if _paused:
            return "Refused: control is paused. Resume first."
        if _limiter_moving(self.limiter):
            return "Refused: a move is in progress."
        current = self._references()
        if any(abs(current.get(a, math.inf) - preview.start[a]) > 1e-9 for a in preview.start):
            return "Refused: the arm or wrist reference moved since this preview. Preview again."
        return None

    # --- PanelBackend ---

    def status(self) -> PanelStatus:
        if self._preview is not None and self._preview.state is not _ee_state:
            self._set_preview(None)     # solved for the other tool point
        raw_pos: dict[int, float | None] = {m: None for m in MOTOR_TO_JOINT}
        ages: dict[int, float | None] = {m: None for m in MOTOR_TO_JOINT}
        faults: dict[int, int | None] = {m: None for m in MOTOR_TO_JOINT}
        if not self.simulation_only:
            fb = si.snapshot()
            raw_pos, _ = si.raw_snapshot()
            now = time.perf_counter()
            for m in MOTOR_TO_JOINT:
                if raw_pos[m] is not None:
                    ages[m] = now - fb.time[m]
                faults[m] = fb.err[m]
        references = self._references()
        rows = [
            MotorRow(motor, MOTOR_TO_JOINT[motor], math.degrees(references[actuator]),
                     raw_pos[motor], ages[motor], faults[motor])
            for actuator, motor in ACTUATOR_TO_MOTOR.items()
        ]

        def fk(q: dict[str, float]) -> tuple[float, float, float]:
            p = self.arm_ik.tool_position_base_on(self._display_data, q)
            return (float(p[0]), float(p[1]), float(p[2]))

        tool_reference = fk(references)
        tool_target = fk(_ik_references({a: self.limiter.requested_position(a) for a in ACTUATOR_TO_MOTOR}, "goal"))
        measured = _measured_references(self.limiter, self.actuator_gear, self.simulation_only)
        tool_encoders = None if measured is None else fk(measured)

        with self._preview_lock:
            preview = self._preview
            busy = self._preview_busy
        text, ok = "", False
        if preview is not None:
            r = preview.result
            target = "(" + ", ".join(f"{1000 * v:.1f}" for v in preview.target_base_m) + ") mm"
            if r.success:
                ok = not r.nearest
                changes = "  ".join(
                    f"{_ik_joint_label(a)} {math.degrees(q - preview.start[a]):+.1f}°"
                    for a, q in r.joint_targets.items()
                )
                state = "Running on the arm." if preview.executed else "The orange ghost in MuJoCo shows the pose."
                if r.warning:
                    changes += f"\nWARNING: {r.warning}"
                if r.nearest:
                    reached = "(" + ", ".join(f"{1000 * v:.1f}" for v in r.achieved_base_m) + ") mm"
                    closer = " Repeating may get closer." if r.window_limited else ""
                    text = (f"{target}: {r.message}.\n"
                            f"Nearest location {reached}, {1000 * r.error_m:.1f} mm away, "
                            f"move {r.move_duration_s:.2f} s.{closer} {state}\nJoint changes: {changes}")
                else:
                    text = (f"Preview {target}: error {1000 * r.error_m:.3f} mm, move {r.move_duration_s:.2f} s. "
                            f"{state}\nJoint changes: {changes}")
            else:
                text = f"No solution for {target}: {r.message}"
        if self.simulation_only:
            control_state = "ACTIVE (simulation)" if not _paused else "PAUSED"
        else:
            control_state = "ACTIVE" if (_control_active and not _paused) else ("PAUSED" if _paused else "IDLE")
        return PanelStatus(
            mode="SIMULATION (no serial port)" if self.simulation_only else f"HARDWARE ({self.port})",
            hardware=not self.simulation_only,
            control_state=control_state,
            motor_power=_motor_power,
            moving=_limiter_moving(self.limiter),
            motors=rows,
            tool_reference_m=tool_reference,
            tool_target_m=tool_target,
            tool_encoders_m=tool_encoders,
            preview_text=text,
            preview_ok=ok,
            preview_runnable=not busy and self._run_refusal() is None,
            preview_busy=busy,
            notice=_notice[0],
            notice_is_alert=_notice[1],
            end_effector_state=_ee_state.value,
            servos=[
                ServoRow(s.name, s.label, s.target_deg, s.commanded_deg if s.last_sent_deg is not None or
                         self.simulation_only else None, s.min_deg, s.max_deg, s.locked)
                for s in self.servos.values()
            ],
        )

    def enable(self) -> str:
        if self.simulation_only:
            return "Unavailable in simulation."
        if _control_active:
            return "Refused: stop control before enabling (each enable blocks the STM32 ~1.3 s)."
        if _motor_power == "enabling":
            return "Refused: already enabling."
        _log_events.append("panel enable")
        _start_enable()
        return "Enabling M1-M4 one at a time (several seconds)..."

    def start(self, arm_at_home: bool) -> str:
        global _request_start, _request_start_home
        if self.simulation_only:
            return "Unavailable in simulation: control is always active."
        if _control_active:
            return "Refused: already running."
        if _motor_power != "enabled":
            return f"Refused: motors are {_motor_power}. Enable them first."
        self._set_preview(None)
        _log_events.append(f"panel start ({'arm at home' if arm_at_home else 'keep reference'})")
        _request_start_home = arm_at_home
        _request_start = True
        return "Starting control" + (" from hardware_zero." if arm_at_home else " from the current reference.")

    def pause(self) -> str:
        global _paused
        _log_events.append("panel pause")
        _paused = True
        return "Paused: no commands are streamed; the motors hold their last command."

    def resume(self) -> str:
        global _paused
        _log_events.append("panel resume")
        _paused = False
        return "Resumed."

    def home(self) -> str:
        global _request_home
        if not _control_active:
            return "Refused: control is not active."
        if _paused:
            return "Refused: control is paused. Resume first."
        _log_events.append("panel home")
        _request_home = True
        return "Going to hardware_zero."

    def hold(self) -> str:
        global _request_hold
        if self.simulation_only:
            return "Unavailable in simulation."
        _log_events.append("panel hold")
        _request_hold = True
        return "HOLD: sending 0X (motors freeze) and stopping streaming. Enable and Start to continue."

    def disable(self) -> str:
        global _request_disable
        if self.simulation_only:
            return "Unavailable in simulation."
        _log_events.append("panel disable")
        _request_disable = True
        return "Disabling: stopping streaming and sending zero-gain frames..."

    def preview(self, target_base_m: tuple[float, float, float]) -> str:
        if _limiter_moving(self.limiter):
            return "Refused: wait for the current move to finish."
        with self._preview_lock:
            if self._preview_busy:
                return "Refused: a solve is already running."
            self._preview_busy = True
        start = self._references()
        state = _ee_state
        arm_ik = self.arm_ik

        def solve() -> None:
            preview = _Preview(tuple(target_base_m), start, IKResult(False, "", np.asarray(target_base_m)))
            try:
                with self.ik_lock:
                    # The panel previews before moving, so refused targets get
                    # a nearest location too; the terminal "ik" moves at once.
                    result = arm_ik.solve_nearest(target_base_m, start, include_refused=True)
                    preview.result = result
                    if result.success:
                        q_start = arm_ik.to_array(start)
                        q_goal = arm_ik.to_array(result.joint_targets)
                        preview.qpos = arm_ik.full_qpos(q_goal)
                        preview.path_world = [
                            arm_ik.base_to_world(p)
                            for p in arm_ik.path_tool_positions_base(q_start, q_goal)
                        ]
            except Exception as exc:   # report it; never kill the controller
                preview = _Preview(
                    tuple(target_base_m), start,
                    IKResult(False, f"solver error: {exc}", np.asarray(target_base_m, dtype=float)),
                )
            preview.state = state
            _log_events.append(
                f"panel preview {tuple(round(v, 4) for v in target_base_m)}: "
                + (f"nearest location {np.round(preview.result.achieved_base_m, 4)} ({preview.result.message})"
                   if preview.result.nearest else "ok" if preview.result.success else preview.result.message)
            )
            with self._preview_lock:
                self._preview = preview
                self._preview_version += 1
                self._preview_busy = False

        threading.Thread(target=solve, daemon=True).start()
        return "Solving..."

    def run_preview(self) -> str:
        global _request_goal
        with self._preview_lock:
            if self._preview_busy:
                return "Refused: a solve is running."
            refusal = self._run_refusal()
            if refusal is not None:
                return refusal
            preview = self._preview
            preview.executed = True
            self._preview_version += 1
        _request_goal = dict(preview.result.joint_targets)
        _log_events.append(
            f"panel run {preview.target_base_m} -> "
            + ", ".join(f"{a}={q:+.5f}" for a, q in preview.result.joint_targets.items())
        )
        destination = tuple(round(float(v), 4) for v in preview.result.achieved_base_m)
        print(f"[ctrl] Panel: running the preview to {destination} m"
              + (f" (nearest location to {preview.target_base_m})" if preview.result.nearest else ""))
        return f"Running: move {preview.result.move_duration_s:.2f} s."

    def clear_preview(self) -> str:
        with self._preview_lock:
            if self._preview_busy:
                return "Refused: a solve is running."
        self._set_preview(None)
        return "Preview cleared."

    def set_end_effector_state(self, name: str) -> str:
        with self._preview_lock:
            if self._preview_busy:
                return "Refused: a solve is running."
        _log_events.append(f"panel state {name}")
        return _request_end_effector_state(EndEffectorState(name), self.limiter, self.ik_lock)

    def set_servo(self, name: str, servo_deg: float) -> str:
        servo = self.servos.get(name)
        if servo is None:
            return f"No servo {name!r}."
        if servo.locked:
            return f"{servo.label} is held at {servo.target_deg:.0f} deg in the {_ee_state.value} state."
        servo.set_target_by_operator(servo_deg)   # the control loop sends it next tick
        where = "simulation" if self.simulation_only else f"'{servo.address}P'"
        return f"{servo.label}: {servo.target_deg:.0f} deg ({where})."

    def quit(self) -> None:
        _log_events.append("panel quit")
        self.viewer.close()

    # --- drawing (main thread, every tick) ---

    def render(self) -> None:
        """Redraw the preview in the viewer's user scene when it changes."""
        with self._preview_lock:
            preview = self._preview
            # An executed preview is removed once its move has finished.
            if preview is not None and preview.executed and not _limiter_moving(self.limiter):
                self._preview = preview = None
                self._preview_version += 1
            version = self._preview_version
        if version == self._drawn_version:
            return
        self._drawn_version = version
        scn = self.viewer.user_scn
        with self.viewer.lock():
            scn.ngeom = 0
            if preview is None:
                return
            if preview.qpos is not None:
                self._ghost_data.qpos[:] = preview.qpos
                mujoco.mj_kinematics(self.model, self._ghost_data)
                first = scn.ngeom
                mujoco.mjv_addGeoms(
                    self.model, self._ghost_data, self._opt, self._pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn
                )
                for i in range(first, scn.ngeom):
                    geom = scn.geoms[i]
                    geom.rgba[:] = GHOST_RGBA
                    geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN   # not selectable
                    geom.objid = -1
            points = preview.path_world
            if len(points) > PATH_MAX_SEGMENTS + 1:
                step = math.ceil((len(points) - 1) / PATH_MAX_SEGMENTS)
                points = points[::step] + [points[-1]]
            for a, b in zip(points, points[1:]):
                if scn.ngeom >= scn.maxgeom - 1 or np.linalg.norm(b - a) < 1e-6:
                    continue
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                    np.array(PATH_RGBA, dtype=np.float32),
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, a, b)
                scn.ngeom += 1
            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.006, 0.0, 0.0]),
                    self.arm_ik.base_to_world(np.array(preview.target_base_m)), np.eye(3).flatten(),
                    np.array(
                        TARGET_OK_RGBA if preview.result.success and not preview.result.nearest else TARGET_BAD_RGBA,
                        dtype=np.float32,
                    ),
                )
                scn.ngeom += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _ser, _control_active, _request_start, _request_pause
    global _paused, _request_goal, _request_home, _request_rezero
    global _request_start_home, _request_hold, _request_disable, _motor_power, _notice, _ik_wrist
    global _ee_state, _request_ee_state

    arguments = sys.argv[1:]
    enable_motors_on_start = "--enable-motors" in arguments
    use_gui = "--gui" in arguments
    port_arguments = [argument for argument in arguments if argument not in ("--enable-motors", "--gui")]
    if len(port_arguments) > 1:
        print("Usage: python3 hardware_integrated_main_controller.py [serial_port] [--enable-motors] [--gui]")
        sys.exit(1)
    port = port_arguments[0] if port_arguments else None
    simulation_only = port is None
    if simulation_only and enable_motors_on_start:
        print("--enable-motors requires a serial port.")
        sys.exit(1)
    if not simulation_only:
        try:
            si.connect(port)
            print(f"[ctrl] Connected to {port} (motors frozen)")
        except Exception as e:
            print(f"[ctrl] Serial error: {e}")
            sys.exit(1)
        if enable_motors_on_start:
            print("[ctrl] Enabling M1-M4; control targets remain inactive until started.")
            si.enable_motors([1, 2, 3, 4])
            _motor_power = "enabled"
    else:
        print("[ctrl] Simulation-only mode: no serial port and no hardware commands.")

    # --- Load model ---
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)

    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HARDWARE_ZERO_KEYFRAME_NAME)
    if keyframe_id < 0:
        raise RuntimeError(f"MJCF has no {HARDWARE_ZERO_KEYFRAME_NAME!r} keyframe.")

    # Build joint index maps.
    qpos_idx: dict[str, int] = {}
    qvel_idx: dict[str, int] = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            qpos_idx[name] = model.jnt_qposadr[i]
            qvel_idx[name] = model.jnt_dofadr[i]

    # Include every joint that the feedback projection may write.  Omitting
    # passive followers makes the first idle sync overwrite their solved
    # hardware_zero positions with zero, visibly breaking the four-bar pose.
    keyframe_joint_names = list(MOTOR_TO_JOINT.values()) + list(CONSTRAINED_JOINTS)
    keyframe_qpos = {
        jname: float(model.key_qpos[keyframe_id, qpos_idx[jname]])
        for jname in keyframe_joint_names
        if jname in qpos_idx
    }

    # Reset to keyframe.
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)

    # --- Load or build controller ---
    if SCHEDULE_PATH.exists():
        controller = MasterController.from_json(SCHEDULE_PATH)
        print(f"[ctrl] Loaded gain schedule from {SCHEDULE_PATH}")
    else:
        controller = _build_default_controller()
        print(f"[ctrl] No gain schedule found at {SCHEDULE_PATH} — using default gains.")
        print(f"[ctrl] Default gains: { {k: v for k, v in DEFAULT_GAINS.items()} }")

    controller.bind(model)

    # --- Reference limiter (MOTION_PROFILE; the cubic one subclasses the
    # trapezoidal one, so the annotations below cover both) ---
    if MOTION_PROFILE == "cubic":
        limiter = CubicSplineReferenceLimiter(model, controller)
    elif MOTION_PROFILE == "trapezoidal":
        limiter = TrapezoidalReferenceLimiter(model, controller)
    else:
        raise ValueError(f"unknown MOTION_PROFILE {MOTION_PROFILE!r}")
    print(f"[ctrl] Motion profile: {MOTION_PROFILE}")

    # --- Inverse kinematics (own MjData; the lock serializes the stdin and
    # named-pipe threads, which can both run commands) ---
    ik_joints = {actuator: MOTOR_TO_JOINT[motor] for actuator, motor in ACTUATOR_TO_MOTOR.items()}
    ik_extra_bounds: dict[str, tuple[float, float]] = {}
    wrist_actuator = SERVO_CHANNELS["wrist"][2]
    wrist_in_ik = (
        WRIST_IK and SERVOS
        and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, wrist_actuator) >= 0
        and WRIST_SERVO_JOINT in qpos_idx
    )
    if wrist_in_ik:
        # The wrist can only go where the servo's 0-180 deg reaches.
        ik_joints[wrist_actuator] = WRIST_SERVO_JOINT
        ik_extra_bounds[wrist_actuator] = tuple(_wrist_joint_rad(deg) for deg in SERVO_LIMITS_DEG["wrist"])
    arm_ik = ArmIK(
        model,
        MODEL_PATH,
        ik_joints,
        CONSTRAINED_JOINTS,
        limiter.max_velocity_rad_s,
        limiter.max_acceleration_rad_s2,
        extra_bounds_rad=ik_extra_bounds,
    )
    if wrist_in_ik:
        i = arm_ik.actuators.index(wrist_actuator)
        print(f"[ctrl] IK includes the wrist ({WRIST_SERVO_JOINT} "
              f"{math.degrees(arm_ik.lower[i]):+.1f}..{math.degrees(arm_ik.upper[i]):+.1f} deg).")
    _arm_iks[EndEffectorState.GRIPPING] = arm_ik
    if wrist_in_ik:
        _arm_iks[EndEffectorState.PUSHING] = ArmIK(
            model,
            MODEL_PATH,
            {actuator: MOTOR_TO_JOINT[motor] for actuator, motor in ACTUATOR_TO_MOTOR.items()},
            CONSTRAINED_JOINTS,
            limiter.max_velocity_rad_s,
            limiter.max_acceleration_rad_s2,
            tool_joint=PUSHING_TOOL_JOINT,
            fixed_joints_rad={WRIST_SERVO_JOINT: _wrist_joint_rad(PUSHING_WRIST_SERVO_DEG)},
        )
    ik_lock = threading.Lock()
    actuator_gear = {
        actuator: float(model.actuator_gear[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator), 0])
        for actuator in ACTUATOR_TO_MOTOR
    }

    # Initialise ctrl to hardware_zero.
    for actuator, ctrl_val in HARDWARE_ZERO_ACTUATOR_CTRL_RAD.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = ctrl_val
    limiter.reset(data)
    # Every servo starts at the firmware's boot angle, not the keyframe ctrl,
    # so the first send does not move it.
    servos: dict[str, _Servo] = {}
    if SERVOS:
        for name, (label, address, actuator) in SERVO_CHANNELS.items():
            servos[name] = _Servo(model, name, label, address, actuator)
            servos[name].sync_control_tab(data)
        print("[ctrl] Servos: " + ", ".join(
            f"{s.label} {s.min_deg:.0f}-{s.max_deg:.0f} deg, starts at {s.park_deg:.0f} -> '{s.address}P'"
            for s in servos.values()
        ) + f"; wrist 90 deg = {WRIST_SERVO_JOINT} {WRIST_SERVO_CENTER_JOINT_RAD:+.3f} rad (uncalibrated)")
    _ik_wrist = servos["wrist"] if wrist_in_ik else None
    if simulation_only:
        # In simulation mode, the Control-tab reference is safe to run immediately.
        _control_active = True
        _paused = False

    # --- Print startup info ---
    loop_mode = "CLOSED-LOOP (encoder feedback)" if CLOSED_LOOP else "OPEN-LOOP (reference only — no encoder feedback)"
    if simulation_only:
        loop_mode = "SIMULATION-ONLY (reference tracking; no serial port)"
    print()
    print(f"[ctrl] *** HARDWARE CONTROLLER READY — {loop_mode} ***")
    if not simulation_only:
        print("[ctrl]     Motors must be enabled before starting the control loop.")
    print("[ctrl]     Use MuJoCo's Control tab sliders as the primary position input.")
    if simulation_only:
        print("[ctrl]     Simulation is active immediately; move a Control tab slider to begin.")
    print()
    print("[ctrl] Commands:")
    if not simulation_only:
        print("         a            — enable all body motors (M1-M4)")
    print("         s            — start control loop")
    print("         g <j> <pos>  — move joint j (1-4); pos examples: 0.35  20deg  +20deg (relative)")
    print("         ik <x> <y> <z> — move the gripper tool point to a base-frame position;")
    print("                        metres, or with mm/cm, e.g.  ik 0.05 0.25 0.22   ik 50mm 250mm 220mm")
    print("         w            — print the tool point position (base frame)")
    print("         state [gripping|pushing] — print or change the end-effector state")
    print("         h            — go to hardware_zero pose")
    print("         p            — pause control loop")
    print("         r            — resume control loop")
    if not simulation_only:
        print("         z            — print current encoder readings")
    print("         f            — toggle live controller telemetry")
    print("         d            — toggle command debug printout")
    print("         q            — quit")
    print()
    home_tool = arm_ik.tool_position_base(arm_ik.home_references_array())
    print("[ctrl] IK base frame: origin on the base yaw axis at the base housing's underside")
    print(f"[ctrl]     ({1000 * (arm_ik.base_pos[2] - arm_ik.table_top_z_world):.1f} mm above the table top); "
          "axes = table axes, +z up.")
    print(f"[ctrl]     Tool point (between the finger tips) at hardware_zero: "
          f"({home_tool[0]:+.4f}, {home_tool[1]:+.4f}, {home_tool[2]:+.4f}) m")
    print()

    def _parse_length_m(text: str) -> float:
        """'0.25', '0.25m', '25cm' or '250mm' -> metres. Raises ValueError."""
        value = text.lower().strip()
        for suffix, scale in (("mm", 1e-3), ("cm", 1e-2), ("m", 1.0)):
            if value.endswith(suffix):
                return float(value[: -len(suffix)]) * scale
        return float(value)

    # --- Shared command handler (used by both stdin and named pipe) ---
    def _handle_command(raw: str) -> None:
        global _control_active, _request_start, _request_pause
        global _paused, _request_goal, _request_home, _request_rezero
        global DEBUG_COMMANDS, PRINT_MOTOR_FEEDBACK

        if not raw:
            return
        _log_events.append(f"cmd {raw}")
        cmd = raw.lower()

        if cmd == "a":
            if simulation_only:
                print("[ctrl] Motor enable is unavailable in simulation-only mode.")
            elif _motor_power == "enabling":
                print("[ctrl] Already enabling.")
            else:
                _start_enable()

        elif cmd == "s":
            if _control_active:
                print("[ctrl] Already running — use 'p' to pause")
            else:
                _request_start = True
                print("[ctrl] Starting control loop...")

        elif cmd.startswith("g "):
            parts = raw.split()
            if len(parts) != 3:
                print("[ctrl] Usage: g <joint 1-4> <pos>  e.g.  g 4 20deg  g 4 +20deg  g 4 0.35")
                return
            try:
                j = int(parts[1])
                val_str = parts[2].lower().strip()
                relative = val_str.startswith('+')
                if relative:
                    val_str = val_str[1:]
                in_deg = val_str.endswith('deg') or val_str.endswith('d')
                if val_str.endswith('deg'):
                    val_str = val_str[:-3]
                elif val_str.endswith('d'):
                    val_str = val_str[:-1]
                value = float(val_str)
                if in_deg:
                    value = math.radians(value)
            except ValueError:
                print("[ctrl] Usage: g <joint 1-4> <pos>  e.g.  g 4 20deg  g 4 +20deg  g 4 0.35")
                return
            if j not in ACTUATOR_TO_MOTOR.values():
                print("[ctrl] Joint must be 1-4")
                return
            actuator = next(a for a, m in ACTUATOR_TO_MOTOR.items() if m == j)
            if relative:
                current = limiter.reference_position(actuator)
                target_rad = current + value
            else:
                target_rad = value
            unit_str = f"{math.degrees(value):+.1f}°" if in_deg else f"{value:+.4f} rad"
            _request_goal = {actuator: target_rad}
            print(f"[ctrl] Goal: {actuator} → {target_rad:.4f} rad  ({'relative ' + unit_str if relative else unit_str})")

        elif cmd == "ik" or cmd.startswith("ik "):
            usage = "[ctrl] Usage: ik <x> <y> <z>  (base frame; metres, or mm/cm)  e.g.  ik 0.05 0.25 0.22"
            parts = raw.split()
            if len(parts) != 4:
                print(usage)
                return
            try:
                target = [_parse_length_m(p) for p in parts[1:]]
            except ValueError:
                print(usage)
                return
            if not _control_active:
                # A goal stored while idle would start moving the arm the
                # moment 's' is pressed.
                print("[ctrl] IK needs an active control loop — type 's' first.")
                return
            if _limiter_moving(limiter):
                # The clearance check assumes a rest-to-rest move from the
                # current reference.
                print("[ctrl] A move is still in progress — wait for it to finish, then retry.")
                return
            start = _ik_references({a: limiter.reference_position(a) for a in ACTUATOR_TO_MOTOR})
            with ik_lock:
                result = _arm_iks[_ee_state].solve_nearest(target, start)
            target_text = f"({target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}) m"
            if not result.success:
                print(f"[ctrl] IK {target_text}: not moving — {result.message}")
                _log_events.append(f"ik {target_text} rejected: {result.message}")
                return
            _request_goal = dict(result.joint_targets)
            if result.warning:
                print(f"[ctrl] WARNING: {result.warning}")
            if result.nearest:
                reached = result.achieved_base_m
                print(f"[ctrl] IK {target_text}: {result.message}")
                print(
                    f"[ctrl] Going to the nearest location ({reached[0]:+.4f}, {reached[1]:+.4f}, {reached[2]:+.4f}) m, "
                    f"{1000 * result.error_m:.1f} mm from the target, move {result.move_duration_s:.2f} s"
                )
                if result.window_limited:
                    print("[ctrl]     A joint reached the per-command limit; repeating the command may get closer.")
            else:
                print(
                    f"[ctrl] IK {target_text}: model error {1000 * result.error_m:.3f} mm, "
                    f"move {result.move_duration_s:.2f} s"
                )
            for actuator, q in result.joint_targets.items():
                print(
                    f"[ctrl]     {_ik_joint_label(actuator)} {actuator}: {start[actuator]:+.4f} -> {q:+.4f} rad joint "
                    f"({math.degrees(q - start[actuator]):+.1f}°)"
                )
            if _paused:
                print("[ctrl]     Control is paused — type 'r' to move.")
            _log_events.append(
                f"ik {target_text} -> "
                + (f"nearest location {np.round(result.achieved_base_m, 4)}: " if result.nearest else "")
                + ", ".join(f"{a}={q:+.5f}" for a, q in result.joint_targets.items())
            )

        elif cmd == "w":
            active_ik = _arm_iks[_ee_state]
            with ik_lock:
                reference = active_ik.tool_position_base(
                    _ik_references({a: limiter.reference_position(a) for a in ACTUATOR_TO_MOTOR})
                )
                goal = active_ik.tool_position_base(
                    _ik_references({a: limiter.requested_position(a) for a in ACTUATOR_TO_MOTOR}, "goal")
                )
                measured_refs = _measured_references(limiter, actuator_gear, simulation_only)
                measured = None if measured_refs is None else active_ik.tool_position_base(measured_refs)
            fmt = lambda p: f"({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) m"
            print(f"[ctrl] {_ee_state.value} tool point, base frame: reference {fmt(reference)}  target {fmt(goal)}")
            if measured is not None:
                print(f"[ctrl]     from encoders {fmt(measured)}  "
                      f"(reference - encoders = {1000 * math.dist(reference, measured):.1f} mm)")

        elif cmd == "state" or cmd.startswith("state "):
            parts = cmd.split()
            states = {state.value.lower(): state for state in EndEffectorState}
            if len(parts) == 1:
                print(f"[ctrl] End-effector state: {_ee_state.value}")
            elif len(parts) == 2 and parts[1] in states:
                print(f"[ctrl] {_request_end_effector_state(states[parts[1]], limiter, ik_lock)}")
            else:
                print("[ctrl] Usage: state [gripping|pushing]")

        elif cmd == "h":
            _request_home = True
            print("[ctrl] Going to hardware_zero pose...")

        elif cmd == "p":
            _paused = True
            print("[ctrl] Control paused — no commands being sent")

        elif cmd == "r":
            _paused = False
            print("[ctrl] Control resumed")

        elif cmd == "z":
            _request_rezero = True

        elif cmd == "f":
            PRINT_MOTOR_FEEDBACK = not PRINT_MOTOR_FEEDBACK
            print(f"[ctrl] Live controller telemetry {'ON' if PRINT_MOTOR_FEEDBACK else 'OFF'}")

        elif cmd == "d":
            DEBUG_COMMANDS = not DEBUG_COMMANDS
            print(f"[ctrl] Command debug {'ON' if DEBUG_COMMANDS else 'OFF'}")

        elif cmd == "q":
            pass  # only stdin thread should quit

    # --- Terminal input thread (stdin) ---
    def _terminal_input() -> None:
        while True:
            try:
                raw = input().strip()
            except EOFError:
                break
            if raw.lower() == "q":
                _log_events.append("cmd q")
                break
            _handle_command(raw)

    threading.Thread(target=_terminal_input, daemon=True).start()

    # Also accept commands from a named pipe so a second terminal can send
    # commands without debug output interfering.
    # Usage in a second terminal:  echo "g 4 20deg" > /tmp/robot_cmd
    FIFO_PATH = "/tmp/robot_cmd"
    try:
        import os
        if not os.path.exists(FIFO_PATH):
            os.mkfifo(FIFO_PATH)
        print(f"[ctrl] Named pipe ready — send commands from another terminal:")
        print(f"[ctrl]   echo 'g 4 20deg' > {FIFO_PATH}")

        def _fifo_input() -> None:
            while True:
                try:
                    with open(FIFO_PATH, "r") as fifo:
                        for line in fifo:
                            _handle_command(line.strip())
                except Exception:
                    time.sleep(0.1)

        threading.Thread(target=_fifo_input, daemon=True).start()
    except Exception as e:
        print(f"[ctrl] Named pipe unavailable: {e}")

    # --- Per-run cycle log (mandatory) ---
    run_metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "script": str(Path(__file__).resolve()),
        "git_revision": _git_revision(),
        "argv": sys.argv,
        "serial_port": port,
        "simulation_only": simulation_only,
        "enable_motors_on_start": enable_motors_on_start,
        "gain_schedule": str(SCHEDULE_PATH) if SCHEDULE_PATH.exists() else None,
        "settings": {
            "UPDATE_HZ": UPDATE_HZ,
            "SEND_EVERY": SEND_EVERY,
            "CLOSED_LOOP": CLOSED_LOOP,
            "ZERO_TFF": ZERO_TFF,
            "MOTION_PROFILE": MOTION_PROFILE,
            "MOTOR_DIRECTION": MOTOR_DIRECTION,
            "MOTOR_SCALE": MOTOR_SCALE,
            "TORQUE_FF_SCALE": TORQUE_FF_SCALE,
            "TAU_FF_LIMIT": si.TAU_FF_LIMIT,
            "DEFAULT_GAINS": DEFAULT_GAINS,
            "CONSTRAINED_JOINTS": CONSTRAINED_JOINTS,
            "HARDWARE_ZERO_ACTUATOR_CTRL_RAD": HARDWARE_ZERO_ACTUATOR_CTRL_RAD,
            "STREAM_KEEPALIVE_S": STREAM_KEEPALIVE_S,
            "SERVOS": SERVOS,
            "SERVO_CHANNELS": SERVO_CHANNELS,
            "SERVO_LIMITS_DEG": SERVO_LIMITS_DEG,
            "PUSHING_WRIST_SERVO_DEG": PUSHING_WRIST_SERVO_DEG,
            "PUSHING_TOOL_JOINT": PUSHING_TOOL_JOINT,
            "SERVO_PARK_DEG": SERVO_PARK_DEG,
            "WRIST_SERVO_CENTER_JOINT_RAD": WRIST_SERVO_CENTER_JOINT_RAD,
            "WRIST_SERVO_DIRECTION": WRIST_SERVO_DIRECTION,
            "SERVO_JOINTS": SERVO_JOINTS,
            "RUNAWAY_STOP": RUNAWAY_STOP,
            "RUNAWAY_POS_ERROR_RAD": RUNAWAY_POS_ERROR_RAD,
            "RUNAWAY_POS_ERROR_S": RUNAWAY_POS_ERROR_S,
            "RUNAWAY_SPEED_RAD_S": RUNAWAY_SPEED_RAD_S,
            "RUNAWAY_SPEED_PACKETS": RUNAWAY_SPEED_PACKETS,
            "RUNAWAY_FEEDBACK_TIMEOUT_S": RUNAWAY_FEEDBACK_TIMEOUT_S,
            "ECHO_CHECK": ECHO_CHECK,
            "ECHO_MISMATCH_COUNT": ECHO_MISMATCH_COUNT,
            "ECHO_WINDOW_S": ECHO_WINDOW_S,
            "ECHO_TOLERANCE_COUNTS": ECHO_TOLERANCE_COUNTS,
            "MOTOR_FAULT_STOP": MOTOR_FAULT_STOP,
            "MOTOR_TYPE": si.MOTOR_TYPE,
        },
    }
    try:
        cycle_log = _open_cycle_log(qpos_idx, limiter, run_metadata)
    except OSError as exc:
        print(f"[ctrl] ERROR: cannot create the run log in {CYCLE_LOG_DIRECTORY}: {exc}")
        print("[ctrl] Refusing to run without a log.")
        si.close()
        sys.exit(1)
    cycle_log_start_time = time.perf_counter()
    run_log_file, run_log_path = cycle_log[0], cycle_log[2]
    atexit.register(_finish_run_log, run_log_file, run_log_path)
    print(f"[ctrl] Logging this run to {run_log_path}")

    dt = 1.0 / UPDATE_HZ
    tick = 0
    streamer = _CommandStreamer()
    runaway = _RunawayMonitor()
    echo_monitor = _EchoMonitor()
    last_servo_print = 0.0
    idle_slider_hint_shown = False
    stm32_print_window_s = 0.0
    stm32_print_count = 0
    latest_command_gains: dict[int, tuple[float, float] | None] = {
        motor: None for motor in MOTOR_TO_JOINT
    }
    last_motor_err: dict[int, int | None] = {motor: None for motor in MOTOR_TO_JOINT}

    disable_pending_check = False
    disable_sent_at = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        panel: ControlPanel | None = None
        panel_backend: _GuiBackend | None = None
        if use_gui:
            panel_backend = _GuiBackend(
                model, limiter, ik_lock, actuator_gear, simulation_only, port, viewer, servos
            )
            panel = ControlPanel(panel_backend, title=f"Robot arm control - {port or 'simulation'}")
            print("[ctrl] Control panel open (closing it twice quits the controller).")
        next_tick  = time.perf_counter()
        last_print = time.perf_counter()
        last_feedback_print = 0.0

        while viewer.is_running():

            # --- Handle deferred terminal commands ---

            if _request_start and not _control_active:
                if _request_start_home:
                    # Control panel "Start - arm is at home": the operator
                    # confirmed the arm is at hardware_zero, so make the
                    # reference match it before the sync below.
                    _request_start_home = False
                    for actuator, ctrl_val in HARDWARE_ZERO_ACTUATOR_CTRL_RAD.items():
                        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
                        if actuator_id >= 0:
                            data.ctrl[actuator_id] = ctrl_val
                    limiter.reset(data)
                    _log_events.append("reference reset to hardware_zero for start")
                # Capture a pending native Control-tab edit before establishing
                # the no-jump hardware-position offset.
                limiter.update(data)
                _control_active = True
                _request_start  = False
                _paused         = False

                pos_raw, _ = si.raw_snapshot()
                missing = [m for m in [1, 2, 3, 4] if pos_raw[m] is None]
                _encoder_feedback_at_sync.clear()
                for motor in MOTOR_TO_JOINT:
                    _encoder_at_sync[motor] = pos_raw[motor] if pos_raw[motor] is not None else 0.0
                    if pos_raw[motor] is not None:
                        _encoder_feedback_at_sync.add(motor)

                for motor, jname in MOTOR_TO_JOINT.items():
                    if jname in qpos_idx:
                        _sim_at_sync[jname] = float(data.qpos[qpos_idx[jname]])
                for follower in CONSTRAINED_JOINTS:
                    if follower in qpos_idx:
                        _sim_at_sync[follower] = float(data.qpos[qpos_idx[follower]])

                if missing:
                    print(f"[ctrl] WARNING: no feedback yet from motors {missing} — their encoder snapshot defaults to 0")
                snap_str = ", ".join(f"M{m}={_encoder_at_sync[m]:.3f}" for m in MOTOR_TO_JOINT)
                print(f"[ctrl] Encoder snapshot: {snap_str}")

                _pos_cmd_offset.clear()
                for actuator, motor in ACTUATOR_TO_MOTOR.items():
                    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
                    # data.ctrl is already in the actuator's motor coordinate.
                    # Applying gear again displaced AK40 home by 0.485 rad.
                    ref_now = MOTOR_DIRECTION[motor] * float(data.ctrl[act_id])
                    _pos_cmd_offset[motor] = _encoder_at_sync[motor] - ref_now
                off_str = ", ".join(f"M{m}={v:+.4f}" for m, v in sorted(_pos_cmd_offset.items()))
                print(f"[ctrl] Position command offsets: {off_str}")
                _log_events.append(
                    f"start snapshot {snap_str}; offsets {off_str}"
                    + (f"; no feedback from {missing}" if missing else "")
                )
                # An IK wrist move interrupted by a hold or stop does not resume.
                if _ik_wrist is not None:
                    _ik_wrist.move = None
                # First tick after start sends a full frame to every motor.
                streamer.reset()
                runaway.reset()
                echo_monitor.reset()

                # Keep the synchronized reference while retaining the target
                # selected through MuJoCo's Control tab.
                print("[ctrl] Control loop active — sim moves relative to sync position.")

            if _request_hold:
                _request_hold = False
                _request_goal = None
                if not simulation_only:
                    si.emergency_stop()
                    _motor_power = "held (0X)"
                _control_active = False
                _paused = False
                _notice = ("HOLD: 0X sent, motors frozen, streaming stopped. Enable and Start to continue.", True)
                _log_events.append("HOLD: 0X sent")
                print("[ctrl] HOLD: 0X sent; no commands are being streamed.")

            if _request_disable:
                _request_disable = False
                _request_goal = None
                _control_active = False
                _paused = False
                if not simulation_only:
                    # Stop streaming first (above), so nothing overwrites the
                    # zero-gain frames.  Kp = Kd = T = 0 is what the firmware
                    # itself sends to a disabled motor: zero torque.  Each
                    # frame is sent twice in case one line is damaged.
                    pos_raw, _ = si.raw_snapshot()
                    disable_sent_at = time.perf_counter()
                    for _ in range(2):
                        for motor in MOTOR_TO_JOINT:
                            si.send_mit_frame(
                                motor, pos_raw[motor] if pos_raw[motor] is not None else 0.0, 0.0, 0.0, 0.0, 0.0
                            )
                    disable_pending_check = True
                    _motor_power = "limp (disabled)"
                    _notice = ("Disabled: zero-gain frames sent; checking the command echoes...", False)
                    _log_events.append("DISABLE: zero-gain frames sent")
                    print("[ctrl] DISABLE: streaming stopped; zero-gain (limp) frames sent to M1-M4.")

            if disable_pending_check and time.perf_counter() - disable_sent_at >= DISABLE_ECHO_CHECK_S:
                # The firmware echoes the CAN frame it sends to each enabled
                # motor; a zero-gain echo newer than the disable confirms it.
                disable_pending_check = False
                echoes, echo_times = si.echo_snapshot()
                confirmed, unconfirmed = [], []
                for motor in MOTOR_TO_JOINT:
                    echo = echoes[motor]
                    limp = (
                        echo is not None and echo_times[motor] > disable_sent_at
                        and abs(echo.kp) < 1e-6 and abs(echo.kd) < 1e-6 and abs(echo.tau_ff) < 0.01
                    )
                    (confirmed if limp else unconfirmed).append(motor)
                if unconfirmed:
                    text = (f"Disable: zero-gain echo confirmed for {confirmed or 'no motor'}; none from "
                            f"{unconfirmed} (a motor that is not enabled already gets zero torque from the "
                            "firmware). Check those motors.")
                    _notice = (text, True)
                else:
                    text = "Disable confirmed: every motor echoes zero gains (limp)."
                    _notice = (text, False)
                _log_events.append(text)
                print(f"[ctrl] {text}")

            if _request_goal is not None and _control_active:
                for actuator, target in _request_goal.items():
                    if actuator in ACTUATOR_TO_MOTOR:
                        limiter.set_requested_position(actuator, target)
                    elif _ik_wrist is not None and actuator == SERVO_CHANNELS["wrist"][2]:
                        # Same profile and limits as the limiter, from the same tick.
                        _ik_wrist.start_move(target, float(data.time), limiter.max_velocity_rad_s,
                                             limiter.max_acceleration_rad_s2)
                _request_goal = None

            if _request_ee_state is not None:
                _ee_state, _request_ee_state = _request_ee_state, None
                if _ik_wrist is not None:
                    _ik_wrist.locked = _ee_state is EndEffectorState.PUSHING
                    if _ik_wrist.locked:
                        _ik_wrist.start_move(_wrist_joint_rad(PUSHING_WRIST_SERVO_DEG), float(data.time),
                                             limiter.max_velocity_rad_s, limiter.max_acceleration_rad_s2)
                _notice = (f"End-effector state: {_ee_state.value}.", False)
                _log_events.append(f"end-effector state {_ee_state.value}")
                print(f"[ctrl] End-effector state: {_ee_state.value}.")

            # Pushing's IK assumes this wrist angle.  A hold during the swing
            # can leave the wrist short of it, so fall back instead of solving
            # for a pose the arm is not in.
            if (
                _ee_state is EndEffectorState.PUSHING and _ik_wrist.move is None
                and abs(_ik_wrist.target_deg - PUSHING_WRIST_SERVO_DEG) > 0.05
            ):
                _ee_state = EndEffectorState.GRIPPING
                _ik_wrist.locked = False
                _notice = (f"Back to Gripping: the wrist stopped at {_ik_wrist.target_deg:.0f} deg, "
                           f"short of {PUSHING_WRIST_SERVO_DEG:.0f}.", True)
                _log_events.append(_notice[0])
                print(f"[ctrl] {_notice[0]}")

            if _request_home and _control_active:
                for actuator, ctrl_val in HARDWARE_ZERO_ACTUATOR_CTRL_RAD.items():
                    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
                    if actuator_id >= 0:
                        gear = float(model.actuator_gear[actuator_id, 0])
                        limiter.set_requested_position(actuator, ctrl_val / gear)
                _request_home = False

            if _request_rezero:
                _request_rezero = False
                if simulation_only:
                    print("[ctrl] Re-zero is unavailable in simulation-only mode.")
                    continue
                pos_raw, vel_raw = si.raw_snapshot()
                print("[ctrl] Current encoder readings:")
                for motor, jname in MOTOR_TO_JOINT.items():
                    p = pos_raw[motor]
                    v = vel_raw[motor]
                    if p is None:
                        print(f"    M{motor} ({jname}): no feedback")
                    else:
                        print(f"    M{motor} ({jname}): pos={p:.5f} rad  vel={v:.5f} rad/s")

            # --- Snapshot encoder feedback (lag-compensated) ---
            # Simulation-only mode intentionally has no serial traffic and
            # drives MuJoCo from the limiter reference below.
            if simulation_only:
                pos_snap = {motor: None for motor in MOTOR_TO_JOINT}
                vel_snap = {motor: 0.0 for motor in MOTOR_TO_JOINT}
                log_pos = {motor: None for motor in MOTOR_TO_JOINT}
                log_vel = {motor: None for motor in MOTOR_TO_JOINT}
                log_age = {motor: None for motor in MOTOR_TO_JOINT}
                log_current: dict[int, float | None] = {motor: None for motor in MOTOR_TO_JOINT}
                log_torque: dict[int, float | None] = {motor: None for motor in MOTOR_TO_JOINT}
                log_temp: dict[int, int | None] = {motor: None for motor in MOTOR_TO_JOINT}
                log_err: dict[int, int | None] = {motor: None for motor in MOTOR_TO_JOINT}
                log_echo = None
                log_echo_age = None
                new_echoes = []
            else:
                fb = si.snapshot()
                pos_snap = fb.pos
                vel_snap = fb.vel
                # The log keeps the raw (non-extrapolated) packet values and
                # how old each one is, so stale feedback is visible.
                log_pos, raw_vel = si.raw_snapshot()
                fb_now = time.perf_counter()
                log_vel = {m: (raw_vel[m] if log_pos[m] is not None else None) for m in MOTOR_TO_JOINT}
                log_age = {m: (fb_now - fb.time[m] if log_pos[m] is not None else None) for m in MOTOR_TO_JOINT}
                log_current = fb.current
                log_torque = fb.torque
                log_temp = fb.temp
                log_err = fb.err
                # Every fault-code change is a run-log event, active or not.
                for m in MOTOR_TO_JOINT:
                    if log_err[m] is not None and log_err[m] != last_motor_err[m]:
                        text = si.MOTOR_ERROR_TEXT.get(log_err[m], "unknown code")
                        _log_events.append(f"M{m} fault code err={log_err[m]} ({text})")
                        if log_err[m] != 0 or last_motor_err[m] is not None:
                            print(f"[ctrl] M{m} fault code err={log_err[m]} ({text})")
                        last_motor_err[m] = log_err[m]
                # Drained every tick, so the echo check never sees echoes
                # from before a start or while paused.
                new_echoes = si.drain_echoes()
                log_echo, echo_time = si.echo_snapshot()
                log_echo_age = {
                    m: (fb_now - echo_time[m] if log_echo[m] is not None else None) for m in MOTOR_TO_JOINT
                }
                # Firmware messages that are not feedback (faults, watchdog,
                # unknown CAN ids): always logged, printed at a capped rate.
                for _, text in si.drain_other_lines():
                    _log_events.append(f"stm32: {text}")
                    if fb_now - stm32_print_window_s >= 1.0:
                        stm32_print_window_s, stm32_print_count = fb_now, 0
                    stm32_print_count += 1
                    if stm32_print_count <= STM32_PRINT_MAX_PER_S:
                        print(f"[stm32] {text}")
                    elif stm32_print_count == STM32_PRINT_MAX_PER_S + 1:
                        print("[stm32] ... more messages this second (see run log events)")
            commanded: dict[int, tuple[float, float, float, float, float, str]] = {}

            # --- Status print ---
            now = time.perf_counter()
            if now - last_print >= 2.0:
                last_print = now
                parts = [
                    f"M{m}={pos_snap[m]:.3f}"
                    for m in [1, 2, 3, 4]
                    if pos_snap[m] is not None
                ]
                status = "ACTIVE" if (_control_active and not _paused) else ("PAUSED" if _paused else "IDLE")
                print(f"[ctrl/{status}] " + (", ".join(parts) if parts else "no feedback yet"))

            # --- Update MuJoCo state ---
            if CLOSED_LOOP and not simulation_only:
                _apply_encoder_state(data, qpos_idx, qvel_idx, keyframe_qpos, pos_snap, vel_snap)
            else:
                _apply_reference_state(data, qpos_idx, qvel_idx, keyframe_qpos, limiter)

            # --- Servos: follow their sliders whenever the port is open,
            # independent of the motors' enable/start state ---
            for servo in servos.values():
                servo.sync_control_tab(data)       # a Control-tab edit cancels an IK move
                servo.advance_move(float(data.time))
                servo_line = servo.update(send=not simulation_only)
                if servo_line:
                    _log_events.append(f"servo {servo.name} {servo.target_deg:.1f} deg: {servo_line}")
                    if DEBUG_COMMANDS and time.perf_counter() - last_servo_print >= 0.5:
                        last_servo_print = time.perf_counter()
                        print(f"[cmd] {servo.name} {servo.target_deg:.1f} deg -> {servo_line}")
            for servo in servos.values():
                _apply_servo_state(data, model, qpos_idx, qvel_idx, servo)
            mujoco.mj_forward(model, data)

            if _control_active and not _paused:

                # Hardware supplies qpos/qvel directly, so no mj_step advances
                # the limiter's clock. Advance it at the real control period.
                data.time += dt

                # 1. Schedule PD gains from current inertia + position.
                gain_updates = controller.update(model, data)

                # 2. Treat native MuJoCo Control-tab sliders as requested
                # targets and write the smooth reference back to data.ctrl.
                limiter.update(data)

                # 3. Compute gravity-compensation feedforward.
                tau_ff = _compute_tau_ff(model, data, qpos_idx, qvel_idx)

                # 4. Runaway check against the encoders, before anything is
                # sent this tick.  Only motors with feedback at sync are
                # monitored; unpowered motors never report and are skipped.
                trip_reason = None
                if MOTOR_FAULT_STOP and not simulation_only:
                    for motor in sorted(_encoder_feedback_at_sync):
                        if log_err[motor]:
                            text = si.MOTOR_ERROR_TEXT.get(log_err[motor], "unknown code")
                            trip_reason = f"M{motor} motor fault err={log_err[motor]} ({text})"
                            break
                # A command the STM32 latched wrongly shows in its echo before
                # the motor moves, so check echoes first.
                if ECHO_CHECK and not simulation_only and not trip_reason:
                    for received, motor, echo in new_echoes:
                        trip_reason = echo_monitor.check(motor, echo, received)
                        if trip_reason:
                            break
                if RUNAWAY_STOP and not simulation_only and not trip_reason:
                    check_now = time.perf_counter()
                    for actuator, motor in ACTUATOR_TO_MOTOR.items():
                        if motor not in _encoder_feedback_at_sync:
                            continue
                        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
                        if actuator_id < 0:
                            continue
                        gear = float(model.actuator_gear[actuator_id, 0])
                        cmd_pos_motor = (
                            MOTOR_DIRECTION[motor] * limiter.reference_position(actuator) * gear
                            + _pos_cmd_offset.get(motor, 0.0)
                        )
                        # Raw (non-extrapolated) packet values, in the same
                        # motor-side frame as the P command.
                        trip_reason = runaway.check(
                            motor, cmd_pos_motor, log_pos[motor], log_vel[motor], fb.time[motor], check_now
                        )
                        if trip_reason:
                            break

                if trip_reason:
                    si.emergency_stop()
                    _control_active = False
                    _paused = False
                    _motor_power = "held (0X)"
                    _notice = (f"RUNAWAY STOP: {trip_reason}. 0X sent; Enable and Start to continue.", True)
                    _log_events.append(f"RUNAWAY STOP: {trip_reason}")
                    print(f"[ctrl] *** RUNAWAY STOP: {trip_reason} ***")
                    print("[ctrl]     Sent 0X; no commands are being streamed. Check the arm,")
                    print("[ctrl]     then type 'a' and 's' to re-enable and re-synchronize.")

                # 5. Send MIT frames every SEND_EVERY ticks.
                elif not simulation_only and tick % SEND_EVERY == 0:
                    for actuator, motor in ACTUATOR_TO_MOTOR.items():
                        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
                        if actuator_id < 0:
                            continue

                        gear = float(model.actuator_gear[actuator_id, 0])

                        ref_pos_joint = limiter.reference_position(actuator)
                        ref_vel_joint = limiter.reference_velocity(actuator)
                        ref_pos_motor = (
                            MOTOR_DIRECTION[motor] * ref_pos_joint * gear + _pos_cmd_offset.get(motor, 0.0)
                        )
                        ref_vel_motor = MOTOR_DIRECTION[motor] * ref_vel_joint * gear

                        if actuator in gain_updates:
                            gu = gain_updates[actuator]
                            kp_motor = gu.kp_output_nm_per_rad / (gear ** 2)
                            kd_motor = gu.kd_output_nm_s_per_rad / (gear ** 2)
                        else:
                            kp_motor, kd_motor = DEFAULT_GAINS[actuator]

                        tau_ff_motor = 0.0 if ZERO_TFF else tau_ff.get(motor, 0.0)
                        # The motor-side gains shown in terminal telemetry.
                        latest_command_gains[motor] = (kp_motor, kd_motor)

                        streamer.send(
                            motor,
                            ref_pos_motor,
                            ref_vel_motor,
                            kp_motor,
                            kd_motor,
                            tau_ff_motor,
                        )
                        echo_monitor.record_sent(
                            motor, streamer.last_sent_values.get(motor, {}), time.perf_counter()
                        )
                        commanded[motor] = (
                            ref_pos_motor,
                            ref_vel_motor,
                            kp_motor,
                            kd_motor,
                            tau_ff_motor,
                            streamer.last_sent_fields.get(motor, ""),
                        )

                        if DEBUG_COMMANDS and tick % UPDATE_HZ == 0:
                            print(
                                f"[cmd] M{motor} "
                                f"P={ref_pos_motor:+.4f} "
                                f"V={ref_vel_motor:+.4f} "
                                f"Kp={kp_motor:.3f} "
                                f"Kd={kd_motor:.3f} "
                                f"T={tau_ff_motor:+.4f}"
                            )

            else:
                # Slider edits select the next target while idle or paused,
                # but cannot move the physical reference until control starts.
                requested_before = {a: limiter.requested_position(a) for a in ACTUATOR_TO_MOTOR}
                limiter.update(data)
                slider_moved = any(
                    abs(limiter.requested_position(a) - p) > 1e-7 for a, p in requested_before.items()
                )
                if slider_moved and not simulation_only and not idle_slider_hint_shown:
                    idle_slider_hint_shown = True
                    state = "paused — type 'r'" if _paused else "idle — type 's'"
                    print(f"[ctrl] Slider target stored, but control is {state} to stream it to the motors.")

            now = time.perf_counter()
            if (
                PRINT_MOTOR_FEEDBACK
                and now - last_feedback_print >= 1.0 / MOTOR_FEEDBACK_PRINT_HZ
            ):
                last_feedback_print = now
                _print_live_telemetry(
                    data,
                    qpos_idx,
                    log_pos,
                    latest_command_gains,
                    simulation_only,
                )

            if cycle_log is not None:
                try:
                    _write_cycle_log(
                        cycle_log[0],
                        cycle_log[1],
                        cycle_log_start_time,
                        tick,
                        simulation_only,
                        data,
                        qpos_idx,
                        qvel_idx,
                        limiter,
                        log_pos,
                        log_vel,
                        log_age,
                        commanded,
                        log_current,
                        log_torque,
                        log_echo,
                        log_echo_age,
                        log_temp,
                        log_err,
                    )
                except OSError as exc:
                    print(f"[ctrl] WARNING: run logging stopped: {exc}")
                    cycle_log[0].close()
                    cycle_log = None

            if panel is not None:
                # Pumped every tick on this thread; the panel never blocks.
                panel.pump()
                panel_backend.render()

            viewer.sync()
            tick += 1

            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    if panel is not None:
        panel.destroy()
    _finish_run_log(run_log_file, run_log_path)
    si.close()
    if simulation_only:
        print("[ctrl] Simulation closed. Bye.")
    else:
        print("[ctrl] Serial closed. Bye.")


if __name__ == "__main__":
    main()
