#!/usr/bin/env python3
"""
hardware_controller.py — Closed-loop control of the physical robot arm.

Combines:
  - Serial encoder feedback from the STM32 (same format as sync_hardware.py)
  - Position-scheduled PD gains from master_controller.py
  - Trapezoidal motion references from trapezoidal_motion.py
  - Gravity-compensation feedforward via mujoco.mj_inverse
  - MIT control mode command sender (P, V, K, D, T over serial)

Control loop (runs at UPDATE_HZ):
  1. Read latest encoder feedback (pos, vel) from serial thread
  2. Write encoder state into MuJoCo data.qpos / data.qvel
  3. Run mj_forward to update kinematics and mass matrix
  4. MasterController.update() schedules kp/kd from inertia + position
  5. TrapezoidalReferenceLimiter.advance() steps the reference trajectory
  6. mj_inverse computes gravity-compensation feedforward torques
  7. Pack and send MIT control frame per motor: P, V, Kp, Kd, tau_ff

Usage:
    python3 hardware_integrated_main_controller.py [serial_port] [--enable-motors]

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
import json
import math
import platform
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import mujoco.viewer

from robot_master_configuration import (
    HARDWARE_ZERO_KEYFRAME_NAME,
    HARDWARE_ZERO_ACTUATOR_CTRL_RAD,
)
from master_controller import MasterController, PositionSchedule
from trapezoidal_motion import TrapezoidalReferenceLimiter
from plot_run_log import plot_run_log
import serial_interface as si

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "robot_master_urdf_table" / "mjcf" / "robot_master.xml"
SCHEDULE_PATH = Path(__file__).parent / "master_controller_schedule.json"

UPDATE_HZ  = 500  # control loop rate (Hz); the firmware loop runs at the same rate
SEND_EVERY = 1    # send commands every N control ticks (1 = every tick)

# Ticks run on a fixed 2 ms schedule.  A tick that starts late is not made up
# by running the next ones back to back: at 84 Hz that sprint ran the
# trajectory 1.6x faster than real time (2026-10-27 logs, up to 11 s behind).
# The schedule restarts from now, the tick counts in the log's late_ticks, and
# the reference advances by the real time since the last tick (capped at
# MAX_TICK_DT_S so a long stall cannot make it jump).
MAX_TICK_DT_S = 4.0 / UPDATE_HZ
# viewer.sync() is far too slow to run every 2 ms tick; the display only
# refreshes at about this rate anyway.  Slider edits reach data.ctrl through
# it, so they arrive up to 1/VIEWER_HZ late.
VIEWER_HZ = 60

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

# Every run writes logs/<branch>_<YYYY-MM-DD>_<HH-MM-SS UTC>.csv (LOG_HZ rows
# per second, written by a background thread so disk writes never delay a
# tick), a matching .json with the run's settings and git revision, and a
# .svg graph (plot_run_log.py) written when the run ends.  Events (commands,
# trips, STM32 messages) are never dropped: each goes into the next row.
# The controller refuses to start without a log.
CYCLE_LOG_DIRECTORY = Path(__file__).parent / "logs"
LOG_HZ = 100

# Command streaming: every tick sends all four motors' commands as one
# checksummed "A" line, the CAN frames packed on the PC (~70 bytes, ~35 kB/s
# at 500 Hz on a 2 Mbaud link that carries 200 kB/s).  The firmware applies
# all four together or, if the line is damaged, none of them; the next line
# replaces it 2 ms later.

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
# Feedback now arrives at up to 500 Hz, so 15 packets is ~30 ms (M1, which
# replies at about half rate, ~60 ms), close to the old 3 packets at 84 Hz.
RUNAWAY_SPEED_RAD_S   = 3.0
RUNAWAY_SPEED_PACKETS = 15
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
# The echo is the frame as packed for CAN, from the counts the PC packed
# itself, so it should match exactly; two counts leave margin.
ECHO_TOLERANCE_COUNTS = 2.0

# A motor that reports a nonzero CubeMars fault code (si.MOTOR_ERROR_TEXT)
# while control is active trips the same stop.  Fault codes are always
# logged (m<n>_encoder_err) and every change is a run-log event.
MOTOR_FAULT_STOP = True

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

# ---------------------------------------------------------------------------
# Control state flags (written by terminal thread, read by main loop)
# ---------------------------------------------------------------------------

_control_active  = False   # True once 's' is pressed
_request_start   = False
_request_pause   = False
_paused          = False
_request_goal: dict[str, float] | None = None   # actuator → target (rad, joint side)
_request_home    = False
_request_rezero  = False

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


class _CycleLogWriter:
    """Write cycle-log rows on a background thread, so CSV formatting and disk
    writes never delay a 2 ms control tick."""

    def __init__(self, writer: csv.DictWriter) -> None:
        self._writer = writer
        self._queue: queue.Queue[dict[str, object] | None] = queue.Queue()
        self.error: OSError | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def put(self, row: dict[str, object]) -> None:
        self._queue.put(row)

    def _run(self) -> None:
        while True:
            row = self._queue.get()
            if row is None:
                return
            try:
                self._writer.writerow(row)
            except OSError as exc:
                self.error = exc
                return

    def close(self) -> None:
        """Write every queued row, then stop.  Safe to call twice."""
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=10.0)


def _finish_run_log(log_file: object, path: Path, log_writer: _CycleLogWriter | None = None) -> None:
    """Close the run's CSV and write its graph next to it. Safe to call twice.

    Registered with atexit as well as called at the normal end, so a crash or
    Ctrl+C still leaves a graph of whatever was logged.
    """
    if path in _finished_logs:
        return
    _finished_logs.add(path)
    if log_writer is not None:
        log_writer.close()
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
        # Timing of the most recent tick: start-to-start interval, time spent
        # before sleeping, and late ticks so far (see MAX_TICK_DT_S).
        "loop_dt_ms",
        "loop_work_ms",
        "late_ticks",
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


def _cycle_log_row(
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
    loop_timing: tuple[float, float, int] | None = None,
) -> dict[str, object]:
    """One controller-cycle record; drains the pending events into it."""
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
    if loop_timing is not None:
        row["loop_dt_ms"], row["loop_work_ms"], row["late_ticks"] = loop_timing
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
    return row

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
# Loop timing
# ---------------------------------------------------------------------------

def _request_fine_timer() -> None:
    """Ask Windows for 1 ms timer resolution.  The default ~15.6 ms made
    time.sleep() overshoot the old 11.9 ms tick (59-75 Hz measured before
    's' on 2026-10-27) and is useless for a 2 ms tick."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        winmm = ctypes.WinDLL("winmm")
        if winmm.timeBeginPeriod(1) == 0:
            atexit.register(winmm.timeEndPeriod, 1)
    except OSError:
        pass


def _sleep_until(deadline: float) -> None:
    """Sleep until time.perf_counter() reaches deadline.  The OS sleeps in ~1 ms
    steps at best, so the last ~1.5 ms is a spin that yields (sleep(0)) so the
    serial reader thread still gets the GIL."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return
        if remaining > 0.0015:
            time.sleep(remaining - 0.001)
        else:
            time.sleep(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _ser, _control_active, _request_start, _request_pause
    global _paused, _request_goal, _request_home, _request_rezero

    _request_fine_timer()
    # The serial reader thread parses up to 500 feedback lines per second;
    # let it take the GIL within 0.5 ms instead of the default 5 ms.
    sys.setswitchinterval(0.0005)

    arguments = sys.argv[1:]
    enable_motors_on_start = "--enable-motors" in arguments
    port_arguments = [argument for argument in arguments if argument != "--enable-motors"]
    if len(port_arguments) > 1:
        print("Usage: python3 hardware_integrated_main_controller.py [serial_port] [--enable-motors]")
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

    # --- Trapezoidal reference ---
    limiter = TrapezoidalReferenceLimiter(model, controller)

    # Initialise ctrl to hardware_zero.
    for actuator, ctrl_val in HARDWARE_ZERO_ACTUATOR_CTRL_RAD.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = ctrl_val
    limiter.reset(data)
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
    print("         h            — go to hardware_zero pose")
    print("         p            — pause control loop")
    print("         r            — resume control loop")
    if not simulation_only:
        print("         z            — print current encoder readings")
    print("         f            — toggle live controller telemetry")
    print("         d            — toggle command debug printout")
    print("         q            — quit")
    print()

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
            else:
                threading.Thread(
                    target=si.enable_motors, args=([1, 2, 3, 4],), daemon=True
                ).start()

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
            "MOTOR_DIRECTION": MOTOR_DIRECTION,
            "MOTOR_SCALE": MOTOR_SCALE,
            "TORQUE_FF_SCALE": TORQUE_FF_SCALE,
            "TAU_FF_LIMIT": si.TAU_FF_LIMIT,
            "DEFAULT_GAINS": DEFAULT_GAINS,
            "CONSTRAINED_JOINTS": CONSTRAINED_JOINTS,
            "HARDWARE_ZERO_ACTUATOR_CTRL_RAD": HARDWARE_ZERO_ACTUATOR_CTRL_RAD,
            "VIEWER_HZ": VIEWER_HZ,
            "LOG_HZ": LOG_HZ,
            "MAX_TICK_DT_S": MAX_TICK_DT_S,
            "BAUD_RATE": si.BAUD_RATE,
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
    log_writer = _CycleLogWriter(cycle_log[1])
    atexit.register(_finish_run_log, run_log_file, run_log_path, log_writer)
    print(f"[ctrl] Logging this run to {run_log_path}")

    dt = 1.0 / UPDATE_HZ
    log_every = max(1, round(UPDATE_HZ / LOG_HZ))
    tick = 0
    late_ticks = 0
    last_work_ms = 0.0
    runaway = _RunawayMonitor()
    # Actuator id and gear per actuator, looked up once instead of every tick.
    actuators: dict[str, tuple[int, int, float]] = {}
    for actuator, motor in ACTUATOR_TO_MOTOR.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        if actuator_id < 0:
            raise RuntimeError(f"MJCF has no actuator {actuator!r}; the A line needs all four motors.")
        actuators[actuator] = (motor, actuator_id, float(model.actuator_gear[actuator_id, 0]))
    echo_monitor = _EchoMonitor()
    idle_slider_hint_shown = False
    stm32_print_window_s = 0.0
    stm32_print_count = 0
    latest_command_gains: dict[int, tuple[float, float] | None] = {
        motor: None for motor in MOTOR_TO_JOINT
    }
    last_motor_err: dict[int, int | None] = {motor: None for motor in MOTOR_TO_JOINT}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_tick  = time.perf_counter()
        last_print = time.perf_counter()
        last_feedback_print = 0.0
        last_viewer_sync = 0.0
        last_tick_start = next_tick

        while viewer.is_running():
            tick_start = time.perf_counter()
            tick_dt = tick_start - last_tick_start
            last_tick_start = tick_start

            # --- Handle deferred terminal commands ---

            if _request_start and not _control_active:
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
                runaway.reset()
                echo_monitor.reset()

                # Keep the synchronized reference while retaining the target
                # selected through MuJoCo's Control tab.
                print("[ctrl] Control loop active — sim moves relative to sync position.")

            if _request_goal is not None and _control_active:
                for actuator, target in _request_goal.items():
                    limiter.set_requested_position(actuator, target)
                _request_goal = None

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
            mujoco.mj_forward(model, data)

            if _control_active and not _paused:

                # Hardware supplies qpos/qvel directly, so no mj_step advances
                # the limiter's clock.  Advance it by the real time since the
                # last tick, so a late tick does not slow or speed the move.
                data.time += min(tick_dt, MAX_TICK_DT_S)

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
                    for actuator, (motor, actuator_id, gear) in actuators.items():
                        if motor not in _encoder_feedback_at_sync:
                            continue
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
                    _log_events.append(f"RUNAWAY STOP: {trip_reason}")
                    print(f"[ctrl] *** RUNAWAY STOP: {trip_reason} ***")
                    print("[ctrl]     Sent 0X; no commands are being streamed. Check the arm,")
                    print("[ctrl]     then type 'a' and 's' to re-enable and re-synchronize.")

                # 5. Send all four motors' MIT frames, as one "A" line, every
                # SEND_EVERY ticks.
                elif not simulation_only and tick % SEND_EVERY == 0:
                    frames: dict[int, tuple[float, float, float, float, float]] = {}
                    for actuator, (motor, actuator_id, gear) in actuators.items():
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

                        frames[motor] = (ref_pos_motor, ref_vel_motor, kp_motor, kd_motor, tau_ff_motor)
                        commanded[motor] = (*frames[motor], "PVKDT")

                        if DEBUG_COMMANDS and tick % UPDATE_HZ == 0:
                            print(
                                f"[cmd] M{motor} "
                                f"P={ref_pos_motor:+.4f} "
                                f"V={ref_vel_motor:+.4f} "
                                f"Kp={kp_motor:.3f} "
                                f"Kd={kd_motor:.3f} "
                                f"T={tau_ff_motor:+.4f}"
                            )

                    si.send_all_frames(frames)
                    sent_at = time.perf_counter()
                    for motor, (pos, vel, kp, kd, tau) in frames.items():
                        echo_monitor.record_sent(
                            motor, {"pos": pos, "vel": vel, "kp": kp, "kd": kd, "tau_ff": tau}, sent_at
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

            if cycle_log is not None and tick % log_every == 0:
                if log_writer.error is not None:
                    print(f"[ctrl] WARNING: run logging stopped: {log_writer.error}")
                    cycle_log = None
                else:
                    log_writer.put(_cycle_log_row(
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
                        (tick_dt * 1000.0, last_work_ms, late_ticks),
                    ))

            if tick_start - last_viewer_sync >= 1.0 / VIEWER_HZ:
                last_viewer_sync = tick_start
                viewer.sync()
            tick += 1

            now = time.perf_counter()
            last_work_ms = (now - tick_start) * 1000.0
            next_tick += dt
            if now > next_tick:
                # Late: restart the schedule from now rather than catching up.
                late_ticks += 1
                next_tick = now
            else:
                _sleep_until(next_tick)

    _finish_run_log(run_log_file, run_log_path, log_writer)
    si.close()
    if simulation_only:
        print("[ctrl] Simulation closed. Bye.")
    else:
        print("[ctrl] Serial closed. Bye.")


if __name__ == "__main__":
    main()
