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

Serial protocol (to STM32):
    <n>P<val>\\n  — position setpoint (rad)
    <n>V<val>\\n  — velocity setpoint (rad/s)
    <n>K<val>\\n  — Kp gain
    <n>D<val>\\n  — Kd gain
    <n>T<val>\\n  — torque feedforward (N·m)
    <n>E\\n       — enable motor

Requires: mujoco, pyserial, numpy
"""

from __future__ import annotations

import atexit
import csv
import json
import math
import platform
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

MODEL_PATH = Path(__file__).parent / "robot_master" / "mjcf" / "robot_master.xml"
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

# Set to True to send zero torque feedforward to all motors.
# Useful for initial testing before verifying TFF sign and magnitude.
ZERO_TFF = True

# Every run writes logs/<branch>_<YYYY-MM-DD>_<HH-MM-SS UTC>.csv (one row per
# control tick, flushed immediately), a matching .json with the run's
# settings and git revision, and a .svg graph (plot_run_log.py) written when
# the run ends.  The controller refuses to start without a log.
CYCLE_LOG_DIRECTORY = Path(__file__).parent / "logs"

# Live slider streaming.  A full P/V/K/D/T frame for all four motors every
# tick is ~220 bytes x 84 Hz = ~18 kB/s, but 115200 baud carries ~11.5 kB/s,
# so the serial write blocks and slider motion reaches the motors late.
# Instead, each tick sends only the fields that changed (P/V while a slider
# move is in progress), plus a full frame per motor every KEEPALIVE_S.
STREAM_POS_EPS_RAD   = 1e-4   # resend P when it moves more than this (motor side)
STREAM_VEL_EPS_RAD_S = 1e-3   # resend V when it changes more than this
STREAM_GAIN_REL_EPS  = 0.02   # resend K/D on a >2% change
STREAM_TFF_EPS_NM    = 0.01   # resend T when it changes more than this
STREAM_KEEPALIVE_S   = 0.25   # full frame per motor at least this often

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
MOTOR_DIRECTION = {1: 1.0, 2: 1.0, 3: 1.0, 4: -1.0}
MOTOR_SCALE = {1: 1.0, 2: 1.0, 3: 1.0, 4: MOTOR_DIRECTION[4] / 1.25}

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
TORQUE_FF_SCALE = {1: 1.0, 2: 1.0, 3: 1.0, 4: MOTOR_DIRECTION[4] / 1.25}

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

class _CommandStreamer:
    """Send each motor's MIT fields only when they change, plus a keepalive."""

    def __init__(self) -> None:
        self._last: dict[int, dict[str, float]] = {}
        self._last_full: dict[int, float] = {}
        # Field letters written on each motor's latest send() ("" = nothing), for the cycle log.
        self.last_sent_fields: dict[int, str] = {}

    def reset(self) -> None:
        self._last.clear()
        self._last_full.clear()
        self.last_sent_fields.clear()

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
        if last is not None and vel == 0.0 and last["vel"] != 0.0:
            fields["vel"] = 0.0
            fields["pos"] = pos
        self.last_sent_fields[motor] = "".join(
            letter for letter, key in zip("PVKDT", fields) if fields[key] is not None
        )
        if all(v is None for v in fields.values()):
            return False

        si.send_mit_update(motor, **fields)
        entry = self._last.setdefault(motor, {})
        for key, value in fields.items():
            if value is not None:
                entry[key] = value
        if full:
            self._last_full[motor] = now
        return True

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
                f"m{motor}_feedback_age_s",
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
        row[f"m{motor}_feedback_age_s"] = feedback_age[motor]
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _ser, _control_active, _request_start, _request_pause
    global _paused, _request_goal, _request_home, _request_rezero

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
            "STREAM_KEEPALIVE_S": STREAM_KEEPALIVE_S,
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
    idle_slider_hint_shown = False
    latest_command_gains: dict[int, tuple[float, float] | None] = {
        motor: None for motor in MOTOR_TO_JOINT
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_tick  = time.perf_counter()
        last_print = time.perf_counter()
        last_feedback_print = 0.0

        while viewer.is_running():

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
                # First tick after start sends a full frame to every motor.
                streamer.reset()

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
                # the limiter's clock. Advance it at the real control period.
                data.time += dt

                # 1. Schedule PD gains from current inertia + position.
                gain_updates = controller.update(model, data)

                # 2. Treat native MuJoCo Control-tab sliders as requested
                # targets and write the smooth reference back to data.ctrl.
                limiter.update(data)

                # 3. Compute gravity-compensation feedforward.
                tau_ff = _compute_tau_ff(model, data, qpos_idx, qvel_idx)

                # 4. Send MIT frames every SEND_EVERY ticks.
                if not simulation_only and tick % SEND_EVERY == 0:
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
                    )
                except OSError as exc:
                    print(f"[ctrl] WARNING: run logging stopped: {exc}")
                    cycle_log[0].close()
                    cycle_log = None

            viewer.sync()
            tick += 1

            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    _finish_run_log(run_log_file, run_log_path)
    si.close()
    if simulation_only:
        print("[ctrl] Simulation closed. Bye.")
    else:
        print("[ctrl] Serial closed. Bye.")


if __name__ == "__main__":
    main()
