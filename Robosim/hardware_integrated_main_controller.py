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
    python3 hardware_integrated_main_controller.py /dev/ttyACM1

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

import sys
import math
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from robot_master_configuration import (
    HARDWARE_ZERO_KEYFRAME_NAME,
    HARDWARE_ZERO_ACTUATOR_CTRL_RAD,
)
from master_controller import MasterController, PositionSchedule
from trapezoidal_motion import TrapezoidalReferenceLimiter
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
CLOSED_LOOP = False

# Print every MIT frame sent to the STM32 (once per second per motor).
# Toggle at runtime by typing 'd' in the terminal.
DEBUG_COMMANDS = True

# Set to True to send zero torque feedforward to all motors.
# Useful for initial testing before verifying TFF sign and magnitude.
ZERO_TFF = False

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
# This mapping is applied symmetrically to feedback, position/velocity commands,
# and feedforward torque. Set M4 to +1.0 only if a low-risk physical test shows
# that its encoder direction already matches the MuJoCo-positive joint motion.
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

    if len(sys.argv) < 2:
        print("Usage: python3 hardware_integrated_main_controller.py <serial_port>")
        sys.exit(1)

    port = sys.argv[1]
    try:
        si.connect(port)
        print(f"[ctrl] Connected to {port} (motors frozen)")
    except Exception as e:
        print(f"[ctrl] Serial error: {e}")
        sys.exit(1)

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

    # Include both driven and constrained joints so fallback pose is correct.
    _all_keyframe_joints = list(MOTOR_TO_JOINT.values()) + list(CONSTRAINED_JOINTS.keys())
    keyframe_qpos = {
        jname: float(model.key_qpos[keyframe_id, qpos_idx[jname]])
        for jname in _all_keyframe_joints
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

    # Initialise ctrl to hardware_zero BEFORE resetting the limiter so that
    # the limiter seeds its internal reference from the correct pose, not zeros.
    for actuator, ctrl_val in HARDWARE_ZERO_ACTUATOR_CTRL_RAD.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = ctrl_val
    limiter.reset(data)

    # --- Print startup info ---
    loop_mode = "CLOSED-LOOP (encoder feedback)" if CLOSED_LOOP else "OPEN-LOOP (reference only — no encoder feedback)"
    print()
    print(f"[ctrl] *** HARDWARE CONTROLLER READY — {loop_mode} ***")
    print("[ctrl]     Motors must be enabled before starting the control loop.")
    print("[ctrl]     Use MuJoCo's Control tab sliders as the primary position input.")
    print()
    print("[ctrl] Commands:")
    print("         a            — enable all body motors (M1-M4)")
    print("         s            — start control loop")
    print("         g <j> <pos>  — move joint j (1-4); pos examples: 0.35  20deg  +20deg (relative)")
    print("         h            — go to hardware_zero pose")
    print("         p            — pause control loop")
    print("         r            — resume control loop")
    print("         z            — print current encoder readings")
    print("         d            — toggle command debug printout")
    print("         q            — quit")
    print()

    # --- Shared command handler (used by both stdin and named pipe) ---
    def _handle_command(raw: str) -> None:
        global _control_active, _request_start, _request_pause
        global _paused, _request_goal, _request_home, _request_rezero
        global DEBUG_COMMANDS

        if not raw:
            return
        cmd = raw.lower()

        if cmd == "a":
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

    dt = 1.0 / UPDATE_HZ
    tick = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_tick  = time.perf_counter()
        last_print = time.perf_counter()

        while viewer.is_running():

            # --- Handle deferred terminal commands ---

            if _request_start and not _control_active:
                limiter.update(data)
                _control_active = True
                _request_start  = False
                _paused         = False

                pos_raw, _ = si.raw_snapshot()
                missing = [m for m in [1, 2, 3, 4] if pos_raw[m] is None]
                for motor in MOTOR_TO_JOINT:
                    _encoder_at_sync[motor] = pos_raw[motor] if pos_raw[motor] is not None else 0.0

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
                    ref_now = MOTOR_DIRECTION[motor] * float(data.ctrl[act_id])
                    _pos_cmd_offset[motor] = _encoder_at_sync[motor] - ref_now
                off_str = ", ".join(f"M{m}={v:+.4f}" for m, v in sorted(_pos_cmd_offset.items()))
                print(f"[ctrl] Position command offsets: {off_str}")
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
                pos_raw, vel_raw = si.raw_snapshot()
                _request_rezero = False
                print("[ctrl] Current encoder readings:")
                for motor, jname in MOTOR_TO_JOINT.items():
                    p = pos_raw[motor]
                    v = vel_raw[motor]
                    if p is None:
                        print(f"    M{motor} ({jname}): no feedback")
                    else:
                        print(f"    M{motor} ({jname}): pos={p:.5f} rad  vel={v:.5f} rad/s")

            # --- Snapshot encoder feedback (lag-compensated) ---
            fb = si.snapshot()
            pos_snap = fb.pos
            vel_snap = fb.vel

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
            if CLOSED_LOOP:
                _apply_encoder_state(data, qpos_idx, qvel_idx, keyframe_qpos, pos_snap, vel_snap)
            else:
                _apply_reference_state(data, qpos_idx, qvel_idx, keyframe_qpos, limiter)
            mujoco.mj_forward(model, data)

            if _control_active and not _paused:

                data.time += dt

                # 1. Schedule PD gains from current inertia + position.
                gain_updates = controller.update(model, data)

                # 2. Capture slider values and advance rate-limited reference.
                limiter.update(data)

                # 3. Compute gravity-compensation feedforward.
                tau_ff = _compute_tau_ff(model, data, qpos_idx, qvel_idx)

                # 4. Send MIT frames every SEND_EVERY ticks.
                if tick % SEND_EVERY == 0:
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

                        si.send_mit_frame(
                            motor,
                            ref_pos_motor,
                            ref_vel_motor,
                            kp_motor,
                            kd_motor,
                            tau_ff_motor,
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
                limiter.update(data)

            viewer.sync()
            tick += 1

            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    si.close()
    print("[ctrl] Serial closed. Bye.")


if __name__ == "__main__":
    main()
