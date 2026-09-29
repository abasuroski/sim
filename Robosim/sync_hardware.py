#!/usr/bin/env python3
"""
sync_hardware.py — Mirror physical robot arm into MuJoCo in real-time.

Reads motor feedback from the STM32 over serial (same format as motor_gui.py)
and drives the MuJoCo sim joints directly via qpos. Moving motors by hand
shows up in the sim immediately.

Usage:
    python3 sync_hardware.py /dev/ttyACM0

Startup workflow:
    1. Viewer opens in SETUP MODE — sim is draggable (Ctrl+drag a body in the viewer).
    2. Drag the sim joints to match the physical arm's current pose.
    3. Type  s  and press Enter to lock that pose as the baseline and start live sync.
       From that point, the sim tracks the *change* in encoder position from the
       hardware's position when you pressed 's' — the raw encoder zero doesn't matter.

Commands (type in terminal while viewer is open, press Enter):
    s        — lock current sim pose as baseline and start live sync
    z        — re-zero encoder baseline to current arm position (re-snap mid-session)
    e        — start passive feedback on M2, M3, M4 (no torque resistance)
    2/3/4    — start passive feedback on individual motor
    p        — freeze sim at REFERENCE_POSE (for old calibration workflow)
    r        — unfreeze sim
    q        — quit

Requires: mujoco, pyserial
"""

import sys
import math
import re
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import serial

MODEL_PATH = Path(__file__).parent / "robot_master" / "mjcf" / "robot_master.xml"
BAUD_RATE  = 115200
UPDATE_HZ  = 50

MOTOR_TO_JOINT = {
    1: "revolute_1",
    2: "revolute_2",
    3: "revolute_3",
    4: "revolute_4",
}

MOTOR_SCALE = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

# Viewer opens at this pose so you have a reasonable starting point to drag from.
# Adjust to roughly match your arm's resting position before you start dragging.
SIM_INIT_POS = {
    "revolute_1": 0.0,
    "revolute_2": 0.5,   # ~28 deg
    "revolute_3": 0.4,   # ~23 deg
    "revolute_4": 0.7,   # ~40 deg
    "revolute_5": -0.7,  # constrained: -revolute_4
}

# Pose shown when you type 'p' (legacy calibration workflow).
REFERENCE_POSE = {
    "revolute_1": 0.0,
    "revolute_2": 0.0,
    "revolute_3": 0.0,
    "revolute_4": 0.0,
    "revolute_5": 0.0,
}

# revolute_5 = -revolute_4 (XML polycoef="0 -1 0 0 0")
CONSTRAINED_JOINTS = {
    "revolute_5": ("revolute_4", -1.0),
}

_ser      = None
_ser_lock = threading.Lock()

_fb_lock      = threading.Lock()
_motor_pos    = {1: None, 2: None, 3: None, 4: None}
_motor_vel    = {1: 0.0,  2: 0.0,  3: 0.0,  4: 0.0}

# Set once when live sync starts: encoder reading at that moment per motor.
_encoder_baseline = {1: None, 2: None, 3: None, 4: None}

# Set once when live sync starts: sim joint angle at that moment per joint.
_sim_baseline = {}

# State flags — written by terminal thread, read by main loop.
_sync_active   = False   # True once user presses 's'
_request_start = False   # set by 's', cleared by main loop after it handles it
_request_rezero = False  # set by 'z', cleared by main loop
_paused        = False   # True = show REFERENCE_POSE

_POS_RE = re.compile(r'pos=(-?\d+\.?\d*)\s*(rad|deg)')
_VEL_RE = re.compile(r'\bvel=(-?\d+\.?\d*)')


def _send(cmd):
    with _ser_lock:
        if _ser and _ser.is_open:
            _ser.write(cmd.encode())


def _enable_motors(motors):
    for m in motors:
        print(f"[sync] Starting passive feedback on M{m} (no resistance)...")
        _send(f"{m}R\n")
        time.sleep(0.1)
    print(f"[sync] Passive feedback active: {motors}")


def _parse_line(line):
    for motor in [1, 2, 3, 4]:
        tag = f"[M{motor}]"
        if not line.startswith(tag):
            continue
        body = line[len(tag):].strip()

        m = _POS_RE.search(body)
        if not m:
            break
        val = float(m.group(1))
        if m.group(2) == "deg":
            val = math.radians(val)

        vel = 0.0
        mv = _VEL_RE.search(body)
        if mv:
            vel = float(mv.group(1))

        with _fb_lock:
            _motor_pos[motor] = val
            _motor_vel[motor] = vel
        break


def _serial_reader():
    while True:
        try:
            line = _ser.readline().decode(errors="replace").strip()
            if line:
                _parse_line(line)
        except Exception:
            time.sleep(0.01)


def main():
    global _ser, _sync_active, _request_start, _request_rezero, _paused
    global _encoder_baseline, _sim_baseline

    if len(sys.argv) < 2:
        print("Usage: python3 sync_hardware.py <serial_port>")
        sys.exit(1)

    port = sys.argv[1]
    try:
        _ser = serial.Serial(port, BAUD_RATE, timeout=1.0)
        print(f"[sync] Connected to {port}")
    except serial.SerialException as e:
        print(f"[sync] Serial error: {e}")
        sys.exit(1)

    threading.Thread(target=_serial_reader, daemon=True).start()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)

    # Build joint index maps.
    qpos_idx = {}
    qvel_idx = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            qpos_idx[name] = model.jnt_qposadr[i]
            qvel_idx[name] = model.jnt_dofadr[i]

    for motor, jname in MOTOR_TO_JOINT.items():
        if jname not in qpos_idx:
            print(f"[sync] WARNING: joint '{jname}' (M{motor}) not found in model")

    # Apply starting pose so viewer opens at a reasonable position.
    for jname, angle in SIM_INIT_POS.items():
        if jname in qpos_idx:
            data.qpos[qpos_idx[jname]] = angle
    mujoco.mj_forward(model, data)

    print()
    print("[sync] *** SETUP MODE — drag the sim to match the physical arm ***")
    print("[sync]     Ctrl+drag a body in the viewer to move joints.")
    print("[sync]     When the sim matches the arm, type  s  and press Enter.")
    print()
    print("[sync] Commands:")
    print("         s        — start live sync (lock current sim pose as baseline)")
    print("         z        — re-zero encoder baseline to current arm position")
    print("         e        — passive feedback on M2, M3, M4 (no resistance)")
    print("         2/3/4    — passive feedback on individual motor")
    print("         p        — freeze sim at reference pose")
    print("         r        — unfreeze sim")
    print("         q        — quit")
    print()

    def _terminal_input():
        global _request_start, _request_rezero, _paused
        while True:
            try:
                cmd = input().strip().lower()
            except EOFError:
                break
            if cmd == "s":
                if _sync_active:
                    print("[sync] Already syncing — use 'z' to re-zero the baseline")
                else:
                    _request_start = True
                    print("[sync] Starting live sync...")
            elif cmd == "z":
                _request_rezero = True
                print("[sync] Re-zeroing encoder baseline...")
            elif cmd == "e":
                threading.Thread(target=_enable_motors, args=([2, 3, 4],), daemon=True).start()
            elif cmd in ("2", "3", "4"):
                threading.Thread(target=_enable_motors, args=([int(cmd)],), daemon=True).start()
            elif cmd == "p":
                _paused = True
                print("[sync] Sim frozen at reference pose")
            elif cmd == "r":
                _paused = False
                print("[sync] Sim unfrozen")
            elif cmd == "q":
                break

    threading.Thread(target=_terminal_input, daemon=True).start()

    dt = 1.0 / UPDATE_HZ

    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_tick  = time.perf_counter()
        last_print = time.perf_counter()

        while viewer.is_running():

            # --- Handle deferred commands from terminal thread ---
            if _request_start and not _sync_active:
                # Snapshot current sim qpos as our sim baseline.
                _sim_baseline = {
                    jname: data.qpos[qpos_idx[jname]]
                    for jname in list(MOTOR_TO_JOINT.values()) + list(CONSTRAINED_JOINTS.keys())
                    if jname in qpos_idx
                }
                # Snapshot current encoder readings as hardware baseline.
                with _fb_lock:
                    for m in [1, 2, 3, 4]:
                        _encoder_baseline[m] = _motor_pos[m]
                _sync_active = True
                _request_start = False
                missing = [m for m in [1, 2, 3, 4] if _encoder_baseline[m] is None]
                if missing:
                    print(f"[sync] WARNING: no feedback yet from motors {missing} — they'll start tracking when feedback arrives")
                sim_angles = ", ".join(
                    f"{jname}={_sim_baseline.get(jname, 0):.3f}"
                    for jname in MOTOR_TO_JOINT.values()
                )
                print(f"[sync] Baseline locked. Sim: {sim_angles}")
                print("[sync] Live sync active.")

            if _request_rezero:
                with _fb_lock:
                    for m in [1, 2, 3, 4]:
                        if _motor_pos[m] is not None:
                            _encoder_baseline[m] = _motor_pos[m]
                # Also re-snap sim baseline from current qpos.
                _sim_baseline = {
                    jname: data.qpos[qpos_idx[jname]]
                    for jname in list(MOTOR_TO_JOINT.values()) + list(CONSTRAINED_JOINTS.keys())
                    if jname in qpos_idx
                }
                _request_rezero = False
                print("[sync] Baseline re-zeroed.")

            # --- Drive sim ---
            with _fb_lock:
                pos_snap = dict(_motor_pos)
                vel_snap = dict(_motor_vel)

            now = time.perf_counter()
            if now - last_print >= 2.0:
                last_print = now
                parts = [f"M{m}={pos_snap[m]:.3f}" for m in [1, 2, 3, 4] if pos_snap[m] is not None]
                status = "SETUP" if not _sync_active else "LIVE"
                print(f"[sync/{status}] " + (", ".join(parts) if parts else "no feedback yet"))

            if _paused:
                for jname, angle in REFERENCE_POSE.items():
                    if jname in qpos_idx:
                        data.qpos[qpos_idx[jname]] = angle
                        data.qvel[qvel_idx[jname]] = 0.0

            elif _sync_active:
                driven_qpos = {}
                for motor, jname in MOTOR_TO_JOINT.items():
                    pos  = pos_snap[motor]
                    enc0 = _encoder_baseline[motor]
                    if pos is None or enc0 is None or jname not in qpos_idx:
                        continue
                    scale    = MOTOR_SCALE[motor]
                    sim_init = _sim_baseline.get(jname, 0.0)
                    driven   = sim_init + (pos - enc0) * scale
                    data.qpos[qpos_idx[jname]] = driven
                    data.qvel[qvel_idx[jname]] = vel_snap[motor] * scale
                    driven_qpos[jname] = driven

                for follower, (source, fscale) in CONSTRAINED_JOINTS.items():
                    if source in driven_qpos and follower in qpos_idx:
                        data.qpos[qpos_idx[follower]] = driven_qpos[source] * fscale
                        if follower in qvel_idx and source in qvel_idx:
                            data.qvel[qvel_idx[follower]] = data.qvel[qvel_idx[source]] * fscale

            # In setup mode: don't touch qpos — viewer mouse perturbation controls it.

            mujoco.mj_forward(model, data)
            viewer.sync()

            next_tick += dt
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    _ser.close()


if __name__ == "__main__":
    main()
