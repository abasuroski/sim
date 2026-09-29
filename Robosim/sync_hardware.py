#!/usr/bin/env python3
"""
sync_hardware.py — Mirror physical robot arm into MuJoCo in real-time.

Reads motor feedback from the STM32 over serial (same format as motor_gui.py)
and drives the MuJoCo sim joints directly via qpos. Moving motors by hand
shows up in the sim immediately.

Usage:
    python3 sync_hardware.py /dev/ttyACM0

Commands (type in this terminal while viewer is open, press Enter):
    e        — enable all motors (M2, M3, M4)
    2 / 3 / 4 — enable individual motor
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

# Offset added after scaling (radians).
MOTOR_OFFSET = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

# Reference pose shown when you type 'p' (radians, sim joint values).
# Set this to whatever looks like the physical arm's natural zero position.
# Type 'p' to freeze sim here → match physical arm → zero motors in GUI → type 'r' to resume.
REFERENCE_POSE = {
    "revolute_1": 0.0,
    "revolute_2": 0.0,
    "revolute_3": 0.0,
    "revolute_4": 0.0,
    "revolute_5": 0.0,
}

# revolute_5 = -revolute_4 (from XML polycoef="0 -1 0 0 0")
CONSTRAINED_JOINTS = {
    "revolute_5": ("revolute_4", -1.0),
}

_ser      = None
_ser_lock = threading.Lock()

_fb_lock   = threading.Lock()
_motor_pos = {1: None, 2: None, 3: None, 4: None}
_motor_vel = {1: 0.0,  2: 0.0,  3: 0.0,  4: 0.0}

_paused = False   # True = sim frozen at REFERENCE_POSE for physical calibration

# Compiled exactly as motor_gui.py uses them (applied to body after stripping tag)
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
    # Mirror motor_gui.py route_feedback exactly:
    # check startswith [M{n}], strip the tag, search the body
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
    global _ser

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
    mujoco.mj_forward(model, data)

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

    print("[sync] Viewer open")
    print("[sync] Commands (type here, press Enter):")
    print("         e        — start passive feedback on M2, M3, M4 (no resistance)")
    print("         2/3/4    — start passive feedback on individual motor")
    print("         p        — freeze sim at reference pose (match arm to it, then zero in GUI)")
    print("         r        — resume live sync")
    print("         q        — quit")

    def _terminal_input():
        global _paused
        while True:
            try:
                cmd = input().strip().lower()
            except EOFError:
                break
            if cmd == "e":
                threading.Thread(target=_enable_motors, args=([2, 3, 4],), daemon=True).start()
            elif cmd in ("2", "3", "4"):
                threading.Thread(target=_enable_motors, args=([int(cmd)],), daemon=True).start()
            elif cmd == "p":
                _paused = True
                print("[sync] Sim frozen at reference pose — match the physical arm to this, then zero motors in the GUI")
            elif cmd == "r":
                _paused = False
                print("[sync] Resuming live sync")
            elif cmd == "q":
                break

    threading.Thread(target=_terminal_input, daemon=True).start()

    dt = 1.0 / UPDATE_HZ

    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_tick  = time.perf_counter()
        last_print = time.perf_counter()

        while viewer.is_running():
            with _fb_lock:
                pos_snap = dict(_motor_pos)
                vel_snap = dict(_motor_vel)

            # Debug: print received positions every 2 seconds
            now = time.perf_counter()
            if now - last_print >= 2.0:
                last_print = now
                parts = [f"M{m}={pos_snap[m]:.3f}" for m in [1, 2, 3, 4] if pos_snap[m] is not None]
                print("[sync] pos: " + (", ".join(parts) if parts else "no feedback yet"))

            if _paused:
                # Freeze sim at reference pose for physical calibration
                for jname, angle in REFERENCE_POSE.items():
                    if jname in qpos_idx:
                        data.qpos[qpos_idx[jname]] = angle
                        data.qvel[qvel_idx[jname]] = 0.0
            else:
                driven_qpos = {}
                for motor, jname in MOTOR_TO_JOINT.items():
                    pos = pos_snap[motor]
                    if pos is None or jname not in qpos_idx:
                        continue
                    scale = MOTOR_SCALE[motor]
                    driven = pos * scale + MOTOR_OFFSET[motor]
                    data.qpos[qpos_idx[jname]] = driven
                    data.qvel[qvel_idx[jname]] = vel_snap[motor] * scale
                    driven_qpos[jname] = driven

                for follower, (source, scale) in CONSTRAINED_JOINTS.items():
                    if source in driven_qpos and follower in qpos_idx:
                        data.qpos[qpos_idx[follower]] = driven_qpos[source] * scale
                        if follower in qvel_idx and source in qvel_idx:
                            data.qvel[qvel_idx[follower]] = data.qvel[qvel_idx[source]] * scale

            mujoco.mj_forward(model, data)
            viewer.sync()

            next_tick += dt
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    _ser.close()


if __name__ == "__main__":
    main()
