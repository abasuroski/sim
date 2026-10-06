#!/usr/bin/env python3
"""
sync_hardware.py — Mirror physical robot arm into MuJoCo in real-time.

Reads motor feedback from the STM32 over serial (same format as motor_gui.py)
and drives the MuJoCo sim joints directly via qpos. Moving motors by hand
shows up in the sim immediately.

The editable ``HARDWARE_ZERO_ENCODER_RAD`` configuration maps raw
encoder positions into the named ``hardware_zero`` model keyframe:
``sim_joint = MOTOR_SCALE * (encoder_position - hardware_zero)``.

Usage:
    python3 sync_hardware.py /dev/ttyACM0

Startup workflow:
    1. Viewer opens in SETUP MODE — sim is draggable (Ctrl+drag a body in the viewer).
    2. Type  s  and press Enter to start calibrated live sync.
       The sim then maps raw encoder values into the shared hardware_zero keyframe.

Commands (type in terminal while viewer is open, press Enter):
    s        — start calibrated live sync
    z        — print current encoder readings as a copyable keyframe block
    e        — start passive feedback on M2, M3, M4 (no torque resistance)
    2/3/4    — start passive feedback on individual motor
    p        — freeze sim at the hardware_zero keyframe
    r        — unfreeze sim
    q        — quit

Requires: mujoco, pyserial
"""

import sys
import math
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from robot_master_configuration import HARDWARE_ZERO_ENCODER_RAD, HARDWARE_ZERO_KEYFRAME_NAME
import serial_interface as si

MODEL_PATH = Path(__file__).parent / "robot_master" / "mjcf" / "robot_master.xml"
UPDATE_HZ  = 50

MOTOR_TO_JOINT = {
    1: "revolute_1",
    2: "revolute_2",
    3: "revolute_3",
    4: "revolute_4",
}

MOTOR_SCALE = {1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0 / 1.25}


def encoder_to_sim_angle(motor: int, encoder_angle_rad: float, sim_angle_at_sync_rad: float, encoder_angle_at_sync_rad: float) -> float:
    """Map encoder delta from the sync snapshot onto the sim pose at sync time.

    The sim stays exactly where it is when 's' is pressed; subsequent encoder
    changes are applied as deltas from that snapshot.
    """
    return sim_angle_at_sync_rad + (encoder_angle_rad - encoder_angle_at_sync_rad) * MOTOR_SCALE[motor]

# revolute_5 = -revolute_4 (XML polycoef="0 -1 0 0 0")
CONSTRAINED_JOINTS = {
    "revolute_5":               ("revolute_4", -1.0), # D NOT CHANGE
    "revolute_7":               ("revolute_4",  1.0),  # same direction as M4, 
    "revolute_8":               ("revolute_4",  -1.0),  # same direction as M4, 
    "revolute_9":               ("revolute_4", -1.0),  # = -revolute_7 = -revolute_4 D NOT CHANGE
    "revolute_6":               ("revolute_4",  1.0),  # = -revolute_8 = -revolute_4 D NOT CHANGE
    "revolute_13":              ("revolute_4", 1.0),  # = -revolute_8 = -revolute_4
    #"revolute_14_loop_closure": ("revolute_4", -1.0),  # = -revolute_8 = -revolute_4
}
# State flags — written by terminal thread, read by main loop.
_sync_active    = False   # True once user presses 's'
_request_start  = False   # set by 's', cleared by main loop after it handles it
_request_rezero = False   # set by 'z', cleared by main loop
_paused         = False   # True = hold the hardware_zero keyframe

# Snapshots captured at the moment live sync starts.
_encoder_at_sync = {}    # {motor: float}
_sim_at_sync     = {}    # {joint_name: float}


def main():
    global _sync_active, _request_start, _request_rezero, _paused

    if len(sys.argv) < 2:
        print("Usage: python3 sync_hardware.py <serial_port>")
        sys.exit(1)

    port = sys.argv[1]
    try:
        si.connect(port)
        print(f"[sync] Connected to {port}")
    except Exception as e:
        print(f"[sync] Serial error: {e}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HARDWARE_ZERO_KEYFRAME_NAME)
    if keyframe_id < 0:
        raise RuntimeError(f"MJCF has no {HARDWARE_ZERO_KEYFRAME_NAME!r} keyframe; regenerate it first.")

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
    keyframe_joint_angles = {
        jname: float(model.key_qpos[keyframe_id, qpos_idx[jname]])
        for jname in MOTOR_TO_JOINT.values()
        if jname in qpos_idx
    }

    # Both the visual viewer and the live hardware synchronizer begin at the
    # same named arm configuration.
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)

    print()
    print(f"[sync] *** SETUP MODE — the viewer starts at {HARDWARE_ZERO_KEYFRAME_NAME!r} ***")
    print("[sync]     Ctrl+drag a body to inspect the mechanism before live sync.")
    print("[sync]     Type  s  and press Enter to begin live hardware synchronization.")
    print()
    print("[sync] Commands:")
    print("         s        — start calibrated live sync")
    print("         z        — print current readings for the hardware-zero keyframe block")
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
                    print("[sync] Already syncing — use 'z' to show the current raw encoder readings")
                else:
                    _request_start = True
                    print("[sync] Starting live sync from the hardware_zero keyframe...")
            elif cmd == "z":
                _request_rezero = True
                print("[sync] Printing hardware-zero keyframe readings...")
            elif cmd == "e":
                threading.Thread(target=si.enable_passive_feedback, args=([2, 3, 4],), daemon=True).start()
            elif cmd in ("2", "3", "4"):
                threading.Thread(target=si.enable_passive_feedback, args=([int(cmd)],), daemon=True).start()
            elif cmd == "p":
                _paused = True
                print("[sync] Sim frozen at the hardware_zero keyframe")
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
                _sync_active = True
                _request_start = False
                pos_raw, _ = si.raw_snapshot()
                missing = [m for m in [1, 2, 3, 4] if pos_raw[m] is None]
                for motor in MOTOR_TO_JOINT:
                    _encoder_at_sync[motor] = pos_raw[motor] if pos_raw[motor] is not None else 0.0
                # Snapshot current sim qpos so the sim doesn't jump —
                # capture both driven joints and constrained followers.
                for motor, jname in MOTOR_TO_JOINT.items():
                    if jname in qpos_idx:
                        _sim_at_sync[jname] = float(data.qpos[qpos_idx[jname]])
                for follower, (source, _) in CONSTRAINED_JOINTS.items():
                    if follower in qpos_idx:
                        _sim_at_sync[follower] = float(data.qpos[qpos_idx[follower]])
                if missing:
                    print(f"[sync] WARNING: no feedback yet from motors {missing} — defaulting their encoder snapshot to 0")
                snap_str = ", ".join(f"M{m}={_encoder_at_sync[m]:.3f}" for m in MOTOR_TO_JOINT)
                print(f"[sync] Encoder snapshot at sync: {snap_str}")
                print("[sync] Live sync active — sim will move relative to this position.")

            if _request_rezero:
                pos_raw, _ = si.raw_snapshot()
                _request_rezero = False
                print("[sync] Current encoder positions; copy a known arm-zero pose into HARDWARE_ZERO_ENCODER_RAD:")
                for motor, jname in MOTOR_TO_JOINT.items():
                    value = pos_raw[motor]
                    if value is None:
                        print(f"    {jname!r}: no feedback")
                    else:
                        print(f"    {jname!r}: {value:.6f},")
                print("[sync] The keyframe was not changed for this session.")

            # --- Drive sim ---
            fb       = si.snapshot()
            pos_snap = fb.pos
            vel_snap = fb.vel

            now = time.perf_counter()
            if now - last_print >= 2.0:
                last_print = now
                parts = [f"M{m}={pos_snap[m]:.3f}" for m in [1, 2, 3, 4] if pos_snap[m] is not None]
                status = "SETUP" if not _sync_active else "LIVE"
                print(f"[sync/{status}] " + (", ".join(parts) if parts else "no feedback yet"))

            if _paused:
                mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

            elif _sync_active:
                driven_qpos = {}
                for motor, jname in MOTOR_TO_JOINT.items():
                    pos  = pos_snap[motor]
                    if pos is None or jname not in qpos_idx:
                        continue
                    scale = MOTOR_SCALE[motor]
                    driven = encoder_to_sim_angle(motor, pos, _sim_at_sync.get(jname, keyframe_joint_angles[jname]), _encoder_at_sync.get(motor, 0.0))
                    data.qpos[qpos_idx[jname]] = driven
                    data.qvel[qvel_idx[jname]] = vel_snap[motor] * scale
                    driven_qpos[jname] = driven

                for follower, (source, fscale) in CONSTRAINED_JOINTS.items():
                    if source in driven_qpos and follower in qpos_idx:
                        # Compute delta from the source's sync snapshot, scale it,
                        # then apply from the follower's own sync snapshot.
                        source_sim_at_sync    = _sim_at_sync.get(source, keyframe_joint_angles.get(source, 0.0))
                        follower_sim_at_sync  = _sim_at_sync.get(follower, float(model.key_qpos[keyframe_id, qpos_idx[follower]]))
                        delta = driven_qpos[source] - source_sim_at_sync
                        data.qpos[qpos_idx[follower]] = follower_sim_at_sync + delta * fscale
                        if follower in qvel_idx and source in qvel_idx:
                            data.qvel[qvel_idx[follower]] = data.qvel[qvel_idx[source]] * fscale

            # In setup mode: don't touch qpos — viewer mouse perturbation controls it.

            mujoco.mj_forward(model, data)
            viewer.sync()

            next_tick += dt
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    si.close()


if __name__ == "__main__":
    main()
