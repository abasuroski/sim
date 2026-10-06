"""
motor_spoof.py — Run hardware_integrated_main_controller with simulated motors.

Patches serial_interface before the controller is imported so the full stack
(MuJoCo viewer, gain scheduling, trapezoidal limiter, gravity compensation)
runs without an STM32 or CubeMars motors attached.

Motor model: per-motor first-order lag with velocity and position limits
derived from actual CubeMars specs.

Gear ratios (internal gearbox — firmware reports output shaft position):
  M1 AK60: 6:1   M2/M3 AK70: 10:1   M4 AK40: 10:1 internal + 1.25:1 arm linkage

Usage:
    python motor_spoof.py

Commands (same as the real controller):
    a            — arm  (no-op in spoof)
    s            — start control loop
    g <j> <rad>  — command joint j (1-4) to position in radians
    h            — go to hardware_zero pose
    p / r        — pause / resume
    q            — quit
"""

import sys
import math
import time
import threading

SIM_RATE_HZ = 500   # motor simulation update rate

# Per-motor specs derived from CubeMars datasheets.
# Positions/velocities here are all in motor OUTPUT SHAFT units (rad, rad/s),
# i.e. what the firmware reports — after the internal gearbox, before any
# external linkage.
#
# time_const_s : first-order closed-loop lag estimate (tunable)
# vel_limit     : no-load output shaft speed in rad/s (hard cap)
# peak_torque   : output torque limit in N·m (informational only)
#
# AK60 (M1): 6:1,   no-load 320 rpm → 33.5 rad/s, peak 9 Nm
# AK70 (M2/M3): 10:1, no-load ~200 rpm → ~20.9 rad/s (Kv=100, 24V), peak 24.8 Nm
# AK40 (M4): 10:1 internal + 1.25 external, no-load 435 rpm → 45.5 rad/s, peak 4.1 Nm
MOTOR_SPECS: dict[int, dict] = {
    1: {"time_const_s": 0.10, "vel_limit": 33.5,  "peak_torque": 9.0},   # AK60
    2: {"time_const_s": 0.14, "vel_limit": 20.9,  "peak_torque": 24.8},  # AK70
    3: {"time_const_s": 0.14, "vel_limit": 20.9,  "peak_torque": 24.8},  # AK70
    4: {"time_const_s": 0.10, "vel_limit": 45.5,  "peak_torque": 4.1},   # AK40
}

import random

import serial_interface as si
from robot_master_configuration import HARDWARE_ZERO_ENCODER_RAD

# ---------------------------------------------------------------------------
# Simulated motor state — starts at hardware zero
# ---------------------------------------------------------------------------

_sim_pos  = {m: random.uniform(-math.pi, math.pi) for m in [1, 2, 3, 4]}
_sim_vel  = {m: 0.0 for m in [1, 2, 3, 4]}
_cmd_pos  = dict(_sim_pos)   # updated by send_mit_frame
_sim_lock = threading.Lock()


def _sim_loop() -> None:
    dt     = 1.0 / SIM_RATE_HZ
    alphas = {m: 1.0 - math.exp(-dt / MOTOR_SPECS[m]["time_const_s"]) for m in [1, 2, 3, 4]}
    while True:
        time.sleep(dt)
        with _sim_lock:
            for m in [1, 2, 3, 4]:
                prev         = _sim_pos[m]
                _sim_pos[m] += alphas[m] * (_cmd_pos[m] - _sim_pos[m])
                raw_vel       = (_sim_pos[m] - prev) / dt
                # Clamp velocity to motor no-load limit.
                vmax          = MOTOR_SPECS[m]["vel_limit"]
                _sim_vel[m]   = max(-vmax, min(vmax, raw_vel))

        # Write directly into serial_interface module state so snapshot() works.
        with si._fb_lock:
            now = time.perf_counter()
            for m in [1, 2, 3, 4]:
                si._motor_pos[m]          = _sim_pos[m]
                si._motor_vel[m]          = _sim_vel[m]
                si._motor_measure_time[m] = now


threading.Thread(target=_sim_loop, daemon=True).start()

# ---------------------------------------------------------------------------
# Patch serial_interface — must happen before controller is imported
# ---------------------------------------------------------------------------

def _mock_connect(port: str) -> None:
    print(f"[spoof] connect({port!r}) — simulated motors active")
    print("[spoof] Random starting positions: " +
          ", ".join(f"M{m}={_sim_pos[m]:.3f}" for m in [1, 2, 3, 4]))


def _mock_send_mit_frame(motor: int, pos: float, vel: float,
                          kp: float, kd: float, tau_ff: float) -> None:
    with _sim_lock:
        _cmd_pos[motor] = pos


def _mock_emergency_stop() -> None:
    with _sim_lock:
        for m in [1, 2, 3, 4]:
            _cmd_pos[m] = _sim_pos[m]
    print("[spoof] emergency stop — motors frozen at current position")


si.connect                 = _mock_connect
si.close                   = lambda: None
si.is_connected            = lambda: True
si.send                    = lambda cmd: None
si.send_mit_frame          = _mock_send_mit_frame
si.enable_motors           = lambda motors: print(f"[spoof] enable_motors({motors}) — no-op")
si.enable_passive_feedback = lambda motors: print(f"[spoof] enable_passive_feedback({motors}) — no-op")
si.emergency_stop          = _mock_emergency_stop

# ---------------------------------------------------------------------------
# Launch controller with a fake port argument
# ---------------------------------------------------------------------------

sys.argv = [sys.argv[0], "SPOOF"]

import hardware_integrated_main_controller
hardware_integrated_main_controller.main()
