"""
serial_interface.py — Serial communication layer for the robot_master arm.

Shared between sync_hardware.py and the main controller. Handles:
  - Serial connection and freeze-on-connect
  - Feedback parsing ([M1]-[M4] packets with pos, vel, timestamp)
  - Lag-compensated position snapshots
  - MIT control mode command sending (P, V, K, D, T)
  - Motor enable with outer PID zeroing

Serial protocol (STM32 UART, 115200 baud):
  Incoming feedback:
    [M<n>] pos=<val> rad  vel=<val>  tau=<val>  T=<val>  err=<val>  t=<val>ms
    [M<n>] pos=<val> deg  spd=<val> eRPM  I=<val> A  T=<val>  err=<val>  t=<val>ms

  Outgoing commands:
    <n>P<val>\\n  — position setpoint (rad)
    <n>V<val>\\n  — velocity setpoint (rad/s)
    <n>K<val>\\n  — inner Kp
    <n>D<val>\\n  — inner Kd
    <n>T<val>\\n  — torque feedforward (N·m)
    <n>E\\n       — enable motor
    <n>R\\n       — passive feedback (no torque)
    0X\\n         — emergency stop all motors

Motor index mapping:
    1 = Base     (AK60, CAN 104) → revolute_1
    2 = Shoulder (AK70, CAN 1)   → revolute_2
    3 = Elbow    (AK70, CAN 2)   → revolute_3
    4 = Linkage  (AK40, CAN 3)   → revolute_4
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass

from robot_master_configuration import AK40_MOTOR_TORQUE_LIMIT_NM

try:
    import serial as _serial_module
except ModuleNotFoundError:
    _serial_module = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BAUD_RATE = 1000000

# Safety: maximum torque feedforward sent per motor (N·m, motor side).
TAU_FF_LIMIT: dict[int, float] = {1: 9.0, 2: 18.0, 3: 18.0, 4: AK40_MOTOR_TORQUE_LIMIT_NM}

# ---------------------------------------------------------------------------
# Shared state — module-level so both sync and controller can import them
# ---------------------------------------------------------------------------

_ser      = None
_ser_lock = threading.Lock()

_fb_lock            = threading.Lock()
_motor_pos:          dict[int, float | None] = {1: None, 2: None, 3: None, 4: None}
_motor_vel:          dict[int, float]        = {1: 0.0,  2: 0.0,  3: 0.0,  4: 0.0}
_motor_measure_time: dict[int, float]        = {1: 0.0,  2: 0.0,  3: 0.0,  4: 0.0}

# ---------------------------------------------------------------------------
# Feedback parsing
# ---------------------------------------------------------------------------

_POS_RE = re.compile(r'pos=(-?\d+\.?\d*)\s*(rad|deg)')
_VEL_RE = re.compile(r'\bvel=(-?\d+\.?\d*)')
_SPD_RE = re.compile(r'\bspd=(-?\d+\.?\d*)')
_TS_RE  = re.compile(r'\bt=(\d+)ms')

_AK60_POLE_PAIRS = 14
_AK60_REDUCTION = 6.0
_ERPM_TO_RAD_S = (2.0 * math.pi) / (60.0 * _AK60_POLE_PAIRS * _AK60_REDUCTION)


def _parse_line(line: str) -> None:
    """Parse one STM32 feedback line and update shared motor state."""
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
        else:
            ms = _SPD_RE.search(body)
            if ms:
                vel = float(ms.group(1)) * _ERPM_TO_RAD_S

        with _fb_lock:
            _motor_pos[motor]          = val
            _motor_vel[motor]          = vel
            _motor_measure_time[motor] = time.perf_counter()
        break


def _serial_reader() -> None:
    """Background thread: continuously read lines from serial and parse them."""
    while True:
        try:
            line = _ser.readline().decode(errors="replace").strip()
            if line:
                _parse_line(line)
        except Exception:
            time.sleep(0.01)

# ---------------------------------------------------------------------------
# Public API — connection
# ---------------------------------------------------------------------------

def connect(port: str) -> None:
    """Open serial connection to the STM32 and freeze motors immediately.

    Raises serial.SerialException on failure.
    Must be called before any other function in this module.
    """
    global _ser
    if _serial_module is None:
        raise RuntimeError("pyserial is not installed. Run: pip install pyserial")

    _ser = _serial_module.Serial(port, BAUD_RATE, timeout=1.0)
    time.sleep(0.05)
    # Freeze all motors immediately — prevents stale commands from a previous
    # session causing unexpected motion on connect.
    _ser.write(b"0X\n")

    threading.Thread(target=_serial_reader, daemon=True).start()


def is_connected() -> bool:
    return _ser is not None and _ser.is_open


def close() -> None:
    """Close the serial connection."""
    global _ser
    with _ser_lock:
        if _ser and _ser.is_open:
            _ser.close()
            _ser = None

# ---------------------------------------------------------------------------
# Public API — feedback snapshot
# ---------------------------------------------------------------------------

@dataclass
class MotorSnapshot:
    """Lag-compensated motor state snapshot for all four body motors."""
    pos:  dict[int, float | None]  # motor → joint position (rad), None = no feedback
    vel:  dict[int, float]         # motor → velocity (rad/s)
    time: dict[int, float]         # motor → PC time of measurement


def snapshot(lag_cap_s: float = 0.1) -> MotorSnapshot:
    """Return lag-compensated motor positions and velocities.

    Extrapolates each motor's position forward from the time the feedback
    packet arrived to now using first-order integration:
        pos_est = pos_received + vel * elapsed

    Args:
        lag_cap_s: maximum extrapolation window in seconds. Packets older
                   than this are not extrapolated further (stale packet guard).
    """
    with _fb_lock:
        pos_raw   = dict(_motor_pos)
        vel_raw   = dict(_motor_vel)
        mtime_raw = dict(_motor_measure_time)

    now = time.perf_counter()
    pos_est: dict[int, float | None] = {}
    for motor in [1, 2, 3, 4]:
        if pos_raw[motor] is None:
            pos_est[motor] = None
            continue
        elapsed = max(0.0, min(lag_cap_s, now - mtime_raw[motor]))
        pos_est[motor] = pos_raw[motor] + vel_raw[motor] * elapsed

    return MotorSnapshot(pos=pos_est, vel=vel_raw, time=mtime_raw)


def raw_snapshot() -> tuple[dict[int, float | None], dict[int, float]]:
    """Return raw (non-extrapolated) pos and vel dicts. Used for zeroing."""
    with _fb_lock:
        return dict(_motor_pos), dict(_motor_vel)

# ---------------------------------------------------------------------------
# Public API — commands
# ---------------------------------------------------------------------------

def send(cmd: str) -> None:
    """Send a raw command string to the STM32."""
    with _ser_lock:
        if _ser and _ser.is_open:
            _ser.write(cmd.encode())


def send_mit_frame(
    motor: int,
    pos: float,
    vel: float,
    kp: float,
    kd: float,
    tau_ff: float,
) -> None:
    """Send one full MIT control mode frame for a single motor.

    Args:
        motor:  motor index (1-4)
        pos:    desired position (rad, motor side)
        vel:    desired velocity (rad/s, motor side)
        kp:     inner position gain (motor side)
        kd:     inner velocity gain (motor side)
        tau_ff: feedforward torque (N·m, motor side), clamped to TAU_FF_LIMIT
    """
    tau_ff = max(-TAU_FF_LIMIT[motor], min(TAU_FF_LIMIT[motor], tau_ff))
    send(f"{motor}P{pos:.5f}\n")
    send(f"{motor}V{vel:.5f}\n")
    send(f"{motor}K{kp:.5f}\n")
    send(f"{motor}D{kd:.5f}\n")
    send(f"{motor}T{tau_ff:.5f}\n")


def send_mit_update(
    motor: int,
    pos: float | None = None,
    vel: float | None = None,
    kp: float | None = None,
    kd: float | None = None,
    tau_ff: float | None = None,
) -> int:
    """Send only the given MIT fields for one motor, in a single serial write.

    The STM32 latches each field, so unchanged fields can be omitted to save
    bandwidth. Returns the number of bytes written.
    """
    parts = []
    if pos is not None:
        parts.append(f"{motor}P{pos:.5f}\n")
    if vel is not None:
        parts.append(f"{motor}V{vel:.5f}\n")
    if kp is not None:
        parts.append(f"{motor}K{kp:.5f}\n")
    if kd is not None:
        parts.append(f"{motor}D{kd:.5f}\n")
    if tau_ff is not None:
        tau_ff = max(-TAU_FF_LIMIT[motor], min(TAU_FF_LIMIT[motor], tau_ff))
        parts.append(f"{motor}T{tau_ff:.5f}\n")
    if not parts:
        return 0
    cmd = "".join(parts)
    send(cmd)
    return len(cmd)


def enable_motors(motors: list[int]) -> None:
    """Enable motors via the STM32 soft-start sequence."""
    for m in motors:
        print(f"[serial] Enabling M{m}...")
        send(f"{m}E\n")
        time.sleep(0.1)
    print(f"[serial] Motors enabled: {motors}")


def enable_passive_feedback(motors: list[int]) -> None:
    """Start passive feedback (0xFC poll) on motors — no torque resistance."""
    for m in motors:
        print(f"[serial] Passive feedback M{m}...")
        send(f"{m}R\n")
        time.sleep(0.1)
    print(f"[serial] Passive feedback active: {motors}")


def emergency_stop() -> None:
    """Send emergency stop — STM32 freezes all motors at current position."""
    send("0X\n")
    print("[serial] Emergency stop sent.")
