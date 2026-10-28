"""
serial_interface.py — Serial communication layer for the robot_master arm.

Shared between sync_hardware.py and the main controller. Handles:
  - Serial connection and freeze-on-connect
  - Feedback parsing (F1-F4 hex CAN data, or legacy [M1]-[M4] text)
  - Lag-compensated position snapshots
  - MIT control mode command sending (one checksummed M line per motor)
  - Motor enable

Serial protocol (STM32 UART, 1000000 baud):
  Incoming feedback (current firmware: raw CAN bytes as hex, decoded here):
    F<n> <16 hex: 8 feedback data bytes> <8 hex: STM32 tick ms>
    C<n> <16 hex: 8 bytes of MIT command as packed for CAN>   (TX_ECHO=1;
         sent when the command changes, plus a heartbeat every 250 ms)
    [UART] rx errors ore=<n> fe=<n> ne=<n> pe=<n> rejected=<n> fb_drop=<n> can_drop=<n> t=<val>ms
    [UART] rejected "<line>"   (latest command line the firmware refused)
    [M<n>] feedback lost — disabled ... / [M<n>] no feedback since enable — disabled ...
    Anything else (faults, unknown CAN id) is kept for the run log.
  Legacy text feedback (older firmware) is still understood:
    [M<n>] pos=<val> rad  vel=<val>  tau=<val>  T=<val>  err=<val>  t=<val>ms
    [M<n>] pos=<val> deg  spd=<val> eRPM  I=<val> A  T=<val>  err=<val>  t=<val>ms
    [TX M<n>] pos=<val> vel=<val> kp=<val> kd=<val> tff=<val>

  Outgoing commands, each line ending "*HH" = XOR of the bytes before '*'
  (required by the firmware on value commands):
    <n>M<pos>,<vel>,<kp>,<kd>,<tff>  — all five MIT fields, applied together
    <n>P / V / K / D / T<val>        — single fields (still accepted)
    <n>E          — enable motor
    <n>R          — passive feedback (no torque)
    0X            — emergency stop all motors

Motor index mapping (must match uart_cmd_init() in the firmware):
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
from collections import deque
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

# Motor type per index, as in the firmware's uart_cmd_init().
MOTOR_TYPE: dict[int, str] = {1: "AK60", 2: "AK70", 3: "AK70", 4: "AK40"}

# MIT command ranges per motor type, as in the firmware's float_to_uint() calls
# (main.c).  The 16-bit position and 12-bit other fields span these ranges.
_COMMAND_RANGE: dict[str, dict[str, tuple[float, float]]] = {
    "AK60": {"pos": (-12.56, 12.56), "vel": (-60.0, 60.0), "tau_ff": (-12.0, 12.0)},
    "AK70": {"pos": (-12.5, 12.5),   "vel": (-30.0, 30.0), "tau_ff": (-18.0, 18.0)},
    "AK40": {"pos": (-12.5, 12.5),   "vel": (-45.5, 45.5), "tau_ff": (-5.0, 5.0)},
}
for _ranges in _COMMAND_RANGE.values():
    _ranges.update({"kp": (0.0, 500.0), "kd": (0.0, 5.0)})
_COMMAND_BITS = {"pos": 16, "vel": 12, "kp": 12, "kd": 12, "tau_ff": 12}

# ---------------------------------------------------------------------------
# Shared state — module-level so both sync and controller can import them
# ---------------------------------------------------------------------------

_ser      = None
_ser_lock = threading.Lock()

_fb_lock            = threading.Lock()
_motor_pos:          dict[int, float | None] = {1: None, 2: None, 3: None, 4: None}
_motor_vel:          dict[int, float]        = {1: 0.0,  2: 0.0,  3: 0.0,  4: 0.0}
_motor_current:      dict[int, float | None] = {1: None, 2: None, 3: None, 4: None}
_motor_torque:       dict[int, float | None] = {1: None, 2: None, 3: None, 4: None}
_motor_temp:         dict[int, int | None]   = {1: None, 2: None, 3: None, 4: None}
_motor_err:          dict[int, int | None]   = {1: None, 2: None, 3: None, 4: None}
_motor_measure_time: dict[int, float]        = {1: 0.0,  2: 0.0,  3: 0.0,  4: 0.0}

# CAN command echo ("C<n>" lines, firmware built with TX_ECHO=1): the
# values the STM32 actually latched and put on the bus, motor side.
_motor_echo:      dict[int, "CommandEcho | None"] = {1: None, 2: None, 3: None, 4: None}
_motor_echo_time: dict[int, float]                = {1: 0.0,  2: 0.0,  3: 0.0,  4: 0.0}
# Every echo in arrival order, as (PC receive time, motor, echo), for the
# controller's echo check: the latest-only dicts above skip echoes that
# arrive between two control ticks.
_echo_queue: deque[tuple[float, int, "CommandEcho"]] = deque(maxlen=2000)

# Every other line from the STM32 (faults, watchdog trips, unknown CAN ids,
# future message types), as (PC receive time, text).  Bounded so a chatty
# firmware cannot grow it without limit if nobody drains it.
_other_lines: deque[tuple[float, str]] = deque(maxlen=500)

# ---------------------------------------------------------------------------
# Command framing and CAN field encoding
# ---------------------------------------------------------------------------

def checksum_line(body: str) -> str:
    """Return "<body>*HH\\n", HH = XOR of every byte of body, as the firmware checks."""
    total = 0
    for byte in body.encode("ascii"):
        total ^= byte
    return f"{body}*{total:02X}\n"


def _uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    """The firmware's uint_to_float()."""
    return x_int * (x_max - x_min) / ((1 << bits) - 1) + x_min


def command_range(motor: int, field: str) -> tuple[float, float]:
    """(min, max) the firmware clamps this MIT command field to, motor side."""
    return _COMMAND_RANGE[MOTOR_TYPE[motor]][field]


def command_resolution(motor: int, field: str) -> float:
    """One CAN count of this MIT command field.  The firmware truncates, so a
    value read back from the CAN frame is up to one count below what was sent."""
    lo, hi = command_range(motor, field)
    return (hi - lo) / ((1 << _COMMAND_BITS[field]) - 1)

# ---------------------------------------------------------------------------
# Feedback parsing
# ---------------------------------------------------------------------------

_HEX_FB_RE   = re.compile(r'^F([1-4]) ([0-9A-F]{16}) ([0-9A-F]{8})$')
_HEX_ECHO_RE = re.compile(r'^C([1-4]) ([0-9A-F]{16})$')

# Legacy text firmware.
_POS_RE  = re.compile(r'pos=(-?\d+\.?\d*)\s*(rad|deg)')
_VEL_RE  = re.compile(r'\bvel=(-?\d+\.?\d*)')
_SPD_RE  = re.compile(r'\bspd=(-?\d+\.?\d*)')
_CUR_RE  = re.compile(r'\bI=(-?\d+\.?\d*)')
_TAU_RE  = re.compile(r'\btau=(-?\d+\.?\d*)')
_TEMP_RE = re.compile(r'\bT=(-?\d+)')
_ERR_RE  = re.compile(r'\berr=(-?\d+)')
_TS_RE   = re.compile(r'\bt=(\d+)ms')
# "[TX M1] pos= 3.865 vel= 0.00 kp=  9.0 kd=0.69 tff= 0.00" (printf widths pad with spaces)
_NUM = r'\s*(-?\d+\.?\d*)'
_ECHO_RE = re.compile(
    rf'^\[TX M([1-4])\]\s*pos={_NUM}\s+vel={_NUM}\s+kp={_NUM}\s+kd={_NUM}\s+tff={_NUM}'
)


@dataclass
class CommandEcho:
    """One CAN command as echoed by the STM32 (motor side)."""
    pos:    float   # rad
    vel:    float   # rad/s
    kp:     float   # N·m/rad
    kd:     float   # N·m·s/rad
    tau_ff: float   # N·m


@dataclass
class MotorFeedback:
    """One decoded motor feedback frame."""
    pos:     float          # rad (AK60: relative to its software origin)
    vel:     float          # rad/s
    current: float | None   # A, AK60 only
    torque:  float | None   # N·m, AK70/AK40 only
    temp:    int | None     # °C (as the firmware reported it)
    err:     int | None     # motor fault code, 0 = none


# CubeMars fault codes (MIT and servo mode).  Check against your motor's manual.
MOTOR_ERROR_TEXT: dict[int, str] = {
    0: "no fault",
    1: "motor over-temperature",
    2: "over-current",
    3: "over-voltage",
    4: "under-voltage",
    5: "encoder fault",
    6: "MOSFET over-temperature",
    7: "motor stall",
}

_AK60_POLE_PAIRS = 14
_AK60_REDUCTION  = 6.0
_ERPM_TO_RAD_S   = (2.0 * math.pi) / (60.0 * _AK60_POLE_PAIRS * _AK60_REDUCTION)


def _int8(value: int) -> int:
    return value - 256 if value >= 128 else value


def _int16(hi: int, lo: int) -> int:
    value = (hi << 8) | lo
    return value - 65536 if value >= 32768 else value


def decode_feedback(motor: int, data: bytes) -> MotorFeedback:
    """Decode the 8 feedback CAN data bytes the firmware forwards as "F<n> ..."."""
    if MOTOR_TYPE[motor] == "AK60":
        # Servo-mode reply: int16 pos (0.1 deg), int16 speed (10 eRPM),
        # int16 current (10 mA), int8 temp, int8 error.
        return MotorFeedback(
            pos=math.radians(_int16(data[0], data[1]) * 0.1),
            vel=_int16(data[2], data[3]) * 10.0 * _ERPM_TO_RAD_S,
            current=_int16(data[4], data[5]) * 0.01,
            torque=None,
            temp=_int8(data[6]),
            err=_int8(data[7]),
        )
    # MIT reply: id, 16-bit pos, 12-bit vel, 12-bit torque, temp + 40, error.
    p_int = (data[1] << 8) | data[2]
    v_int = (data[3] << 4) | (data[4] >> 4)
    i_int = ((data[4] & 0xF) << 8) | data[5]
    p_min, p_max = command_range(motor, "pos")
    v_min, v_max = command_range(motor, "vel")
    t_max = command_range(motor, "tau_ff")[1]
    return MotorFeedback(
        pos=_uint_to_float(p_int, p_min, p_max, 16),
        vel=_uint_to_float(v_int, v_min, v_max, 12),
        current=None,
        torque=_uint_to_float(i_int, -t_max, t_max, 12),
        temp=_int8((data[6] - 40) & 0xFF),
        err=_int8(data[7]),
    )


def decode_command(motor: int, data: bytes) -> CommandEcho:
    """Decode an 8-byte MIT command frame, as echoed in "C<n> ..." lines."""
    if MOTOR_TYPE[motor] == "AK60":
        # KP/KD/Pos/Vel/Torque byte order.
        counts = {
            "kp": (data[0] << 4) | (data[1] >> 4),
            "kd": ((data[1] & 0xF) << 8) | data[2],
            "pos": (data[3] << 8) | data[4],
            "vel": (data[5] << 4) | (data[6] >> 4),
            "tau_ff": ((data[6] & 0xF) << 8) | data[7],
        }
    else:
        # Pos/Vel/KP/KD/Torque byte order.
        counts = {
            "pos": (data[0] << 8) | data[1],
            "vel": (data[2] << 4) | (data[3] >> 4),
            "kp": ((data[3] & 0xF) << 8) | data[4],
            "kd": (data[5] << 4) | (data[6] >> 4),
            "tau_ff": ((data[6] & 0xF) << 8) | data[7],
        }
    return CommandEcho(**{
        field: _uint_to_float(count, *command_range(motor, field), _COMMAND_BITS[field])
        for field, count in counts.items()
    })


def _store_feedback(motor: int, fb: MotorFeedback) -> None:
    with _fb_lock:
        _motor_pos[motor]          = fb.pos
        _motor_vel[motor]          = fb.vel
        _motor_current[motor]      = fb.current
        _motor_torque[motor]       = fb.torque
        _motor_temp[motor]         = fb.temp
        _motor_err[motor]          = fb.err
        _motor_measure_time[motor] = time.perf_counter()


def _store_echo(motor: int, values: CommandEcho) -> None:
    now = time.perf_counter()
    with _fb_lock:
        _motor_echo[motor]      = values
        _motor_echo_time[motor] = now
        _echo_queue.append((now, motor, values))


def _parse_line(line: str) -> None:
    """Parse one STM32 line: motor feedback, a command echo, or anything else."""
    hex_fb = _HEX_FB_RE.match(line)
    if hex_fb:
        motor = int(hex_fb.group(1))
        _store_feedback(motor, decode_feedback(motor, bytes.fromhex(hex_fb.group(2))))
        return

    hex_echo = _HEX_ECHO_RE.match(line)
    if hex_echo:
        motor = int(hex_echo.group(1))
        _store_echo(motor, decode_command(motor, bytes.fromhex(hex_echo.group(2))))
        return

    echo = _ECHO_RE.match(line)
    if echo:
        _store_echo(int(echo.group(1)), CommandEcho(*(float(echo.group(i)) for i in range(2, 7))))
        return

    for motor in [1, 2, 3, 4]:
        tag = f"[M{motor}]"
        if not line.startswith(tag):
            continue
        body = line[len(tag):].strip()

        m = _POS_RE.search(body)
        if not m:
            # e.g. "[M1] feedback lost — disabled (re-enable required)"
            _other_lines.append((time.perf_counter(), line))
            return
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

        mc = _CUR_RE.search(body)
        mt = _TAU_RE.search(body)
        mtemp = _TEMP_RE.search(body)
        merr = _ERR_RE.search(body)
        _store_feedback(motor, MotorFeedback(
            pos=val,
            vel=vel,
            current=float(mc.group(1)) if mc else None,
            torque=float(mt.group(1)) if mt else None,
            temp=int(mtemp.group(1)) if mtemp else None,
            err=int(merr.group(1)) if merr else None,
        ))
        return

    _other_lines.append((time.perf_counter(), line))


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
    _ser.write(checksum_line("0X").encode())

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
    pos:     dict[int, float | None]  # motor → joint position (rad), None = no feedback
    vel:     dict[int, float]         # motor → velocity (rad/s)
    current: dict[int, float | None]  # motor → phase current (A), AK60 only; None if unavailable
    torque:  dict[int, float | None]  # motor → measured torque (N·m), AK70/AK40 only; None if unavailable
    time:    dict[int, float]         # motor → PC time of measurement
    temp:    dict[int, int | None]    # motor → temperature (°C); None if unavailable
    err:     dict[int, int | None]    # motor → fault code (0 = none); None if unavailable


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
        pos_raw     = dict(_motor_pos)
        vel_raw     = dict(_motor_vel)
        current_raw = dict(_motor_current)
        torque_raw  = dict(_motor_torque)
        temp_raw    = dict(_motor_temp)
        err_raw     = dict(_motor_err)
        mtime_raw   = dict(_motor_measure_time)

    now = time.perf_counter()
    pos_est: dict[int, float | None] = {}
    for motor in [1, 2, 3, 4]:
        if pos_raw[motor] is None:
            pos_est[motor] = None
            continue
        elapsed = max(0.0, min(lag_cap_s, now - mtime_raw[motor]))
        pos_est[motor] = pos_raw[motor] + vel_raw[motor] * elapsed

    return MotorSnapshot(
        pos=pos_est, vel=vel_raw, current=current_raw, torque=torque_raw,
        time=mtime_raw, temp=temp_raw, err=err_raw,
    )


def raw_snapshot() -> tuple[dict[int, float | None], dict[int, float]]:
    """Return raw (non-extrapolated) pos and vel dicts. Used for zeroing."""
    with _fb_lock:
        return dict(_motor_pos), dict(_motor_vel)


def echo_snapshot() -> tuple[dict[int, CommandEcho | None], dict[int, float]]:
    """Return the latest echoed CAN command per motor and its PC receive time.

    None until the firmware sends an echo (needs TX_ECHO=1; enabled motors only).
    """
    with _fb_lock:
        return dict(_motor_echo), dict(_motor_echo_time)


def drain_echoes() -> list[tuple[float, int, CommandEcho]]:
    """Remove and return every echo received since the last call, oldest first."""
    with _fb_lock:
        echoes = list(_echo_queue)
        _echo_queue.clear()
    return echoes


def drain_other_lines() -> list[tuple[float, str]]:
    """Remove and return STM32 lines that were neither feedback nor echoes."""
    lines = []
    while _other_lines:
        lines.append(_other_lines.popleft())
    return lines

# ---------------------------------------------------------------------------
# Public API — commands
# ---------------------------------------------------------------------------

def send(cmd: str) -> None:
    """Send command lines to the STM32, adding the "*HH" checksum to each."""
    framed = "".join(checksum_line(line) for line in cmd.splitlines() if line)
    with _ser_lock:
        if _ser and _ser.is_open:
            _ser.write(framed.encode())


def format_mit_frame(motor: int, pos: float, vel: float, kp: float, kd: float, tau_ff: float) -> str:
    """The "<n>M..." body for one MIT frame, tau_ff clamped to TAU_FF_LIMIT.

    Digits are kept to what the firmware's CAN packing can resolve (pos
    ~0.0004 rad, vel 0.015-0.03 rad/s, kp 0.12, kd 0.0012, tau 0.002-0.006 N·m).
    """
    tau_ff = max(-TAU_FF_LIMIT[motor], min(TAU_FF_LIMIT[motor], tau_ff))
    return f"{motor}M{pos:.4f},{vel:.3f},{kp:.2f},{kd:.4f},{tau_ff:.4f}"


def send_mit_frame(
    motor: int,
    pos: float,
    vel: float,
    kp: float,
    kd: float,
    tau_ff: float,
) -> int:
    """Send one full MIT control mode frame for a single motor, as one line.

    The firmware applies all five fields together or, if the line is
    damaged, none of them.  Returns the number of bytes written.

    Args:
        motor:  motor index (1-4)
        pos:    desired position (rad, motor side)
        vel:    desired velocity (rad/s, motor side)
        kp:     inner position gain (motor side)
        kd:     inner velocity gain (motor side)
        tau_ff: feedforward torque (N·m, motor side), clamped to TAU_FF_LIMIT
    """
    line = checksum_line(format_mit_frame(motor, pos, vel, kp, kd, tau_ff))
    with _ser_lock:
        if _ser and _ser.is_open:
            _ser.write(line.encode())
    return len(line)


# The firmware's enable soft-start blocks its main loop for ~1.1 s (AK60) to
# ~1.3 s (AK70/AK40), and no other motor is sent a frame meanwhile.  An AK70 or
# AK40 only replies when commanded, so back-to-back enables starved M2 of
# feedback for 2.57 s and its 2 s watchdog disabled it (run 2026-10-27
# 21-31-36: "[M2] feedback lost — disabled").  Each enable is therefore sent
# only after the previous motor's first post-enable echo, which the firmware
# emits once that enable has finished; the loop then services every enabled
# motor before the next enable starts, so no motor waits through more than one.
# Echoes earlier than ENABLE_MIN_S after the E line are ignored: an already
# enabled motor keeps sending 250 ms heartbeats until its new soft-start (at
# least 1.0 s of ramp) begins, and those must not end the wait early.
ENABLE_MIN_S = 0.5
ENABLE_ECHO_TIMEOUT_S = 3.0
ENABLE_SETTLE_S = 0.1


def enable_motors(motors: list[int]) -> None:
    """Enable motors one at a time (the firmware soft-starts each one)."""
    enabled: list[int] = []
    for m in motors:
        print(f"[serial] Enabling M{m}...")
        sent_at = time.perf_counter()
        send(f"{m}E\n")
        deadline = sent_at + ENABLE_ECHO_TIMEOUT_S
        while True:
            with _fb_lock:
                echoed = _motor_echo_time[m] > sent_at + ENABLE_MIN_S
            if echoed or time.perf_counter() > deadline:
                break
            time.sleep(0.01)
        if echoed:
            enabled.append(m)
        else:
            print(
                f"[serial] WARNING: no command echo from M{m} within {ENABLE_ECHO_TIMEOUT_S:g} s "
                "of its enable; it may not be enabled"
            )
        time.sleep(ENABLE_SETTLE_S)
    print(f"[serial] Motors enabled: {enabled}")


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
