"""
measure_lag.py — STM32 → PC serial latency measurement

How the clock-offset works (NTP algorithm)
-------------------------------------------
The STM32 and PC clocks are not synchronised.  To find the true one-way lag
we send a PING, the STM32 replies immediately with "PONG t=<stm_ms>ms".

    pc_send  ──PING──►  STM32  ──PONG──►  pc_recv
                         stm_ms (when PONG was generated)

    rtt    = pc_recv - pc_send
    offset = (pc_send + rtt/2) - stm_ms   ← NTP formula

After calibration, for every motor message with t=<stm_ms>ms:

    true_lag = (pc_recv_ms - stm_ms) - offset

A positive lag means the message arrived later than the clock offset predicts.
We run CALIB_PINGS pings and keep the estimate from the one with the
smallest RTT (least OS scheduling jitter).

Usage
-----
    pip install pyserial
    python measure_lag.py              # prompts for port
    python measure_lag.py COM3         # specify port
    python measure_lag.py COM3 --csv   # also save to lag_log.csv
"""

import sys
import time
import re
import serial
import serial.tools.list_ports

# ── config ────────────────────────────────────────────────────────────────────
BAUD         = 115200
CALIB_PINGS  = 20      # number of pings used to find best clock offset
PING_PAUSE   = 0.05    # seconds between pings during calibration
PRINT_EVERY  = 1       # print every Nth motor message (1 = all)
# ─────────────────────────────────────────────────────────────────────────────

_TS_RE   = re.compile(r't=(\d+)ms')
_PONG_RE = re.compile(r'PONG t=(\d+)us')


def pick_port(arg=None):
    if arg:
        return arg
    ports = serial.tools.list_ports.comports()
    if not ports:
        sys.exit("No serial ports found.")
    if len(ports) == 1:
        print(f"Using {ports[0].device}  ({ports[0].description})")
        return ports[0].device
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  {p.description}")
    idx = int(input("Select port index: "))
    return ports[idx].device


def calibrate(ser):
    """
    Send CALIB_PINGS pings.  For each one measure RTT and compute the
    NTP clock offset.  Return the offset from the ping with the
    smallest RTT (cleanest estimate), plus stats.
    """
    print(f"  Sending {CALIB_PINGS} pings …")
    results = []

    for i in range(CALIB_PINGS):
        ser.reset_input_buffer()
        pc_send = time.time() * 1000.0
        ser.write(b"PING\n")

        # Read lines until we see a PONG (motor feedback may arrive first)
        deadline = time.time() + 1.0
        pong_line = None
        while time.time() < deadline:
            raw = ser.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if _PONG_RE.search(line):
                pong_line = line
                break

        pc_recv = time.time() * 1000.0

        if pong_line is None:
            print(f"    ping {i+1}: no PONG (timeout)")
            continue

        stm_ms = int(_PONG_RE.search(pong_line).group(1)) / 1000.0  # us → ms
        rtt    = pc_recv - pc_send
        offset = (pc_send + rtt / 2.0) - stm_ms
        results.append((rtt, offset))
        print(f"    ping {i+1:2d}: rtt={rtt:.2f}ms  offset={offset:.2f}ms")
        time.sleep(PING_PAUSE)

    if not results:
        sys.exit("Calibration failed: no PONG responses received.\n"
                 "Make sure the firmware was flashed with the PING handler.")

    best_rtt, best_offset = min(results, key=lambda x: x[0])
    avg_rtt = sum(r for r, _ in results) / len(results)
    return best_offset, best_rtt, avg_rtt


def main():
    save_csv = "--csv" in sys.argv
    port_arg = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    port = pick_port(port_arg)

    csv_file = None
    if save_csv:
        csv_file = open("lag_log.csv", "w")
        csv_file.write("motor,stm_ms,pc_ms,rtt_ms,lag_ms\n")
        print("Saving to lag_log.csv")

    print(f"\nOpening {port} @ {BAUD} baud …")
    with serial.Serial(port, BAUD, timeout=1) as ser:
        print("Connected.\n")

        # ── calibration ───────────────────────────────────────────────────
        print("=== Calibration (NTP clock-offset estimation) ===")
        offset_ms, best_rtt, avg_rtt = calibrate(ser)
        print(f"\n  Best RTT  : {best_rtt:.2f} ms  → one-way estimate {best_rtt/2:.2f} ms")
        print(f"  Avg  RTT  : {avg_rtt:.2f} ms")
        print(f"  Clock offset: {offset_ms:.2f} ms  (PC ahead of STM32 by this much)\n")

        # ── live measurement ───────────────────────────────────────────────
        print(f"  {'Motor':<8} {'stm_ms':>10} {'lag_ms':>10}  {'min':>8}  {'avg':>8}  {'max':>8}")
        print(f"  {'-'*8} {'-'*10} {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

        count       = 0
        lag_history = []

        while True:
            raw = ser.readline()
            if not raw:
                continue

            pc_ms = time.time() * 1000.0

            try:
                line = raw.decode("utf-8", errors="replace").rstrip()
            except Exception:
                continue

            # Ignore PONG lines that arrive after calibration
            if line.startswith("PONG"):
                continue

            m = _TS_RE.search(line)
            if not m:
                print(f"  (no ts) {line}")
                continue

            stm_ms = float(m.group(1))  # already ms
            lag_ms = (pc_ms - stm_ms) - offset_ms
            lag_history.append(lag_ms)
            count += 1

            motor_m = re.match(r'\[M(\d+)\]', line)
            motor   = motor_m.group(1) if motor_m else "?"

            if csv_file:
                csv_file.write(f"{motor},{stm_ms},{pc_ms:.1f},{best_rtt:.2f},{lag_ms:.2f}\n")
                csv_file.flush()

            if count % PRINT_EVERY == 0:
                mn  = min(lag_history)
                avg = sum(lag_history) / len(lag_history)
                mx  = max(lag_history)
                print(f"  M{motor:<7} {stm_ms:>10}   {lag_ms:>+9.2f}  {mn:>8.2f}  {avg:>8.2f}  {mx:>8.2f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
