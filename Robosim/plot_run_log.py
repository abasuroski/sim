"""Plot a hardware-controller run log as an SVG next to it.

For every motor that reported feedback (or, failing that, was commanded), four
panels share one time axis: encoder vs commanded position, estimated torque
output, Kp and Kd. Shaded bands mark stale feedback; dashed lines mark terminal
commands. The controller calls ``plot_run_log`` when each run ends; run this
file directly to plot existing logs:

    python plot_run_log.py logs/<run log>.csv [more.csv ...]

Standard library only, so it runs wherever the controller does.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from html import escape
from pathlib import Path

MOTORS = (1, 2, 3, 4)
STALE_FEEDBACK_S = 0.1     # a reading older than this is shaded as stale
MAX_POINTS = 2_000         # per series, after min/max downsampling

WIDTH = 1200
LEFT, RIGHT = 84, 24
PANEL_HEIGHT = 130
PANEL_GAP = 50
GROUP_GAP = 56

ACTUAL_COLOR = "#2a78d6"
DESIRED_COLOR = "#eb6834"
STALE_FILL = "#e34948"
EVENT_LABELS = {"a": "a · enable", "s": "s · start", "p": "p · pause", "r": "r · resume", "h": "h · home", "q": "q · quit"}


def _value(row: dict[str, str], key: str) -> float | None:
    text = row.get(key)
    return float(text) if text not in ("", None) else None


def _load(csv_path: Path) -> tuple[list[float], dict[int, dict[str, list[float | None]]], list[tuple[float, str]]]:
    """Return times, per-motor series, and (time, command) events from one log."""
    times: list[float] = []
    motors: dict[int, dict[str, list[float | None]]] = {
        m: {k: [] for k in ("pos", "vel", "age", "cmd", "cmd_vel", "kp", "kd", "tff", "torque")} for m in MOTORS
    }
    events: list[tuple[float, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as log_file:
        for row in csv.DictReader(log_file):
            t = _value(row, "wall_time_s")
            if t is None:
                continue
            times.append(t)
            for entry in (row.get("events") or "").split(" | "):
                if entry.startswith("cmd "):
                    events.append((t, entry[4:].split()[0] if entry[4:].split() else entry[4:]))
            for m in MOTORS:
                s = motors[m]
                pos = _value(row, f"m{m}_encoder_position_rad")
                vel = _value(row, f"m{m}_encoder_velocity_rad_s")
                cmd = _value(row, f"m{m}_cmd_position_rad")
                cmd_vel = _value(row, f"m{m}_cmd_velocity_rad_s")
                kp = _value(row, f"m{m}_cmd_kp")
                kd = _value(row, f"m{m}_cmd_kd")
                tff = _value(row, f"m{m}_cmd_tau_ff_nm")
                s["pos"].append(pos)
                s["vel"].append(vel)
                s["age"].append(_value(row, f"m{m}_feedback_age_s"))
                s["cmd"].append(cmd)
                s["cmd_vel"].append(cmd_vel)
                s["kp"].append(kp)
                s["kd"].append(kd)
                s["tff"].append(tff)
                # MIT law: tau = Kp (P - pos) + Kd (V - vel) + T_ff, using the last reading.
                torque = None
                if None not in (pos, cmd, kp, kd):
                    torque = kp * (cmd - pos) + kd * ((cmd_vel or 0.0) - (vel or 0.0)) + (tff or 0.0)
                s["torque"].append(torque)
    return times, motors, events


def _segments(times: list[float], values: list[float | None]) -> list[list[tuple[float, float]]]:
    """Split a series into runs of consecutive non-missing points."""
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for t, v in zip(times, values):
        if v is None:
            if current:
                segments.append(current)
                current = []
        else:
            current.append((t, v))
    if current:
        segments.append(current)
    return segments


def _downsample(points: list[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    """Keep ordered extrema per bucket so short spikes survive."""
    if len(points) <= max_points or max_points < 4:
        return points
    bucket_size = math.ceil(len(points) / (max_points // 2))
    reduced = [points[0]]
    for start in range(1, len(points) - 1, bucket_size):
        bucket = points[start : min(start + bucket_size, len(points) - 1)]
        low = min(bucket, key=lambda p: p[1])
        high = max(bucket, key=lambda p: p[1])
        reduced.extend(sorted({low, high}, key=lambda p: p[0]))
    reduced.append(points[-1])
    return reduced


def _ticks(low: float, high: float, count: int = 5) -> tuple[float, float, list[float]]:
    """Return a padded, rounded range and its tick values."""
    if not math.isfinite(low) or not math.isfinite(high):
        low, high = 0.0, 1.0
    if high - low < 1e-9:
        pad = max(abs(low) * 0.1, 0.5)
        low, high = low - pad, high + pad
    raw = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    ticks = []
    value = start
    while value <= end + step * 1e-6:
        ticks.append(round(value, 10))
        value += step
    return start, end, ticks


def _fmt(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _stale_intervals(times: list[float], ages: list[float | None]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    start = None
    for t, age in zip(times, ages):
        stale = age is not None and age > STALE_FEEDBACK_S
        if stale and start is None:
            start = t - age
        elif not stale and start is not None:
            intervals.append((start, t))
            start = None
    if start is not None:
        intervals.append((start, times[-1]))
    return intervals


def _motors_to_plot(motors: dict[int, dict[str, list[float | None]]]) -> list[int]:
    reporting = [m for m in MOTORS if any(v is not None for v in motors[m]["pos"])]
    if reporting:
        return reporting
    return [m for m in MOTORS if any(v is not None for v in motors[m]["cmd"])]


def build_svg(csv_path: Path) -> str:
    times, motors, events = _load(csv_path)
    metadata_path = csv_path.with_suffix(".json")
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = {}

    plot_motors = _motors_to_plot(motors) if times else []
    panels = [
        ("Position (motor rad)", [("pos", "Encoder", ACTUAL_COLOR), ("cmd", "Desired", DESIRED_COLOR)]),
        ("Torque output, estimated (N·m)", [("torque", "Kp·(P−pos) + Kd·(V−vel) + T", ACTUAL_COLOR)]),
        ("Kp (N·m/rad)", [("kp", "Kp sent, axis from zero", ACTUAL_COLOR)]),
        ("Kd (N·m·s/rad)", [("kd", "Kd sent, axis from zero", ACTUAL_COLOR)]),
    ]
    header = 96
    group_height = len(panels) * (PANEL_HEIGHT + PANEL_GAP) + 30
    height = header + max(1, len(plot_motors)) * (group_height + GROUP_GAP)
    plot_width = WIDTH - LEFT - RIGHT
    t0, t1 = (times[0], times[-1]) if times else (0.0, 1.0)
    t_low, t_high, t_ticks = _ticks(t0, t1, 10)

    def x(t: float) -> float:
        return LEFT + (t - t_low) / (t_high - t_low) * plot_width

    subtitle_parts = [
        metadata.get("started_utc", ""),
        f"port {metadata['serial_port']}" if metadata.get("serial_port") else "simulation only",
        f"git {metadata['git_revision']}" if metadata.get("git_revision") else "",
    ]
    settings = metadata.get("settings", {})
    if settings:
        subtitle_parts.append(f"CLOSED_LOOP={settings.get('CLOSED_LOOP')}  ZERO_TFF={settings.get('ZERO_TFF')}")
    subtitle = "   ·   ".join(p for p in subtitle_parts if p)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img">',
        "<style>",
        "text { font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; fill: #0b0b0b; }",
        ".title { font-size: 20px; font-weight: 600; } .subtitle, .note { font-size: 12px; fill: #52514e; }",
        ".group { font-size: 16px; font-weight: 600; } .panel { font-size: 13px; font-weight: 600; }",
        ".tick { font-size: 11px; fill: #52514e; font-variant-numeric: tabular-nums; } .legend { font-size: 12px; fill: #52514e; }",
        ".grid { stroke: #e1e0d9; stroke-width: 1; } .frame { fill: none; stroke: #c3c2b7; stroke-width: 1; }",
        ".trace { fill: none; stroke-width: 1.6; stroke-linejoin: round; stroke-linecap: round; }",
        ".event { stroke: #898781; stroke-width: 1; stroke-dasharray: 3 3; } .event-label { font-size: 11px; fill: #52514e; }",
        "</style>",
        f'<rect width="{WIDTH}" height="{height}" fill="#fcfcfb"/>',
        f'<text x="{LEFT}" y="34" class="title">{escape(csv_path.name)}</text>',
        f'<text x="{LEFT}" y="56" class="subtitle">{escape(subtitle)}</text>',
    ]
    legend_x = LEFT
    for label, color, swatch in (
        ("Encoder (measured)", ACTUAL_COLOR, "line"),
        ("Desired (commanded)", DESIRED_COLOR, "line"),
        (f"Stale feedback (> {STALE_FEEDBACK_S:g} s old)", STALE_FILL, "band"),
        ("Terminal command", "#898781", "rule"),
    ):
        if swatch == "line":
            svg.append(f'<line x1="{legend_x}" y1="76" x2="{legend_x + 18}" y2="76" stroke="{color}" stroke-width="2.5"/>')
        elif swatch == "band":
            svg.append(f'<rect x="{legend_x}" y="70" width="18" height="12" fill="{color}" fill-opacity="0.15"/>')
        else:
            svg.append(f'<line x1="{legend_x + 9}" y1="69" x2="{legend_x + 9}" y2="83" class="event"/>')
        svg.append(f'<text x="{legend_x + 24}" y="80" class="legend">{escape(label)}</text>')
        legend_x += 30 + 7 * len(label)

    if not plot_motors:
        svg.append(f'<text x="{LEFT}" y="{header + 30}" class="group">No motor feedback or commands in this run.</text>')

    top = header
    for motor in plot_motors:
        series = motors[motor]
        reported = any(v is not None for v in series["pos"])
        note = "" if reported else "  (no feedback this run; commands only)"
        svg.append(f'<text x="{LEFT}" y="{top + 18}" class="group">M{motor}{escape(note)}</text>')
        stale = _stale_intervals(times, series["age"]) if reported else []
        panel_top = top + 30 + PANEL_GAP - 12
        for title, keys in panels:
            values = [v for key, _, _ in keys for v in series[key] if v is not None]
            from_zero = keys[0][0] in ("kp", "kd")  # gains: show size, not tiny wiggles
            if values:
                low, high, y_ticks = _ticks(min(0.0, min(values)) if from_zero else min(values), max(values))
            else:
                low, high, y_ticks = _ticks(0.0, 1.0)
            bottom = panel_top + PANEL_HEIGHT

            def y(v: float, low: float = low, high: float = high, bottom: float = bottom) -> float:
                return bottom - (v - low) / (high - low) * PANEL_HEIGHT

            svg.append(f'<text x="{LEFT}" y="{panel_top - 8:.1f}" class="panel">{escape(title)}</text>')
            if len(keys) == 1:
                svg.append(f'<text x="{WIDTH - RIGHT}" y="{panel_top - 8:.1f}" text-anchor="end" class="note">{escape(keys[0][1])}</text>')
            for start, end in stale:
                x0, x1 = max(x(start), LEFT), min(x(end), WIDTH - RIGHT)
                if x1 > x0:
                    svg.append(f'<rect x="{x0:.1f}" y="{panel_top:.1f}" width="{x1 - x0:.1f}" height="{PANEL_HEIGHT}" fill="{STALE_FILL}" fill-opacity="0.12"/>')
            for tick in y_ticks:
                svg.append(f'<line x1="{LEFT}" y1="{y(tick):.1f}" x2="{WIDTH - RIGHT}" y2="{y(tick):.1f}" class="grid"/>')
                svg.append(f'<text x="{LEFT - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" class="tick">{_fmt(tick)}</text>')
            for t, label in events:
                svg.append(f'<line x1="{x(t):.1f}" y1="{panel_top:.1f}" x2="{x(t):.1f}" y2="{bottom:.1f}" class="event"/>')
                if title == panels[0][0]:
                    svg.append(f'<text x="{x(t) + 3:.1f}" y="{panel_top + 12:.1f}" class="event-label">{escape(EVENT_LABELS.get(label, label))}</text>')
            svg.append(f'<rect x="{LEFT}" y="{panel_top:.1f}" width="{plot_width}" height="{PANEL_HEIGHT}" class="frame"/>')
            if not values:
                svg.append(f'<text x="{LEFT + plot_width / 2:.1f}" y="{panel_top + PANEL_HEIGHT / 2:.1f}" text-anchor="middle" class="note">No data</text>')
            for key, _, color in keys:
                segments = _segments(times, series[key])
                total = sum(len(s) for s in segments) or 1
                for segment in segments:
                    points = _downsample(segment, max(4, MAX_POINTS * len(segment) // total))
                    path = "M" + " L".join(f"{x(t):.1f},{y(v):.1f}" for t, v in points)
                    svg.append(f'<path d="{path}" class="trace" stroke="{color}"/>')
            for tick in t_ticks:
                svg.append(f'<line x1="{x(tick):.1f}" y1="{bottom:.1f}" x2="{x(tick):.1f}" y2="{bottom + 4:.1f}" class="frame"/>')
                svg.append(f'<text x="{x(tick):.1f}" y="{bottom + 16:.1f}" text-anchor="middle" class="tick">{_fmt(tick)}</text>')
            panel_top = bottom + PANEL_GAP
        svg.append(f'<text x="{LEFT + plot_width / 2:.1f}" y="{panel_top - PANEL_GAP + 30:.1f}" text-anchor="middle" class="note">Seconds since the program started</text>')
        top += group_height + GROUP_GAP
    svg.append("</svg>")
    return "\n".join(svg)


def plot_run_log(csv_path: Path | str) -> Path:
    """Write <log>.svg next to the CSV and return its path."""
    csv_path = Path(csv_path)
    svg_path = csv_path.with_suffix(".svg")
    svg_path.write_text(build_svg(csv_path), encoding="utf-8")
    return svg_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for argument in sys.argv[1:]:
        print(plot_run_log(argument))
