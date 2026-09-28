"""Create a comparison plot from the three motor test logs.

The log containing ``I=... A`` is treated as the measured-current test.  The
two logs containing only ``tau=...`` records are converted with I = tau / kT.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import math
import re
import statistics


KT_NM_PER_A = 0.123
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "motor-current-comparison.svg"
RECORD = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}\.\d{3}).*?\b(?P<field>I|tau)="
    r"(?P<value>-?\d+(?:\.\d+)?)"
)


def parse_log(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Return relative seconds and raw values keyed by their log field."""
    records: dict[str, list[tuple[float, float]]] = {"I": [], "tau": []}
    starts: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RECORD.match(line)
        if not match:
            continue
        field = match.group("field")
        timestamp = datetime.strptime(match.group("time"), "%H:%M:%S.%f")
        seconds = (
            timestamp.hour * 3600
            + timestamp.minute * 60
            + timestamp.second
            + timestamp.microsecond / 1_000_000
        )
        starts.setdefault(field, seconds)
        records[field].append((seconds - starts[field], float(match.group("value"))))
    return records


def min_max_downsample(points: list[tuple[float, float]], max_points: int = 2_000) -> list[tuple[float, float]]:
    """Keep ordered extrema per bucket, preserving short current spikes."""
    if len(points) <= max_points:
        return points
    buckets = max_points // 2
    bucket_size = math.ceil(len(points) / buckets)
    reduced = [points[0]]
    for start in range(1, len(points) - 1, bucket_size):
        bucket = points[start : min(start + bucket_size, len(points) - 1)]
        low = min(bucket, key=lambda point: point[1])
        high = max(bucket, key=lambda point: point[1])
        reduced.extend(sorted((low, high), key=lambda point: point[0]))
    reduced.append(points[-1])
    return reduced


def nice_step(span: float, target_ticks: int = 5) -> float:
    raw = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    for candidate in (1, 2, 2.5, 5, 10):
        if normalized <= candidate:
            return candidate * magnitude
    return 10 * magnitude


def tick_values(low: float, high: float, count: int = 5) -> list[float]:
    step = nice_step(high - low, count)
    start = math.ceil(low / step) * step
    values: list[float] = []
    value = start
    while value <= high + step * 0.001:
        values.append(value)
        value += step
    return values


def fmt(value: float) -> str:
    return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"


def build_svg(series: list[dict[str, object]]) -> str:
    width, height = 1200, 990
    left, right, top, bottom = 105, 50, 130, 55
    gap = 35
    plot_width = width - left - right
    panel_height = (height - top - bottom - gap * (len(series) - 1)) / len(series)

    all_values = [value for item in series for _, value in item["points"]]  # type: ignore[index]
    value_low, value_high = min(all_values), max(all_values)
    padding = max((value_high - value_low) * 0.07, 0.4)
    y_low, y_high = value_low - padding, value_high + padding
    y_ticks = tick_values(y_low, y_high, 6)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Motor current comparison</title>',
        '<desc id="description">Three motor test traces. M1 is measured current. M2 and M4 are torque readings converted to current using a torque constant of 0.123 newton metres per ampere.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#17212b}.title{font-size:28px;font-weight:700}.subtitle{font-size:15px;fill:#52616b}.axis{font-size:13px;fill:#394955}.tick{font-size:12px;fill:#52616b}.panel-title{font-size:18px;font-weight:700}.detail{font-size:13px;fill:#52616b}.grid{stroke:#d7dfe5;stroke-width:1}.frame{stroke:#60707c;stroke-width:1;fill:none}.trace{fill:none;stroke-linejoin:round;stroke-linecap:round;stroke-width:1.5}</style>',
        '<text x="105" y="45" class="title">Motor current comparison</text>',
        '<text x="105" y="72" class="subtitle">M1: measured current  •  M2 and M4: inferred as I = τ / 0.123 N·m/A</text>',
        '<text x="28" y="560" class="axis" transform="rotate(-90 28 560)">Current (A)</text>',
    ]

    for index, item in enumerate(series):
        points: list[tuple[float, float]] = item["points"]  # type: ignore[assignment]
        display = min_max_downsample(points)
        duration = points[-1][0] if points else 1.0
        duration = max(duration, 0.001)
        panel_top = top + index * (panel_height + gap)
        panel_bottom = panel_top + panel_height

        def x_coord(seconds: float) -> float:
            return left + seconds / duration * plot_width

        def y_coord(amps: float) -> float:
            return panel_bottom - (amps - y_low) / (y_high - y_low) * panel_height

        label = escape(str(item["label"]))
        detail = escape(str(item["detail"]))
        color = str(item["color"])
        svg.append(f'<text x="{left}" y="{panel_top - 10:.1f}" class="panel-title">{label}</text>')
        svg.append(f'<text x="{width - right}" y="{panel_top - 10:.1f}" text-anchor="end" class="detail">{detail}</text>')

        for tick in y_ticks:
            y = y_coord(tick)
            if panel_top <= y <= panel_bottom:
                svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" class="grid"/>')
                svg.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" class="tick">{fmt(tick)}</text>')
        svg.append(f'<rect x="{left}" y="{panel_top:.2f}" width="{plot_width}" height="{panel_height:.2f}" class="frame"/>')

        path = " ".join(
            f"{'M' if point_index == 0 else 'L'}{x_coord(seconds):.2f},{y_coord(amps):.2f}"
            for point_index, (seconds, amps) in enumerate(display)
        )
        svg.append(f'<path d="{path}" class="trace" stroke="{color}"/>')

        for tick in tick_values(0, duration, 5):
            x = x_coord(tick)
            svg.append(f'<line x1="{x:.2f}" y1="{panel_bottom:.2f}" x2="{x:.2f}" y2="{panel_bottom + 6:.2f}" class="frame"/>')
            svg.append(f'<text x="{x:.2f}" y="{panel_bottom + 22:.2f}" text-anchor="middle" class="tick">{fmt(tick)}</text>')
        svg.append(f'<text x="{left + plot_width / 2:.2f}" y="{panel_bottom + 42:.2f}" text-anchor="middle" class="axis">Elapsed time from first sample (s)</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def main() -> None:
    logs = sorted(ROOT.glob("log*.txt"))
    if len(logs) != 3:
        raise RuntimeError(f"Expected exactly three log*.txt files, found {len(logs)}")
    parsed = {path: parse_log(path) for path in logs}
    measured_logs = [path for path, records in parsed.items() if records["I"]]
    torque_logs = [path for path, records in parsed.items() if records["tau"] and not records["I"]]
    if len(measured_logs) != 1 or len(torque_logs) != 2:
        raise RuntimeError("Expected one log with current records and two torque-only logs")

    current_log = measured_logs[0]
    measured = parsed[current_log]["I"]
    series: list[dict[str, object]] = [
        {
            "label": f"M1 — measured current ({current_log.name})",
            "points": measured,
            "detail": f"{len(measured):,} samples  |  mean {statistics.fmean(value for _, value in measured):.2f} A  |  range {min(value for _, value in measured):.2f}–{max(value for _, value in measured):.2f} A",
            "color": "#0072B2",
        }
    ]
    colors = ("#D55E00", "#009E73")
    for path, color in zip(torque_logs, colors):
        torque = parsed[path]["tau"]
        inferred = [(seconds, newton_metres / KT_NM_PER_A) for seconds, newton_metres in torque]
        motor = "M2" if "14-54" in path.name else "M4"
        series.append(
            {
                "label": f"{motor} — current inferred from torque ({path.name})",
                "points": inferred,
                "detail": f"{len(inferred):,} samples  |  mean {statistics.fmean(value for _, value in inferred):.2f} A  |  range {min(value for _, value in inferred):.2f}–{max(value for _, value in inferred):.2f} A",
                "color": color,
            }
        )
    OUTPUT.write_text(build_svg(series), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
