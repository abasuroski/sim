r"""Run reproducible position-PD step tests for the robot_master MJCF.

The generated MJCF uses MuJoCo ``position`` actuators.  This program adjusts
the selected actuator's proportional and derivative gains in memory, applies a
step in its position reference, and reports tracking metrics.  The XML on disk
is never changed, so every invocation begins from the same model.

Examples (from the repository root):

    & .\Robosim\.venv\Scripts\python.exe .\Robosim\tune_robot_master_controller.py --list-actuators
    & .\Robosim\.venv\Scripts\python.exe .\Robosim\tune_robot_master_controller.py --actuator ak60_revolute_1 --target 0.15 --kp 12 --kd 2.2 --csv .\Robosim\results\ak60_trial.csv
    & .\Robosim\.venv\Scripts\python.exe .\Robosim\tune_robot_master_controller.py --actuator ak40_revolute_4 --target 0.4 --kp 3 --kd 0.5 --viewer
    & .\Robosim\.venv\Scripts\python.exe .\Robosim\tune_robot_master_controller.py --actuator ak60_revolute_1 --turns 1 --ramp-duration 3 --duration 5 --viewer

``kp`` is in joint-output N m/rad and ``kd`` is in joint-output N m s/rad.
Torque remains limited by the motor actuator's ``forcerange`` in
robot_master.xml, with transmission scaling applied to reported joint torque.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Iterable

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).resolve().parent / "robot_master" / "mjcf" / "robot_master.xml"
EPSILON = 1e-9


@dataclass(frozen=True)
class Sample:
    """One physics-step sample for the actuator under test."""

    time: float
    target: float
    position: float
    velocity: float
    torque: float
    contact_count: int


def actuator_name(model: mujoco.MjModel, actuator_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    return name if name is not None else f"actuator_{actuator_id}"


def joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    return name if name is not None else f"joint_{joint_id}"


def actuator_gear(model: mujoco.MjModel, actuator_id: int) -> float:
    """Return the scalar motor-to-joint transmission ratio for a joint actuator."""
    gear = float(model.actuator_gear[actuator_id, 0])
    if abs(gear) <= EPSILON:
        raise ValueError(f"{actuator_name(model, actuator_id)} has a zero transmission gear")
    return gear


def print_actuators(model: mujoco.MjModel) -> None:
    """Print the control surface exposed by this particular MJCF."""
    print("id  actuator              joint        joint target range      gear    output torque (N m)  default kp/kd")
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        gear = actuator_gear(model, actuator_id)
        joint_range = model.jnt_range[joint_id]
        force_range = model.actuator_forcerange[actuator_id] * abs(gear)
        kp = model.actuator_gainprm[actuator_id, 0] * gear**2
        kd = -model.actuator_biasprm[actuator_id, 2] * gear**2
        range_text = f"{joint_range[0]:7.3f} .. {joint_range[1]:7.3f}" if model.jnt_limited[joint_id] else "unlimited"
        print(
            f"{actuator_id:>2}  {actuator_name(model, actuator_id):<20} "
            f"{joint_name(model, joint_id):<12} {range_text:<23} {gear:>4.2f}    "
            f"{force_range[0]:7.2f} .. {force_range[1]:7.2f}   {kp:.3g}/{kd:.3g}"
        )


def resolve_actuator(model: mujoco.MjModel, value: str) -> int:
    """Resolve a numeric actuator id or an actuator name with a useful error."""
    if value.isdecimal():
        actuator_id = int(value)
        if 0 <= actuator_id < model.nu:
            return actuator_id
    else:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, value)
        if actuator_id >= 0:
            return actuator_id
    names = ", ".join(actuator_name(model, index) for index in range(model.nu))
    raise ValueError(f"Unknown actuator {value!r}. Choose an id from 0 to {model.nu - 1} or one of: {names}")


def configure_position_pd(model: mujoco.MjModel, actuator_id: int, kp: float | None, kd: float | None) -> tuple[float, float]:
    """Set a position actuator's affine gain terms to ``kp`` and ``kd``.

    MuJoCo expands ``<position kp=... kv=...>`` to a fixed gain and affine
    bias: gainprm=[kp, ...] and biasprm=[0, -kp, -kv, ...].  Changing these
    arrays before stepping is equivalent to choosing different XML gains.
    """
    gear = actuator_gear(model, actuator_id)
    selected_kp = float(model.actuator_gainprm[actuator_id, 0] * gear**2) if kp is None else kp
    selected_kd = float(-model.actuator_biasprm[actuator_id, 2] * gear**2) if kd is None else kd
    if not math.isfinite(selected_kp) or selected_kp <= 0:
        raise ValueError("kp must be greater than zero")
    if not math.isfinite(selected_kd) or selected_kd < 0:
        raise ValueError("kd must be zero or greater")

    actuator_kp = selected_kp / gear**2
    actuator_kd = selected_kd / gear**2
    model.actuator_gainprm[actuator_id, 0] = actuator_kp
    model.actuator_biasprm[actuator_id, 0] = 0.0
    model.actuator_biasprm[actuator_id, 1] = -actuator_kp
    model.actuator_biasprm[actuator_id, 2] = -actuator_kd
    return selected_kp, selected_kd


def check_target_range(model: mujoco.MjModel, actuator_id: int, target: float) -> None:
    if not math.isfinite(target):
        raise ValueError("target must be finite")
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    if not model.jnt_limited[joint_id]:
        return
    low, high = model.jnt_range[joint_id]
    if not low <= target <= high:
        raise ValueError(
            f"Target {target:.6g} rad is outside {joint_name(model, joint_id)}'s "
            f"mechanical range [{low:.6g}, {high:.6g}] rad."
        )


def make_other_actuators_passive(model: mujoco.MjModel, actuator_id: int) -> None:
    """Release non-tested position servos so they do not fight the step.

    A one-actuator test should not silently command the other motors to zero.
    The option to hold them at zero remains available for testing the complete
    multi-drive controller state.
    """
    for other_id in range(model.nu):
        if other_id == actuator_id:
            continue
        model.actuator_gainprm[other_id, 0] = 0.0
        model.actuator_biasprm[other_id, 0:3] = 0.0


def is_finite(data: mujoco.MjData) -> bool:
    return all(math.isfinite(value) for value in data.qpos) and all(math.isfinite(value) for value in data.qvel)


def active_contact_pairs(model: mujoco.MjModel, data: mujoco.MjData) -> set[tuple[str, str]]:
    """Return body-name pairs participating in the current contact constraints."""
    pairs: set[tuple[str, str]] = set()
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom1]))
        body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom2]))
        if body1 is not None and body2 is not None:
            pairs.add(tuple(sorted((body1, body2))))
    return pairs


def step_test(
    model: mujoco.MjModel,
    actuator_id: int,
    target: float,
    duration: float,
    ramp_duration: float,
    show_viewer: bool,
) -> tuple[float, list[Sample], set[tuple[str, str]]]:
    """Run a reference-position step or linear ramp with the selected actuator."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    joint_id = int(model.actuator_trnid[actuator_id, 0])
    qpos_address = int(model.jnt_qposadr[joint_id])
    qvel_address = int(model.jnt_dofadr[joint_id])
    gear = actuator_gear(model, actuator_id)
    initial_position = float(data.qpos[qpos_address])

    def reference_at(simulation_time: float) -> float:
        if ramp_duration <= EPSILON:
            return target
        progress = min(simulation_time / ramp_duration, 1.0)
        return initial_position + progress * (target - initial_position)

    data.ctrl[actuator_id] = gear * reference_at(data.time)
    mujoco.mj_forward(model, data)

    samples = [
        Sample(
            time=float(data.time),
            target=reference_at(data.time),
            position=initial_position,
            velocity=float(data.qvel[qvel_address]),
            torque=float(data.actuator_force[actuator_id] * gear),
            contact_count=int(data.ncon),
        )
    ]
    contact_pairs = active_contact_pairs(model, data)

    def advance_one_step() -> None:
        data.ctrl[actuator_id] = gear * reference_at(data.time)
        mujoco.mj_step(model, data)
        if not is_finite(data):
            raise RuntimeError(f"Simulation became non-finite at t={data.time:.4f} s")
        contact_pairs.update(active_contact_pairs(model, data))
        samples.append(
            Sample(
                time=float(data.time),
                target=reference_at(data.time),
                position=float(data.qpos[qpos_address]),
                velocity=float(data.qvel[qvel_address]),
                torque=float(data.actuator_force[actuator_id] * gear),
                contact_count=int(data.ncon),
            )
        )

    if not show_viewer:
        while data.time + EPSILON < duration:
            advance_one_step()
    else:
        print("Viewer open. Close its window to stop the test early.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_wall_time = time.perf_counter()
            while viewer.is_running() and data.time + EPSILON < duration:
                advance_one_step()
                viewer.sync()
                # Keep GUI trials close to real time without affecting the
                # deterministic physics sequence used in headless runs.
                remaining = start_wall_time + data.time - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

    return initial_position, samples, contact_pairs


def first_time(samples: Iterable[Sample], predicate: Callable[[Sample], bool]) -> float | None:
    for sample in samples:
        if predicate(sample):
            return sample.time
    return None


def format_seconds(value: float | None) -> str:
    return "not reached" if value is None else f"{value:.3f} s"


def report_metrics(
    samples: list[Sample], initial_position: float, target: float, torque_limit: float, ramp_duration: float
) -> None:
    positions = [sample.position for sample in samples]
    errors = [sample.position - sample.target for sample in samples]
    torques = [abs(sample.torque) for sample in samples]
    step_size = target - initial_position
    rms_error = math.sqrt(sum(error * error for error in errors) / len(errors))
    peak_torque = max(torques)
    saturation = 100.0 * sum(torque >= torque_limit - 1e-6 for torque in torques) / len(torques)
    peak_contacts = max(sample.contact_count for sample in samples)
    contact_time = 100.0 * sum(sample.contact_count > 0 for sample in samples) / len(samples)

    print(f"Completed {samples[-1].time:.3f} s ({len(samples)} physics samples).")
    print(f"Final position: {positions[-1]:.5f} rad; final error: {errors[-1]:+.5f} rad; RMS error: {rms_error:.5f} rad")
    print(f"Peak actuator torque: {peak_torque:.3f} N m; time at torque limit: {saturation:.1f}%")
    print(f"Collision constraints: peak {peak_contacts} contacts; active for {contact_time:.1f}% of samples.")

    if ramp_duration > EPSILON:
        print(f"Reference was ramped from {initial_position:.5f} to {target:.5f} rad over {ramp_duration:.3f} s.")
        print("Step-response metrics are omitted for a ramped reference.")
        return

    if abs(step_size) <= EPSILON:
        return

    direction = 1.0 if step_size > 0 else -1.0
    normalized = [direction * (position - initial_position) / abs(step_size) for position in positions]
    t10 = first_time(samples, lambda sample: direction * (sample.position - initial_position) / abs(step_size) >= 0.10)
    t90 = first_time(samples, lambda sample: direction * (sample.position - initial_position) / abs(step_size) >= 0.90)
    rise_time = None if t10 is None or t90 is None else t90 - t10
    overshoot = max(0.0, max(normalized) - 1.0) * 100.0
    tolerance = 0.02 * abs(step_size)
    last_outside = max((index for index, error in enumerate(errors) if abs(error) > tolerance), default=-1)
    settling_time = 0.0 if last_outside < 0 else (samples[last_outside + 1].time if last_outside + 1 < len(samples) else None)
    print(f"10-90% rise time: {format_seconds(rise_time)}; overshoot: {overshoot:.1f}%; 2% settling time: {format_seconds(settling_time)}")


def write_csv(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            ("time_s", "target_rad", "position_rad", "velocity_rad_per_s", "joint_output_torque_nm", "contact_count")
        )
        writer.writerows(
            (sample.time, sample.target, sample.position, sample.velocity, sample.torque, sample.contact_count)
            for sample in samples
        )
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune one robot_master MuJoCo position actuator with a step or ramp test.")
    parser.add_argument("--list-actuators", action="store_true", help="show actuator ids, target ranges, limits, and default gains")
    parser.add_argument("--actuator", default="0", help="actuator name or id (default: 0)")
    parser.add_argument("--target", type=float, help="final position target in radians (default: 0.1)")
    parser.add_argument(
        "--turns",
        type=float,
        help="number of turns from the reset pose; requires --ramp-duration so the cyclic hinge reference moves continuously",
    )
    parser.add_argument("--kp", type=float, help="joint-output position gain in N m/rad; default is the MJCF value")
    parser.add_argument("--kd", type=float, help="joint-output velocity damping in N m s/rad; default is the MJCF value")
    parser.add_argument("--duration", type=float, default=5.0, help="test duration in seconds (default: 5)")
    parser.add_argument(
        "--ramp-duration",
        type=float,
        default=0.0,
        help="linearly move the target from the initial angle over this many seconds; zero is a step (default: 0)",
    )
    parser.add_argument(
        "--hold-other-actuators",
        action="store_true",
        help="keep all non-tested position servos at their zero-radian target instead of releasing them",
    )
    parser.add_argument(
        "--show-contacts",
        action="store_true",
        help="list body pairs that made contact during the trial",
    )
    parser.add_argument("--viewer", action="store_true", help="display the same test in MuJoCo's interactive viewer")
    parser.add_argument("--csv", type=Path, help="optional CSV path for the raw simulated trace")
    arguments = parser.parse_args()
    if arguments.target is not None and arguments.turns is not None:
        parser.error("use either --target or --turns, not both")
    if arguments.turns is not None:
        if not math.isfinite(arguments.turns):
            parser.error("--turns must be finite")
        if arguments.ramp_duration <= EPSILON:
            parser.error("--turns requires a positive --ramp-duration")
        arguments.target = arguments.turns * math.tau
    elif arguments.target is None:
        arguments.target = 0.1
    if not math.isfinite(arguments.duration) or arguments.duration <= 0:
        parser.error("--duration must be finite and greater than zero")
    if not math.isfinite(arguments.ramp_duration) or arguments.ramp_duration < 0:
        parser.error("--ramp-duration must be finite and zero or greater")
    if arguments.ramp_duration > arguments.duration:
        parser.error("--ramp-duration cannot be longer than --duration")
    return arguments


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    if args.list_actuators:
        print_actuators(model)
        return

    actuator_id = resolve_actuator(model, args.actuator)
    check_target_range(model, actuator_id, args.target)
    kp, kd = configure_position_pd(model, actuator_id, args.kp, args.kd)
    if not args.hold_other_actuators:
        make_other_actuators_passive(model, actuator_id)
    torque_limit = max(abs(value) for value in model.actuator_forcerange[actuator_id]) * abs(actuator_gear(model, actuator_id))
    print(
        f"Testing {actuator_name(model, actuator_id)} at {args.target:.4f} rad "
        f"with output kp={kp:.6g} N m/rad and output kd={kd:.6g} N m s/rad."
    )
    if args.ramp_duration > EPSILON:
        print(f"Reference will ramp continuously over {args.ramp_duration:.3f} s.")
    if args.hold_other_actuators:
        print("Non-tested actuators hold their zero-radian references.")
    else:
        print("Non-tested actuators are passive for this isolated test.")
    initial_position, samples, contact_pairs = step_test(
        model, actuator_id, args.target, args.duration, args.ramp_duration, args.viewer
    )
    report_metrics(samples, initial_position, args.target, torque_limit, args.ramp_duration)
    if args.show_contacts:
        if contact_pairs:
            print("Contacting body pairs:")
            for body1, body2 in sorted(contact_pairs):
                print(f"  {body1} <-> {body2}")
        else:
            print("No collision contacts occurred during the trial.")
    if args.csv is not None:
        write_csv(args.csv, samples)


if __name__ == "__main__":
    main()
