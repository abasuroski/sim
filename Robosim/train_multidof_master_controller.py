"""Collision-screened, multi-DOF training for ``master_controller.py``.

The existing single-joint tuner is intentionally conservative.  This program
instead samples simultaneous four-actuator targets over the complete *modelled*
command windows, ramps to each target from ``hardware_zero``, and discards a
candidate whenever MuJoCo reports a collision constraint, non-finite state, or
unreachable final state.  It then jointly tunes all position-scheduled PD
controllers on a diverse subset of the accepted multi-joint trajectories.

For continuous actuators, MuJoCo has no finite mechanical stop.  Their
``ctrlrange`` of -2*pi..+2*pi is therefore the explicit exploration window;
it is not a claim that the physical hardware can rotate forever.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from master_controller import MasterController, PositionSchedule, actuator_gear, actuator_name, joint_name, output_pd_gains
from robot_master_configuration import (
    HARDWARE_ZERO_KEYFRAME_NAME,
    TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2,
    TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S,
)
from trapezoidal_motion import trapezoidal_duration, trapezoidal_position


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "robot_master" / "mjcf" / "robot_master.xml"
DEFAULT_SCHEDULE = ROOT / "master_controller_schedule.json"
EPSILON = 1e-9
MAX_FREQUENCY_RAD_S = 60.0
MIN_FREQUENCY_RAD_S = 1.0


@dataclass(frozen=True)
class DrivenJoint:
    actuator_id: int
    actuator: str
    joint: str
    qpos_address: int
    gear: float
    command_min_rad: float
    command_max_rad: float
    schedule_min_rad: float
    schedule_max_rad: float


@dataclass(frozen=True)
class Rollout:
    safe: bool
    cost: float
    rms_error_rad: float
    max_error_rad: float
    peak_contacts: int
    peak_saturation_percent: float


def keyframe_id(model: mujoco.MjModel) -> int:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HARDWARE_ZERO_KEYFRAME_NAME)
    if key_id < 0:
        raise RuntimeError(f"MJCF has no {HARDWARE_ZERO_KEYFRAME_NAME!r} keyframe; regenerate it first")
    return key_id


def modelled_command_bounds(model: mujoco.MjModel, actuator_id: int) -> tuple[float, float]:
    """Return the finite joint-coordinate exploration window for one actuator.

    Even MuJoCo's unlimited motors have a deliberate ``ctrlrange`` for the
    interactive slider.  That range is used as the explicit finite window for
    full-range simulation exploration.
    """
    gear = actuator_gear(model, actuator_id)
    low, high = (float(value / gear) for value in model.actuator_ctrlrange[actuator_id])
    return min(low, high), max(low, high)


def driven_joints(model: mujoco.MjModel, key_id: int) -> list[DrivenJoint]:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    result: list[DrivenJoint] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        address = int(model.jnt_qposadr[joint_id])
        command_min, command_max = modelled_command_bounds(model, actuator_id)
        actual_home = float(data.qpos[address])
        result.append(
            DrivenJoint(
                actuator_id=actuator_id,
                actuator=actuator_name(model, actuator_id),
                joint=joint_name(model, joint_id),
                qpos_address=address,
                gear=actuator_gear(model, actuator_id),
                command_min_rad=command_min,
                command_max_rad=command_max,
                schedule_min_rad=min(command_min, actual_home),
                schedule_max_rad=max(command_max, actual_home),
            )
        )
    return result


def halton(index: int, base: int) -> float:
    """Return one deterministic low-discrepancy coordinate in [0, 1)."""
    fraction = 1.0
    value = 0.0
    while index:
        fraction /= base
        value += fraction * (index % base)
        index //= base
    return value


def candidate_targets(driven: list[DrivenJoint], count: int) -> list[np.ndarray]:
    """Generate corners plus low-discrepancy simultaneous multi-joint targets."""
    targets: list[np.ndarray] = []
    # Binary corners deliberately exercise the extrema of every configured
    # joint range.  Collision screening below decides which are usable.
    for corner in range(1 << len(driven)):
        targets.append(
            np.array(
                [joint.command_max_rad if corner & (1 << index) else joint.command_min_rad for index, joint in enumerate(driven)],
                dtype=np.float64,
            )
        )
    primes = (2, 3, 5, 7)
    for index in range(1, count + 1):
        targets.append(
            np.array(
                [
                    joint.command_min_rad
                    + halton(index, primes[dimension]) * (joint.command_max_rad - joint.command_min_rad)
                    for dimension, joint in enumerate(driven)
                ],
                dtype=np.float64,
            )
        )
    return targets


def bounded_frequency_pair(midpoint: float, ratio: float) -> tuple[float, float]:
    """Return an exact endpoint ratio while keeping both frequencies bounded."""
    root = math.sqrt(ratio)
    midpoint = min(
        MAX_FREQUENCY_RAD_S * min(root, 1.0 / root),
        max(MIN_FREQUENCY_RAD_S * max(root, 1.0 / root), midpoint),
    )
    return midpoint / root, midpoint * root


def load_initial_parameters(
    model: mujoco.MjModel, data: mujoco.MjData, driven: list[DrivenJoint], initial_schedule: Path
) -> tuple[np.ndarray, list[float]]:
    """Use the last trained schedule when possible, otherwise infer MuJoCo defaults."""
    by_name: dict[str, PositionSchedule] = {}
    if initial_schedule.exists():
        by_name = {schedule.actuator: schedule for schedule in MasterController.from_json(initial_schedule).schedules}
    matrix = np.empty((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, matrix)
    values: list[float] = []
    ratios: list[float] = []
    for item in driven:
        prior = by_name.get(item.actuator)
        if prior is not None:
            midpoint = math.sqrt(prior.natural_frequency_low_rad_s * prior.natural_frequency_high_rad_s)
            zeta = prior.damping_ratio
            ratio = prior.natural_frequency_high_rad_s / prior.natural_frequency_low_rad_s
        else:
            joint_id = int(model.actuator_trnid[item.actuator_id, 0])
            dof = int(model.jnt_dofadr[joint_id])
            inertia = max(1e-6, float(matrix[dof, dof]))
            kp, kd = output_pd_gains(model, item.actuator_id)
            midpoint = math.sqrt(max(kp, 1e-6) / inertia)
            zeta = kd / (2.0 * inertia * midpoint)
            ratio = 1.25
        values.append(math.log(min(MAX_FREQUENCY_RAD_S, max(MIN_FREQUENCY_RAD_S, midpoint))))
        values.append(math.log(min(4.0, max(0.35, zeta))))
        # Preserve the previously selected direction while enforcing at least
        # the requested 1.25:1 position dependence.
        magnitude = max(1.25, ratio, 1.0 / ratio)
        ratios.append(magnitude if ratio >= 1.0 else 1.0 / magnitude)
    return np.array(values, dtype=np.float64), ratios


def schedules_from_parameters(driven: list[DrivenJoint], parameters: np.ndarray, ratios: list[float]) -> list[PositionSchedule]:
    schedules: list[PositionSchedule] = []
    for index, item in enumerate(driven):
        midpoint = math.exp(float(parameters[2 * index]))
        zeta = math.exp(float(parameters[2 * index + 1]))
        low_frequency, high_frequency = bounded_frequency_pair(midpoint, ratios[index])
        schedules.append(
            PositionSchedule(
                actuator=item.actuator,
                joint=item.joint,
                position_min_rad=item.schedule_min_rad,
                position_max_rad=item.schedule_max_rad,
                natural_frequency_low_rad_s=low_frequency,
                natural_frequency_high_rad_s=high_frequency,
                damping_ratio=min(4.0, max(0.35, zeta)),
            )
        )
    return schedules


def rollout(
    model: mujoco.MjModel,
    key_id: int,
    driven: list[DrivenJoint],
    schedules: Iterable[PositionSchedule],
    target_joint_rad: np.ndarray,
    duration_s: float,
    settle_duration_s: float,
    reach_tolerance_rad: float,
    baseline_gainprm: np.ndarray,
    baseline_biasprm: np.ndarray,
    max_joint_velocity_rad_s: float,
    max_joint_acceleration_rad_s2: float,
) -> Rollout:
    """Move all driven joints to one target and score every driven coordinate."""
    model.actuator_gainprm[:] = baseline_gainprm
    model.actuator_biasprm[:] = baseline_biasprm
    controller = MasterController(schedules)
    controller.bind(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    home_ctrl = data.ctrl.copy()
    home_joint = np.array([home_ctrl[item.actuator_id] / item.gear for item in driven])
    trajectory_duration_s = max(
        trapezoidal_duration(
            target - home,
            max_velocity_rad_s=max_joint_velocity_rad_s,
            max_acceleration_rad_s2=max_joint_acceleration_rad_s2,
        )
        for target, home in zip(target_joint_rad, home_joint)
    )
    total_duration_s = max(duration_s, trajectory_duration_s + settle_duration_s)
    force_limits = np.array(
        [max(abs(value) for value in model.actuator_forcerange[item.actuator_id]) * abs(item.gear) for item in driven]
    )

    integrated_error = 0.0
    integrated_effort = 0.0
    peak_error = 0.0
    peak_contacts = 0
    saturated = 0
    samples = 0
    while data.time + EPSILON < total_duration_s:
        commanded_joint = np.array(
            [
                trapezoidal_position(
                    home,
                    target,
                    data.time,
                    max_velocity_rad_s=max_joint_velocity_rad_s,
                    max_acceleration_rad_s2=max_joint_acceleration_rad_s2,
                )
                for home, target in zip(home_joint, target_joint_rad)
            ]
        )
        data.ctrl[:] = home_ctrl
        for value, item in zip(commanded_joint, driven):
            data.ctrl[item.actuator_id] = item.gear * value
        mujoco.mj_forward(model, data)
        controller.update(model, data)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            return Rollout(False, 1e12, math.inf, math.inf, peak_contacts, 100.0)
        peak_contacts = max(peak_contacts, int(data.ncon))
        if data.ncon:
            # Any active contact is outside the collision-free training set.
            return Rollout(False, 1e9 + data.ncon, math.inf, math.inf, peak_contacts, 100.0)

        actual = np.array([data.qpos[item.qpos_address] for item in driven])
        error = actual - commanded_joint
        torque = np.array([data.actuator_force[item.actuator_id] * item.gear for item in driven])
        integrated_error += float(np.mean(error**2)) * model.opt.timestep
        integrated_effort += float(np.mean((torque / force_limits) ** 2)) * model.opt.timestep
        peak_error = max(peak_error, float(np.max(np.abs(error))))
        saturated += int(np.any(np.abs(torque) >= force_limits * 0.995))
        samples += 1

    final = np.array([data.qpos[item.qpos_address] for item in driven])
    final_error = float(np.max(np.abs(final - target_joint_rad)))
    safe = final_error <= reach_tolerance_rad
    rms_error = math.sqrt(integrated_error / max(total_duration_s, EPSILON))
    saturation_percent = 100.0 * saturated / max(samples, 1)
    cost = integrated_error + 0.003 * integrated_effort
    if not safe:
        cost += 100.0 * final_error
    return Rollout(safe, cost, rms_error, peak_error, peak_contacts, saturation_percent)


def select_diverse(targets: list[np.ndarray], home: np.ndarray, count: int, spans: np.ndarray) -> list[np.ndarray]:
    """Choose collision-free targets that cover the accepted workspace."""
    if len(targets) <= count:
        return targets
    normalized = [(target - home) / spans for target in targets]
    selected = [max(range(len(targets)), key=lambda index: float(np.linalg.norm(normalized[index])))]
    while len(selected) < count:
        choice = max(
            (index for index in range(len(targets)) if index not in selected),
            key=lambda index: min(float(np.linalg.norm(normalized[index] - normalized[other])) for other in selected),
        )
        selected.append(choice)
    return [targets[index] for index in selected]


def evaluate(
    model: mujoco.MjModel,
    key_id: int,
    driven: list[DrivenJoint],
    schedules: list[PositionSchedule],
    targets: list[np.ndarray],
    duration_s: float,
    settle_duration_s: float,
    reach_tolerance_rad: float,
    baseline_gainprm: np.ndarray,
    baseline_biasprm: np.ndarray,
    max_joint_velocity_rad_s: float,
    max_joint_acceleration_rad_s2: float,
) -> tuple[float, list[Rollout]]:
    reports = [
        rollout(
            model,
            key_id,
            driven,
            schedules,
            target,
            duration_s,
            settle_duration_s,
            reach_tolerance_rad,
            baseline_gainprm,
            baseline_biasprm,
            max_joint_velocity_rad_s,
            max_joint_acceleration_rad_s2,
        )
        for target in targets
    ]
    return float(sum(report.cost for report in reports) / len(reports)), reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train collision-screened multi-DOF PD schedules for robot_master.")
    parser.add_argument("--samples", type=int, default=24, help="Halton workspace samples in addition to 16 range corners (default: 24)")
    parser.add_argument("--training-cases", type=int, default=4, help="diverse collision-free multi-DOF trajectories used for gain training")
    parser.add_argument(
        "--screen-duration",
        type=float,
        default=1.5,
        help="minimum seconds for each collision-screening trajectory; longer moves extend automatically to respect the joint-speed limit",
    )
    parser.add_argument(
        "--trial-duration",
        type=float,
        default=1.5,
        help="minimum seconds for each gain-training trajectory; longer moves extend automatically to respect the joint-speed limit",
    )
    parser.add_argument("--settle-duration", type=float, default=0.5, help="time retained after each motion profile reaches its target")
    parser.add_argument(
        "--max-joint-velocity",
        type=float,
        default=TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S,
        help="per-joint trapezoid velocity limit in rad/s (default: configuration value)",
    )
    parser.add_argument(
        "--max-joint-acceleration",
        type=float,
        default=TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2,
        help="per-joint trapezoid acceleration limit in rad/s^2 (default: configuration value)",
    )
    parser.add_argument("--iterations", type=int, default=1, help="joint pattern-search passes over all frequency and damping parameters")
    parser.add_argument("--reach-tolerance", type=float, default=0.35, help="maximum final joint-coordinate error accepted as reachable (rad)")
    parser.add_argument("--schedule-in", type=Path, default=DEFAULT_SCHEDULE, help="previous schedule used as the optimization starting point")
    parser.add_argument("--output", type=Path, default=DEFAULT_SCHEDULE, help="where to write the trained schedule")
    arguments = parser.parse_args()
    for name in (
        "screen_duration",
        "trial_duration",
        "settle_duration",
        "reach_tolerance",
        "max_joint_velocity",
        "max_joint_acceleration",
    ):
        value = getattr(arguments, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and greater than zero")
    if arguments.samples < 0 or arguments.training_cases < 1 or arguments.iterations < 0:
        parser.error("--samples and --iterations must be zero or greater; --training-cases must be at least one")
    return arguments


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    key_id = keyframe_id(model)
    driven = driven_joints(model, key_id)
    baseline_gainprm = model.actuator_gainprm.copy()
    baseline_biasprm = model.actuator_biasprm.copy()
    initial_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, initial_data, key_id)
    mujoco.mj_forward(model, initial_data)
    parameters, ratios = load_initial_parameters(model, initial_data, driven, args.schedule_in)
    schedules = schedules_from_parameters(driven, parameters, ratios)

    command_ranges = np.array([[item.command_min_rad, item.command_max_rad] for item in driven])
    print("Collision-screening simultaneous targets over modelled command windows:")
    for item in driven:
        print(f"  {item.actuator}: {item.command_min_rad:.3f} .. {item.command_max_rad:.3f} rad")
    accepted: list[np.ndarray] = []
    rejected_collision = 0
    rejected_unreachable = 0
    for target in candidate_targets(driven, args.samples):
        report = rollout(
            model,
            key_id,
            driven,
            schedules,
            target,
            args.screen_duration,
            args.settle_duration,
            args.reach_tolerance,
            baseline_gainprm,
            baseline_biasprm,
            args.max_joint_velocity,
            args.max_joint_acceleration,
        )
        if report.safe:
            accepted.append(target)
        elif report.peak_contacts:
            rejected_collision += 1
        else:
            rejected_unreachable += 1
    if len(accepted) < args.training_cases:
        raise RuntimeError(
            f"only {len(accepted)} collision-free reachable multi-DOF poses were found; "
            "increase --screen-duration or --reach-tolerance, or review the modelled collision geometry"
        )
    home = np.array([model.key_ctrl[key_id, item.actuator_id] / item.gear for item in driven])
    selected = select_diverse(accepted, home, args.training_cases, command_ranges[:, 1] - command_ranges[:, 0])
    print(
        f"Accepted {len(accepted)} collision-free reachable targets; rejected "
        f"{rejected_collision} for collision and {rejected_unreachable} as unreachable. "
        f"Training on {len(selected)} diverse simultaneous-motion trajectories."
    )

    def score(candidate: np.ndarray) -> tuple[float, list[PositionSchedule]]:
        bounded = candidate.copy()
        for index in range(len(driven)):
            bounded[2 * index] = min(math.log(MAX_FREQUENCY_RAD_S), max(math.log(MIN_FREQUENCY_RAD_S), bounded[2 * index]))
            bounded[2 * index + 1] = min(math.log(4.0), max(math.log(0.35), bounded[2 * index + 1]))
        candidate_schedules = schedules_from_parameters(driven, bounded, ratios)
        value, _ = evaluate(
            model,
            key_id,
            driven,
            candidate_schedules,
            selected,
            args.trial_duration,
            args.settle_duration,
            args.reach_tolerance,
            baseline_gainprm,
            baseline_biasprm,
            args.max_joint_velocity,
            args.max_joint_acceleration,
        )
        return value, candidate_schedules

    best_cost, best_schedules = score(parameters)
    steps = np.tile(np.array([0.30, 0.25], dtype=np.float64), len(driven))
    for _ in range(args.iterations):
        best_parameters = parameters
        improved = False
        for dimension in range(len(parameters)):
            for direction in (-1.0, 1.0):
                candidate = parameters.copy()
                candidate[dimension] += direction * steps[dimension]
                candidate_cost, candidate_schedules = score(candidate)
                if candidate_cost + 1e-12 < best_cost:
                    best_cost = candidate_cost
                    best_schedules = candidate_schedules
                    best_parameters = candidate
                    improved = True
        parameters = best_parameters
        if not improved:
            steps *= 0.55

    final_cost, final_reports = evaluate(
        model,
        key_id,
        driven,
        best_schedules,
        selected,
        args.trial_duration,
        args.settle_duration,
        args.reach_tolerance,
        baseline_gainprm,
        baseline_biasprm,
        args.max_joint_velocity,
        args.max_joint_acceleration,
    )
    controller = MasterController(best_schedules)
    document = controller.to_document(
        model="robot_master",
        reference_keyframe=HARDWARE_ZERO_KEYFRAME_NAME,
        training={
            "mode": "multi_dof_collision_screened",
            "actuator_order": [item.actuator for item in driven],
            "modelled_command_ranges_joint_rad": command_ranges.tolist(),
            "screened_candidates": len(candidate_targets(driven, args.samples)),
            "collision_free_reachable_candidates": len(accepted),
            "rejected_for_collision": rejected_collision,
            "rejected_as_unreachable": rejected_unreachable,
            "safe_targets_joint_rad": [target.tolist() for target in selected],
            "screen_duration_s": args.screen_duration,
            "trial_duration_s": args.trial_duration,
            "settle_duration_s": args.settle_duration,
            "motion_profile": {
                "type": "trapezoidal",
                "max_joint_velocity_rad_s": args.max_joint_velocity,
                "max_joint_acceleration_rad_s2": args.max_joint_acceleration,
            },
            "reach_tolerance_rad": args.reach_tolerance,
            "pattern_search_iterations": args.iterations,
            "combined_validation_mean_cost": final_cost,
            "validation": [report.__dict__ for report in final_reports],
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(document, output_file, indent=2)
        output_file.write("\n")
    for schedule in best_schedules:
        print(
            f"{schedule.actuator}: omega_n {schedule.natural_frequency_low_rad_s:.2f} -> "
            f"{schedule.natural_frequency_high_rad_s:.2f} rad/s, zeta={schedule.damping_ratio:.2f}"
        )
    print(f"Multi-DOF validation cost: {final_cost:.5f}; wrote {args.output}")


if __name__ == "__main__":
    main()
