r"""Train position-scheduled PD pole targets in the robot_master simulation.

The optimizer is intentionally dependency-free and deterministic.  It uses a
pattern search over each actuator's low-position natural frequency,
high-position natural frequency, and damping ratio, with all four motors
holding the shared ``hardware_zero`` pose between trials.

Example (from the repository root):

    & .\Robosim\.venv\Scripts\python.exe .\Robosim\train_master_controller.py

The result is JSON that :class:`master_controller.MasterController` can load.
It is a simulation result, not a hardware-safe motor configuration.
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

from master_controller import (
    MasterController,
    PositionSchedule,
    actuator_gear,
    actuator_name,
    joint_name,
    output_pd_gains,
)
from robot_master_configuration import HARDWARE_ZERO_KEYFRAME_NAME


MODEL_PATH = Path(__file__).resolve().parent / "robot_master" / "mjcf" / "robot_master.xml"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "master_controller_schedule.json"
EPSILON = 1e-9


@dataclass(frozen=True)
class TrainingCase:
    actuator_id: int
    target_joint_rad: float
    position_min_rad: float
    position_max_rad: float


@dataclass(frozen=True)
class RolloutScore:
    cost: float
    rms_error_rad: float
    peak_error_rad: float
    peak_output_torque_nm: float
    saturation_percent: float
    peak_contacts: int


def resolve_actuators(model: mujoco.MjModel, requested: str) -> list[int]:
    """Resolve ``all`` or a comma-separated list of actuator names or ids."""
    if requested.strip().lower() == "all":
        return list(range(model.nu))
    resolved: list[int] = []
    for token in (item.strip() for item in requested.split(",")):
        if not token:
            continue
        if token.isdecimal():
            actuator_id = int(token)
        else:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, token)
        if not 0 <= actuator_id < model.nu:
            choices = ", ".join(actuator_name(model, index) for index in range(model.nu))
            raise ValueError(f"unknown actuator {token!r}; choose one of {choices}")
        if actuator_id not in resolved:
            resolved.append(actuator_id)
    if not resolved:
        raise ValueError("--actuators must name at least one actuator")
    return resolved


def hardware_zero_id(model: mujoco.MjModel) -> int:
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HARDWARE_ZERO_KEYFRAME_NAME)
    if keyframe_id < 0:
        raise RuntimeError(f"MJCF has no {HARDWARE_ZERO_KEYFRAME_NAME!r} keyframe; regenerate it first")
    return keyframe_id


def joint_target_bounds(model: mujoco.MjModel, actuator_id: int, home_joint_rad: float, motion_rad: float) -> tuple[float, float]:
    """Return the sampled target interval, respecting explicit actuator limits."""
    gear = actuator_gear(model, actuator_id)
    if model.actuator_ctrllimited[actuator_id]:
        control_low, control_high = model.actuator_ctrlrange[actuator_id]
        low, high = sorted((float(control_low / gear), float(control_high / gear)))
    else:
        low, high = home_joint_rad - motion_rad, home_joint_rad + motion_rad
    low = max(low, home_joint_rad - motion_rad)
    high = min(high, home_joint_rad + motion_rad)
    if high - low < 0.01:
        raise ValueError(
            f"{actuator_name(model, actuator_id)} has less than 0.01 rad of usable target travel around its home reference"
        )
    return low, high


def make_cases(model: mujoco.MjModel, keyframe_id: int, actuator_ids: Iterable[int], motion_rad: float) -> list[TrainingCase]:
    """Create a negative and positive step case per actuator where available."""
    cases: list[TrainingCase] = []
    home_ctrl = model.key_ctrl[keyframe_id]
    for actuator_id in actuator_ids:
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_address = int(model.jnt_qposadr[joint_id])
        gear = actuator_gear(model, actuator_id)
        home_command_joint = float(home_ctrl[actuator_id] / gear)
        low, high = joint_target_bounds(model, actuator_id, home_command_joint, motion_rad)

        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        mujoco.mj_forward(model, data)
        home_actual_joint = float(data.qpos[qpos_address])
        position_min = min(low, home_actual_joint)
        position_max = max(high, home_actual_joint)
        for target in (low, high):
            if abs(target - home_actual_joint) >= 0.01:
                cases.append(TrainingCase(actuator_id, target, position_min, position_max))
    if not cases:
        raise RuntimeError("no usable training motions were generated")
    return cases


def baseline_pole_parameters(model: mujoco.MjModel, keyframe_id: int, actuator_id: int) -> tuple[float, float]:
    """Estimate an initial omega_n and zeta from the currently loaded PD gains."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    matrix = np.empty((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, matrix)
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    inertia = max(1e-6, float(matrix[model.jnt_dofadr[joint_id], model.jnt_dofadr[joint_id]]))
    kp, kd = output_pd_gains(model, actuator_id)
    omega = min(45.0, max(2.0, math.sqrt(max(kp, 1e-6) / inertia)))
    zeta = min(3.0, max(0.4, kd / (2.0 * inertia * omega)))
    return omega, zeta


def schedule_from_parameters(
    model: mujoco.MjModel,
    actuator_id: int,
    position_min: float,
    position_max: float,
    log_mid_frequency: float,
    log_frequency_ratio: float,
    log_damping_ratio: float,
) -> PositionSchedule:
    """Build a schedule from bounded search coordinates."""
    ratio = math.exp(log_frequency_ratio)
    ratio_root = math.sqrt(ratio)
    # Keep both endpoints inside the allowed frequency band *without* clipping
    # either endpoint individually.  Individual clipping would silently weaken
    # --minimum-frequency-ratio whenever the optimizer reaches 60 rad/s.
    minimum_midpoint = max(ratio_root, 1.0 / ratio_root)
    maximum_midpoint = 60.0 * min(ratio_root, 1.0 / ratio_root)
    midpoint = min(maximum_midpoint, max(minimum_midpoint, math.exp(log_mid_frequency)))
    low_frequency = midpoint / ratio_root
    high_frequency = midpoint * ratio_root
    damping_ratio = min(4.0, max(0.35, math.exp(log_damping_ratio)))
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    return PositionSchedule(
        actuator=actuator_name(model, actuator_id),
        joint=joint_name(model, joint_id),
        position_min_rad=position_min,
        position_max_rad=position_max,
        natural_frequency_low_rad_s=low_frequency,
        natural_frequency_high_rad_s=high_frequency,
        damping_ratio=damping_ratio,
    )


def rollout(
    model: mujoco.MjModel,
    keyframe_id: int,
    schedules: Iterable[PositionSchedule],
    case: TrainingCase,
    duration_s: float,
    effort_weight: float,
    collision_weight: float,
    baseline_gainprm: np.ndarray,
    baseline_biasprm: np.ndarray,
) -> RolloutScore:
    """Score one closed-loop step motion from the fixed hardware-zero state."""
    model.actuator_gainprm[:] = baseline_gainprm
    model.actuator_biasprm[:] = baseline_biasprm
    controller = MasterController(schedules)
    controller.bind(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    home_ctrl = data.ctrl.copy()

    actuator_id = case.actuator_id
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    qpos_address = int(model.jnt_qposadr[joint_id])
    gear = actuator_gear(model, actuator_id)
    force_limit = max(abs(value) for value in model.actuator_forcerange[actuator_id]) * abs(gear)
    target_control = case.target_joint_rad * gear

    integrated_error_squared = 0.0
    integrated_effort = 0.0
    peak_error = 0.0
    peak_torque = 0.0
    saturated_samples = 0
    peak_contacts = 0
    samples = 0
    while data.time + EPSILON < duration_s:
        data.ctrl[:] = home_ctrl
        data.ctrl[actuator_id] = target_control
        mujoco.mj_forward(model, data)
        controller.update(model, data)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            return RolloutScore(1e12, math.inf, math.inf, math.inf, 100.0, int(data.ncon))

        error = float(data.qpos[qpos_address] - case.target_joint_rad)
        torque = float(data.actuator_force[actuator_id] * gear)
        normalized_effort = torque / max(force_limit, EPSILON)
        integrated_error_squared += error**2 * model.opt.timestep
        integrated_effort += normalized_effort**2 * model.opt.timestep
        peak_error = max(peak_error, abs(error))
        peak_torque = max(peak_torque, abs(torque))
        saturated_samples += int(abs(torque) >= force_limit * 0.995)
        peak_contacts = max(peak_contacts, int(data.ncon))
        samples += 1

    rms_error = math.sqrt(integrated_error_squared / max(duration_s, EPSILON))
    saturation_percent = 100.0 * saturated_samples / max(samples, 1)
    cost = integrated_error_squared + effort_weight * integrated_effort + collision_weight * peak_contacts
    return RolloutScore(cost, rms_error, peak_error, peak_torque, saturation_percent, peak_contacts)


def evaluate(
    model: mujoco.MjModel,
    keyframe_id: int,
    schedules: Iterable[PositionSchedule],
    cases: Iterable[TrainingCase],
    duration_s: float,
    effort_weight: float,
    collision_weight: float,
    baseline_gainprm: np.ndarray,
    baseline_biasprm: np.ndarray,
) -> tuple[float, list[RolloutScore]]:
    """Return mean cost and raw scores for a group of deterministic rollouts."""
    scores = [
        rollout(
            model,
            keyframe_id,
            schedules,
            case,
            duration_s,
            effort_weight,
            collision_weight,
            baseline_gainprm,
            baseline_biasprm,
        )
        for case in cases
    ]
    return sum(score.cost for score in scores) / len(scores), scores


def train_one_actuator(
    model: mujoco.MjModel,
    keyframe_id: int,
    actuator_id: int,
    cases: list[TrainingCase],
    duration_s: float,
    iterations: int,
    minimum_frequency_ratio: float,
    effort_weight: float,
    collision_weight: float,
    baseline_gainprm: np.ndarray,
    baseline_biasprm: np.ndarray,
) -> tuple[PositionSchedule, float]:
    """Train one smooth omega_n(q), zeta schedule by deterministic pattern search."""
    relevant_cases = [case for case in cases if case.actuator_id == actuator_id]
    position_min = min(case.position_min_rad for case in relevant_cases)
    position_max = max(case.position_max_rad for case in relevant_cases)
    base_frequency, base_zeta = baseline_pole_parameters(model, keyframe_id, actuator_id)
    minimum_log_ratio = math.log(minimum_frequency_ratio)
    parameters = np.array([math.log(base_frequency), 0.0, math.log(base_zeta)], dtype=np.float64)
    steps = np.array([0.45, 0.60, 0.35], dtype=np.float64)

    def score(candidate: np.ndarray, slope_sign: float) -> tuple[float, PositionSchedule, np.ndarray]:
        bounded = candidate.copy()
        bounded[0] = min(math.log(60.0), max(math.log(1.0), bounded[0]))
        if minimum_log_ratio > EPSILON:
            # Keep a non-zero position dependence.  The optimizer chooses the
            # sign below, then searches the magnitude from this safe minimum
            # up to a 6:1 endpoint ratio.
            bounded[1] = slope_sign * min(math.log(6.0), max(minimum_log_ratio, abs(bounded[1])))
        else:
            bounded[1] = min(math.log(6.0), max(math.log(1.0 / 6.0), bounded[1]))
        bounded[2] = min(math.log(4.0), max(math.log(0.35), bounded[2]))
        schedule = schedule_from_parameters(model, actuator_id, position_min, position_max, *bounded)
        value, _ = evaluate(
            model,
            keyframe_id,
            [schedule],
            relevant_cases,
            duration_s,
            effort_weight,
            collision_weight,
            baseline_gainprm,
            baseline_biasprm,
        )
        return value, schedule, bounded

    # A static schedule is mathematically optimal for a uniform tracking cost
    # on a near-linear range.  When scheduling is requested, evaluate both
    # possible directions at the minimum variation and retain the one the
    # simulated mechanism prefers.
    candidate_signs = (1.0, -1.0) if minimum_log_ratio > EPSILON else (1.0,)
    best_cost = math.inf
    best_schedule: PositionSchedule | None = None
    slope_sign = 1.0
    for candidate_sign in candidate_signs:
        initial = parameters.copy()
        initial[1] = candidate_sign * minimum_log_ratio
        candidate_cost, candidate_schedule, bounded = score(initial, candidate_sign)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_schedule = candidate_schedule
            parameters = bounded
            slope_sign = candidate_sign
    assert best_schedule is not None
    for _ in range(iterations):
        best_candidate = parameters
        improved = False
        for dimension in range(len(parameters)):
            for direction in (-1.0, 1.0):
                candidate = parameters.copy()
                candidate[dimension] += direction * steps[dimension]
                candidate_cost, candidate_schedule, bounded = score(candidate, slope_sign)
                if candidate_cost + 1e-12 < best_cost:
                    best_cost = candidate_cost
                    best_candidate = bounded
                    best_schedule = candidate_schedule
                    improved = True
        parameters = best_candidate
        if not improved:
            steps *= 0.55
    return best_schedule, best_cost


def format_pole(pole: complex) -> str:
    """Compact display for a continuous-time real or complex pole."""
    if abs(pole.imag) < 1e-8:
        return f"{pole.real:.2f}"
    return f"{pole.real:.2f}{pole.imag:+.2f}j"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train position-scheduled PD gains for robot_master in MuJoCo.")
    parser.add_argument("--actuators", default="all", help="'all' or a comma-separated actuator-name/id list (default: all)")
    parser.add_argument("--motion", type=float, default=0.15, help="maximum training step away from home, in joint radians")
    parser.add_argument("--duration", type=float, default=1.5, help="duration of each simulated step trial in seconds")
    parser.add_argument("--iterations", type=int, default=4, help="pattern-search iterations per actuator (default: 4)")
    parser.add_argument(
        "--minimum-frequency-ratio",
        type=float,
        default=1.25,
        help="minimum high/low or low/high omega_n ratio across the trained range; use 1 to allow static poles (default: 1.25)",
    )
    parser.add_argument("--effort-weight", type=float, default=0.003, help="cost weight on squared normalized torque")
    parser.add_argument("--collision-weight", type=float, default=0.10, help="cost penalty for each simultaneous collision")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"schedule JSON output (default: {DEFAULT_OUTPUT})")
    arguments = parser.parse_args()
    if not math.isfinite(arguments.motion) or arguments.motion <= 0.0:
        parser.error("--motion must be finite and greater than zero")
    if not math.isfinite(arguments.duration) or arguments.duration <= 0.0:
        parser.error("--duration must be finite and greater than zero")
    if arguments.iterations < 0:
        parser.error("--iterations must be zero or greater")
    if not math.isfinite(arguments.minimum_frequency_ratio) or arguments.minimum_frequency_ratio < 1.0:
        parser.error("--minimum-frequency-ratio must be finite and at least one")
    if not math.isfinite(arguments.effort_weight) or arguments.effort_weight < 0.0:
        parser.error("--effort-weight must be finite and zero or greater")
    if not math.isfinite(arguments.collision_weight) or arguments.collision_weight < 0.0:
        parser.error("--collision-weight must be finite and zero or greater")
    return arguments


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    keyframe_id = hardware_zero_id(model)
    actuator_ids = resolve_actuators(model, args.actuators)
    cases = make_cases(model, keyframe_id, actuator_ids, args.motion)
    baseline_gainprm = model.actuator_gainprm.copy()
    baseline_biasprm = model.actuator_biasprm.copy()

    print(
        f"Training {len(actuator_ids)} actuator schedule(s) with {len(cases)} step cases "
        f"from {HARDWARE_ZERO_KEYFRAME_NAME!r}."
    )
    schedules: list[PositionSchedule] = []
    individual_costs: dict[str, float] = {}
    for actuator_id in actuator_ids:
        schedule, cost = train_one_actuator(
            model,
            keyframe_id,
            actuator_id,
            cases,
            args.duration,
            args.iterations,
            args.minimum_frequency_ratio,
            args.effort_weight,
            args.collision_weight,
            baseline_gainprm,
            baseline_biasprm,
        )
        schedules.append(schedule)
        individual_costs[schedule.actuator] = cost
        low_pole, _ = schedule.desired_poles(schedule.position_min_rad)
        high_pole, _ = schedule.desired_poles(schedule.position_max_rad)
        print(
            f"{schedule.actuator}: omega_n {schedule.natural_frequency_low_rad_s:.2f} -> "
            f"{schedule.natural_frequency_high_rad_s:.2f} rad/s, zeta={schedule.damping_ratio:.2f}, "
            f"dominant pole {format_pole(low_pole)} -> {format_pole(high_pole)} rad/s, cost={cost:.5f}"
        )

    combined_cost, validation = evaluate(
        model,
        keyframe_id,
        schedules,
        cases,
        args.duration,
        args.effort_weight,
        args.collision_weight,
        baseline_gainprm,
        baseline_biasprm,
    )
    controller = MasterController(schedules)
    document = controller.to_document(
        model="robot_master",
        reference_keyframe=HARDWARE_ZERO_KEYFRAME_NAME,
        training={
            "motion_rad": args.motion,
            "trial_duration_s": args.duration,
            "pattern_search_iterations": args.iterations,
            "minimum_frequency_ratio": args.minimum_frequency_ratio,
            "individual_mean_cost": individual_costs,
            "combined_validation_mean_cost": combined_cost,
            "validation": [score.__dict__ for score in validation],
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(document, output_file, indent=2)
        output_file.write("\n")
    print(f"Combined scheduled-controller validation mean cost: {combined_cost:.5f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
