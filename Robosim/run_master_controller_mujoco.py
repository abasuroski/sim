"""Show the trained position-scheduled PD controller in the MuJoCo viewer.

Each motor moves through its trained joint-position interval in turn while the
other motors hold the hardware-zero controls.  This keeps the four-bar linkage
easy to inspect and makes the changing scheduled gains visible in a safe,
repeatable simulation-only demonstration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer

from master_controller import MasterController, actuator_gear
from robot_master_configuration import HARDWARE_ZERO_KEYFRAME_NAME
from trapezoidal_motion import TrapezoidalReferenceLimiter


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "robot_master" / "mjcf" / "robot_master.xml"
SCHEDULE_PATH = ROOT / "master_controller_schedule.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained master-controller demonstration in MuJoCo.")
    parser.add_argument(
        "--seconds-per-joint",
        type=float,
        default=4.0,
        help="time for one complete smooth sweep of one actuator (default: 4)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after this many simulation seconds; zero repeats until the viewer is closed (default: 0)",
    )
    parser.add_argument("--schedule", type=Path, default=SCHEDULE_PATH, help="trained schedule JSON path")
    arguments = parser.parse_args()
    if not math.isfinite(arguments.seconds_per_joint) or arguments.seconds_per_joint <= 0.0:
        parser.error("--seconds-per-joint must be finite and greater than zero")
    if not math.isfinite(arguments.duration) or arguments.duration < 0.0:
        parser.error("--duration must be finite and zero or greater")
    return arguments


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    controller = MasterController.from_json(args.schedule)
    with args.schedule.open(encoding="utf-8") as schedule_file:
        schedule_document = json.load(schedule_file)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HARDWARE_ZERO_KEYFRAME_NAME)
    if keyframe_id < 0:
        raise RuntimeError(f"MJCF has no {HARDWARE_ZERO_KEYFRAME_NAME!r} keyframe; regenerate it first")

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    home_controls = data.ctrl.copy()
    limiter = TrapezoidalReferenceLimiter(model, controller)
    limiter.reset(data)
    scheduled: list[tuple[str, int, float, float, float]] = []
    for schedule in controller.schedules:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, schedule.actuator)
        if actuator_id < 0:
            raise RuntimeError(f"scheduled actuator {schedule.actuator!r} is absent from this MJCF")
        center = 0.5 * (schedule.position_min_rad + schedule.position_max_rad)
        amplitude = 0.5 * (schedule.position_max_rad - schedule.position_min_rad)
        scheduled.append((schedule.actuator, actuator_id, center, amplitude, actuator_gear(model, actuator_id)))

    mujoco.mj_forward(model, data)
    safe_targets = schedule_document.get("training", {}).get("safe_targets_joint_rad")
    use_multidof_poses = (
        isinstance(safe_targets, list)
        and bool(safe_targets)
        and all(isinstance(target, list) and len(target) == len(scheduled) for target in safe_targets)
    )
    print(f"Loaded {args.schedule.name} and reset to {HARDWARE_ZERO_KEYFRAME_NAME!r}.")
    if use_multidof_poses:
        print(f"Cycling {len(safe_targets)} collision-screened multi-joint trajectories. Close the viewer to stop.")
    else:
        print("Each actuator sweeps alone through its trained range. Close the MuJoCo viewer to stop.")
        print("Order: " + ", ".join(name for name, *_ in scheduled))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()
        while viewer.is_running() and (args.duration <= 0.0 or data.time < args.duration):
            segment = int(data.time / args.seconds_per_joint)
            phase = math.tau * (data.time % args.seconds_per_joint) / args.seconds_per_joint
            data.ctrl[:] = home_controls
            if use_multidof_poses:
                # Smoothly depart from and return to hardware_zero for every
                # screened target.  This reproduces the same all-joints-active
                # path that the multi-DOF trainer accepted without contacts.
                target = safe_targets[segment % len(safe_targets)]
                half_phase = 2.0 * (data.time % args.seconds_per_joint) / args.seconds_per_joint
                if half_phase > 1.0:
                    half_phase = 2.0 - half_phase
                blend = half_phase * half_phase * (3.0 - 2.0 * half_phase)
                for value, (_, actuator_id, _, _, gear) in zip(target, scheduled):
                    data.ctrl[actuator_id] = home_controls[actuator_id] + blend * (gear * value - home_controls[actuator_id])
            else:
                # Legacy single-joint visualization for schedules trained by
                # train_master_controller.py.
                _, actuator_id, center, amplitude, gear = scheduled[segment % len(scheduled)]
                data.ctrl[actuator_id] = gear * (center + amplitude * math.sin(phase))
            # Keep visual demonstrations consistent with the live control
            # panel and the multi-DOF trainer: no joint reference may exceed
            # the configured trapezoidal velocity/acceleration limits.
            limiter.update(data)
            mujoco.mj_forward(model, data)
            controller.update(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()

            # Viewer interaction stays responsive while simulation advances at
            # real time, independently of rendering performance.
            remaining = wall_start + data.time - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
