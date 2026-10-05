"""Open robot_master with MuJoCo's native Control sliders and scheduled PD gains.

Use the viewer's built-in **Control** panel to change ``data.ctrl``.  MuJoCo
invokes the registered control callback before every simulation step, so the
position-scheduled controller recalculates ``kp`` and ``kd`` from the live
joint state without taking control away from those sliders.
"""

import argparse
import math
from pathlib import Path

import mujoco
import mujoco.viewer

from master_controller import MasterController
from robot_master_configuration import HARDWARE_ZERO_KEYFRAME_NAME
from trapezoidal_motion import TrapezoidalReferenceLimiter


MODEL_PATH = Path(__file__).resolve().parent / "robot_master" / "mjcf" / "robot_master.xml"
SCHEDULE_PATH = Path(__file__).resolve().parent / "master_controller_schedule.json"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open robot_master with native Control sliders and scheduled PD gains.")
    parser.add_argument(
        "--gain-report-period",
        type=float,
        default=0.25,
        help="print live target, position, kp, kd, and omega_n every many simulation seconds; zero disables reports (default: 0.25)",
    )
    arguments = parser.parse_args()
    if not math.isfinite(arguments.gain_report_period) or arguments.gain_report_period < 0.0:
        parser.error("--gain-report-period must be finite and zero or greater")
    return arguments


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HARDWARE_ZERO_KEYFRAME_NAME)
    if keyframe_id < 0:
        raise RuntimeError(f"MJCF has no {HARDWARE_ZERO_KEYFRAME_NAME!r} keyframe; regenerate it first.")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    controller = MasterController.from_json(SCHEDULE_PATH)
    limiter = TrapezoidalReferenceLimiter(model, controller)
    limiter.reset(data)
    last_report_time = -math.inf

    def scheduled_control_callback(callback_model: mujoco.MjModel, callback_data: mujoco.MjData) -> None:
        """Update scheduled gains while preserving the viewer-owned ctrl values."""
        nonlocal last_report_time
        limiter.update(callback_data)
        updates = controller.update(callback_model, callback_data)
        if args.gain_report_period <= 0.0 or callback_data.time < last_report_time + args.gain_report_period:
            return
        last_report_time = float(callback_data.time)
        fields = []
        for actuator, update in updates.items():
            fields.append(
                f"{actuator}: request={limiter.requested_position(actuator):+.3f}, "
                f"ref={limiter.reference_position(actuator):+.3f}, q={update.position_rad:+.3f}, "
                f"ref_vel={limiter.reference_velocity(actuator):+.2f}, "
                f"kp={update.kp_output_nm_per_rad:.2f}/{update.kp_output_limit_nm_per_rad:.2f}, "
                f"kd={update.kd_output_nm_s_per_rad:.2f}/{update.kd_output_limit_nm_s_per_rad:.2f}, "
                f"omega={update.natural_frequency_rad_s:.2f}"
            )
        print(f"[scheduled gains t={callback_data.time:.2f}] " + " | ".join(fields))

    print("MuJoCo viewer starting at hardware_zero with position-scheduled PD gains.")
    print("Use its native Control panel sliders to set joint targets; every joint ramps at <= 1 rad/s.")
    print("Live requested/ref/actual positions and applied gains print below.")
    mujoco.set_mjcb_control(scheduled_control_callback)
    try:
        mujoco.viewer.launch(model, data)
    finally:
        # The callback is process-global in MuJoCo's Python binding. Do not leave a
        # closure referring to this viewer's controller after its window closes.
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
