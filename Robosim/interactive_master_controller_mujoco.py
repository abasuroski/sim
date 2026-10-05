"""Interactively test the trained master controller with joint-space sliders.

The Tk panel owns the four desired joint positions; MuJoCo's passive viewer
shows the arm.  Every physics step applies the position-scheduled gains from
``master_controller_schedule.json`` before stepping the model.

This is strictly a simulation test tool.  It latches and rolls back on a
MuJoCo collision contact, but that is not a substitute for hardware limits or
an emergency stop on the physical arm.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
import tkinter as tk

import mujoco
import mujoco.viewer
import numpy as np

from master_controller import MasterController, actuator_gear
from robot_master_configuration import HARDWARE_ZERO_KEYFRAME_NAME
from trapezoidal_motion import TrapezoidalReferenceLimiter


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "robot_master" / "mjcf" / "robot_master.xml"
DEFAULT_SCHEDULE = ROOT / "master_controller_schedule.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the trained master controller with interactive joint sliders.")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE, help="trained schedule JSON path")
    return parser.parse_args()


def output_target_bounds(model: mujoco.MjModel, actuator_id: int) -> tuple[float, float]:
    """Return the finite joint-output target window exposed by this MJCF."""
    gear = actuator_gear(model, actuator_id)
    low, high = (float(value / gear) for value in model.actuator_ctrlrange[actuator_id])
    return min(low, high), max(low, high)


class InteractiveControllerTest:
    """Keep a Tk slider panel and MuJoCo passive viewer synchronized."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, controller: MasterController, keyframe_id: int) -> None:
        self.model = model
        self.data = data
        self.controller = controller
        self.keyframe_id = keyframe_id
        self.root = tk.Tk()
        self.root.title("robot_master scheduled PD test")
        self.root.resizable(False, False)
        self.running = True
        self.paused = False
        self.reset_requested = False
        self.status = tk.StringVar(value="Running — move a slider to command its joint target.")
        self.targets: dict[str, tuple[int, float, tk.DoubleVar, tk.StringVar]] = {}
        self.limiter = TrapezoidalReferenceLimiter(model, controller)
        self._reset_to_hardware_zero()
        self._build_panel()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _reset_to_hardware_zero(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        mujoco.mj_forward(self.model, self.data)
        self.limiter.reset(self.data)
        self.last_safe_qpos = self.data.qpos.copy()
        self.last_safe_qvel = self.data.qvel.copy()
        self.last_safe_ctrl = self.data.ctrl.copy()
        self.paused = False

    def _build_panel(self) -> None:
        tk.Label(self.root, text="Joint targets (rad)", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=3, padx=10, pady=(10, 4), sticky="w"
        )
        tk.Label(
            self.root,
            text="The range is the modelled control window. Commands use a 1 rad/s trapezoid. Collision contact pauses and rolls back the simulation.",
            wraplength=560,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, padx=10, pady=(0, 8), sticky="w")

        for row, schedule in enumerate(self.controller.schedules, start=2):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, schedule.actuator)
            if actuator_id < 0:
                raise RuntimeError(f"scheduled actuator {schedule.actuator!r} is absent from this MJCF")
            gear = actuator_gear(self.model, actuator_id)
            low, high = output_target_bounds(self.model, actuator_id)
            initial = float(self.data.ctrl[actuator_id] / gear)
            target = tk.DoubleVar(value=initial)
            readout = tk.StringVar(value="")
            tk.Label(self.root, text=schedule.actuator, width=20, anchor="w").grid(row=row, column=0, padx=(10, 2), pady=3)
            tk.Scale(
                self.root,
                variable=target,
                from_=low,
                to=high,
                resolution=0.001,
                orient=tk.HORIZONTAL,
                length=330,
                showvalue=True,
            ).grid(row=row, column=1, padx=2, pady=3)
            tk.Label(self.root, textvariable=readout, width=31, anchor="w", justify="left").grid(
                row=row, column=2, padx=(2, 10), pady=3
            )
            self.targets[schedule.actuator] = (actuator_id, gear, target, readout)

        button_row = len(self.controller.schedules) + 2
        tk.Button(self.root, text="Reset to hardware zero", command=self._request_reset).grid(
            row=button_row, column=0, padx=10, pady=(10, 5), sticky="w"
        )
        self.pause_button = tk.Button(self.root, text="Pause", command=self._toggle_pause)
        self.pause_button.grid(row=button_row, column=1, padx=2, pady=(10, 5), sticky="w")
        tk.Label(self.root, textvariable=self.status, fg="#174ea6", wraplength=560, justify="left").grid(
            row=button_row + 1, column=0, columnspan=3, padx=10, pady=(2, 10), sticky="w"
        )

    def _close(self) -> None:
        self.running = False

    def _request_reset(self) -> None:
        self.reset_requested = True

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        self.status.set("Paused — adjust sliders, then Resume." if self.paused else "Running.")

    def _restore_last_safe_state(self) -> None:
        self.data.qpos[:] = self.last_safe_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.last_safe_ctrl
        mujoco.mj_forward(self.model, self.data)

    def _update_readouts(self, updates: dict) -> None:
        for actuator, update in updates.items():
            _, _, target, readout = self.targets[actuator]
            readout.set(
                f"q={update.position_rad:+.3f}; request={target.get():+.3f}; ref={self.limiter.reference_position(actuator):+.3f}\n"
                f"kp={update.kp_output_nm_per_rad:.3f}; kd={update.kd_output_nm_s_per_rad:.3f}"
            )

    def run(self) -> None:
        """Run both windows on the main thread until either one is closed."""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            wall_start = time.perf_counter()
            last_ui_update = -math.inf
            while self.running and viewer.is_running():
                try:
                    self.root.update_idletasks()
                    self.root.update()
                except tk.TclError:
                    break

                if self.reset_requested:
                    self._reset_to_hardware_zero()
                    for actuator, (actuator_id, gear, target, _) in self.targets.items():
                        target.set(float(self.data.ctrl[actuator_id] / gear))
                    self.reset_requested = False
                    self.status.set("Reset to hardware_zero.")

                if not self.paused:
                    for actuator, (_, _, target, _) in self.targets.items():
                        self.limiter.set_requested_position(actuator, target.get())
                    self.limiter.advance(self.data)
                    mujoco.mj_forward(self.model, self.data)
                    updates = self.controller.update(self.model, self.data)
                    self.last_safe_qpos[:] = self.data.qpos
                    self.last_safe_qvel[:] = self.data.qvel
                    self.last_safe_ctrl[:] = self.data.ctrl
                    mujoco.mj_step(self.model, self.data)
                    if self.data.ncon:
                        self._restore_last_safe_state()
                        self.limiter.reset(self.data)
                        self.paused = True
                        self.pause_button.configure(text="Resume")
                        self.status.set(
                            f"Collision contact detected — rolled back and paused. Reset or move targets, then Resume."
                        )
                else:
                    mujoco.mj_forward(self.model, self.data)
                    updates = self.controller.update(self.model, self.data)

                if time.perf_counter() - last_ui_update >= 0.05:
                    self._update_readouts(updates)
                    last_ui_update = time.perf_counter()
                viewer.sync()
                remaining = wall_start + self.data.time - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                elif self.paused:
                    time.sleep(0.01)
        self.root.destroy()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    controller = MasterController.from_json(args.schedule)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HARDWARE_ZERO_KEYFRAME_NAME)
    if keyframe_id < 0:
        raise RuntimeError(f"MJCF has no {HARDWARE_ZERO_KEYFRAME_NAME!r} keyframe; regenerate it first")
    InteractiveControllerTest(model, data, controller, keyframe_id).run()


if __name__ == "__main__":
    main()
