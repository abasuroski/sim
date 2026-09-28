r"""Make the Unitree A1 perform a simple diagonal trot for up to 3 metres.

Run with the 3D viewer:
    .\.venv\Scripts\python.exe .\a1_walk_3m.py

For a quick non-graphical check:
    .\.venv\Scripts\python.exe .\a1_walk_3m.py --headless

This is an educational open-loop gait, not a production locomotion controller.
"""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


SCENE = Path(__file__).parent / "mujoco_menagerie" / "unitree_a1" / "scene.xml"
TARGET_DISTANCE = 3.0
HOME = np.array([0.0, 0.9, -1.8])
# FR and RL form one diagonal pair; FL and RR form the other.
LEG_PHASES = np.array([0.0, np.pi, np.pi, 0.0])


def trot_targets(sim_time: float) -> np.ndarray:
    """Return 12 position-actuator targets for a slow, simple diagonal trot."""
    targets = np.tile(HOME, 4)
    phases = 2.0 * np.pi * 0.8 * sim_time + LEG_PHASES
    stride = np.sin(phases)

    for leg, value in enumerate(stride):
        actuator = leg * 3
        # The hip sweep cycles each foot; bending the knee during its swing
        # phase lifts it clear of the ground before the next planted stride.
        targets[actuator + 1] += 0.22 * value
        targets[actuator + 2] += 0.35 * max(-value, 0.0)

    return targets


def run(model: mujoco.MjModel, data: mujoco.MjData, show_viewer: bool) -> float:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    start_x = float(data.qpos[0])

    def step() -> bool:
        data.ctrl[:] = trot_targets(data.time)
        mujoco.mj_step(model, data)
        return abs(float(data.qpos[0]) - start_x) < TARGET_DISTANCE and data.time < 45.0

    if not show_viewer:
        while step():
            pass
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and step():
                viewer.sync()
                # Keep the viewer near real time while the physics uses 2 ms steps.
                time.sleep(model.opt.timestep)

    return abs(float(data.qpos[0]) - start_x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without opening the 3D viewer")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    if not args.headless:
        print("A1 walking demo open. Close the viewer window to stop early.")
    distance = run(model, data, show_viewer=not args.headless)
    print(f"Finished after {data.time:.2f} s; distance travelled: {distance:.2f} m")


if __name__ == "__main__":
    main()
