r"""Launch the Unitree A1 quadruped in MuJoCo's interactive 3D viewer.

Run from PowerShell:
    .\.venv\Scripts\python.exe .\unitree_a1_demo.py

The robot begins in the Menagerie "home" standing pose. Its built-in position
actuators hold that pose while you inspect it in the viewer.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer


SCENE = Path(__file__).parent / "mujoco_menagerie" / "unitree_a1" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    # Keyframe 0 supplies both the free-base standing pose and joint targets.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    standing_targets = model.key_ctrl[0].copy()

    print("Unitree A1 viewer open. Close its window to quit.")
    print("Use left-drag to rotate, scroll to zoom, and right-drag to pan.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()

            # Advance several 2 ms physics steps per rendered frame. The position
            # actuators use the standing targets from the model's home keyframe.
            for _ in range(4):
                data.ctrl[:] = standing_targets
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(max(0, 0.008 - (time.perf_counter() - frame_start)))


if __name__ == "__main__":
    main()
