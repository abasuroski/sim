"""Open the converted robot_master MJCF in MuJoCo's interactive viewer."""

from pathlib import Path

import mujoco
import mujoco.viewer

from robot_master_configuration import HARDWARE_ZERO_KEYFRAME_NAME


MODEL_PATH = Path(__file__).resolve().parent / "robot_master" / "mjcf" / "robot_master.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HARDWARE_ZERO_KEYFRAME_NAME)
if keyframe_id < 0:
    raise RuntimeError(f"MJCF has no {HARDWARE_ZERO_KEYFRAME_NAME!r} keyframe; regenerate it first.")
mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
mujoco.mj_forward(model, data)
mujoco.viewer.launch(model, data)
