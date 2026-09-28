"""Open the converted robot_master MJCF in MuJoCo's interactive viewer."""

from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).resolve().parent / "robot_master" / "mjcf" / "robot_master.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
mujoco.viewer.launch(model, data)
