r"""A minimal, headless MuJoCo simulation.

Run from PowerShell:
    .\.venv\Scripts\python.exe .\falling_ball.py
"""

import time

import mujoco
import mujoco.viewer


XML = """
<mujoco model="falling_ball">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.2 0.3 0.2 1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="0.5" rgba="0.9 0.2 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    # Compile the XML description into a model, then allocate its mutable state.
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # The viewer stays open until you close its window. `sync()` both redraws it
    # and applies any UI changes (for example, pausing the simulation).
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
