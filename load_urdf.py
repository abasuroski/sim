import mujoco
import mujoco.viewer
import sys
import time

URDF_PATH = "/media/sf_CSC_FOLDER/robot_master/urdf/robot_master.urdf"

if len(sys.argv) > 1:
    URDF_PATH = sys.argv[1]

print(f"Loading: {URDF_PATH}")

spec = mujoco.MjSpec.from_file(URDF_PATH)
for mesh in spec.meshes:
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
model = spec.compile()
data  = mujoco.MjData(model)

print(f"Model loaded — {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0, 0, 0.3]
    viewer.cam.distance   = 1.5
    viewer.cam.azimuth    = 90
    viewer.cam.elevation  = -20

    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))
