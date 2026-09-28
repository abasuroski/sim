# robot_master MuJoCo simulation

`Robosim` is the active MuJoCo project.  It contains the CAD-derived
`robot_master` model, a converter from the source URDF/GLTF assets, a simple
viewer, and a reproducible single-joint controller-tuning test.

## Start here

MuJoCo 3.12.0 is already installed in `Robosim/.venv`.  From PowerShell at the
repository root, use that interpreter directly (the global `python` command is
not registered on this machine):

```powershell
& .\Robosim\.venv\Scripts\python.exe .\Robosim\tune_robot_master_controller.py --list-actuators
& .\Robosim\.venv\Scripts\python.exe .\Robosim\tune_robot_master_controller.py --actuator ak60_revolute_1 --target 0.15 --kp 12 --kd 2.2 --csv .\Robosim\results\ak60_trial.csv
```

The first command prints the four driven joints, their safe target ranges,
torque limits, and the gains currently loaded from MJCF.  The second runs a
headless 5-second step response and reports rise time, overshoot, settling,
tracking error, and torque saturation.  By default it releases the other three
servos so they cannot fight an isolated step; add `--hold-other-actuators` to
test the complete controller state with those servos holding zero.  Add
`--viewer` to watch exactly the same trial in MuJoCo.

For a visual-only model inspection, run:

```powershell
& .\Robosim\.venv\Scripts\python.exe .\Robosim\launch_robot_master_mujoco.py
```

## Controller being tuned

Each MJCF position actuator acts as a saturated PD position servo:

```text
torque = kp * (target_position - joint_position) - kd * joint_velocity
```

The tuner changes `kp` and `kd` only in memory; it does not overwrite
`robot_master.xml`.  Its torque output remains constrained to the motor limits
in that file.  This makes headless trials repeatable and keeps generated model
assets separate from controller experiments.  MuJoCo's position-actuator
mapping is documented in the [XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html).

The present model is fixed-base, and its CAD visual meshes intentionally have
collision disabled.  It is therefore appropriate for actuator/trajectory
tracking gains, but it cannot yet validate ground contact, whole-robot balance,
or hardware safety.  Before transferring a result to the robot, calibrate
encoder zero/sign, gearing, torque limits, friction, and payload against the
real mechanism.

## Model workflow

The source model is `Robosim/robot_master/urdf/robot_master.urdf`; the model
that MuJoCo actually loads is `Robosim/robot_master/mjcf/robot_master.xml`.
If the URDF or GLTF assets change, regenerate and validate the MJCF with:

```powershell
& .\Robosim\.venv\Scripts\python.exe .\Robosim\convert_robot_master_to_mjcf.py
```

The converter adds the closed-loop constraints and the four actuator limits.
Review those constants whenever the physical transmission or motor selection
changes.
