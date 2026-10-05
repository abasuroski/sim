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

## Hardware zero alignment

`Robosim/robot_master_configuration.py` is the single place to change the
common starting configuration. Its `HARDWARE_ZERO_ACTUATOR_CTRL_RAD` block
defines the requested actuator references for the named `hardware_zero`
keyframe, which is loaded by both the normal MuJoCo viewer and the live
hardware synchronizer:

| actuator | home reference |
| --- | ---: |
| `ak60_revolute_1` | 1.45 rad |
| `ak70_revolute_2` | 0.113 rad |
| `ak70_revolute_3` | -1.29 rad |
| `ak40_revolute_4` | 1.94 rad motor-side (1.552 rad joint-side) |

The converter settles the closed-loop linkage for 10 seconds before writing the
keyframe, so all passive joints start in a constraint-consistent state. Live
sync maps encoder deltas from `HARDWARE_ZERO_ENCODER_RAD` onto that solved
state; AK40 feedback is scaled by 0.8 for its 1.25:1 reduction. Press `z` to
print current raw readings in a copyable form when recalibrating. Hardware sync
requires `pyserial==3.5`, which is included in `Robosim/requirements.txt`.

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

Each limited position actuator rejects references outside its CAD mechanical
range; continuous actuators remain unbounded.  For an unlimited hinge, command
a full rotation as a continuous ramp, not a static `2*pi` step:

```powershell
& .\Robosim\.venv\Scripts\python.exe .\Robosim\tune_robot_master_controller.py --actuator ak60_revolute_1 --turns 1 --ramp-duration 3 --duration 5 --viewer
```

The CAD model's closed-loop linkage constraints remain active.  For coupled
joints, a target inside one joint's range can still require compatible targets
from the other motors to form a feasible mechanism pose.

The MuJoCo viewer now gives the continuous AK60 and AK70 controls a `-2*pi` to
`+2*pi` slider span.  This is a display/input span only: their controller
commands remain unclamped.

The AK40 `revolute_4` transmission models a 48-tooth motor pinion driving a
60-tooth output gear (`1.25:1`).  Its motor-side 4.1 N m limit therefore maps
to 5.125 N m at the joint before gearbox losses.  The tuner accepts and reports
joint-output angles, gains, and torques; it applies the transmission conversion
internally.

The CAD visual meshes are not used as collision geometry: MuJoCo reduces a
mesh collider to a convex hull, which makes the intermeshed gears, fasteners,
and nested CAD hardware collide falsely.  Instead, the model adds five 15 mm
radius capsule colliders along the articulated linkage centerlines.  Adjacent
links are excluded where their capsules meet at a hinge; all remaining proxy
pairs generate MuJoCo contact constraints.  This prevents the modeled linkage
bars from passing through one another without artificially blocking a joint at
about 90 degrees.  The colliders are intentionally conservative, so they are
not a replacement for manufacturing-grade collision geometry.

Each tuner trial reports the peak number of collision contacts. Add
`--show-contacts` to list the body pairs that made contact during the trial.

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

## Current state:

```MuJoCo sim mirrors real arm via serial encoder feedback (sync_hardware.py)
4 driven joints (revolute 1-4) tracked via encoders
Passive joints (revolute 5,6,7,8,9,13) derived via CONSTRAINED_JOINTS deltas
revolute_14_loop_closure left out, to be handled by weld constraint in full sim
Serial-to-CAN bridge on STM32 at ~84Hz
Motors accept MIT control mode: (q_ref, v_ref, Kp, Kd, tau_ff)```

## Architecture plan:

PC runs all heavy computation, STM32 is purely a serial-to-CAN bridge
Motors driven via MIT control mode frames

## Control stack (bottom to top):

FOC — onboard motor driver, already handled
Onboard PD — low gains, just for stability, part of MIT control mode
Computed torque / inverse dynamics — mj_inverse on PC, outputs tau_ff
MPC outer loop — runs trajectory optimization, outputs q_des, qd_des, qadd_des

## State estimation:

Predict: mj_step forward between encoder packets
Update: replace driven joint qpos with real encoder values when packet arrives
Passive joints computed from driven joints via kinematic constraints
Velocity: finite difference or velocity observer (encoders give position + velocity already)

## TODO:

Finish and validate kinematic mirror (sync_hardware.py)
Implement serial command sender for MIT control frames
Validate computed torque with mj_inverse in open loop
Add state estimator
Implement MPC/iLQR on top
Switch from mj_forward to mj_step for full simulation mode
