# robot_master arm: MuJoCo controller and STM32 bridge

PC-side control of a 4-motor robot arm (CubeMars AK60, 2x AK70, AK40) built
around a CAD-derived MuJoCo model.  The PC does all of the computation; an
STM32 Nucleo turns serial commands into CAN frames for the motors and streams
the motors' feedback back.  The operator drags the MuJoCo viewer's **Control**
sliders, and the real arm follows the simulated reference.

## Repository layout

| Path | Contents |
| --- | --- |
| `Mujoco_sim/` | The PC controller, the MuJoCo model, and helper tools (see below). |
| `Mujoco_sim/old_scripts/` | Older simulation-only tuning, training, and viewer scripts. |
| `Mujoco_sim/robot_master_urdf_table/` | Source URDF (`urdf/`) and the MJCF that MuJoCo loads (`mjcf/robot_master.xml`). |
| `Mujoco_sim/logs/` | One CSV + JSON + SVG per controller run. |
| `STM32_firmware/cubemarscontrol-pid-main-CUBEIDE-project/` | STM32CubeIDE project for the serial-to-CAN bridge (STM32F446RE Nucleo + Waveshare CAN shield). |
| `simple_gui/motor_gui.py` | Tkinter GUI for sending single commands to individual motors and servos. |

## Setup (Windows)

The global `python` command is not registered on this machine, so use a venv
inside `Mujoco_sim/` (it is git-ignored).  From PowerShell at the repository
root:

```powershell
py -3 -m venv .\Mujoco_sim\.venv
& .\Mujoco_sim\.venv\Scripts\python.exe -m pip install -r .\Mujoco_sim\requirements.txt
```

`requirements.txt` pins `mujoco==3.12.0` and `pyserial==3.5`.  The commands
below assume this venv.

## Running the controller

```powershell
& .\Mujoco_sim\.venv\Scripts\python.exe .\Mujoco_sim\hardware_integrated_main_controller.py COM5   # real arm
& .\Mujoco_sim\.venv\Scripts\python.exe .\Mujoco_sim\hardware_integrated_main_controller.py        # simulation only
& .\Mujoco_sim\.venv\Scripts\python.exe .\Mujoco_sim\motor_spoof.py                                 # full stack, simulated motors
```

Pass `--enable-motors` after the port to enable M1-M4 at startup.  On connect
the controller sends `0X` (freeze all motors), so nothing moves until you
start it.

Hardware workflow, typed in the controller terminal:

1. `a`: enable all body motors (M1-M4).
2. `s`: start.  The controller records the encoder readings and the current
   sim pose, then follows *changes* in the reference.  Nothing moves on start.
3. Drag the Control-tab sliders, or type `g <j> <pos>`, for example `g 1 0.5`,
   `g 4 20deg`, or `g 4 +20deg` (relative).

Other commands: `h` go to `hardware_zero`, `p` / `r` pause / resume, `z` print
encoder readings, `f` toggle telemetry, `d` toggle command printout, `q` quit.

Behaviour flags are at the top of `hardware_integrated_main_controller.py`.
The current values on `main` are:

| Flag | Value | Meaning |
| --- | --- | --- |
| `UPDATE_HZ` | 84 | Control loop rate. |
| `CLOSED_LOOP` | `True` | Write encoder feedback into the sim.  `False` tracks the reference only (open loop). |
| `ZERO_TFF` | `True` | Send zero torque feedforward (see known issues). |
| `RUNAWAY_STOP` | `True` | Freeze (`0X`) and return to idle when a motor leaves its command, moves too fast, or stops reporting. |
| `ECHO_CHECK` | `True` | Trip the same stop when the STM32's echo of a CAN command disagrees with what the PC sent. |
| `MOTOR_FAULT_STOP` | `True` | Trip the stop when a motor reports a nonzero CubeMars fault code. |

After a stop, type `a` and then `s` to resynchronize before the arm moves
again.

### Control loop (each tick)

1. Read the latest feedback (`serial_interface.snapshot()`, lag-extrapolated).
2. Write the state into MuJoCo (encoders if `CLOSED_LOOP`, otherwise the
   reference), then `mj_forward`.
3. `MasterController.update()` schedules `kp`/`kd`; the trapezoidal limiter
   advances the reference from the sliders.
4. Compute gravity feedforward with `mj_inverse` (zeroed while `ZERO_TFF`).
5. Send each changed motor's full MIT frame, plus a keepalive frame per motor
   every 0.25 s.
6. Write one row to the run log.

### Run logs

Every run writes `Mujoco_sim/logs/<branch>_<YYYY-MM-DD>_<HH-MM-SS UTC>.csv`
(one row per tick), a `.json` with the run settings and git revision, and an
`.svg` graph when the run ends.  To re-plot existing logs:

```powershell
& .\Mujoco_sim\.venv\Scripts\python.exe .\Mujoco_sim\plot_run_log.py .\Mujoco_sim\logs\<run>.csv
```

## Serial protocol

The UART runs at 1,000,000 baud.  Every value command ends in `*HH`, the XOR
of all bytes before `*` as two hex digits; the firmware rejects value commands
without it.

| PC to STM32 | Meaning |
| --- | --- |
| `<n>M<pos>,<vel>,<kp>,<kd>,<tff>*HH` | All five MIT fields for motor `n`, applied together or not at all. |
| `<n>P` / `V` / `K` / `D` / `T<val>*HH` | Single field (still accepted; used by `motor_gui.py`). |
| `<n>E` | Enable motor. |
| `<n>R` | Passive feedback (no torque). |
| `<n>O` | Set origin at the current position. |
| `0X` | Emergency stop: hold all motors. |

| STM32 to PC | Meaning |
| --- | --- |
| `F<n> <16 hex> <8 hex>` | 8 bytes of feedback CAN data plus the STM32 tick (ms). |
| `C<n> <16 hex>` | Echo of the MIT command as packed for CAN (`TX_ECHO=1`), on change plus every 250 ms. |
| `[UART] ...`, `[M<n>] ...`, `PONG ...` | Plain-text status, errors, and latency replies. |

The firmware disables any enabled motor that sends no feedback for
`FB_WATCHDOG_MS` (2000 ms).  `measure_lag.py` measures the STM32-to-PC latency
using `PING`/`PONG`.

## Motors and coordinates

| Motor | Joint | Motor | CAN ID | Gear | `MOTOR_DIRECTION` |
| --- | --- | --- | --- | --- | --- |
| M1 | `revolute_1` (base) | AK60 | 104 | 1 | +1 |
| M2 | `revolute_2` (shoulder) | AK70 | 1 | 1 | -1 |
| M3 | `revolute_3` (elbow) | AK70 | 2 | 1 | +1 |
| M4 | `revolute_4` (linkage) | AK40 | 3 | 1.25 | -1 |

- M2 is -1 only because the MJCF `revolute_2` coordinate is reversed (hard
  stop = 0, `REVOLUTE_2_REVERSED_REF_RAD`); the physical motor sign is
  unchanged.
- M4 drives the joint through a 48:60 gear pair (1.25:1).  Its actuator `ctrl`
  and serial commands are motor-side, while `qpos` is joint-side:
  joint = motor x direction / 1.25, and gains convert as
  `k_motor = k_joint / 1.25^2`.
- `revolute_5, 6, 7, 8, 9, 13` form a passive four-bar linkage driven by
  `revolute_4`.  The MJCF closes the loop with joint equalities and a weld;
  the controller sets the followers kinematically through
  `CONSTRAINED_JOINTS`.
- At `hardware_zero`, `revolute_4` rests on its 1.552 rad joint limit
  (a hard stop).

### Hardware zero

`Mujoco_sim/robot_master_configuration.py` is the single place to change the
starting pose.  `HARDWARE_ZERO_ACTUATOR_CTRL_RAD` defines the named
`hardware_zero` keyframe:

| Actuator | Home reference |
| --- | ---: |
| `ak60_revolute_1` | 1.45 rad |
| `ak70_revolute_2` | 0.113 rad (URDF frame) |
| `ak70_revolute_3` | -1.29 rad |
| `ak40_revolute_4` | 1.94 rad motor-side (1.552 rad joint-side) |

The same file holds the trapezoid limits (1 rad/s, 2 rad/s^2 by default),
torque-based gain caps, the AK40 motor torque limit, and joint-range
overrides.

## Gain schedule

`master_controller.py` loads `master_controller_schedule.json` and schedules
joint-side PD gains from the measured position:

```text
omega_n(q) = smooth (log-space) interpolation across the trained range
kp(q)      = I_eff(q) * omega_n(q)^2
kd(q)      = 2 * I_eff(q) * zeta * omega_n(q)
```

`I_eff(q)` comes from MuJoCo's mass matrix (`mj_fullM`) on every update, and
the gains are capped so the commanded torque stays within the motor limits.
This is diagonal gain scheduling, not coupled-arm pole placement.

To retrain the schedule on simultaneous multi-joint motions (16 control-window
extremes plus low-discrepancy targets, all collision-screened):

```powershell
& .\Mujoco_sim\.venv\Scripts\python.exe .\Mujoco_sim\train_multidof_master_controller.py
```

### Older simulation-only scripts

`Mujoco_sim/old_scripts/` keeps the earlier tools.  They import from
`Mujoco_sim/`, so run them from the repository root like the other scripts:

| Script | Purpose |
| --- | --- |
| `tune_robot_master_controller.py` | Headless single-joint step test (`--list-actuators`, `--actuator`, `--target`, `--kp`, `--kd`, `--viewer`, `--show-contacts`). |
| `train_master_controller.py` | Per-joint schedule training around `hardware_zero`. |
| `run_master_controller_mujoco.py` | Replays the saved schedule in the viewer. |
| `interactive_master_controller_mujoco.py` | Tk slider panel plus viewer; pauses on collision. |
| `launch_robot_master_mujoco.py` | Viewer running the scheduled-PD callback from the Control sliders. |

`sync_hardware.py` is the older passive mirror: it shows the arm's encoder
readings in the sim and sends no commands.

## Model workflow

The source is `Mujoco_sim/robot_master_urdf_table/urdf/robot_master.urdf`;
MuJoCo loads `Mujoco_sim/robot_master_urdf_table/mjcf/robot_master.xml`.
After the URDF or meshes change, regenerate the MJCF:

```powershell
& .\Mujoco_sim\.venv\Scripts\python.exe .\Mujoco_sim\convert_robot_master_to_mjcf.py
```

The converter adds the loop-closure constraints, the actuator limits, and the
`hardware_zero` keyframe (settling the linkage for 10 s first).  Hand edits to
`robot_master.xml`, such as M4's `forcerange`, are lost on regeneration.

Collisions use five 15 mm capsule proxies along the linkage centerlines, not
the CAD meshes (MuJoCo would reduce them to convex hulls that collide
falsely).  The model is fixed-base and cannot validate hardware safety.

## Firmware

Open `STM32_firmware/` in STM32CubeIDE and build the
`cubemarscontrol-pid-main-CUBEIDE-project`.  The firmware is a thin
passthrough: it parses the UART commands above, packs MIT CAN frames (AK60
uses extended IDs, AK70/AK40 standard IDs), forwards feedback as hex, and
drives three MG995 end-effector servos (motors 5-7) on TIM2 PWM.  The motor
index-to-CAN-ID mapping is in `uart_cmd_init()` and must match
`serial_interface.py`.

## Known issues

- **M4 torque feedforward from `mj_inverse` is wrong** once M4 leaves
  `hardware_zero`: the `revolute_4` row of `qfrc_inverse` picks up the stiff
  loop-closure weld's reaction (hundreds of N m instead of ~0.2 N m).  `main`
  avoids this with `ZERO_TFF = True`.  A verified virtual-work fix is in
  commit `a0a7821` on `10_17_experiments` and is not merged.
- M4's gains are soft (Kp about 0.4 N m/rad motor-side), so unmodelled load
  leaves a visible steady-state error.
- Synchronization is relative: a constant pose difference between the sim and
  the arm at `s` persists.
- The named-pipe command input (`/tmp/robot_cmd`) works only on Linux/macOS;
  on Windows it reports "unavailable".

## Branches

| Branch | Contents |
| --- | --- |
| `main` | Working controller, 84 Hz loop, change-only command streaming. |
| `500hz-loop` | 500 Hz loop with all four motors' frames in one packed line per tick (new `mit_frame` firmware module). |
| `uart-dma-checksum-protocol` | UART DMA receive and the checksummed protocol. |
| `cleanup-and-cubic-spline` | Cubic-spline motion profile and closed-loop fixes (not merged). |
| `10_17_experiments` | M4 feedforward fix (`a0a7821`). |
| `MattBranch` | Matt's experiments. |

## Roadmap

Control stack, bottom to top:

1. FOC on the motor driver (done).
2. MIT-mode onboard PD with low gains for stability (done).
3. Computed torque: `mj_inverse` feedforward on the PC (pending the M4 fix).
4. MPC / iLQR outer loop producing `q_des`, `qd_des`, `qdd_des`.

Remaining work:

- Merge and validate the M4 feedforward fix, then enable `ZERO_TFF = False`.
- Add a state estimator: `mj_step` prediction between packets, with encoder
  updates for the driven joints.
- Implement MPC/iLQR on top.
- Switch from `mj_forward` to `mj_step` for full simulation mode.
