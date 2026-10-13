# Hardware Controller Behavior

`hardware_integrated_main_controller.py` has two launch modes.

- `python hardware_integrated_main_controller.py` opens MuJoCo in
  simulation-only mode. It does not open a serial port or send anything to an
  STM32.
- `python hardware_integrated_main_controller.py COMx --enable-motors` opens
  the specified STM32 serial port at 115200 baud, sends `0X` to freeze motors
  on connection, then enables M1--M4 and clears the STM32 outer PID gains.

## Commands sent to the STM32

On connection, the controller sends:

```text
0X
```

With `--enable-motors`, or after the terminal command `a`, each motor is sent:

```text
1E
1G0.0
1H0.0
1J0.0
```

The motor number replaces `1` for M2--M4. `E` enables the motor; `G`, `H`, and
`J` clear the STM32 outer-loop PID terms so the PC-side controller supplies
the control request.

After the terminal command `s`, the program transmits five newline-terminated
ASCII fields to each M1--M4 actuator at 84 Hz:

```text
1P<position_rad>
1V<velocity_rad_per_s>
1K<kp>
1D<kd>
1T<feedforward_torque_Nm>
```

`P` and `V` are motor-side position and velocity references. `K` and `D` are
the motor-side scheduled PD gains. `T` is gravity feedforward; its magnitude
is clamped by the configured motor torque limit.

## Encoder-relative command frame

The controller does not replace an absolute encoder reading with a relative
sensor. It captures the raw encoder reading when `s` is pressed and uses it as
a software reference for outgoing position commands:

```text
P_command = encoder_at_start + requested_motion
```

No actuator receives an automatic motion when `s` is pressed. The first
position command for each actuator is its measured encoder position, so M1 and
M4 hold their synchronized positions. Motion begins only after an operator
changes a MuJoCo Control-tab value or issues a `g` command.

## Terminal telemetry

While the program runs, telemetry is printed four times per second by default.
For each motor it reports:

- `sim`: current MuJoCo joint position;
- `enc`: raw STM32 encoder position;
- `enc_sim`: encoder position projected into the synchronized MuJoCo joint
  coordinate;
- `error`: `sim - enc_sim`, in joint radians;
- `Kp` and `Kd`: the motor-side values being used for the MIT commands.

Before a hardware synchronization or without encoder feedback, `enc_sim` and
`error` are shown as `n/a`. Press `f` to toggle this terminal telemetry. Change
`MOTOR_FEEDBACK_PRINT_HZ` in `hardware_integrated_main_controller.py` to alter
the print rate.

`p` stops sending MIT frames but does not transmit `0X`; closing the program
closes the serial port. The explicit all-motor freeze command is `0X`.
