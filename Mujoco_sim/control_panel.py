"""Tk control panel for hardware_integrated_main_controller.py (--gui).

The panel only draws widgets and forwards clicks to a backend object that the
controller provides (see ``PanelBackend``).  It runs on the controller's main
thread: the control loop calls ``ControlPanel.pump()`` once per tick, so the
panel never owns an event loop and can never stall motor streaming.  For the
same reason it uses no modal dialogs (a dialog runs its own event loop and
would freeze the control loop while open); risky buttons need a second click
within CONFIRM_WINDOW_S instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
import tkinter as tk
from tkinter import ttk
from typing import Protocol


REFRESH_PERIOD_S = 0.1     # how often labels are refreshed (events are pumped every tick)
CONFIRM_WINDOW_S = 4.0     # second click must come within this time


@dataclass
class MotorRow:
    motor: int
    joint: str
    reference_deg: float            # limiter reference, joint side
    encoder_raw_rad: float | None   # raw STM32 reading, motor side
    feedback_age_s: float | None
    fault: int | None


@dataclass
class PanelStatus:
    mode: str                       # e.g. "SIMULATION" or "HARDWARE (COM5)"
    hardware: bool
    control_state: str              # "IDLE", "ACTIVE" or "PAUSED"
    motor_power: str                # "not enabled", "enabling", "enabled", "limp (disabled)", "held (0X)"
    moving: bool
    motors: list[MotorRow]
    tool_reference_m: tuple[float, float, float]
    tool_target_m: tuple[float, float, float]
    tool_encoders_m: tuple[float, float, float] | None
    preview_text: str               # description of the current preview (or "")
    preview_ok: bool                # the preview solved
    preview_runnable: bool          # "Run on robot" would be accepted now
    preview_busy: bool              # a solve is in progress
    notice: str = ""                # latest controller event worth showing
    notice_is_alert: bool = False


class PanelBackend(Protocol):
    """What the controller provides.  Actions return a message for the status line."""

    def status(self) -> PanelStatus: ...
    def enable(self) -> str: ...
    def start(self, arm_at_home: bool) -> str: ...
    def pause(self) -> str: ...
    def resume(self) -> str: ...
    def home(self) -> str: ...
    def hold(self) -> str: ...
    def disable(self) -> str: ...
    def preview(self, target_base_m: tuple[float, float, float]) -> str: ...
    def run_preview(self) -> str: ...
    def clear_preview(self) -> str: ...
    def quit(self) -> None: ...


@dataclass
class _Confirm:
    """Two-click confirmation state for one button."""

    button: ttk.Button
    label: str
    confirm_label: str
    armed_at: float = field(default=-math.inf)

    def click(self) -> bool:
        """Return True on the confirming (second) click."""
        now = time.monotonic()
        if now - self.armed_at <= CONFIRM_WINDOW_S:
            self.armed_at = -math.inf
            self.button.configure(text=self.label)
            return True
        self.armed_at = now
        self.button.configure(text=self.confirm_label)
        return False

    def expire(self) -> None:
        if self.armed_at != -math.inf and time.monotonic() - self.armed_at > CONFIRM_WINDOW_S:
            self.armed_at = -math.inf
            self.button.configure(text=self.label)


class ControlPanel:
    def __init__(self, backend: PanelBackend, title: str = "Robot arm control") -> None:
        self.backend = backend
        self.closed = False
        self._last_refresh = 0.0
        self._quit_armed_at = -math.inf

        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.resizable(False, False)
        style = ttk.Style(self.root)
        style.configure("Alert.TLabel", foreground="#b00020")
        style.configure("Ok.TLabel", foreground="#1b5e20")
        style.configure("Stop.TButton", foreground="#b00020")

        pad = {"padx": 6, "pady": 3}
        status = self.backend.status()

        # --- Status ---
        frame = ttk.LabelFrame(self.root, text="Status")
        frame.grid(row=0, column=0, sticky="ew", **pad)
        self.mode_var = tk.StringVar()
        self.state_var = tk.StringVar()
        self.power_var = tk.StringVar()
        for row, (name, var) in enumerate((("Mode", self.mode_var), ("Control", self.state_var), ("Motors", self.power_var))):
            ttk.Label(frame, text=f"{name}:").grid(row=row, column=0, sticky="w", **pad)
            ttk.Label(frame, textvariable=var, width=46).grid(row=row, column=1, sticky="w", **pad)

        # --- Motor controls ---
        frame = ttk.LabelFrame(self.root, text="Motors")
        frame.grid(row=1, column=0, sticky="ew", **pad)
        self.enable_btn = ttk.Button(frame, text="Enable motors", command=lambda: self._act(self.backend.enable))
        self.start_home_btn = ttk.Button(frame, text="Start - arm is at home", command=self._start_home)
        self.start_keep_btn = ttk.Button(frame, text="Start - keep current pose", command=self._start_keep)
        self.pause_btn = ttk.Button(frame, text="Pause", command=lambda: self._act(self.backend.pause))
        self.resume_btn = ttk.Button(frame, text="Resume", command=lambda: self._act(self.backend.resume))
        self.home_btn = ttk.Button(frame, text="Go home", command=lambda: self._act(self.backend.home))
        self.hold_btn = ttk.Button(frame, text="HOLD (0X freeze)", style="Stop.TButton",
                                   command=lambda: self._act(self.backend.hold))
        self.disable_btn = ttk.Button(frame, text="Disable (limp)", command=self._disable)
        buttons = (
            (self.enable_btn, 0, 0), (self.start_home_btn, 0, 1), (self.start_keep_btn, 0, 2),
            (self.pause_btn, 1, 0), (self.resume_btn, 1, 1), (self.home_btn, 1, 2),
            (self.hold_btn, 2, 0), (self.disable_btn, 2, 1),
        )
        for button, row, column in buttons:
            button.grid(row=row, column=column, sticky="ew", **pad)
        self._disable_confirm = _Confirm(self.disable_btn, "Disable (limp)", "Click again: arm goes limp")
        self._start_home_confirm = _Confirm(
            self.start_home_btn, "Start - arm is at home", "Click again: arm IS at home"
        )
        self._start_keep_confirm = _Confirm(
            self.start_keep_btn, "Start - keep current pose", "Click again: arm matches sim"
        )

        # --- Motor table ---
        frame = ttk.LabelFrame(self.root, text="Joints")
        frame.grid(row=2, column=0, sticky="ew", **pad)
        headings = ("Motor", "Joint", "Reference (deg, joint)", "Encoder (rad, raw)", "Feedback age", "Fault")
        for column, text in enumerate(headings):
            ttk.Label(frame, text=text).grid(row=0, column=column, sticky="w", padx=6)
        self.motor_vars: dict[int, list[tk.StringVar]] = {}
        for row, motor in enumerate(status.motors, start=1):
            variables = [tk.StringVar() for _ in headings]
            self.motor_vars[motor.motor] = variables
            for column, variable in enumerate(variables):
                ttk.Label(frame, textvariable=variable, width=(6 if column == 0 else 13)).grid(
                    row=row, column=column, sticky="w", padx=6
                )

        # --- Tool point ---
        frame = ttk.LabelFrame(self.root, text="Tool point (base frame, mm)")
        frame.grid(row=3, column=0, sticky="ew", **pad)
        self.tool_ref_var = tk.StringVar()
        self.tool_target_var = tk.StringVar()
        self.tool_enc_var = tk.StringVar()
        for row, (name, var) in enumerate(
            (("Reference", self.tool_ref_var), ("Target", self.tool_target_var), ("From encoders", self.tool_enc_var))
        ):
            ttk.Label(frame, text=f"{name}:").grid(row=row, column=0, sticky="w", **pad)
            ttk.Label(frame, textvariable=var, width=46).grid(row=row, column=1, sticky="w", **pad)

        # --- IK target ---
        frame = ttk.LabelFrame(self.root, text="Move tool point (base frame)")
        frame.grid(row=4, column=0, sticky="ew", **pad)
        self.target_vars = []
        for column, axis in enumerate("xyz"):
            ttk.Label(frame, text=f"{axis} (mm)").grid(row=0, column=2 * column, sticky="e", **pad)
            variable = tk.StringVar()
            entry = ttk.Entry(frame, textvariable=variable, width=9)
            entry.grid(row=0, column=2 * column + 1, sticky="w", **pad)
            entry.bind("<Return>", lambda _event: self._preview())
            self.target_vars.append(variable)
        self._fill_target(status.tool_reference_m)
        self.use_current_btn = ttk.Button(frame, text="Use current", command=self._use_current)
        self.preview_btn = ttk.Button(frame, text="Preview in MuJoCo", command=self._preview)
        self.run_btn = ttk.Button(frame, text="Run on robot", command=lambda: self._act(self.backend.run_preview))
        self.clear_btn = ttk.Button(frame, text="Clear preview", command=lambda: self._act(self.backend.clear_preview))
        # The entries above use columns 0-5 (label, entry per axis).
        self.use_current_btn.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)
        self.preview_btn.grid(row=1, column=2, columnspan=2, sticky="ew", **pad)
        self.run_btn.grid(row=1, column=4, columnspan=2, sticky="ew", **pad)
        self.clear_btn.grid(row=1, column=6, sticky="ew", **pad)
        self.preview_var = tk.StringVar()
        self.preview_label = ttk.Label(frame, textvariable=self.preview_var, justify="left", width=70)
        self.preview_label.grid(row=2, column=0, columnspan=7, sticky="w", **pad)

        # --- Messages ---
        self.message_var = tk.StringVar(value="Ready.")
        self.message_label = ttk.Label(self.root, textvariable=self.message_var, width=80)
        self.message_label.grid(row=5, column=0, sticky="w", **pad)
        self.notice_var = tk.StringVar()
        self.notice_label = ttk.Label(self.root, textvariable=self.notice_var, width=80)
        self.notice_label.grid(row=6, column=0, sticky="w", **pad)

        self._refresh(status)

    # ------------------------------------------------------------------
    # Event pumping (called by the control loop)
    # ------------------------------------------------------------------

    def pump(self) -> None:
        """Process pending UI events; refresh labels at REFRESH_PERIOD_S."""
        if self.closed:
            return
        now = time.monotonic()
        try:
            if now - self._last_refresh >= REFRESH_PERIOD_S:
                self._last_refresh = now
                self._refresh(self.backend.status())
            self.root.update()
        except tk.TclError:
            # The window was destroyed underneath us.
            self.closed = True

    def destroy(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.root.destroy()
            except tk.TclError:
                pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _act(self, action) -> None:
        self._say(action())

    def _say(self, message: str) -> None:
        self.message_var.set(message)
        is_error = message.startswith(("Cannot", "Not ", "Refused", "No ", "Unavailable"))
        self.message_label.configure(style="Alert.TLabel" if is_error else "TLabel")

    def _start_home(self) -> None:
        if self._start_home_confirm.click():
            self._act(lambda: self.backend.start(arm_at_home=True))
        else:
            self._say("Start resets the reference to hardware_zero. Only confirm if the arm is physically at home.")

    def _start_keep(self) -> None:
        if self._start_keep_confirm.click():
            self._act(lambda: self.backend.start(arm_at_home=False))
        else:
            self._say("Start keeps the current reference. Only confirm if the real arm matches the sim pose.")

    def _disable(self) -> None:
        if self._disable_confirm.click():
            self._act(self.backend.disable)
        else:
            self._say("Disable makes every motor limp: the arm drops unless it rests on its hard stops.")

    def _read_target_m(self) -> tuple[float, float, float] | None:
        try:
            values = tuple(float(v.get()) / 1000.0 for v in self.target_vars)
        except ValueError:
            self._say("Not a number: enter x, y and z in millimetres.")
            return None
        if not all(math.isfinite(v) for v in values):
            self._say("Not a number: enter x, y and z in millimetres.")
            return None
        return values  # type: ignore[return-value]

    def _preview(self) -> None:
        target = self._read_target_m()
        if target is not None:
            self._say(self.backend.preview(target))

    def _fill_target(self, position_m: tuple[float, float, float]) -> None:
        for variable, value in zip(self.target_vars, position_m):
            variable.set(f"{1000.0 * value:.1f}")

    def _use_current(self) -> None:
        self._fill_target(self.backend.status().tool_reference_m)
        self._say("Target set to the current reference tool point.")

    def _on_close(self) -> None:
        now = time.monotonic()
        if now - self._quit_armed_at <= CONFIRM_WINDOW_S:
            self.backend.quit()
            self.destroy()
            return
        self._quit_armed_at = now
        self._say("Close the window again within 4 s to quit the controller.")

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    @staticmethod
    def _mm(position_m: tuple[float, float, float] | None) -> str:
        if position_m is None:
            return "n/a"
        return "  ".join(f"{axis}={1000.0 * v:+8.1f}" for axis, v in zip("xyz", position_m))

    def _refresh(self, s: PanelStatus) -> None:
        for confirm in (self._disable_confirm, self._start_home_confirm, self._start_keep_confirm):
            confirm.expire()

        self.mode_var.set(s.mode)
        self.state_var.set(s.control_state + ("  (moving)" if s.moving else ""))
        self.power_var.set(s.motor_power if s.hardware else "n/a (simulation)")

        active = s.control_state != "IDLE"            # running or paused
        running = s.control_state.startswith("ACTIVE")  # also "ACTIVE (simulation)"
        enabled = s.motor_power == "enabled"

        def set_state(button: ttk.Button, allowed: bool) -> None:
            button.state(["!disabled"] if allowed else ["disabled"])

        set_state(self.enable_btn, s.hardware and not active and s.motor_power != "enabling")
        set_state(self.start_home_btn, s.hardware and not active and enabled)
        set_state(self.start_keep_btn, s.hardware and not active and enabled)
        set_state(self.pause_btn, running)
        set_state(self.resume_btn, s.control_state == "PAUSED")
        set_state(self.home_btn, running)
        set_state(self.hold_btn, s.hardware and active)
        set_state(self.disable_btn, s.hardware and s.motor_power not in ("not enabled", "limp (disabled)"))
        set_state(self.preview_btn, not s.preview_busy)
        set_state(self.run_btn, s.preview_runnable)
        set_state(self.clear_btn, bool(s.preview_text) and not s.preview_busy)

        for row in s.motors:
            values = (
                f"M{row.motor}",
                row.joint,
                f"{row.reference_deg:+8.2f}",
                "n/a" if row.encoder_raw_rad is None else f"{row.encoder_raw_rad:+.4f}",
                "n/a" if row.feedback_age_s is None else f"{1000.0 * row.feedback_age_s:.0f} ms",
                "n/a" if row.fault is None else str(row.fault),
            )
            for variable, value in zip(self.motor_vars[row.motor], values):
                variable.set(value)

        self.tool_ref_var.set(self._mm(s.tool_reference_m))
        self.tool_target_var.set(self._mm(s.tool_target_m))
        self.tool_enc_var.set(self._mm(s.tool_encoders_m) if s.hardware else "n/a (simulation)")

        self.preview_var.set("Solving..." if s.preview_busy else s.preview_text)
        self.preview_label.configure(style=("Ok.TLabel" if s.preview_ok else "Alert.TLabel") if s.preview_text else "TLabel")
        self.notice_var.set(s.notice)
        self.notice_label.configure(style="Alert.TLabel" if s.notice_is_alert else "TLabel")
