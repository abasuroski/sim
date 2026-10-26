"""
Robot Arm Control GUI (Body + End Effector)
Sends commands over serial to the STM32 UART command parser.

Protocol (motor index prefix):
  <n>P<val>\n  — position (rad for motors, degrees for servos)
  <n>V<val>\n  — velocity (rad/s)
  <n>K<val>\n  — Kp
  <n>D<val>\n  — Kd
  <n>T<val>\n  — torque feedforward (N·m)
  <n>E\n       — enable motor (AK70/80/AK40 only)
  <n>O\n       — set origin (zero current position)
  <n>I<val>\n  — set CAN ID
  <n>A<val>\n  — set motor type (60, 70, or 40)

  Body motors: n=1 (Base), 2 (Shoulder), 3 (Elbow), 4 (Linkage)
  End effector servos: n=5 (Wrist 1), 6 (Wrist 2), 7 (Gripper)

Servo PWM reference (from servotest project):
  TIM2, Prescaler=180-1, Period=10000-1 → 50Hz
  Pulse 250 = 0 deg (500us), Pulse 1250 = 180 deg (2500us)
  MG995 TowerPro servos, channels TIM2_CH2, TIM2_CH3, TIM2_CH4

Feedback from STM32 is prefixed: [M<n>] ...

Usage:
  pip install pyserial
  python motor_gui.py
"""

import math
import re
import time
import tkinter as tk
from tkinter import ttk
import serial
import serial.tools.list_ports
import threading
from datetime import datetime


AK60_LIMIT_DEG = 45.0

# All angular params (P, V) use degrees in the GUI; converted to rad before sending
ANGULAR_CMDS = {"P", "V"}

AK60_PARAMS = {
    "Position (deg)": ("P", math.degrees(-12.56), math.degrees(12.56), 0.0),
    "Velocity (deg/s)": ("V", math.degrees(-60.0), math.degrees(60.0), 0.0),
    "Kp": ("K", 0.0, 500.0, 2.0),
    "Kd": ("D", 0.0, 5.0, 1.0),
    "Torque FF (N·m)": ("T", -12.0, 12.0, 0.0),
}

AK70_PARAMS = {
    "Position (deg)": ("P", math.degrees(-12.5), math.degrees(12.5), 0.0),
    "Velocity (deg/s)": ("V", math.degrees(-30.0), math.degrees(30.0), 0.0),
    "Kp (inner)": ("K", 0.0, 500.0, 5.0),
    "Kd (inner)": ("D", 0.0, 5.0, 0.3),
    "Torque FF (N·m)": ("T", -18.0, 18.0, 0.0),
    "Outer Kp": ("G", 0.0, 50.0, 5.0),
    "Outer Kd": ("H", 0.0, 20.0, 0.0),
    "Outer Ki": ("J", 0.0, 1.0, 0.0),
}

AK40_PARAMS = {
    "Position (deg)": ("P", math.degrees(-12.5), math.degrees(12.5), 0.0),
    "Velocity (deg/s)": ("V", math.degrees(-45.5), math.degrees(45.5), 0.0),
    "Kp (inner)": ("K", 0.0, 500.0, 5.0),
    "Kd (inner)": ("D", 0.0, 5.0, 0.3),
    "Torque FF (N·m)": ("T", -5.0, 5.0, 0.0),
    "Outer Kp": ("G", 0.0, 50.0, 4.0),
    "Outer Kd": ("H", 0.0, 20.0, 0.0),
    "Outer Ki": ("J", 0.0, 1.0, 0.0),
}


def convert_rad_feedback(line):
    """Convert pos/vel rad values in AK70/AK40 feedback lines to degrees."""
    if " rad" not in line:
        return line
    line = re.sub(
        r'pos=(-?\d+\.?\d*) rad',
        lambda m: f"pos={math.degrees(float(m.group(1))):.1f} deg",
        line
    )
    line = re.sub(
        r'vel=(-?\d+\.?\d*)',
        lambda m: f"vel={math.degrees(float(m.group(1))):.1f} deg/s",
        line
    )
    return line


def cosine_interp(t, duration, start, end):
    """Cosine interpolation: zero velocity at both endpoints, smooth acceleration.
    Returns (position, velocity) in the same units as start/end."""
    s = t / duration
    pos = start + (end - start) * (1.0 - math.cos(math.pi * s)) / 2.0
    vel = (end - start) * math.pi / (2.0 * duration) * math.sin(math.pi * s)
    return pos, vel


DEMO_SEGMENTS = [
    {"name": "Lowering shoulder",  "duration": 5.0, "targets": {2: (0, -45)}, "tff": {2: -0.3}},
    {"name": "Pause",              "duration": 1.0, "targets": {}},
    {"name": "Sweep right",        "duration": 5.0, "targets": {1: (0, 90)}},
    {"name": "Pause",              "duration": 1.0, "targets": {}},
    {"name": "Sweep left",         "duration": 6.0, "targets": {1: (90, -90)}},
    {"name": "Pause",              "duration": 1.0, "targets": {}},
    {"name": "Return base",        "duration": 4.0, "targets": {1: (-90, 0)}},
    {"name": "Pause",              "duration": 1.0, "targets": {}},
    {"name": "Raising shoulder",   "duration": 4.0, "targets": {2: (-45, 0)}},
]


class MotorPanel:
    def __init__(self, parent, motor_idx, label, motor_type, can_id, send_fn,
                 pos_min=None, pos_max=None):
        self.motor_idx = motor_idx
        self.send_fn = send_fn
        self.sliders = {}
        self.slider_widgets = {}
        self.ak60_home = 0.0
        self.pos_min = pos_min
        self.pos_max = pos_max

        self.frame = ttk.LabelFrame(parent, text=label)
        self.frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # --- Type + CAN ID row ---
        config_frame = ttk.Frame(self.frame)
        config_frame.pack(fill="x", padx=5, pady=3)

        ttk.Label(config_frame, text="Type:").pack(side="left")
        self.type_var = tk.StringVar(value=motor_type)
        self.type_combo = ttk.Combobox(config_frame, textvariable=self.type_var,
                                       values=["AK60", "AK70/80", "AK40"], width=8, state="readonly")
        self.type_combo.pack(side="left", padx=3)
        self.type_combo.bind("<<ComboboxSelected>>", self.on_type_change)

        ttk.Label(config_frame, text="CAN ID:").pack(side="left", padx=(10, 0))
        self.canid_var = tk.IntVar(value=can_id)
        self.canid_entry = ttk.Entry(config_frame, textvariable=self.canid_var, width=5)
        self.canid_entry.pack(side="left", padx=3)

        ttk.Button(config_frame, text="Set ID", command=self.send_canid).pack(side="left", padx=3)

        # --- Enable button ---
        self.enable_frame = ttk.Frame(self.frame)
        self.enable_frame.pack(fill="x", padx=5, pady=2)
        self.enable_btn = ttk.Button(self.enable_frame, text="Enable Motor", command=self.send_enable)
        self.enable_btn.pack(side="left")
        self.origin_btn = ttk.Button(self.enable_frame, text="Set Origin", command=self.send_set_origin)
        self.origin_btn.pack(side="left", padx=(6, 0))
        self.limit_label = ttk.Label(self.enable_frame, text="", font=("Consolas", 8), foreground="blue")
        self.fixed_limit_label = ttk.Label(self.enable_frame, text="", font=("Consolas", 8), foreground="red")

        # --- Sliders frame ---
        self.slider_frame = ttk.LabelFrame(self.frame, text="Parameters")
        self.slider_frame.pack(fill="x", padx=5, pady=3)

        self.build_sliders()

        # --- GO button ---
        self.go_btn = ttk.Button(self.frame, text="GO", command=self.send_all_params)
        self.go_btn.pack(fill="x", padx=5, pady=5, ipady=6)

        # --- Feedback ---
        fb_frame = ttk.LabelFrame(self.frame, text="Feedback")
        fb_frame.pack(fill="both", expand=True, padx=5, pady=3)

        self.feedback_text = tk.Text(fb_frame, height=6, width=40, state="disabled", font=("Consolas", 8))
        self.feedback_text.pack(fill="both", expand=True, padx=3, pady=3)

        self.update_enable_visibility()

    def build_sliders(self):
        for widget in self.slider_frame.winfo_children():
            widget.destroy()
        self.sliders.clear()
        self.slider_widgets.clear()

        t = self.type_var.get()
        if "70" in t:
            params = dict(AK70_PARAMS)
        elif "40" in t:
            params = dict(AK40_PARAMS)
        else:
            lo = self.ak60_home - AK60_LIMIT_DEG
            hi = self.ak60_home + AK60_LIMIT_DEG
            params = dict(AK60_PARAMS)
            params["Position (deg)"] = ("P", lo, hi, self.ak60_home)

        if self.pos_min is not None or self.pos_max is not None:
            cmd, lo, hi, default = params["Position (deg)"]
            if self.pos_min is not None:
                lo = self.pos_min
            if self.pos_max is not None:
                hi = self.pos_max
            default = max(lo, min(hi, default))
            params["Position (deg)"] = (cmd, lo, hi, default)

        for i, (label, (cmd, lo, hi, default)) in enumerate(params.items()):
            ttk.Label(self.slider_frame, text=label).grid(row=i, column=0, sticky="w", padx=3, pady=1)

            var = tk.DoubleVar(value=default)
            slider = ttk.Scale(self.slider_frame, from_=lo, to=hi, variable=var,
                               orient="horizontal", length=180)
            slider.grid(row=i, column=1, padx=3, pady=1)

            entry = ttk.Entry(self.slider_frame, textvariable=var, width=7)
            entry.grid(row=i, column=2, padx=3, pady=1)

            self.sliders[cmd] = var

    def on_type_change(self, event=None):
        self.ak60_home = 0.0
        self.build_sliders()
        self.update_enable_visibility()
        t = self.type_var.get()
        if "70" in t:
            code = "70"
        elif "40" in t:
            code = "40"
        else:
            code = "60"
        self.send_fn(f"{self.motor_idx}A{code}\n")

    def update_enable_visibility(self):
        t = self.type_var.get()
        if "60" in t:
            lo = self.ak60_home - AK60_LIMIT_DEG
            hi = self.ak60_home + AK60_LIMIT_DEG
            self.limit_label.config(text=f"  [{lo:.1f}, {hi:.1f}] deg")
            self.limit_label.pack(side="left", padx=(4, 0))
        else:
            self.limit_label.pack_forget()
        if self.pos_min is not None or self.pos_max is not None:
            lo_txt = f"{self.pos_min:.1f}" if self.pos_min is not None else "-∞"
            hi_txt = f"{self.pos_max:.1f}" if self.pos_max is not None else "+∞"
            self.fixed_limit_label.config(text=f"  lim [{lo_txt}, {hi_txt}] deg")
            self.fixed_limit_label.pack(side="left", padx=(4, 0))
        else:
            self.fixed_limit_label.pack_forget()
        self.enable_btn.pack(side="left")

    def send_canid(self):
        self.send_fn(f"{self.motor_idx}I{self.canid_var.get()}\n")

    def send_enable(self):
        self.send_fn(f"{self.motor_idx}E\n")

    def send_set_origin(self):
        self.send_fn(f"{self.motor_idx}O\n")
        t = self.type_var.get()
        if "60" in t:
            self.ak60_home = 0.0
            self.build_sliders()
            lo = self.ak60_home - AK60_LIMIT_DEG
            hi = self.ak60_home + AK60_LIMIT_DEG
            self.limit_label.config(text=f"  [{lo:.1f}, {hi:.1f}] deg")

    def send_all_params(self):
        import time
        t = self.type_var.get()
        for cmd, var in self.sliders.items():
            val = var.get()
            if "60" in t and cmd == "P":
                lo = self.ak60_home - AK60_LIMIT_DEG
                hi = self.ak60_home + AK60_LIMIT_DEG
                val = max(lo, min(hi, val))
                var.set(val)
            if cmd == "P" and (self.pos_min is not None or self.pos_max is not None):
                lo = self.pos_min if self.pos_min is not None else -math.inf
                hi = self.pos_max if self.pos_max is not None else math.inf
                val = max(lo, min(hi, val))
                var.set(val)
            if cmd in ANGULAR_CMDS:
                val = math.radians(val)
            self.send_fn(f"{self.motor_idx}{cmd}{val:.4f}\n")
            time.sleep(0.005)

    def append_feedback(self, line):
        line = convert_rad_feedback(line)
        self.feedback_text.config(state="normal")
        self.feedback_text.insert("end", line + "\n")
        self.feedback_text.see("end")
        if int(self.feedback_text.index("end-1c").split(".")[0]) > 200:
            self.feedback_text.delete("1.0", "2.0")
        self.feedback_text.config(state="disabled")


class WristDifferentialPanel:
    """Differential wrist: pitch and roll map to two servos.
    servo1 = pitch + roll, servo2 = pitch - roll"""

    def __init__(self, parent, send_fn):
        self.send_fn = send_fn

        self.frame = ttk.LabelFrame(parent, text="Wrist (Differential)")
        self.frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        ttk.Label(self.frame, text="MG995 TowerPro x2", font=("", 8, "italic")).pack(padx=5, pady=2)

        # --- Pitch slider ---
        pitch_frame = ttk.LabelFrame(self.frame, text="Pitch (degrees)")
        pitch_frame.pack(fill="x", padx=5, pady=3)

        self.pitch_var = tk.DoubleVar(value=0.0)
        ttk.Scale(pitch_frame, from_=-120, to=120, variable=self.pitch_var,
                  orient="horizontal", length=200).pack(padx=5, pady=3)

        pf = ttk.Frame(pitch_frame)
        pf.pack(fill="x", padx=5, pady=2)
        ttk.Entry(pf, textvariable=self.pitch_var, width=7).pack(side="left", padx=3)
        ttk.Label(pf, text="deg").pack(side="left")

        # --- Roll slider ---
        roll_frame = ttk.LabelFrame(self.frame, text="Roll (degrees)")
        roll_frame.pack(fill="x", padx=5, pady=3)

        self.roll_var = tk.DoubleVar(value=0.0)
        ttk.Scale(roll_frame, from_=-120, to=120, variable=self.roll_var,
                  orient="horizontal", length=200).pack(padx=5, pady=3)

        rf = ttk.Frame(roll_frame)
        rf.pack(fill="x", padx=5, pady=2)
        ttk.Entry(rf, textvariable=self.roll_var, width=7).pack(side="left", padx=3)
        ttk.Label(rf, text="deg").pack(side="left")

        # --- Computed servo angles display ---
        calc_frame = ttk.Frame(self.frame)
        calc_frame.pack(fill="x", padx=5, pady=2)
        self.calc_label = ttk.Label(calc_frame, text="S1=0.0  S2=0.0", font=("Consolas", 8))
        self.calc_label.pack()

        # --- Preset buttons ---
        preset_frame = ttk.Frame(self.frame)
        preset_frame.pack(fill="x", padx=5, pady=3)
        ttk.Button(preset_frame, text="Zero", width=5,
                   command=lambda: self.set_preset(0, 0)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="P+60", width=5,
                   command=lambda: self.set_preset(60, 0)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="P-60", width=5,
                   command=lambda: self.set_preset(-60, 0)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="R+60", width=5,
                   command=lambda: self.set_preset(0, 60)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="R-60", width=5,
                   command=lambda: self.set_preset(0, -60)).pack(side="left", padx=2)

        # --- GO button ---
        self.go_btn = ttk.Button(self.frame, text="GO", command=self.send_position)
        self.go_btn.pack(fill="x", padx=5, pady=5, ipady=6)

        # --- Feedback ---
        fb_frame = ttk.LabelFrame(self.frame, text="Feedback")
        fb_frame.pack(fill="both", expand=True, padx=5, pady=3)

        self.feedback_text = tk.Text(fb_frame, height=4, width=30, state="disabled", font=("Consolas", 8))
        self.feedback_text.pack(fill="both", expand=True, padx=3, pady=3)

    def set_preset(self, pitch, roll):
        self.pitch_var.set(pitch)
        self.roll_var.set(roll)

    def send_position(self):
        import time
        pitch = max(-120.0, min(120.0, self.pitch_var.get()))
        roll = max(-120.0, min(120.0, self.roll_var.get()))
        self.pitch_var.set(pitch)
        self.roll_var.set(roll)

        s1 = max(-120.0, min(120.0, pitch + roll))
        s2 = max(-120.0, min(120.0, pitch - roll))
        self.calc_label.config(text=f"S1={s1:.1f}  S2={s2:.1f}")

        self.send_fn(f"5P{s1:.1f}\n")
        time.sleep(0.005)
        self.send_fn(f"6P{s2:.1f}\n")

    def append_feedback(self, line):
        self.feedback_text.config(state="normal")
        self.feedback_text.insert("end", line + "\n")
        self.feedback_text.see("end")
        if int(self.feedback_text.index("end-1c").split(".")[0]) > 100:
            self.feedback_text.delete("1.0", "2.0")
        self.feedback_text.config(state="disabled")


class ServoPanel:
    def __init__(self, parent, servo_idx, label, send_fn):
        self.servo_idx = servo_idx
        self.send_fn = send_fn

        self.frame = ttk.LabelFrame(parent, text=label)
        self.frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        ttk.Label(self.frame, text="MG995 TowerPro", font=("", 8, "italic")).pack(padx=5, pady=2)

        # --- Position slider ---
        pos_frame = ttk.LabelFrame(self.frame, text="Position (degrees)")
        pos_frame.pack(fill="x", padx=5, pady=5)

        self.pos_var = tk.DoubleVar(value=0.0)

        slider = ttk.Scale(pos_frame, from_=-120, to=120, variable=self.pos_var,
                           orient="horizontal", length=200)
        slider.pack(padx=5, pady=3)

        entry_frame = ttk.Frame(pos_frame)
        entry_frame.pack(fill="x", padx=5, pady=3)

        self.pos_entry = ttk.Entry(entry_frame, textvariable=self.pos_var, width=7)
        self.pos_entry.pack(side="left", padx=3)
        ttk.Label(entry_frame, text="deg").pack(side="left")

        # --- Preset buttons ---
        preset_frame = ttk.Frame(self.frame)
        preset_frame.pack(fill="x", padx=5, pady=3)
        for angle in [-120, -60, 0, 60, 120]:
            ttk.Button(preset_frame, text=f"{angle}°", width=4,
                       command=lambda a=angle: self.set_angle(a)).pack(side="left", padx=2)

        # --- GO button ---
        self.go_btn = ttk.Button(self.frame, text="GO", command=self.send_position)
        self.go_btn.pack(fill="x", padx=5, pady=5, ipady=6)

        # --- Feedback ---
        fb_frame = ttk.LabelFrame(self.frame, text="Feedback")
        fb_frame.pack(fill="both", expand=True, padx=5, pady=3)

        self.feedback_text = tk.Text(fb_frame, height=4, width=30, state="disabled", font=("Consolas", 8))
        self.feedback_text.pack(fill="both", expand=True, padx=3, pady=3)

    def set_angle(self, angle):
        self.pos_var.set(angle)

    def send_position(self):
        angle = max(-120.0, min(120.0, self.pos_var.get()))
        self.pos_var.set(angle)
        self.send_fn(f"{self.servo_idx}P{angle:.1f}\n")

    def append_feedback(self, line):
        self.feedback_text.config(state="normal")
        self.feedback_text.insert("end", line + "\n")
        self.feedback_text.see("end")
        if int(self.feedback_text.index("end-1c").split(".")[0]) > 100:
            self.feedback_text.delete("1.0", "2.0")
        self.feedback_text.config(state="disabled")


class RobotArmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Arm Control")
        self.ser = None

        # --- Connection frame ---
        conn_frame = ttk.LabelFrame(root, text="Connection")
        conn_frame.pack(fill="x", padx=10, pady=5)

        self.port_var = tk.StringVar()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo = ttk.Combobox(conn_frame, textvariable=self.port_var, values=ports, width=15)
        self.port_combo.pack(side="left", padx=5, pady=5)
        if ports:
            self.port_combo.current(0)

        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.toggle_connect)
        self.connect_btn.pack(side="left", padx=5)

        self.status_label = ttk.Label(conn_frame, text="Disconnected")
        self.status_label.pack(side="left", padx=10)

        # --- Recording frame ---
        rec_frame = ttk.LabelFrame(root, text="Recording")
        rec_frame.pack(fill="x", padx=10, pady=5)

        self.record_btn = ttk.Button(rec_frame, text="Record", command=self.start_recording)
        self.record_btn.pack(side="left", padx=5, pady=5)

        self.stop_btn = ttk.Button(rec_frame, text="Stop Record", command=self.stop_recording, state="disabled")
        self.stop_btn.pack(side="left", padx=5, pady=5)

        self.rec_label = ttk.Label(rec_frame, text="")
        self.rec_label.pack(side="left", padx=10)

        self.log_file = None

        # --- Demo frame ---
        demo_frame = ttk.LabelFrame(root, text="Demo Sequence")
        demo_frame.pack(fill="x", padx=10, pady=5)

        self.demo_run_btn = ttk.Button(demo_frame, text="Run Demo", command=self.start_demo)
        self.demo_run_btn.pack(side="left", padx=5, pady=5)

        self.demo_abort_btn = ttk.Button(demo_frame, text="Abort", command=self.abort_demo, state="disabled")
        self.demo_abort_btn.pack(side="left", padx=5, pady=5)

        ttk.Label(demo_frame, text="Speed:").pack(side="left", padx=(10, 2))
        self.demo_speed_var = tk.DoubleVar(value=1.0)
        self.demo_speed_scale = ttk.Scale(demo_frame, from_=0.25, to=2.0,
                                          variable=self.demo_speed_var,
                                          orient="horizontal", length=100)
        self.demo_speed_scale.pack(side="left", padx=2)
        self.demo_speed_label = ttk.Label(demo_frame, text="1.00x", width=5)
        self.demo_speed_label.pack(side="left")
        self.demo_speed_var.trace_add("write", lambda *_: self.demo_speed_label.config(
            text=f"{self.demo_speed_var.get():.2f}x"))

        self.demo_status_label = ttk.Label(demo_frame, text="Ready", font=("Consolas", 9))
        self.demo_status_label.pack(side="left", padx=15)

        self.demo_abort = False
        self.demo_thread = None
        self.demo_last_pos = {}
        self.motor_fb_pos = {1: None, 2: None, 3: None, 4: None}

        # --- Tabbed notebook ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # === Body Tab ===
        body_frame = ttk.Frame(self.notebook)
        self.notebook.add(body_frame, text="Body")

        self.panels = []
        motor_configs = [
            dict(idx=1, label="Base",     mtype="AK60",    cid=104),
            dict(idx=2, label="Shoulder", mtype="AK70/80", cid=2,  pos_min=-90.0, pos_max=0.0),
            dict(idx=3, label="Elbow",    mtype="AK70/80", cid=1,  pos_min=0.0, pos_max=math.degrees(0.4)),
            dict(idx=4, label="Linkage",  mtype="AK40",    cid=3,  pos_min=0.0, pos_max=math.degrees(1.782)),
        ]
        for cfg in motor_configs:
            panel = MotorPanel(body_frame, cfg["idx"], cfg["label"], cfg["mtype"], cfg["cid"],
                               self.send_cmd,
                               pos_min=cfg.get("pos_min"), pos_max=cfg.get("pos_max"))
            self.panels.append(panel)

        # === End Effector Tab ===
        ee_frame = ttk.Frame(self.notebook)
        self.notebook.add(ee_frame, text="End Effector")

        self.wrist_panel = WristDifferentialPanel(ee_frame, self.send_cmd)
        self.gripper_panel = ServoPanel(ee_frame, 7, "Gripper", self.send_cmd)

        # For feedback routing: index 5,6 → wrist, 7 → gripper
        self.servo_panels = [self.wrist_panel, self.wrist_panel, self.gripper_panel]

        # --- GO ALL servos button ---
        self.go_all_btn = ttk.Button(ee_frame, text="GO ALL", command=self.send_all_servos)
        self.go_all_btn.pack(fill="x", padx=10, pady=5, ipady=6)

        # --- Serial reader ---
        self.running = True
        self.read_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.read_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _set_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for panel in self.panels:
            panel.go_btn.config(state=state)
            panel.enable_btn.config(state=state)
            panel.origin_btn.config(state=state)
        self.wrist_panel.go_btn.config(state=state)
        self.gripper_panel.go_btn.config(state=state)
        self.go_all_btn.config(state=state)

    def _demo_set_status(self, text):
        self.root.after(0, self.demo_status_label.config, {"text": text})

    def start_demo(self):
        if not self.ser or not self.ser.is_open:
            self.demo_status_label.config(text="Not connected!")
            return
        self.demo_abort = False
        self.demo_run_btn.config(state="disabled")
        self.demo_abort_btn.config(state="normal")
        self.demo_speed_scale.config(state="disabled")
        self._set_controls_enabled(False)
        self.demo_last_pos = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.demo_thread = threading.Thread(target=self._demo_thread, daemon=True)
        self.demo_thread.start()

    def abort_demo(self):
        self.demo_abort = True

    def _demo_finish(self, status_text):
        self.root.after(0, self._set_controls_enabled, True)
        self.root.after(0, self.demo_run_btn.config, {"state": "normal"})
        self.root.after(0, self.demo_abort_btn.config, {"state": "disabled"})
        self.root.after(0, self.demo_speed_scale.config, {"state": "normal"})
        self._demo_set_status(status_text)

    def _demo_send(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write(cmd.encode())

    def _demo_log(self, msg):
        if self.log_file:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.log_file.write(f"{timestamp}  [DEMO] {msg}\n")
            self.log_file.flush()

    def _demo_thread(self):
        speed = self.demo_speed_var.get()
        if speed < 0.1:
            speed = 0.1

        # Inner Kp during demo — outer PID stays active for gravity compensation.
        # 2× the firmware defaults gives tighter trajectory tracking.
        DEMO_INNER_KP = {1: 3.0, 2: 5.0, 3: 5.0, 4: 5.0}
        KD_DEMO = {1: 0.5, 2: 1.2, 3: 0.5, 4: 0.5}

        saved_gains = {}
        try:
            # --- Phase 1: Save inner Kp/Kd and outer PID, apply demo gains ---
            self._demo_log("Demo start")
            self._demo_set_status("Preparing gains...")
            for motor in [1, 2, 3, 4]:
                panel = self.panels[motor - 1]
                saved_gains[motor] = {}
                for cmd in ["K", "D", "G", "H", "J"]:
                    if cmd in panel.sliders:
                        saved_gains[motor][cmd] = panel.sliders[cmd].get()
                self._demo_send(f"{motor}K{DEMO_INNER_KP[motor]:.4f}\n")
                time.sleep(0.005)
                self._demo_send(f"{motor}D{KD_DEMO[motor]:.4f}\n")
                time.sleep(0.005)
            time.sleep(0.1)

            if self.demo_abort:
                return

            # --- Phase 1b: Enable motors ---
            self._demo_set_status("Enabling motors...")
            self._demo_log("Enabling motors")
            for motor in [1, 2, 3, 4]:
                if self.demo_abort:
                    return
                self._demo_set_status(f"Enabling motor {motor}...")
                self._demo_log(f"Enable M{motor}")
                self._demo_send(f"{motor}E\n")
                self._demo_wait(2.0)

            if self.demo_abort:
                return

            # --- Phase 2: Capture current positions as demo zero (no origin set) ---
            self._demo_set_status("Reading start positions...")
            demo_offset = {}
            for motor in [1, 2, 3, 4]:
                pos = self.motor_fb_pos.get(motor)
                demo_offset[motor] = pos if pos is not None else 0.0
                self._demo_log(f"M{motor} start offset: {demo_offset[motor]:.4f} rad")

            for motor in [1, 2, 3, 4]:
                self._demo_send(f"{motor}P{demo_offset[motor]:.4f}\n")
                time.sleep(0.005)
                self._demo_send(f"{motor}V0.0\n")
                time.sleep(0.005)
                self.demo_last_pos[motor] = demo_offset[motor]
            self._demo_wait(0.5)

            if self.demo_abort:
                return

            # Zero outer PID gains — prevents inner+outer stiffness doubling
            # which causes oscillation when motor overshoots during trajectory.
            self._demo_log("Zeroing outer PID gains (G/H/J) for trajectory")
            for motor in [1, 2, 3, 4]:
                self._demo_send(f"{motor}G0.0\n")
                time.sleep(0.005)
                self._demo_send(f"{motor}H0.0\n")
                time.sleep(0.005)
                self._demo_send(f"{motor}J0.0\n")
                time.sleep(0.005)

            if self.demo_abort:
                return

            # --- Phase 3: Run demo segments ---
            for seg in DEMO_SEGMENTS:
                if self.demo_abort:
                    return
                name = seg["name"]
                duration = seg["duration"] / speed
                targets = seg["targets"]
                tff = seg.get("tff", {})

                for motor, val in tff.items():
                    self._demo_send(f"{motor}T{val:.4f}\n")
                    time.sleep(0.005)

                if not targets:
                    self._demo_set_status(f"Pause...")
                    self._demo_log(f"Pause ({duration:.1f}s)")
                    self._demo_wait(duration)
                else:
                    self._demo_set_status(name)
                    self._demo_log(f"Segment: {name} ({duration:.1f}s)")
                    self._run_segment(targets, duration, demo_offset)

                for motor in tff:
                    self._demo_send(f"{motor}T0.0000\n")
                    time.sleep(0.005)

            if self.demo_abort:
                return

            # --- Phase 4: Settle ---
            self._demo_set_status("Settling...")
            self._demo_log("Settling")
            for motor in [1, 2, 3, 4]:
                self._demo_send(f"{motor}P{demo_offset[motor]:.4f}\n")
                time.sleep(0.005)
                self._demo_send(f"{motor}V0.0\n")
                time.sleep(0.005)
                self.demo_last_pos[motor] = demo_offset[motor]
            self._demo_wait(1.0)

            self._demo_log("Demo complete")
            self._demo_finish("Demo complete")

        except Exception as e:
            self._demo_log(f"Error: {e}")
            self._demo_finish(f"Error: {e}")
        finally:
            # Restore inner Kp/Kd and outer PID gains (G/H/J), clear tff
            self._demo_log("Restoring gains")
            for motor in [1, 2, 3, 4]:
                self._demo_send(f"{motor}T0.0000\n")
                time.sleep(0.005)
            for motor, gains in saved_gains.items():
                for cmd, val in gains.items():
                    self._demo_send(f"{motor}{cmd}{val:.4f}\n")
                    time.sleep(0.005)
            if self.demo_abort:
                # Emergency stop: zero velocity, hold position
                self._demo_log("Demo aborted — holding position")
                for motor in [1, 2, 3, 4]:
                    self._demo_send(f"{motor}V0.0\n")
                    time.sleep(0.003)
                for motor in [1, 2, 3, 4]:
                    pos = self.demo_last_pos.get(motor, 0.0)
                    self._demo_send(f"{motor}P{pos:.4f}\n")
                    time.sleep(0.003)
                self._demo_finish("Aborted")

    def _demo_wait(self, duration):
        t0 = time.perf_counter()
        while not self.demo_abort:
            if time.perf_counter() - t0 >= duration:
                break
            time.sleep(0.020)

    def _run_segment(self, targets, duration, demo_offset=None):
        """Run a cosine-interpolated motion segment.
        targets: {motor_num: (start_deg, end_deg)}
        demo_offset: {motor_num: float} — added to all p_des values so targets are
                     relative to the arm's position when the demo started."""
        if demo_offset is None:
            demo_offset = {}
        DT = 0.050  # 20 Hz — gives the outer PID time to settle between updates
        t0 = time.perf_counter()
        next_tick = t0

        targets_rad = {}
        for motor, (start_deg, end_deg) in targets.items():
            offset = demo_offset.get(motor, 0.0)
            targets_rad[motor] = (math.radians(start_deg) + offset,
                                  math.radians(end_deg) + offset)

        while not self.demo_abort:
            elapsed = time.perf_counter() - t0
            if elapsed >= duration:
                break

            for motor, (start_rad, end_rad) in targets_rad.items():
                pos, vel = cosine_interp(elapsed, duration, start_rad, end_rad)
                self._demo_send(f"{motor}P{pos:.4f}\n")
                time.sleep(0.003)
                self._demo_send(f"{motor}V{vel:.4f}\n")
                time.sleep(0.003)
                self.demo_last_pos[motor] = pos

            next_tick += DT
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Send final position with zero velocity
        for motor, (start_rad, end_rad) in targets_rad.items():
            self._demo_send(f"{motor}P{end_rad:.4f}\n")
            time.sleep(0.003)
            self._demo_send(f"{motor}V0.0\n")
            time.sleep(0.003)
            self.demo_last_pos[motor] = end_rad

    def send_all_servos(self):
        self.wrist_panel.send_position()
        time.sleep(0.005)
        self.gripper_panel.send_position()
        time.sleep(0.005)

    def toggle_connect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None
            self.connect_btn.config(text="Connect")
            self.status_label.config(text="Disconnected")
        else:
            port = self.port_var.get()
            if not port:
                return
            try:
                self.ser = serial.Serial(port, 115200, timeout=0.1)
                time.sleep(0.05)
                # Freeze motors at current position to prevent stale commands
                # from a previous session causing unexpected motion
                self.ser.write(b"0X\n")
                self.connect_btn.config(text="Disconnect")
                self.status_label.config(text=f"Connected: {port}")
            except serial.SerialException as e:
                self.status_label.config(text=f"Error: {e}")

    def send_cmd(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write(cmd.encode())

    def read_serial(self):
        while self.running:
            if self.ser and self.ser.is_open:
                try:
                    line = self.ser.readline().decode(errors="replace").strip()
                    if line:
                        self.route_feedback(line)
                except Exception:
                    pass

    def start_recording(self):
        filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.txt")
        self.log_file = open(filename, "w")
        self.record_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.rec_label.config(text=f"Recording: {filename}")

    def stop_recording(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.record_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.rec_label.config(text="Stopped")

    def route_feedback(self, line):
        if self.log_file:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.log_file.write(f"{timestamp}  {line}\n")
            self.log_file.flush()

        routed = False
        # Check motor panels [M1]-[M4]
        for i, panel in enumerate(self.panels, start=1):
            tag = f"[M{i}]"
            if line.startswith(tag):
                body = line[len(tag):].strip()
                self.root.after(0, panel.append_feedback, body)
                # Parse position for demo feedback tracking
                m = re.search(r'pos=(-?\d+\.?\d*)\s*(rad|deg)', body)
                if m:
                    val = float(m.group(1))
                    if m.group(2) == "deg":
                        val = math.radians(val)
                    self.motor_fb_pos[i] = val
                routed = True
                break
        # Check servo panels [M5]-[M7]
        if not routed:
            for i, sp in enumerate(self.servo_panels, start=5):
                tag = f"[M{i}]"
                if line.startswith(tag):
                    self.root.after(0, sp.append_feedback, line[len(tag):].strip())
                    routed = True
                    break
        if not routed:
            self.root.after(0, self.panels[0].append_feedback, line)

    def on_close(self):
        self.demo_abort = True
        self.running = False
        if self.demo_thread and self.demo_thread.is_alive():
            self.demo_thread.join(timeout=2.0)
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RobotArmGUI(root)
    root.mainloop()
