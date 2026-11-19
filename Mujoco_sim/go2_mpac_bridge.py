"""go2_mpac_bridge.py — MuJoCo physics/DDS bridge for MPAC's --io_mode=mujoco.

MPAC's mujoco_io subscribes to rt/lowcmd and publishes rt/lowstate on
DDS domain 1, loopback ("lo").  This bridge mirrors that:
  - subscribes to rt/lowcmd and applies PD + tau_ff to the 12 leg actuators
  - runs the combined Go2+arm physics at 2 kHz (dt=0.0005 s)
  - publishes rt/lowstate after every physics step
  - exposes sync_to_data() so the arm controller reads the live dog pose

The arm is frozen at the hardware_zero keyframe during the physics sim; its
mass (once real values are filled in) acts as the balance controller's payload.

Joint index mapping between MPAC sim-mode DDS indices and MuJoCo joint names:
  DDS  0-2 : FL_hip_joint, FL_thigh_joint, FL_calf_joint
  DDS  3-5 : FR_hip_joint, FR_thigh_joint, FR_calf_joint
  DDS  6-8 : RL_hip_joint, RL_thigh_joint, RL_calf_joint
  DDS  9-11: RR_hip_joint, RR_thigh_joint, RR_calf_joint

Foot-force mapping (sim mode: FL_=0, FR_=1, RL_=2, RR_=3).

Dependencies (Linux only):
  pip install unitree_sdk2_python   (Unitree Python SDK for DDS)
  pip install mujoco numpy

Usage:
  bridge = Go2MpacBridge(Path("robot_master_urdf_quadruped/mjcf/go2_with_arm.xml"))
  bridge.start()
  ...
  bridge.sync_to_data(arm_data, arm_model)   # call each control tick
  bridge.stop()
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DDS topics (must match mujoco_io.cpp)
_TOPIC_LOWCMD   = "rt/lowcmd"
_TOPIC_LOWSTATE = "rt/lowstate"
_DDS_DOMAIN     = 1    # loopback ("lo") for MPAC --io_mode=mujoco
_DDS_IFACE      = "lo"

# Physics timestep MPAC expects (500 µs)
PHYSICS_DT = 0.0005

# PosStopF / VelStopF sentinels from mujoco_io.cpp
_POS_STOP_F = 2.146e9
_VEL_STOP_F = 16000.0

# MPAC sim-mode DDS index → Go2 joint name
_DDS_JOINT_NAMES: list[str] = [
    "FL_hip_joint",  "FL_thigh_joint",  "FL_calf_joint",   # 0–2
    "FR_hip_joint",  "FR_thigh_joint",  "FR_calf_joint",   # 3–5
    "RL_hip_joint",  "RL_thigh_joint",  "RL_calf_joint",   # 6–8
    "RR_hip_joint",  "RR_thigh_joint",  "RR_calf_joint",   # 9–11
]

# Foot body names → DDS foot_force index (sim mode: FL=0, FR=1, RL=2, RR=3)
_FOOT_BODIES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


class Go2MpacBridge:
    """MuJoCo physics simulation bridge for MPAC --io_mode=mujoco.

    Call start() before the arm controller's main loop.
    Call sync_to_data(data, model) each control tick to update the arm
    controller's combined model with the live dog pose.
    Call stop() on exit.
    """

    def __init__(self, model_path: Path, *, physics_dt: float = PHYSICS_DT) -> None:
        self._physics_dt = physics_dt

        # --- Physics model (separate from arm controller's model) ---
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data  = mujoco.MjData(self._model)

        # Override timestep to what MPAC expects
        self._model.opt.timestep = physics_dt

        keyframe_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_KEY, "hardware_zero")
        if keyframe_id < 0:
            raise RuntimeError("Combined model has no 'hardware_zero' keyframe.")
        mujoco.mj_resetDataKeyframe(self._model, self._data, keyframe_id)
        mujoco.mj_forward(self._model, self._data)

        # Keep arm at keyframe; only leg ctrl[0:12] will change
        self._keyframe_ctrl = self._model.key_ctrl[keyframe_id].copy()

        # --- Joint maps (by name, robust to ordering changes) ---
        self._jnt_qposadr: dict[str, int] = {}   # joint name → qpos start
        self._jnt_dofadr:  dict[str, int] = {}   # joint name → dof start
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self._jnt_qposadr[name] = int(self._model.jnt_qposadr[i])
                self._jnt_dofadr[name]  = int(self._model.jnt_dofadr[i])

        # --- Actuator map: joint name → ctrl index ---
        self._jnt_to_act: dict[str, int] = {}
        for i in range(self._model.nu):
            jid = int(self._model.actuator_trnid[i, 0])
            if jid >= 0:
                jname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if jname:
                    self._jnt_to_act[jname] = i

        # --- Freejoint address (base_link) ---
        # The unnamed freejoint on base_link is joint 0 (first declared joint).
        base_link_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if base_link_id < 0:
            raise RuntimeError("Combined model has no 'base_link' body.")
        # Find the freejoint of base_link by traversing its joints
        self._free_qposadr = -1
        self._free_dofadr  = -1
        for jid in range(self._model.njnt):
            if (self._model.jnt_bodyid[jid] == base_link_id
                    and self._model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE):
                self._free_qposadr = int(self._model.jnt_qposadr[jid])
                self._free_dofadr  = int(self._model.jnt_dofadr[jid])
                break
        if self._free_qposadr < 0:
            raise RuntimeError("base_link has no freejoint.")

        # --- Foot body IDs ---
        self._foot_body_ids: list[int] = []
        for name in _FOOT_BODIES:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(f"Foot body '{name}' not found in model.")
            self._foot_body_ids.append(bid)

        # --- Shared state (bridge thread → arm controller) ---
        self._state_lock = threading.Lock()
        # 7 freejoint qpos + 12 leg qpos = 19 values
        n_leg_qpos = sum(1 for n in _DDS_JOINT_NAMES)
        self._go2_qpos = np.zeros(7 + n_leg_qpos)   # [freejoint(7), legs(12)]
        self._go2_qvel = np.zeros(6 + n_leg_qpos)   # [freejoint(6), legs(12)]

        # --- Latest lowcmd from MPAC ---
        self._cmd_lock   = threading.Lock()
        self._latest_cmd = None   # will be set by DDS callback

        # --- DDS (initialised in start()) ---
        self._publisher  = None
        self._subscriber = None

        self._thread  : threading.Thread | None = None
        self._running = False
        print("[bridge] Go2MpacBridge initialised; "
              f"freejoint qposadr={self._free_qposadr}, dofadr={self._free_dofadr}")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Initialise DDS and start the physics loop thread."""
        self._init_dds()
        self._running = True
        self._thread = threading.Thread(target=self._physics_loop, name="go2-bridge", daemon=True)
        self._thread.start()
        print("[bridge] Physics loop started.")

    def stop(self) -> None:
        """Stop the physics loop and clean up."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[bridge] Stopped.")

    def sync_to_data(self, data: mujoco.MjData, model: mujoco.MjModel) -> None:
        """Copy the bridge's live Go2 state into the arm controller's data.

        Call this once per control tick, before mj_forward(model, data).
        """
        with self._state_lock:
            go2_qpos = self._go2_qpos.copy()
            go2_qvel = self._go2_qvel.copy()

        # Freejoint (7 qpos, 6 dof)
        data.qpos[self._free_qposadr : self._free_qposadr + 7] = go2_qpos[:7]
        data.qvel[self._free_dofadr  : self._free_dofadr  + 6] = go2_qvel[:6]

        # Leg joints
        for i, jname in enumerate(_DDS_JOINT_NAMES):
            if jname not in self._jnt_qposadr:
                continue
            # Look up the address in the ARM controller's model (same joint names)
            arm_qposadr = _joint_qposadr(model, jname)
            arm_dofadr  = _joint_dofadr(model, jname)
            if arm_qposadr >= 0:
                data.qpos[arm_qposadr] = go2_qpos[7 + i]
            if arm_dofadr >= 0:
                data.qvel[arm_dofadr]  = go2_qvel[6 + i]

    def get_base_height_m(self) -> float:
        """Return the Go2 base_link z position from the physics sim."""
        with self._state_lock:
            return float(self._go2_qpos[2])

    # -----------------------------------------------------------------------
    # DDS
    # -----------------------------------------------------------------------

    def _init_dds(self) -> None:
        try:
            from unitree_sdk2py.core.channel import (  # type: ignore
                ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
            from unitree_sdk2py.idl.unitree_go.msg.dds_.LowCmd_ import LowCmd_    # type: ignore
            from unitree_sdk2py.idl.unitree_go.msg.dds_.LowState_ import LowState_  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "unitree_sdk2_python is required for --go2 mpac.  "
                "Install it on your Linux machine with:  pip install unitree_sdk2_python"
            ) from exc

        self._LowState_ = LowState_

        ChannelFactoryInitialize(_DDS_DOMAIN, _DDS_IFACE)

        self._publisher = ChannelPublisher(_TOPIC_LOWSTATE, LowState_)
        self._publisher.Init()

        self._subscriber = ChannelSubscriber(_TOPIC_LOWCMD, LowCmd_)
        self._subscriber.Init(self._on_lowcmd, 10)
        print(f"[bridge] DDS ready on domain {_DDS_DOMAIN} iface '{_DDS_IFACE}'.")

    def _on_lowcmd(self, msg: object) -> None:
        with self._cmd_lock:
            self._latest_cmd = msg

    # -----------------------------------------------------------------------
    # Physics loop
    # -----------------------------------------------------------------------

    def _physics_loop(self) -> None:
        dt = self._physics_dt
        last_print = 0.0
        while self._running:
            t0 = time.perf_counter()

            # Apply latest MPAC command to leg motors; arm held at keyframe
            with self._cmd_lock:
                cmd = self._latest_cmd
            self._apply_command(cmd)

            # Step physics
            mujoco.mj_step(self._model, self._data)

            # Update shared state
            with self._state_lock:
                self._go2_qpos[:7] = self._data.qpos[
                    self._free_qposadr : self._free_qposadr + 7]
                self._go2_qvel[:6] = self._data.qvel[
                    self._free_dofadr : self._free_dofadr + 6]
                for i, jname in enumerate(_DDS_JOINT_NAMES):
                    if jname in self._jnt_qposadr:
                        self._go2_qpos[7 + i] = self._data.qpos[self._jnt_qposadr[jname]]
                        self._go2_qvel[6 + i] = self._data.qvel[self._jnt_dofadr[jname]]

            # Publish LowState back to MPAC
            if self._publisher is not None:
                self._publisher.Write(self._build_lowstate())

            # Throttle to physics_dt
            elapsed = time.perf_counter() - t0
            remaining = dt - elapsed
            if remaining > 1e-6:
                time.sleep(remaining)

            if time.perf_counter() - last_print > 5.0:
                z = float(self._data.qpos[self._free_qposadr + 2])
                print(f"[bridge] base_link z={z:.3f} m  time={self._data.time:.1f} s")
                last_print = time.perf_counter()

    def _apply_command(self, cmd: object | None) -> None:
        """Write ctrl from MPAC's LowCmd_ into the bridge's physics data."""
        # Reset to keyframe ctrl every tick (holds arm in place)
        self._data.ctrl[:] = self._keyframe_ctrl

        if cmd is None:
            return

        for dds_i, jname in enumerate(_DDS_JOINT_NAMES):
            if jname not in self._jnt_qposadr or jname not in self._jnt_to_act:
                continue

            motor = cmd.motor_cmd[dds_i]
            q_act  = float(self._data.qpos[self._jnt_qposadr[jname]])
            dq_act = float(self._data.qvel[self._jnt_dofadr[jname]])

            q_des  = float(motor.q)
            dq_des = float(motor.dq)
            kp     = float(motor.kp)
            kd     = float(motor.kd)
            tau_ff = float(motor.tau)

            # Sentinel values → disable that term
            if q_des  >= _POS_STOP_F:
                q_des = q_act;  kp = 0.0
            if dq_des >= _VEL_STOP_F:
                dq_des = 0.0;   kd = 0.0

            tau = tau_ff + kp * (q_des - q_act) + kd * (dq_des - dq_act)
            self._data.ctrl[self._jnt_to_act[jname]] = tau

    def _build_lowstate(self) -> object:
        """Construct a LowState_ from the current physics sim state."""
        state = self._LowState_()

        # --- IMU ---
        # MuJoCo freejoint qpos[3:7] = [qw, qx, qy, qz]
        q = self._data.qpos[self._free_qposadr + 3 : self._free_qposadr + 7]
        state.imu_state.quaternion[0] = float(q[0])   # w
        state.imu_state.quaternion[1] = float(q[1])   # x
        state.imu_state.quaternion[2] = float(q[2])   # y
        state.imu_state.quaternion[3] = float(q[3])   # z

        # gyroscope: world-frame angular velocity (mujoco_io.cpp rotates it to body)
        w_world = self._data.qvel[self._free_dofadr + 3 : self._free_dofadr + 6]
        state.imu_state.gyroscope[0] = float(w_world[0])
        state.imu_state.gyroscope[1] = float(w_world[1])
        state.imu_state.gyroscope[2] = float(w_world[2])

        # accelerometer: specific force in body frame = R^T(a_world − g_world)
        a_world = self._data.qacc[self._free_dofadr : self._free_dofadr + 3]
        g = self._model.opt.gravity              # [0, 0, -9.81]
        sf_world = np.array(a_world) - np.array(g)
        R = _quat_to_mat(q)                      # world→body rotation
        sf_body = R.T @ sf_world
        state.imu_state.accelerometer[0] = float(sf_body[0])
        state.imu_state.accelerometer[1] = float(sf_body[1])
        state.imu_state.accelerometer[2] = float(sf_body[2])

        # --- Motor state ---
        # DDS sim-mode ordering: FL0-2, FR0-2, RL0-2, RR0-2
        for dds_i, jname in enumerate(_DDS_JOINT_NAMES):
            if jname not in self._jnt_qposadr:
                continue
            act_i = self._jnt_to_act.get(jname, -1)
            q_val  = float(self._data.qpos[self._jnt_qposadr[jname]])
            dq_val = float(self._data.qvel[self._jnt_dofadr[jname]])
            tau_val = float(self._data.actuator_force[act_i]) if act_i >= 0 else 0.0
            state.motor_state[dds_i].q       = q_val
            state.motor_state[dds_i].dq      = dq_val
            state.motor_state[dds_i].tau_est = tau_val

        # --- Foot forces ---
        # cfrc_ext[body_id] is [fx,fy,fz,tx,ty,tz] in world frame after mj_step.
        # Foot force is the z component of contact force on the foot body.
        for fi, body_id in enumerate(self._foot_body_ids):
            fz = float(self._data.cfrc_ext[body_id, 2])
            state.foot_force[fi] = max(0, int(fz))

        return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion [w, x, y, z] → 3×3 rotation matrix (world→body)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _joint_qposadr(model: mujoco.MjModel, jname: str) -> int:
    """Return qpos start address of a named joint, or -1 if not found."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return int(model.jnt_qposadr[jid]) if jid >= 0 else -1


def _joint_dofadr(model: mujoco.MjModel, jname: str) -> int:
    """Return dof start address of a named joint, or -1 if not found."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return int(model.jnt_dofadr[jid]) if jid >= 0 else -1


# ---------------------------------------------------------------------------
# CLI smoke-test (no MPAC needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Smoke-test: run Go2 physics without DDS.")
    parser.add_argument("--model", default="robot_master_urdf_quadruped/mjcf/go2_with_arm.xml")
    parser.add_argument("--seconds", type=float, default=5.0, help="how long to run (default 5 s)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    # Run without DDS: create bridge, skip init_dds, run loop directly
    bridge = Go2MpacBridge(model_path)
    bridge._running = True
    print(f"Running physics-only smoke test for {args.seconds} s …")
    t_end = time.perf_counter() + args.seconds
    steps = 0
    while time.perf_counter() < t_end:
        bridge._apply_command(None)
        mujoco.mj_step(bridge._model, bridge._data)
        steps += 1
        z = float(bridge._data.qpos[bridge._free_qposadr + 2])
        if steps % 10000 == 0:
            print(f"  step {steps:7d}  t={bridge._data.time:.3f} s  base_z={z:.4f} m")

    print(f"Done. {steps} steps in {args.seconds} s = {steps/args.seconds:.0f} Hz.")
    print(f"Final base_link z = {float(bridge._data.qpos[bridge._free_qposadr + 2]):.4f} m")
