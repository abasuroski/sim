"""Position-scheduled PD control primitives for the robot_master MuJoCo model.

The controller exposes PD gains in *joint-output* units.  Each schedule is a
second-order pole specification rather than a hard-coded pair of gains:

    kp(q) = I_eff(q) * omega_n(q)^2
    kd(q) = 2 * I_eff(q) * zeta * omega_n(q)

``omega_n`` is log-linearly interpolated from its low-position value to its
high-position value, so the target poles change continuously with the measured
joint position.  The diagonal of MuJoCo's joint-space mass matrix supplies
``I_eff``.  This is a deliberately diagonal, gain-scheduled PD controller; it
does not claim exact MIMO pole placement for the arm's coupled linkage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from robot_master_configuration import (
    SCHEDULED_PD_POSITION_ERROR_FOR_FULL_TORQUE_RAD,
    SCHEDULED_PD_VELOCITY_ERROR_FOR_FULL_TORQUE_RAD_S,
)


EPSILON = 1e-12


def actuator_name(model: mujoco.MjModel, actuator_id: int) -> str:
    """Return a human-readable actuator name."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    return name if name is not None else f"actuator_{actuator_id}"


def joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    """Return a human-readable joint name."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    return name if name is not None else f"joint_{joint_id}"


def actuator_gear(model: mujoco.MjModel, actuator_id: int) -> float:
    """Return the scalar actuator transmission from control to joint space."""
    gear = float(model.actuator_gear[actuator_id, 0])
    if not math.isfinite(gear) or abs(gear) <= EPSILON:
        raise ValueError(f"{actuator_name(model, actuator_id)} has an invalid zero transmission gear")
    return gear


def output_pd_gains(model: mujoco.MjModel, actuator_id: int) -> tuple[float, float]:
    """Read MuJoCo position-actuator gains in joint-output units."""
    gear = actuator_gear(model, actuator_id)
    kp = float(model.actuator_gainprm[actuator_id, 0] * gear**2)
    kd = float(-model.actuator_biasprm[actuator_id, 2] * gear**2)
    return kp, kd


def set_output_pd_gains(model: mujoco.MjModel, actuator_id: int, kp: float, kd: float) -> None:
    """Set a MuJoCo ``position`` actuator from joint-output PD gains.

    MuJoCo stores the actuator-side affine force law.  Applying the gear
    conversion here keeps schedule files independent of the AK40's 1.25:1
    reduction: all stored gains are N m/rad and N m s/rad at the output joint.
    """
    if not math.isfinite(kp) or kp <= 0.0:
        raise ValueError("kp must be finite and greater than zero")
    if not math.isfinite(kd) or kd < 0.0:
        raise ValueError("kd must be finite and zero or greater")

    gear = actuator_gear(model, actuator_id)
    actuator_kp = kp / gear**2
    actuator_kd = kd / gear**2
    model.actuator_gainprm[actuator_id, 0] = actuator_kp
    model.actuator_biasprm[actuator_id, 0] = 0.0
    model.actuator_biasprm[actuator_id, 1] = -actuator_kp
    model.actuator_biasprm[actuator_id, 2] = -actuator_kd


@dataclass(frozen=True)
class PositionSchedule:
    """Desired local closed-loop poles over one joint-position interval."""

    actuator: str
    joint: str
    position_min_rad: float
    position_max_rad: float
    natural_frequency_low_rad_s: float
    natural_frequency_high_rad_s: float
    damping_ratio: float

    def __post_init__(self) -> None:
        values = (
            self.position_min_rad,
            self.position_max_rad,
            self.natural_frequency_low_rad_s,
            self.natural_frequency_high_rad_s,
            self.damping_ratio,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all position-schedule values must be finite")
        if self.position_max_rad <= self.position_min_rad:
            raise ValueError("position_max_rad must be greater than position_min_rad")
        if self.natural_frequency_low_rad_s <= 0.0 or self.natural_frequency_high_rad_s <= 0.0:
            raise ValueError("natural frequencies must be greater than zero")
        if self.damping_ratio <= 0.0:
            raise ValueError("damping_ratio must be greater than zero")

    @property
    def position_span_rad(self) -> float:
        return self.position_max_rad - self.position_min_rad

    def normalized_position(self, position_rad: float) -> float:
        """Map a joint position to [0, 1], clamping outside the trained range."""
        return min(1.0, max(0.0, (position_rad - self.position_min_rad) / self.position_span_rad))

    def natural_frequency(self, position_rad: float) -> float:
        """Return a positive, smoothly position-scheduled natural frequency."""
        fraction = self.normalized_position(position_rad)
        low = math.log(self.natural_frequency_low_rad_s)
        high = math.log(self.natural_frequency_high_rad_s)
        return math.exp(low + fraction * (high - low))

    def desired_poles(self, position_rad: float) -> tuple[complex, complex]:
        """Return the continuous-time poles requested at ``position_rad``."""
        frequency = self.natural_frequency(position_rad)
        zeta = self.damping_ratio
        discriminant = complex(zeta * zeta - 1.0, 0.0) ** 0.5
        return (-zeta * frequency + frequency * discriminant, -zeta * frequency - frequency * discriminant)

    def gains(self, position_rad: float, effective_inertia_kg_m2: float) -> tuple[float, float, float]:
        """Return ``kp``, ``kd``, and omega_n for the measured position."""
        if not math.isfinite(effective_inertia_kg_m2) or effective_inertia_kg_m2 <= 0.0:
            raise ValueError("effective inertia must be finite and greater than zero")
        frequency = self.natural_frequency(position_rad)
        kp = effective_inertia_kg_m2 * frequency**2
        kd = 2.0 * effective_inertia_kg_m2 * self.damping_ratio * frequency
        return kp, kd, frequency

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "position_range_rad": [self.position_min_rad, self.position_max_rad],
            "natural_frequency_rad_s": [
                self.natural_frequency_low_rad_s,
                self.natural_frequency_high_rad_s,
            ],
            "damping_ratio": self.damping_ratio,
        }

    @classmethod
    def from_dict(cls, actuator: str, value: dict[str, Any]) -> "PositionSchedule":
        position_range = value["position_range_rad"]
        frequencies = value["natural_frequency_rad_s"]
        if len(position_range) != 2 or len(frequencies) != 2:
            raise ValueError(f"{actuator}: position_range_rad and natural_frequency_rad_s each require two values")
        return cls(
            actuator=actuator,
            joint=str(value["joint"]),
            position_min_rad=float(position_range[0]),
            position_max_rad=float(position_range[1]),
            natural_frequency_low_rad_s=float(frequencies[0]),
            natural_frequency_high_rad_s=float(frequencies[1]),
            damping_ratio=float(value["damping_ratio"]),
        )


@dataclass(frozen=True)
class GainUpdate:
    """The gain and pole request applied to one actuator at one control step."""

    actuator: str
    joint: str
    position_rad: float
    effective_inertia_kg_m2: float
    kp_output_nm_per_rad: float
    kd_output_nm_s_per_rad: float
    kp_output_limit_nm_per_rad: float
    kd_output_limit_nm_s_per_rad: float
    natural_frequency_rad_s: float
    poles_rad_s: tuple[complex, complex]


class MasterController:
    """Apply position-scheduled PD gains to MuJoCo position actuators.

    Call :meth:`update` after ``mj_forward`` and immediately before every
    ``mj_step``.  This class only changes actuator gains; caller-owned
    ``data.ctrl`` remains the desired position command in normal MuJoCo units.
    """

    def __init__(
        self,
        schedules: Iterable[PositionSchedule],
        minimum_inertia_kg_m2: float = 1e-6,
        position_error_for_full_torque_rad: float = SCHEDULED_PD_POSITION_ERROR_FOR_FULL_TORQUE_RAD,
        velocity_error_for_full_torque_rad_s: float = SCHEDULED_PD_VELOCITY_ERROR_FOR_FULL_TORQUE_RAD_S,
    ) -> None:
        self.schedules = tuple(schedules)
        if not self.schedules:
            raise ValueError("at least one actuator schedule is required")
        if minimum_inertia_kg_m2 <= 0.0 or not math.isfinite(minimum_inertia_kg_m2):
            raise ValueError("minimum_inertia_kg_m2 must be finite and greater than zero")
        if position_error_for_full_torque_rad <= 0.0 or not math.isfinite(position_error_for_full_torque_rad):
            raise ValueError("position_error_for_full_torque_rad must be finite and greater than zero")
        if velocity_error_for_full_torque_rad_s <= 0.0 or not math.isfinite(velocity_error_for_full_torque_rad_s):
            raise ValueError("velocity_error_for_full_torque_rad_s must be finite and greater than zero")
        self.minimum_inertia_kg_m2 = minimum_inertia_kg_m2
        self.position_error_for_full_torque_rad = position_error_for_full_torque_rad
        self.velocity_error_for_full_torque_rad_s = velocity_error_for_full_torque_rad_s
        self._bound_model: mujoco.MjModel | None = None
        self._bindings: tuple[tuple[PositionSchedule, int, int, int, float, float], ...] = ()
        self._mass_matrix: np.ndarray | None = None

    @classmethod
    def from_json(cls, path: Path | str) -> "MasterController":
        with Path(path).open(encoding="utf-8") as input_file:
            document = json.load(input_file)
        actuator_data = document.get("actuators")
        if not isinstance(actuator_data, dict) or not actuator_data:
            raise ValueError("schedule file must contain a non-empty 'actuators' object")
        safety_limits = document.get("safety_limits", {})
        return cls(
            (PositionSchedule.from_dict(name, value) for name, value in actuator_data.items()),
            position_error_for_full_torque_rad=float(
                safety_limits.get("position_error_for_full_torque_rad", SCHEDULED_PD_POSITION_ERROR_FOR_FULL_TORQUE_RAD)
            ),
            velocity_error_for_full_torque_rad_s=float(
                safety_limits.get(
                    "velocity_error_for_full_torque_rad_s", SCHEDULED_PD_VELOCITY_ERROR_FOR_FULL_TORQUE_RAD_S
                )
            ),
        )

    def to_document(self, **metadata: Any) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema_version": 1,
            "controller": "position_scheduled_pd",
            "units": {
                "position": "rad",
                "natural_frequency": "rad/s",
                "kp_output": "N m/rad",
                "kd_output": "N m s/rad",
            },
            "safety_limits": {
                "position_error_for_full_torque_rad": self.position_error_for_full_torque_rad,
                "velocity_error_for_full_torque_rad_s": self.velocity_error_for_full_torque_rad_s,
            },
            "actuators": {schedule.actuator: schedule.to_dict() for schedule in self.schedules},
        }
        document.update(metadata)
        return document

    def bind(self, model: mujoco.MjModel) -> None:
        """Resolve names once and allocate the MuJoCo mass-matrix workspace."""
        bindings: list[tuple[PositionSchedule, int, int, int, float, float]] = []
        for schedule in self.schedules:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, schedule.actuator)
            if actuator_id < 0:
                raise ValueError(f"unknown scheduled actuator {schedule.actuator!r}")
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            resolved_joint = joint_name(model, joint_id)
            if resolved_joint != schedule.joint:
                raise ValueError(
                    f"{schedule.actuator!r} drives {resolved_joint!r}, not scheduled joint {schedule.joint!r}"
                )
            output_torque_limit = max(abs(value) for value in model.actuator_forcerange[actuator_id]) * abs(
                actuator_gear(model, actuator_id)
            )
            bindings.append(
                (
                    schedule,
                    actuator_id,
                    int(model.jnt_qposadr[joint_id]),
                    int(model.jnt_dofadr[joint_id]),
                    output_torque_limit / self.position_error_for_full_torque_rad,
                    output_torque_limit / self.velocity_error_for_full_torque_rad_s,
                )
            )
        self._bound_model = model
        self._bindings = tuple(bindings)
        self._mass_matrix = np.empty((model.nv, model.nv), dtype=np.float64)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, GainUpdate]:
        """Schedule and apply output-space PD gains for the present state."""
        if self._bound_model is not model:
            self.bind(model)
        assert self._mass_matrix is not None
        mujoco.mj_fullM(model, data, self._mass_matrix)

        applied: dict[str, GainUpdate] = {}
        for schedule, actuator_id, qpos_address, dof_address, kp_limit, kd_limit in self._bindings:
            position = float(data.qpos[qpos_address])
            inertia = max(self.minimum_inertia_kg_m2, float(self._mass_matrix[dof_address, dof_address]))
            kp, kd, frequency = schedule.gains(position, inertia)
            kp = min(kp, kp_limit)
            kd = min(kd, kd_limit)
            set_output_pd_gains(model, actuator_id, kp, kd)
            applied[schedule.actuator] = GainUpdate(
                actuator=schedule.actuator,
                joint=schedule.joint,
                position_rad=position,
                effective_inertia_kg_m2=inertia,
                kp_output_nm_per_rad=kp,
                kd_output_nm_s_per_rad=kd,
                kp_output_limit_nm_per_rad=kp_limit,
                kd_output_limit_nm_s_per_rad=kd_limit,
                natural_frequency_rad_s=frequency,
                poles_rad_s=schedule.desired_poles(position),
            )
        return applied
