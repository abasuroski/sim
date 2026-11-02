"""Position inverse kinematics for the robot_master gripper.

Solves for the four driven joint references (revolute_1-4, joint side, the
same coordinates ``TrapezoidalReferenceLimiter.set_requested_position`` takes)
that put the tool point at a requested position in the base frame.

Kinematic model (identical to the controller's reference display,
``_apply_reference_state``):
  * driven joints are set to their references;
  * the passive four-bar followers are offset from the hardware_zero keyframe
    by ``scale * (revolute_4 - keyframe revolute_4)`` (``CONSTRAINED_JOINTS``);
  * every other joint (revolute_12 wrist servo, revolute_14 loop closure)
    stays at its keyframe value.  The controller never commands the wrist
    servo, so the real wrist must be at the keyframe angle for the tool point
    to land where the model says.

The linear four-bar coupling leaves a loop-closure gap of up to 1.7 mm away
from hardware_zero; against settled MuJoCo physics the tool point differs by
at most ~1.2 mm (at revolute_4 near 0).  That is a property of the coupling
model, not of this solver, whose own residual is below IK_ACCEPT_ERROR_M.

The four joints give one redundant degree of freedom for a 3-D position.  The
solver picks the solution closest (Euclidean, joint radians) to the start
references: damped least squares with a null-space pull toward the start,
seeded from the start; if that fails, seeds at increasing distances from the
start, then uniform random ones, and the closest passing solution wins.

Every accepted solution is inside the actuator control ranges, joint ranges
and follower ranges, and keeps every moving mesh vertex at least
IK_TABLE_CLEARANCE_M above the table top, at the target pose and along the
limiter's rest-to-rest path to it (except IK_CLEARANCE_IGNORE_BODIES, a stray
CAD part).  The table top is treated as an infinite plane, which is
conservative where the arm reaches past the table's edge.  The MJCF has no collision geometry for
the arm links or gripper, so self-collision is NOT checked; instead no joint
may change by more than IK_MAX_JOINT_CHANGE_RAD in one solve.

Command-line use (simulation only, no hardware):
    python inverse_kinematics.py <x> <y> <z>        # metres, base frame; opens the viewer
    python inverse_kinematics.py <x> <y> <z> --no-view
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import mujoco
import numpy as np

from robot_master_configuration import (
    HARDWARE_ZERO_ACTUATOR_CTRL_RAD,
    HARDWARE_ZERO_KEYFRAME_NAME,
    IK_ACCEPT_ERROR_M,
    IK_BASE_BODY,
    IK_CLEARANCE_IGNORE_BODIES,
    IK_MAX_JOINT_CHANGE_RAD,
    IK_TABLE_BODY,
    IK_TABLE_CLEARANCE_M,
    IK_TOOL_BODY,
    IK_TOOL_POINT_IN_BODY_M,
    TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2,
    TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S,
)
from trapezoidal_motion import trapezoidal_duration, trapezoidal_position


# Solver settings.  Errors are in metres, steps in joint radians.
SOLVER_TOLERANCE_M = 1e-6          # stop iterating below this position error
SOLVER_MAX_ITERATIONS = 400
SOLVER_DAMPING_M = 0.01            # DLS damping; arm Jacobian columns are ~0.1-0.3 m/rad
SOLVER_MAX_STEP_RAD = 0.2          # largest joint change per iteration
SOLVER_NULLSPACE_GAIN = 0.2        # pull toward the start pose, per iteration
SOLVER_FALLBACK_SEEDS = 32         # uniform random seeds, the last resort
SOLVER_LOCAL_SEED_SIGMAS_RAD = (0.25, 0.5, 1.0, 2.0)   # seed tiers around the start pose ...
SOLVER_LOCAL_SEEDS_PER_TIER = 8                        # ... this many seeds each
SOLVER_STALL_ITERATIONS = 25       # stop a seed after this many iterations ...
SOLVER_STALL_RELATIVE = 1e-3       # ... without this relative error improvement
PATH_CHECK_STEP_S = 0.02           # path clearance sample spacing (limiter time)


@dataclass
class IKResult:
    """Outcome of one solve.  ``joint_targets`` is only meaningful on success."""

    success: bool
    message: str
    target_base_m: np.ndarray
    joint_targets: dict[str, float] = field(default_factory=dict)   # actuator -> joint rad
    achieved_base_m: np.ndarray | None = None
    error_m: float = math.inf
    move_duration_s: float = 0.0


class ArmIK:
    """Tool-point position IK on the controller's kinematic arm model.

    Uses its own ``MjData``, so it never disturbs the controller's live data
    and can run in another thread (the model is only read).  ``model_path``
    must be the file ``model`` was loaded from; it is recompiled once to get
    the meshes' convex hulls for the clearance check.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        model_path: str | Path,
        actuator_to_joint: dict[str, str],
        constrained_joints: dict[str, tuple[str, float]],
        max_velocity_rad_s: float = TRAPEZOIDAL_MAX_JOINT_VELOCITY_RAD_S,
        max_acceleration_rad_s2: float = TRAPEZOIDAL_MAX_JOINT_ACCELERATION_RAD_S2,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.max_velocity_rad_s = max_velocity_rad_s
        self.max_acceleration_rad_s2 = max_acceleration_rad_s2

        keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HARDWARE_ZERO_KEYFRAME_NAME)
        if keyframe_id < 0:
            raise ValueError(f"MJCF has no {HARDWARE_ZERO_KEYFRAME_NAME!r} keyframe")
        self.keyframe_qpos = model.key_qpos[keyframe_id].copy()

        def joint_id(name: str) -> int:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint {name!r} is absent from this MJCF")
            if model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(f"joint {name!r} is not a 1-DOF joint")
            return jid

        # Driven joints, in actuator order given by the caller.
        self.actuators = list(actuator_to_joint)
        n = len(self.actuators)
        self._qadr = np.zeros(n, dtype=int)
        self._dofadr = np.zeros(n, dtype=int)
        lower = np.full(n, -math.inf)
        upper = np.full(n, math.inf)
        joint_index: dict[str, int] = {}
        first_joint = -1
        for i, actuator in enumerate(self.actuators):
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            if act_id < 0:
                raise ValueError(f"actuator {actuator!r} is absent from this MJCF")
            jname = actuator_to_joint[actuator]
            jid = joint_id(jname)
            if model.actuator_trnid[act_id, 0] != jid:
                raise ValueError(f"actuator {actuator!r} does not drive joint {jname!r}")
            joint_index[jname] = i
            if i == 0:
                first_joint = jid
            self._qadr[i] = model.jnt_qposadr[jid]
            self._dofadr[i] = model.jnt_dofadr[jid]
            # Control range is in actuator coordinates (motor side for AK40):
            # joint = ctrl / gear.  It bounds the solution even where MuJoCo
            # does not enforce it (revolute_1 and revolute_3 compile with
            # ctrllimited false but declare +-2 pi).
            lo_ctrl, hi_ctrl = (float(c) for c in model.actuator_ctrlrange[act_id])
            if lo_ctrl < hi_ctrl:
                gear = float(model.actuator_gear[act_id, 0])
                a, b = sorted(float(c) / gear for c in model.actuator_ctrlrange[act_id])
                lower[i], upper[i] = max(lower[i], a), min(upper[i], b)
            if model.jnt_limited[jid]:
                lower[i] = max(lower[i], float(model.jnt_range[jid, 0]))
                upper[i] = min(upper[i], float(model.jnt_range[jid, 1]))

        # Followers: qpos = keyframe + scale * (source - keyframe source).
        self._followers: list[tuple[int, int, int, float, float, float]] = []
        for follower, (source, scale) in constrained_joints.items():
            if source not in joint_index:
                raise ValueError(f"follower {follower!r} has undriven source {source!r}")
            fid = joint_id(follower)
            src = joint_index[source]
            f_qadr = int(model.jnt_qposadr[fid])
            src_kf = float(self.keyframe_qpos[self._qadr[src]])
            f_kf = float(self.keyframe_qpos[f_qadr])
            self._followers.append((src, f_qadr, int(model.jnt_dofadr[fid]), float(scale), src_kf, f_kf))
            # The follower's own range limits its source.
            if model.jnt_limited[fid] and scale != 0.0:
                a, b = sorted(src_kf + (float(r) - f_kf) / scale for r in model.jnt_range[fid])
                lower[src], upper[src] = max(lower[src], a), min(upper[src], b)
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise ValueError("every driven joint needs a finite control or joint range")
        if np.any(lower > upper):
            raise ValueError(f"empty joint range for {[self.actuators[i] for i in np.where(lower > upper)[0]]}")
        self.lower = lower
        self.upper = upper

        # Tool point and base frame.
        self._tool_body = self._body_id(IK_TOOL_BODY)
        self._tool_point = np.asarray(IK_TOOL_POINT_IN_BODY_M, dtype=float)
        self._base_body = self._body_id(IK_BASE_BODY)
        self._require_fixed(self._base_body)
        table_body = self._body_id(IK_TABLE_BODY)
        self._require_fixed(table_body)

        # Fixed bodies do not move, so their frames can be read at any pose.
        self._set_driven(self.home_references_array())
        self.base_pos = self.data.xpos[self._base_body].copy()
        self.base_rot = self.data.xmat[self._base_body].reshape(3, 3).copy()
        self.table_top_z_world = float(self._body_vertices_world(table_body)[:, 2].max())

        # Convex-hull vertices of every body that moves with the arm (the
        # subtree of the first driven joint), in body coordinates, for the
        # clearance check.  A body's lowest point is always a hull vertex, so
        # this is exact, and ~10x fewer points than the full meshes.
        self._hull_index = self._mesh_hull_indices(model, model_path)
        root = int(model.jnt_bodyid[first_joint])
        ignored = {self._body_id(name) for name in IK_CLEARANCE_IGNORE_BODIES}
        bodies, chunks = [], []
        for body in range(model.nbody):
            if body not in ignored and self._is_in_subtree(body, root):
                verts = self._body_vertices_local(body, hull_only=True)
                if verts is not None:
                    bodies.append(np.full(len(verts), body))
                    chunks.append(verts)
        self._moving_vertex_body = np.concatenate(bodies)
        self._moving_vertices = np.vstack(chunks)

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body {name!r} is absent from this MJCF")
        return bid

    def _require_fixed(self, body: int) -> None:
        b = body
        while b != 0:
            if self.model.body_jntnum[b] != 0:
                raise ValueError(f"body {self.model.body(body).name!r} is not fixed to the world")
            b = int(self.model.body_parentid[b])

    def _is_in_subtree(self, body: int, root: int) -> bool:
        b = body
        while True:
            if b == root:
                return True
            if b == 0:
                return False
            b = int(self.model.body_parentid[b])

    @staticmethod
    def _mesh_hull_indices(model: mujoco.MjModel, model_path: str | Path) -> list[np.ndarray]:
        """Per mesh, the indices of its convex-hull vertices (MuJoCo's qhull).

        MuJoCo only builds hulls for meshes used in collision, so a copy of
        the model is compiled with collision enabled on every geom.
        """
        spec = mujoco.MjSpec.from_file(str(model_path))
        for geom in spec.geoms:
            geom.contype = 1
            geom.conaffinity = 1
        hull_model = spec.compile()
        if hull_model.nmesh != model.nmesh or not np.array_equal(hull_model.mesh_vert, model.mesh_vert):
            raise ValueError(f"{model_path} does not match the loaded model's meshes")
        indices = []
        for mesh in range(model.nmesh):
            adr = int(hull_model.mesh_graphadr[mesh])
            if adr < 0:
                raise ValueError(f"MuJoCo built no convex hull for mesh {mesh}")
            count = int(hull_model.mesh_graph[adr])
            # mesh_graph layout: numvert, numface, vert_edgeadr[numvert], vert_globalid[numvert], ...
            indices.append(hull_model.mesh_graph[adr + 2 + count: adr + 2 + 2 * count].astype(int))
        return indices

    def _body_vertices_local(self, body: int, hull_only: bool = False) -> np.ndarray | None:
        """Mesh vertices of one body's geoms, in that body's frame."""
        m = self.model
        chunks = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = m.geom_dataid[g]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            verts = m.mesh_vert[start:start + count]
            if hull_only:
                verts = verts[self._hull_index[mesh]]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, m.geom_quat[g])
            chunks.append(verts @ rot.reshape(3, 3).T + m.geom_pos[g])
        return np.vstack(chunks) if chunks else None

    def _body_vertices_world(self, body: int) -> np.ndarray:
        verts = self._body_vertices_local(body)
        if verts is None:
            raise ValueError(f"body {self.model.body(body).name!r} has no mesh geometry")
        return verts @ self.data.xmat[body].reshape(3, 3).T + self.data.xpos[body]

    def home_references_array(self) -> np.ndarray:
        """hardware_zero references (joint side) in actuator order."""
        out = []
        for actuator in self.actuators:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            out.append(HARDWARE_ZERO_ACTUATOR_CTRL_RAD[actuator] / float(self.model.actuator_gear[act_id, 0]))
        return np.array(out)

    def to_array(self, references: dict[str, float]) -> np.ndarray:
        return np.array([float(references[a]) for a in self.actuators])

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def full_qpos(self, q: np.ndarray | dict[str, float]) -> np.ndarray:
        """Complete qpos for driven references q (no side effects; thread-safe)."""
        if isinstance(q, dict):
            q = self.to_array(q)
        qpos = self.keyframe_qpos.copy()
        qpos[self._qadr] = q
        for src, f_qadr, _, scale, src_kf, f_kf in self._followers:
            qpos[f_qadr] = f_kf + scale * (q[src] - src_kf)
        return qpos

    def _set_driven(self, q: np.ndarray) -> None:
        d = self.data
        d.qpos[:] = self.full_qpos(q)
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)   # cdof, needed by mj_jac

    def _tool_world(self) -> np.ndarray:
        return self.data.xpos[self._tool_body] + self.data.xmat[self._tool_body].reshape(3, 3) @ self._tool_point

    def world_to_base(self, p_world: np.ndarray) -> np.ndarray:
        return self.base_rot.T @ (np.asarray(p_world, dtype=float) - self.base_pos)

    def base_to_world(self, p_base: np.ndarray) -> np.ndarray:
        return self.base_rot @ np.asarray(p_base, dtype=float) + self.base_pos

    def tool_position_base(self, q: np.ndarray | dict[str, float]) -> np.ndarray:
        """Tool point in the base frame (m) for driven references q (joint rad)."""
        if isinstance(q, dict):
            q = self.to_array(q)
        self._set_driven(np.asarray(q, dtype=float))
        return self.world_to_base(self._tool_world())

    def tool_position_base_on(self, data: mujoco.MjData, q: np.ndarray | dict[str, float]) -> np.ndarray:
        """Like tool_position_base, but computed on the caller's ``data``.

        Does not touch this object's own MjData, so it needs no lock and can
        run while a solve is in progress in another thread.
        """
        data.qpos[:] = self.full_qpos(q)
        mujoco.mj_kinematics(self.model, data)
        body = self._tool_body
        return self.world_to_base(data.xpos[body] + data.xmat[body].reshape(3, 3) @ self._tool_point)

    def _position_and_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool point (base frame) and its 3 x n Jacobian w.r.t. the driven references."""
        self._set_driven(q)
        p_world = self._tool_world()
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, None, p_world, self._tool_body)
        jac = jacp[:, self._dofadr].copy()
        for src, _, f_dof, scale, _, _ in self._followers:
            jac[:, src] += scale * jacp[:, f_dof]
        return self.world_to_base(p_world), self.base_rot.T @ jac

    # ------------------------------------------------------------------
    # Clearance
    # ------------------------------------------------------------------

    def min_clearance_m(self, q: np.ndarray) -> float:
        """Lowest moving mesh vertex height above the table top (m)."""
        self._set_driven(q)
        body = self._moving_vertex_body
        # World z of each vertex: row 2 of its body's rotation (xmat[6:9]).
        z = np.einsum("ij,ij->i", self.data.xmat[body, 6:9], self._moving_vertices) + self.data.xpos[body, 2]
        return float(z.min()) - self.table_top_z_world

    def path_references(self, q_start: np.ndarray, q_goal: np.ndarray) -> tuple[list[np.ndarray], float]:
        """Samples of the limiter's rest-to-rest path (every joint its own
        trapezoid, as TrapezoidalReferenceLimiter moves them), and its duration."""
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)
        durations = [
            trapezoidal_duration(g - s, self.max_velocity_rad_s, self.max_acceleration_rad_s2)
            for s, g in zip(q_start, q_goal)
        ]
        duration = max(durations, default=0.0)
        samples = max(2, int(math.ceil(duration / PATH_CHECK_STEP_S)) + 1)
        path = [
            np.array([
                trapezoidal_position(s, g, t, self.max_velocity_rad_s, self.max_acceleration_rad_s2)
                for s, g in zip(q_start, q_goal)
            ])
            for t in np.linspace(0.0, duration, samples)
        ]
        return path, duration

    def path_tool_positions_base(self, q_start: np.ndarray, q_goal: np.ndarray) -> list[np.ndarray]:
        """Tool point (base frame) along the limiter's path from q_start to q_goal."""
        path, _ = self.path_references(q_start, q_goal)
        return [self.tool_position_base(q) for q in path]

    def _path_min_clearance_m(self, q_start: np.ndarray, q_goal: np.ndarray) -> tuple[float, float]:
        """Minimum clearance along the limiter's rest-to-rest path; also its duration."""
        path, duration = self.path_references(q_start, q_goal)
        return min(self.min_clearance_m(q) for q in path), duration

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def _solve_from(self, target: np.ndarray, seed: np.ndarray, rest: np.ndarray) -> tuple[np.ndarray, float]:
        """Damped least squares from one seed, pulled toward ``rest`` in the null space."""
        q = np.clip(seed, self.lower, self.upper)
        n = q.size
        damping2 = SOLVER_DAMPING_M ** 2
        best_q, best_err = q.copy(), math.inf
        stalled = 0
        for _ in range(SOLVER_MAX_ITERATIONS):
            p, jac = self._position_and_jacobian(q)
            e = target - p
            err = float(np.linalg.norm(e))
            if err < SOLVER_TOLERANCE_M:
                best_q, best_err = q.copy(), err
                break
            # Give up on a seed that has stopped improving (out of reach, or
            # stuck on a limit).
            stalled = 0 if err < best_err * (1.0 - SOLVER_STALL_RELATIVE) else stalled + 1
            if err < best_err:
                best_q, best_err = q.copy(), err
            if stalled >= SOLVER_STALL_ITERATIONS:
                break
            # Damping is scaled down near the target so the final iterations
            # converge quickly instead of creeping.
            lam2 = damping2 * min(1.0, err / 0.01)
            # Active set: a joint on a limit that the step would push further
            # out is removed from the step and the step recomputed.  Clipping
            # it afterwards instead would cancel part of the task step and
            # turn the null-space pull into tool-point motion, leaving a
            # residual error.
            free = np.ones(n, dtype=bool)
            for _ in range(n):
                jac_free = jac * free
                dq = jac_free.T @ np.linalg.solve(jac_free @ jac_free.T + lam2 * np.eye(3), e)
                # Exact null-space projector (SVD), so the pull toward
                # ``rest`` never moves the tool point to first order.
                _, sv, vt = np.linalg.svd(jac_free)
                row_space = vt[: int(np.sum(sv > 1e-9))]
                null = np.eye(n) - row_space.T @ row_space
                dq = dq + SOLVER_NULLSPACE_GAIN * (null @ ((rest - q) * free))
                blocked = free & (
                    ((q <= self.lower + 1e-12) & (dq < 0.0)) | ((q >= self.upper - 1e-12) & (dq > 0.0))
                )
                if not blocked.any():
                    break
                free &= ~blocked
            largest = float(np.max(np.abs(dq)))
            if largest > SOLVER_MAX_STEP_RAD:
                dq *= SOLVER_MAX_STEP_RAD / largest
            q = np.clip(q + dq, self.lower, self.upper)
        return best_q, best_err

    def solve(self, target_base_m: np.ndarray | list[float], start: dict[str, float]) -> IKResult:
        """Find references that put the tool point at ``target_base_m``.

        ``start`` is the current (at-rest) reference of every actuator, joint
        side.  The returned solution is the closest one found to it that
        passes every check; nothing is commanded here.
        """
        target = np.asarray(target_base_m, dtype=float).reshape(3)
        result = IKResult(False, "", target)
        if not np.all(np.isfinite(target)):
            result.message = "target is not finite"
            return result
        q_start = self.to_array(start)
        target_world = self.base_to_world(target)
        if target_world[2] - self.table_top_z_world < IK_TABLE_CLEARANCE_M:
            result.message = (
                f"target is {1000 * (target_world[2] - self.table_top_z_world):.1f} mm above the table top; "
                f"at least {1000 * IK_TABLE_CLEARANCE_M:.0f} mm is required"
            )
            return result

        # Seeds in tiers of increasing distance from the start pose.  After
        # each tier the converged solutions are checked closest-first, so a
        # far-away branch (e.g. the base swung half a turn) is only used when
        # nothing nearer works.
        rng = np.random.default_rng(0)
        tiers: list[list[np.ndarray]] = [[q_start]]
        for sigma in SOLVER_LOCAL_SEED_SIGMAS_RAD:
            tiers.append([
                np.clip(q_start + rng.normal(0.0, sigma, q_start.size), self.lower, self.upper)
                for _ in range(SOLVER_LOCAL_SEEDS_PER_TIER)
            ])
        tiers.append([
            self.lower + rng.random(q_start.size) * (self.upper - self.lower) for _ in range(SOLVER_FALLBACK_SEEDS)
        ])
        closest_err = math.inf
        rejected: list[str] = []
        seen: list[np.ndarray] = []        # distinct converged solutions
        for tier in tiers:
            candidates: list[np.ndarray] = []
            for seed in tier:
                q, err = self._solve_from(target, seed, q_start)
                closest_err = min(closest_err, err)
                if err > IK_ACCEPT_ERROR_M:
                    continue
                if any(np.max(np.abs(q - s)) < 1e-3 for s in seen):
                    continue   # same solution as an earlier seed
                seen.append(q)
                candidates.append(q)
            candidates.sort(key=lambda q: float(np.linalg.norm(q - q_start)))
            for q in candidates:
                verdict = self._check(result, q_start, q)
                if verdict is None:
                    return result
                rejected.append(verdict)

        if rejected:
            result.message = f"no solution passed the checks ({rejected[0]})"
        else:
            result.message = (
                f"target is out of reach: closest tool point is {1000 * closest_err:.1f} mm away"
            )
            result.error_m = closest_err
        return result

    def _check(self, result: IKResult, q_start: np.ndarray, q: np.ndarray) -> str | None:
        """Fill ``result`` and return None if ``q`` is acceptable, else a reason."""
        if np.any(q < self.lower) or np.any(q > self.upper):
            return "solution outside the joint ranges"
        change = np.abs(q - q_start)
        if np.max(change) > IK_MAX_JOINT_CHANGE_RAD:
            worst = int(np.argmax(change))
            return (
                f"{self.actuators[worst]} would turn {math.degrees(change[worst]):.0f} deg, more than the "
                f"{math.degrees(IK_MAX_JOINT_CHANGE_RAD):.0f} deg allowed per command (self-collision is not modelled)"
            )
        achieved = self.tool_position_base(q)
        err = float(np.linalg.norm(achieved - result.target_base_m))
        if err > IK_ACCEPT_ERROR_M:
            return f"residual {1000 * err:.3f} mm"
        final_clearance = self.min_clearance_m(q)
        if final_clearance < IK_TABLE_CLEARANCE_M:
            return f"arm would be {1000 * final_clearance:.1f} mm above the table at the target"
        path_clearance, duration = self._path_min_clearance_m(q_start, q)
        if path_clearance < IK_TABLE_CLEARANCE_M:
            return f"arm would pass {1000 * path_clearance:.1f} mm above the table on the way"
        result.success = True
        result.message = "ok"
        result.joint_targets = {a: float(v) for a, v in zip(self.actuators, q)}
        result.achieved_base_m = achieved
        result.error_m = err
        result.move_duration_s = duration
        return None


def _show_pose(ik: ArmIK, q: np.ndarray, target_base_m: np.ndarray, solved: bool) -> None:
    """Open the MuJoCo viewer on pose ``q``, with markers, until it is closed.

    Kinematic display only: no physics step runs, so the arm does not sag and
    the Control-tab sliders do not move it.  Markers: the target (green if
    solved, red if not) and the model's tool point (blue).
    """
    import time

    import mujoco.viewer

    model = ik.model
    data = mujoco.MjData(model)
    ik._set_driven(q)
    data.qpos[:] = ik.data.qpos
    # Show the references on the Control tab too (actuator ctrl = gear * joint).
    for actuator, value in zip(ik.actuators, q):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        data.ctrl[act_id] = float(model.actuator_gear[act_id, 0]) * value
    mujoco.mj_forward(model, data)
    tool_world = ik.base_to_world(ik.tool_position_base(q))
    markers = [
        (ik.base_to_world(target_base_m), 0.006, (0.1, 0.9, 0.1, 0.6) if solved else (0.9, 0.1, 0.1, 0.8)),
        (tool_world, 0.003, (0.1, 0.3, 1.0, 1.0)),
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            for i, (pos, radius, rgba) in enumerate(markers):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([radius, 0.0, 0.0]),
                    pos=np.asarray(pos, dtype=float),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(rgba, dtype=np.float32),
                )
            viewer.user_scn.ngeom = len(markers)
        print("[ik] Viewer open; close the window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def _main() -> None:
    import argparse

    from hardware_integrated_main_controller import ACTUATOR_TO_MOTOR, CONSTRAINED_JOINTS, MODEL_PATH, MOTOR_TO_JOINT

    parser = argparse.ArgumentParser(
        description="Solve tool-point IK from hardware_zero and show the pose in the MuJoCo viewer (no hardware)."
    )
    parser.add_argument("x", type=float, help="base-frame target, metres")
    parser.add_argument("y", type=float, help="base-frame target, metres")
    parser.add_argument("z", type=float, help="base-frame target, metres")
    parser.add_argument("--no-view", action="store_true", help="print the solution only; do not open the viewer")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(Path(MODEL_PATH)))
    ik = ArmIK(model, MODEL_PATH, {a: MOTOR_TO_JOINT[m] for a, m in ACTUATOR_TO_MOTOR.items()}, CONSTRAINED_JOINTS)
    home_q = np.clip(ik.home_references_array(), ik.lower, ik.upper)
    home = dict(zip(ik.actuators, home_q))
    print(f"[ik] Tool point at hardware_zero: {np.round(ik.tool_position_base(home), 4)} m (base frame)")
    target = np.array([args.x, args.y, args.z])
    result = ik.solve(target, home)
    if result.success:
        for actuator, q in result.joint_targets.items():
            print(f"[ik]   {actuator}: {q:+.5f} rad joint ({math.degrees(q):+.2f} deg)")
        print(
            f"[ik] Reaches {np.round(result.achieved_base_m, 5)} m, error {1000 * result.error_m:.4f} mm, "
            f"move {result.move_duration_s:.2f} s"
        )
    else:
        print(f"[ik] No solution: {result.message}")
        if not args.no_view:
            print("[ik] Showing hardware_zero with the target marked in red.")
    if not args.no_view:
        pose = ik.to_array(result.joint_targets) if result.success else home_q
        _show_pose(ik, pose, target, result.success)
    if not result.success:
        raise SystemExit(2)


if __name__ == "__main__":
    _main()
