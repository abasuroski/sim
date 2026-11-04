"""Map the gripper's reachable workspace (simulation only, no hardware).

Samples the four driven joints uniformly inside the same limits the IK
solver uses (``ArmIK.lower``/``upper``: actuator control ranges, joint ranges
and four-bar follower ranges), keeps the poses whose every moving mesh vertex
clears the table by IK_TABLE_CLEARANCE_M, and records the tool point in the
base frame -- the same x/y/z the control panel's IK target takes.

Kinematics are ArmIK's (followers from CONSTRAINED_JOINTS, wrist servo at its
keyframe angle), so a mapped point is one ``ik`` can solve.  Reaching it from
the current pose may take several moves if IK_MAX_JOINT_CHANGE_RAD caps how far
an IK move may change a joint.  Self-collision is not checked (the
MJCF has no arm collision geometry).

Usage:
    python map_workspace.py                     # 5000 samples (~1 min), opens the viewer
    python map_workspace.py --samples 20000 --csv workspace.csv
    python map_workspace.py --base-angle 83     # one slice: M1 fixed at 83 deg
    python map_workspace.py --no-view
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from hardware_integrated_main_controller import ACTUATOR_TO_MOTOR, CONSTRAINED_JOINTS, MODEL_PATH, MOTOR_TO_JOINT
from inverse_kinematics import ArmIK
from robot_master_configuration import IK_TABLE_CLEARANCE_M

POINT_RADIUS_M = 0.003


def sample_workspace(
    ik: ArmIK, samples: int, base_angle_rad: float | None, seed: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (joint references [k x 4], tool points base frame m [k x 3],
    number rejected for table clearance)."""
    rng = np.random.default_rng(seed)
    lower, upper = ik.lower.copy(), ik.upper.copy()
    # revolute_1 declares +-2 pi; one turn covers every base angle.
    lower[0], upper[0] = max(lower[0], -math.pi), min(upper[0], math.pi)
    q_all = lower + rng.random((samples, lower.size)) * (upper - lower)
    if base_angle_rad is not None:
        q_all[:, 0] = base_angle_rad
    kept_q, kept_p, rejected = [], [], 0
    for q in q_all:
        if ik.min_clearance_m(q) < IK_TABLE_CLEARANCE_M:
            rejected += 1
            continue
        kept_q.append(q)
        kept_p.append(ik.tool_position_base(q))
    return np.array(kept_q), np.array(kept_p), rejected


def _height_colour(z: float, z_lo: float, z_hi: float) -> np.ndarray:
    """Blue (low) -> green -> red (high)."""
    t = 0.0 if z_hi <= z_lo else (z - z_lo) / (z_hi - z_lo)
    r = min(1.0, max(0.0, 2.0 * t - 1.0))
    b = min(1.0, max(0.0, 1.0 - 2.0 * t))
    return np.array([r, 1.0 - r - b, b, 0.8], dtype=np.float32)


def show(ik: ArmIK, points_base_m: np.ndarray) -> None:
    """Open the viewer at hardware_zero with the mapped points drawn as spheres."""
    import mujoco.viewer

    model = ik.model
    data = mujoco.MjData(model)
    home = np.clip(ik.home_references_array(), ik.lower, ik.upper)
    data.qpos[:] = ik.full_qpos(home)
    mujoco.mj_forward(model, data)   # kinematic display only; no physics step

    with mujoco.viewer.launch_passive(model, data) as viewer:
        scn = viewer.user_scn
        shown = points_base_m
        if len(shown) > scn.maxgeom:
            shown = shown[np.linspace(0, len(shown) - 1, scn.maxgeom).astype(int)]
            print(f"[map] Showing {len(shown)} of {len(points_base_m)} points (viewer limit).")
        z_lo, z_hi = float(shown[:, 2].min()), float(shown[:, 2].max())
        with viewer.lock():
            for i, p in enumerate(shown):
                mujoco.mjv_initGeom(
                    scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([POINT_RADIUS_M, 0.0, 0.0]),
                    pos=ik.base_to_world(p),
                    mat=np.eye(3).flatten(),
                    rgba=_height_colour(float(p[2]), z_lo, z_hi),
                )
            scn.ngeom = len(shown)
        print("[map] Viewer open (colour = height: blue low, red high); close the window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--samples", type=int, default=5000, help="random joint samples (default 5000, ~1 min)")
    parser.add_argument("--base-angle", type=float, default=None,
                        help="fix revolute_1 at this angle (deg, joint side) for a 2-D slice")
    parser.add_argument("--csv", type=Path, default=None, help="write joint references and tool points here")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-view", action="store_true", help="print the summary only")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(Path(MODEL_PATH)))
    ik = ArmIK(model, MODEL_PATH, {a: MOTOR_TO_JOINT[m] for a, m in ACTUATOR_TO_MOTOR.items()}, CONSTRAINED_JOINTS)
    base_angle = None if args.base_angle is None else math.radians(args.base_angle)

    started = time.perf_counter()
    q, p, rejected = sample_workspace(ik, args.samples, base_angle, args.seed)
    print(f"[map] {len(p)} reachable poses, {rejected} rejected for table clearance "
          f"({time.perf_counter() - started:.1f} s)")
    if len(p) == 0:
        raise SystemExit(1)

    mm = 1000.0 * p
    radial = np.hypot(mm[:, 0], mm[:, 1])
    for axis, column in zip("xyz", mm.T):
        print(f"[map]   {axis}: {column.min():+8.1f} .. {column.max():+8.1f} mm (base frame)")
    print(f"[map]   distance from base axis: {radial.min():.1f} .. {radial.max():.1f} mm")
    print(f"[map]   farthest point from base origin: {np.linalg.norm(mm, axis=1).max():.1f} mm")

    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{a}_joint_rad" for a in ik.actuators] + ["x_base_m", "y_base_m", "z_base_m"])
            writer.writerows(np.hstack([q, p]).tolist())
        print(f"[map] Wrote {args.csv}")

    if not args.no_view:
        show(ik, p)


if __name__ == "__main__":
    main()
