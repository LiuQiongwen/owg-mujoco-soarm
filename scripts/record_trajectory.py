#!/usr/bin/env python3
"""Record a grasp trajectory from MuJoCo simulation for later sim-to-real replay.

Loads a YCB object, samples or uses a fixed grasp action, executes it via
MujocoBackend with a TrajectoryRecorder attached, and saves the resulting
Trajectory JSON for replay on the physical SO-ARM101.

Usage
-----
    # Record one grasp of a banana (random seed 42)
    conda run -n owg-mujoco python scripts/record_trajectory.py \
        --obj banana --seed 42 --out trajs/banana_42.json

    # Try up to 5 random grasps until one succeeds
    conda run -n owg-mujoco python scripts/record_trajectory.py \
        --obj mustard --seed 7 --n-tries 5 --out trajs/mustard_7.json

    # Record with visualisation (for debugging)
    conda run -n owg-mujoco python scripts/record_trajectory.py \
        --obj banana --seed 1 --vis --out trajs/banana_vis.json

    # Save regardless of success (useful for failure analysis)
    conda run -n owg-mujoco python scripts/record_trajectory.py \
        --obj drill --seed 3 --save-all --out trajs/drill_3.json
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from datasets.episode import GraspAction
from tango_robot.env_soarm import EnvironmentSoArm, TABLE_TOP_Z
from robots import MujocoBackend, TrajectoryRecorder

# ── Object catalogue ──────────────────────────────────────────────────────────

OBJECT_REGISTRY = {
    "banana":   "YcbBanana",
    "pear":     "YcbPear",
    "mustard":  "YcbMustardBottle",
    "cracker":  "YcbCrackerBox",
    "drill":    "YcbPowerDrill",
    "can":      "YcbTomatoSoupCan",
    "cylinder": "YcbMediumClamp",
}

# ── Spawn / sampling constants ────────────────────────────────────────────────

_CENTRE_Y    = -0.40
_SPREAD_XY   =  0.06
_DROP_Z      = TABLE_TOP_Z + 0.12
_SETTLE_STEPS = 300
_Z_OFFSET    =  0.025   # metres above settled object centroid
_YAW_LO      = -math.pi / 2
_YAW_HI      =  math.pi / 2
_OPEN_LO     =  0.04
_OPEN_HI     =  0.09
_OBJ_HEIGHT  =  0.05    # assumed object height for IK target


# ── Grasp sampling ────────────────────────────────────────────────────────────

def _sample_action(obj_pos: np.ndarray, rng: np.random.Generator,
                   spread: float = 0.04) -> GraspAction:
    x       = float(obj_pos[0] + rng.uniform(-spread, spread))
    y       = float(obj_pos[1] + rng.uniform(-spread, spread))
    z       = float(obj_pos[2] + _Z_OFFSET)
    yaw     = float(rng.uniform(_YAW_LO, _YAW_HI))
    opening = float(rng.uniform(_OPEN_LO, _OPEN_HI))
    return GraspAction(
        eef_pos    = np.array([x, y, z], dtype=np.float32),
        yaw        = yaw,
        opening_m  = opening,
        obj_height = _OBJ_HEIGHT,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record a grasp trajectory from MuJoCo simulation."
    )
    p.add_argument("--obj",        default="banana",
                   choices=list(OBJECT_REGISTRY), help="Object to grasp")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for spawn position and grasp sampling")
    p.add_argument("--out",        default=None,
                   help="Output trajectory JSON path (default: trajs/<obj>_<seed>.json)")
    p.add_argument("--n-tries",    type=int, default=1,
                   help="Maximum grasp attempts; stops at first success")
    p.add_argument("--grasp-mode", default="physics_weld_after_bilateral",
                   help="MuJoCo grasp execution mode")
    p.add_argument("--vis",        action="store_true",
                   help="Enable MuJoCo viewer (slower)")
    p.add_argument("--save-all",   action="store_true",
                   help="Save trajectory even when grasp fails")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_path = Path(args.out) if args.out else Path(f"trajs/{args.obj}_{args.seed}.json")
    ycb_name = OBJECT_REGISTRY[args.obj]
    rng      = np.random.default_rng(args.seed)

    # ── build environment ─────────────────────────────────────────────────────
    print(f"[record] obj={args.obj}  seed={args.seed}  grasp_mode={args.grasp_mode}")
    env = EnvironmentSoArm(vis=args.vis, grasp_mode=args.grasp_mode)

    # spawn object at randomised tabletop position
    cx = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
    cy = _CENTRE_Y + float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
    obj_id = env.load_obj(ycb_name, name=args.obj, pos=[cx, cy, _DROP_Z])
    env._steps(_SETTLE_STEPS)

    obj_pos = env.get_obj_pos(obj_id)
    print(f"[record] object settled at ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")

    # ── attach recorder ───────────────────────────────────────────────────────
    recorder = TrajectoryRecorder()
    backend  = MujocoBackend(env, recorder=recorder)

    result   = None
    saved    = False

    for attempt in range(1, args.n_tries + 1):
        action = _sample_action(obj_pos, rng)
        print(
            f"[record] attempt {attempt}/{args.n_tries}  "
            f"eef=({action.eef_pos[0]:.3f},{action.eef_pos[1]:.3f},{action.eef_pos[2]:.3f})  "
            f"yaw={math.degrees(action.yaw):.1f}°  opening={action.opening_m:.3f}m"
        )

        result = backend.execute_grasp(action)

        print(
            f"[record] result: success={result.success}  dz={result.dz:.4f}m  "
            f"points={result.trajectory.n_points if result.trajectory else 0}"
        )

        if result.success or attempt == args.n_tries:
            break

        # failed — reload object for next attempt
        env.reset_robot()
        # re-read settled position (object may have moved on first attempt)
        obj_pos = env.get_obj_pos(obj_id)

    env.close()

    # ── save trajectory ───────────────────────────────────────────────────────
    traj = result.trajectory if result else None
    if traj is None:
        print("[record] no trajectory captured (recorder not active?)")
        return

    if not result.success and not args.save_all:
        print(f"[record] grasp failed after {args.n_tries} attempt(s). "
              "Pass --save-all to save failed trajectories.")
        return

    # enrich metadata
    traj.metadata.update({
        "obj_name": args.obj,
        "ycb_name": ycb_name,
        "seed":     args.seed,
    })

    saved_path = traj.save(out_path)
    print(
        f"[record] saved  →  {saved_path}\n"
        f"         {traj.n_points} points  "
        f"{traj.duration:.2f}s  "
        f"success={traj.metadata.get('success')}  "
        f"dz={traj.metadata.get('dz', 0):.4f}m"
    )
    saved = True

    if not saved:
        print("[record] nothing written.")


if __name__ == "__main__":
    main()
