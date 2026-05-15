#!/usr/bin/env python3
"""
Collect grasp-transition data from MuJoCo + SO-ARM101.

Each episode:
  1. Load a YCB object at a random tabletop position
  2. Let physics settle (300 steps)
  3. Sample a random grasp pose near the object centroid
  4. Execute the grasp (approach → close → lift)
  5. Record obs_before / action / obs_after / labels → .npz

Usage:
  # 100-episode smoke test (banana + cylinder, alternating yaw modes)
  MUJOCO_GL=egl conda run -n bridge python scripts/collect_mujoco_transitions.py

  # 500 episodes across all YCB objects
  MUJOCO_GL=egl conda run -n bridge python scripts/collect_mujoco_transitions.py \\
    --n-episodes 500 --objects all

  # Quick 10-episode sanity check
  MUJOCO_GL=egl conda run -n bridge python scripts/collect_mujoco_transitions.py \\
    --n-episodes 10 --quiet
"""

import math
import os
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from owg_robot.env_soarm import (
    EnvironmentSoArm,
    TABLE_TOP_Z,
    ARM_JOINTS,
)
from data.transition_logger import (
    Transition,
    TransitionLogger,
    compute_pc_stats,
    compute_pose_delta,
    TRANSITIONS_DIR,
)

# ── Object catalogue ──────────────────────────────────────────────────────────

OBJECTS = {
    "banana":   "YcbBanana",
    "cylinder": "YcbTomatoSoupCan",
    "cracker":  "YcbCrackerBox",
    "mustard":  "YcbMustardBottle",
    "drill":    "YcbPowerDrill",
}

_CENTRE_Y       = -0.40   # arm reachable table centre
_SPREAD_XY      = 0.06    # max random XY offset from centre
_DROP_Z         = TABLE_TOP_Z + 0.12   # initial drop height
_SETTLE_STEPS   = 300
_GRASP_Z_OFFSET = 0.025   # metres above settled centroid
_LIFT_STEPS     = 80
_FELL_OFF_Z     = TABLE_TOP_Z - 0.10  # below this → fell off table
_LIFT_THRESHOLD = 0.07    # metres above table → successfully lifted

YAW_MODES = ["xyz_only", "top_down_yaw"]


# ── Grasp sampling ────────────────────────────────────────────────────────────

def sample_grasp(obj_pos: np.ndarray, rng: np.random.Generator,
                 spread: float = 0.04) -> np.ndarray:
    """
    Random grasp pose near the object centroid.
    Returns (6,): [x, y, z, yaw, opening_len, obj_height]
    """
    x   = float(obj_pos[0] + rng.uniform(-spread, spread))
    y   = float(obj_pos[1] + rng.uniform(-spread, spread))
    z   = float(obj_pos[2] + _GRASP_Z_OFFSET)
    yaw = float(rng.uniform(-math.pi / 2, math.pi / 2))
    opening = float(rng.uniform(0.04, 0.09))
    return np.array([x, y, z, yaw, opening, 0.05], dtype=np.float32)


# ── Yaw helpers ───────────────────────────────────────────────────────────────

def _apply_top_down_yaw(env: EnvironmentSoArm, yaw: float):
    """Command wrist_roll to approximate end-effector yaw (near-vertical arm)."""
    try:
        idx = ARM_JOINTS.index("wrist_roll")
        lo  = env.model.jnt_range[env._arm_jnt_ids[idx], 0]
        hi  = env.model.jnt_range[env._arm_jnt_ids[idx], 1]
        cmd = float(np.clip(yaw, lo, hi))
        env.data.ctrl[env._arm_act_ids[idx]] = cmd
        env.data.qpos[env._arm_qpos_adr[idx]] = cmd
        env._steps(80)
    except Exception:
        pass


# ── Grasp execution ───────────────────────────────────────────────────────────

def execute_grasp(env: EnvironmentSoArm, obj_id: int,
                  grasp: np.ndarray, yaw_mode: str) -> bool:
    """
    Attempt one grasp. Returns True on contact or successful lift.
    Does NOT reset or reload the object — caller is responsible.
    """
    x, y, z     = float(grasp[0]), float(grasp[1]), float(grasp[2])
    yaw         = float(grasp[3])
    opening_len = float(grasp[4])

    try:
        env.reset_robot()
        env.move_gripper(opening_len)
        env.move_ee([x, y, env.GRIPPER_MOVING_HEIGHT, None])

        if yaw_mode == "top_down_yaw":
            _apply_top_down_yaw(env, yaw)

        env.move_ee([x, y, z, None])
        env.auto_close_gripper(check_contact=False)
        env._steps(60)

        contact = obj_id in env.check_grasped_id()
        env.move_ee([x, y, env.GRIPPER_MOVING_HEIGHT, None])
        env._steps(_LIFT_STEPS)

        grasped = obj_id in env.check_grasped_id()
        obj_z   = env.get_obj_pos(obj_id)[2]
        lifted  = obj_z > TABLE_TOP_Z + _LIFT_THRESHOLD

        return contact or grasped or lifted

    except Exception:
        return False


# ── Collection loop ───────────────────────────────────────────────────────────

def run_collection(
    env: EnvironmentSoArm,
    logger: TransitionLogger,
    n_episodes: int,
    obj_keys: list,
    rng: np.random.Generator,
    verbose: bool = True,
) -> dict:
    """
    Main loop. Each iteration:
      reset → load → settle → obs_before → grasp → obs_after → log
    """
    ep_id     = logger.n_episodes
    n_success = 0

    for i in range(n_episodes):
        obj_key  = obj_keys[i % len(obj_keys)]
        obj_name = OBJECTS[obj_key]
        yaw_mode = YAW_MODES[i % len(YAW_MODES)]

        # ── load object ───────────────────────────────────────────────────────
        env.reset_robot()
        env.remove_all_obj()
        cx  = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
        cy  = _CENTRE_Y + float(rng.uniform(-0.04, 0.04))
        oid = env.load_obj(obj_name, name=obj_key, pos=[cx, cy, _DROP_Z])
        env._steps(_SETTLE_STEPS)

        # ── pre-grasp state ───────────────────────────────────────────────────
        pos_before  = env.get_obj_pos(oid).copy()
        quat_before = env.get_obj_pose(oid)["quaternion"].copy()
        obs_before  = env.get_obs(pointcloud=True)
        pc_stats    = compute_pc_stats(obs_before, oid)
        dep_mean    = float(obs_before["depth"].mean())

        # ── sample + execute grasp ────────────────────────────────────────────
        grasp   = sample_grasp(pos_before, rng)
        success = execute_grasp(env, oid, grasp, yaw_mode)

        # ── post-grasp state ──────────────────────────────────────────────────
        env._steps(40)
        pos_after  = env.get_obj_pos(oid).copy()
        quat_after = env.get_obj_pose(oid)["quaternion"].copy()

        dz       = float(pos_after[2] - pos_before[2])
        fell_off = bool(pos_after[2] < _FELL_OFF_Z)
        delta    = compute_pose_delta(pos_before, quat_before, pos_after, quat_after)

        # ── log ───────────────────────────────────────────────────────────────
        t = Transition(
            episode_id      = ep_id,
            obj_name        = obj_key,
            obj_id          = oid,
            yaw_mode        = yaw_mode,
            obj_pos_before  = pos_before,
            obj_quat_before = quat_before,
            pc_stats_before = pc_stats,
            depth_mean_before = dep_mean,
            grasp_pose      = grasp,
            obj_pos_after   = pos_after,
            obj_quat_after  = quat_after,
            success         = success,
            dz              = dz,
            fell_off        = fell_off,
            pose_delta      = delta,
        )
        logger.log(t)
        n_success += int(success)
        ep_id     += 1

        if verbose:
            sym = "✓" if success else "✗"
            ff  = " [fell]" if fell_off else ""
            print(f"  ep {ep_id-1:04d}  {obj_key:<10}  {yaw_mode:<14}"
                  f"  dz={dz:+.3f}  {sym}{ff}")

    sr = n_success / n_episodes if n_episodes else 0.0
    print(f"\nCollected {n_episodes} episodes — success_rate={sr:.2f}")
    return logger.summary()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-episodes", type=int, default=100,
                    help="Number of grasp attempts (default: 100)")
    ap.add_argument("--objects", default="banana,cylinder",
                    help="Comma-separated keys, or 'all' (default: banana,cylinder)")
    ap.add_argument("--out-dir", default=str(TRANSITIONS_DIR),
                    help=f"Output directory (default: {TRANSITIONS_DIR})")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.objects == "all":
        obj_keys = list(OBJECTS.keys())
    else:
        obj_keys = [k.strip() for k in args.objects.split(",")]
    for k in obj_keys:
        if k not in OBJECTS:
            ap.error(f"Unknown object '{k}'. Valid keys: {list(OBJECTS.keys())}")

    rng    = np.random.default_rng(args.seed)
    env    = EnvironmentSoArm(vis=False, debug=False)
    logger = TransitionLogger(out_dir=Path(args.out_dir))

    print(f"Collecting {args.n_episodes} transitions  →  {args.out_dir}")
    print(f"Objects: {obj_keys}  |  existing episodes: {logger.n_episodes}\n")

    try:
        summary = run_collection(
            env, logger, args.n_episodes, obj_keys,
            rng, verbose=not args.quiet,
        )
    finally:
        env.close()

    print("\nDataset summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
