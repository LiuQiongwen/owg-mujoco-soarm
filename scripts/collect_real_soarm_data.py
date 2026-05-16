#!/usr/bin/env python3
"""
Collect grasp-episode data from a real SO-ARM101 robot (or simulation).

For simulation (default):
  MUJOCO_GL=egl conda run -n owg2 python scripts/collect_real_soarm_data.py

For real hardware (not yet supported — hardware drivers pending):
  python scripts/collect_real_soarm_data.py --real --camera realsense

Data is saved in two formats simultaneously:
  data/transitions/          ← legacy format (compatible with TransitionLogger)
  data/lerobot/              ← LeRobot-compatible format

Usage
-----
  # 50 simulation episodes, banana + mustard alternating
  MUJOCO_GL=egl conda run -n owg2 \\
    python scripts/collect_real_soarm_data.py --n-episodes 50

  # Save per-step RGB-D frames (large files)
  MUJOCO_GL=egl conda run -n owg2 \\
    python scripts/collect_real_soarm_data.py --n-episodes 20 --record-frames

  # Physics mode only (for world-model training labels)
  MUJOCO_GL=egl conda run -n owg2 \\
    python scripts/collect_real_soarm_data.py --mode physics
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from owg_robot.env_soarm import (
    EnvironmentSoArm,
    TABLE_TOP_Z,
    GRASP_MODE_PHYSICS,
    GRASP_MODE_DEMO_ATTACH,
)
from robots.soarm101.robot import SOARM101Robot
from datasets.episode import GraspAction
from datasets.writer import EpisodeWriter
from data.transition_logger import compute_pc_stats

# ── object catalogue ──────────────────────────────────────────────────────────

OBJECTS = {
    "banana":   "YcbBanana",
    "mustard":  "YcbMustardBottle",
    "drill":    "YcbPowerDrill",
    "pear":     "YcbPear",
    "cylinder": "YcbMediumClamp",
}

# ── collection parameters ─────────────────────────────────────────────────────

_CENTRE_Y       = -0.40
_SPREAD_XY      = 0.06
_DROP_Z         = TABLE_TOP_Z + 0.12
_SETTLE_STEPS   = 300
_GRASP_Z_OFFSET = 0.025
_LIFT_THRESHOLD = 0.07
_FELL_OFF_Z     = TABLE_TOP_Z - 0.10
YAW_MODES       = ["xyz_only", "top_down_yaw"]


# ── grasp sampling ────────────────────────────────────────────────────────────

def sample_grasp(obj_pos: np.ndarray, rng: np.random.Generator,
                 spread: float = 0.04) -> GraspAction:
    x   = float(obj_pos[0] + rng.uniform(-spread, spread))
    y   = float(obj_pos[1] + rng.uniform(-spread, spread))
    z   = float(obj_pos[2] + _GRASP_Z_OFFSET)
    yaw = float(rng.uniform(-math.pi / 2, math.pi / 2))
    opening = float(rng.uniform(0.04, 0.09))
    return GraspAction(
        eef_pos    = np.array([x, y, z], dtype=np.float32),
        yaw        = yaw,
        opening_m  = opening,
        obj_height = 0.05,
    )


# ── main collection loop ──────────────────────────────────────────────────────

def run(args):
    mode  = args.mode
    n_eps = args.n_episodes
    seed  = args.seed
    rng   = np.random.default_rng(seed)

    obj_keys = (
        list(OBJECTS.keys()) if args.objects == "all"
        else [o.strip() for o in args.objects.split(",")]
    )
    for k in obj_keys:
        if k not in OBJECTS:
            raise ValueError(f"Unknown object '{k}'. Choose from: {list(OBJECTS)}")

    print(f"[collect] mode={mode}  episodes={n_eps}  objects={obj_keys}  seed={seed}")

    # ── writer (saves both legacy + LeRobot formats) ──────────────────────────
    writer = EpisodeWriter(
        out_dir_legacy  = "data/transitions",
        out_dir_lerobot = "data/lerobot",
        record_frames   = args.record_frames,
    )

    if args.real:
        raise NotImplementedError(
            "Real-hardware driver not yet implemented. "
            "Implement SOARM101HardwareRobot in robots/soarm101/hardware.py "
            "then call it here."
        )

    # ── simulation ────────────────────────────────────────────────────────────
    obj_names = [OBJECTS[k] for k in obj_keys[:1]]   # start with one object in pool
    env = EnvironmentSoArm(
        obj_names   = obj_names,
        vis         = args.vis,
        grasp_mode  = mode,
    )
    robot = SOARM101Robot(
        env          = env,
        writer       = writer,
        record_steps = args.record_steps,
        record_frames= args.record_frames,
    )

    ep_id     = len(writer._meta)   # continue from existing dataset
    n_success = 0

    for i in range(n_eps):
        obj_key  = obj_keys[i % len(obj_keys)]
        obj_name = OBJECTS[obj_key]
        yaw_mode = YAW_MODES[i % len(YAW_MODES)]

        # reset + load
        env.reset_robot()
        env.remove_all_obj()
        cx  = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
        cy  = _CENTRE_Y + float(rng.uniform(-0.04, 0.04))
        oid = env.load_obj(obj_name, name=obj_key, pos=[cx, cy, _DROP_Z])
        env._steps(_SETTLE_STEPS)

        # pre-grasp state
        pos_before  = env.get_obj_pos(oid).copy()
        quat_before = env.get_obj_pose(oid)["quaternion"].copy()
        obs_before  = env.get_obs(pointcloud=True)
        pc_stats    = compute_pc_stats(obs_before, oid)
        dep_mean    = float(obs_before["depth"].mean())

        # sample grasp
        action = sample_grasp(pos_before, rng)

        # start episode recording
        ep = robot.begin_episode(
            episode_id     = ep_id,
            obj_name       = obj_name,
            obj_id         = oid,
            yaw_mode       = yaw_mode,
            execution_mode = mode,
        )
        robot.log_step(action=action)

        # execute
        env.reset_robot()
        env.move_gripper(action.opening_m)
        env.move_ee([*action.eef_pos[:2], env.GRIPPER_MOVING_HEIGHT, None])
        env.move_ee([*action.eef_pos, None])
        env.auto_close_gripper()
        env._steps(60)

        contact = oid in env.check_grasped_id()
        env.move_ee([*action.eef_pos[:2], env.GRIPPER_MOVING_HEIGHT, None])
        env._steps(80)

        pos_after  = env.get_obj_pos(oid).copy()
        quat_after = env.get_obj_pose(oid)["quaternion"].copy()
        dz         = float(pos_after[2] - pos_before[2])
        fell_off   = bool(pos_after[2] < _FELL_OFF_Z)
        lifted     = bool(pos_after[2] > TABLE_TOP_Z + _LIFT_THRESHOLD)
        success    = contact or lifted

        if success:
            n_success += 1

        robot.end_episode(
            success           = success,
            dz                = dz,
            fell_off          = fell_off,
            obj_pos_before    = pos_before,
            obj_quat_before   = quat_before,
            pc_stats_before   = pc_stats,
            depth_mean_before = dep_mean,
            obj_pos_after     = pos_after,
            obj_quat_after    = quat_after,
            grasp_action      = action,
        )

        if not args.quiet:
            print(
                f"  ep {ep_id:4d}  {obj_name:20s}  {yaw_mode:16s}  "
                f"{'OK' if success else '--'}  dz={dz:+.3f}"
            )
        ep_id += 1

    env.close()

    print(
        f"\n[collect] done.  {n_eps} episodes, "
        f"success={n_success}/{n_eps} ({100*n_success/max(n_eps,1):.1f}%)"
    )
    print(f"  legacy: data/transitions/  ({n_eps} new episodes)")
    print(f"  lerobot: data/lerobot/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-episodes",    type=int,  default=50,
                   help="Number of episodes to collect (default: 50)")
    p.add_argument("--objects",       type=str,  default="banana,mustard",
                   help="Comma-separated object keys, or 'all' (default: banana,mustard)")
    p.add_argument("--mode",          type=str,  default=GRASP_MODE_PHYSICS,
                   choices=[GRASP_MODE_PHYSICS, GRASP_MODE_DEMO_ATTACH],
                   help="Grasp execution mode (default: physics)")
    p.add_argument("--seed",          type=int,  default=0,
                   help="RNG seed (default: 0)")
    p.add_argument("--real",          action="store_true",
                   help="Use real hardware (raises NotImplementedError until driver is ready)")
    p.add_argument("--camera",        type=str,  default="simulated",
                   choices=["simulated", "realsense"],
                   help="Camera backend (default: simulated)")
    p.add_argument("--record-steps",  action="store_true",
                   help="Log per-step joint/EEF state to LeRobot format")
    p.add_argument("--record-frames", action="store_true",
                   help="Save per-step RGB-D images (implies --record-steps, large files)")
    p.add_argument("--vis",           action="store_true",
                   help="Open MuJoCo viewer during collection")
    p.add_argument("--quiet",         action="store_true",
                   help="Suppress per-episode output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.record_frames:
        args.record_steps = True
    run(args)
