#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal MuJoCo grasp demo for SO-ARM101.

Usage:
    python grasp_6dof/demo_mujoco.py
    python grasp_6dof/demo_mujoco.py --grasps grasp_6dof/out/cylinder_0p08.json --n 5
    python grasp_6dof/demo_mujoco.py --vis          # save rendered frames
"""
import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grasp_6dof.sim.mujoco_env import MujocoGraspEnv
from grasp_6dof.robots.soarm101_adapter import SoArm101Adapter
from grasp_6dof.data.transition_logger import (
    TransitionLogger, make_before_state, make_after_state
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grasps", type=str, default=None,
                   help="Path to grasp JSON (position/rpy/width/score list). "
                        "If omitted, uses a random box object with random grasps.")
    p.add_argument("--n",      type=int, default=3,  help="Number of grasp trials")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--log",    type=str, default="logs/demo_mujoco.jsonl")
    p.add_argument("--vis",    action="store_true", help="Save rendered PNG frames")
    p.add_argument("--out",    type=str, default="grasp_6dof/out/demo_frames")
    return p.parse_args()


def load_grasps(path: str):
    with open(path) as f:
        data = json.load(f)
    # support both {"grasps": [...]} and direct list
    if isinstance(data, dict):
        data = data.get("grasps", [])
    return data


def random_grasp_above(obj_pos, width=0.04) -> dict:
    """Fallback: straight-down grasp above the object."""
    return {
        "position": [obj_pos[0], obj_pos[1], obj_pos[2] + 0.01],
        "rpy":      [0.0, 1.5708, 0.0],
        "width":    width,
        "score":    0.5,
    }


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # ── build environment ────────────────────────────────────────────────────
    env   = MujocoGraspEnv(seed=args.seed)
    robot = SoArm101Adapter().attach(env)

    if args.vis:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    # ── load grasps ──────────────────────────────────────────────────────────
    grasp_library = None
    if args.grasps and os.path.exists(args.grasps):
        grasp_library = load_grasps(args.grasps)
        print(f"[INFO] Loaded {len(grasp_library)} grasps from {args.grasps}")
    else:
        print("[INFO] No grasp file provided — using random top-down grasps")

    # ── run trials ───────────────────────────────────────────────────────────
    successes = 0

    with TransitionLogger(args.log) as logger:
        for trial in range(args.n):
            logger.new_episode()

            # reset
            obj_pos = [rng.uniform(-0.06, 0.06),
                       rng.uniform(0.03, 0.10),
                       0.04 + 0.002]
            obs = env.reset(obj_pos=obj_pos)

            # pick a grasp
            if grasp_library:
                g = grasp_library[trial % len(grasp_library)]
            else:
                g = random_grasp_above(obj_pos)

            print(f"\n[Trial {trial+1}/{args.n}]")
            print(f"  object pos : {np.round(obj_pos, 3).tolist()}")
            print(f"  grasp pos  : {np.round(g['position'], 3)}")
            print(f"  grasp score: {g.get('score', '?'):.3f}" if 'score' in g else "")

            before = make_before_state(env, robot)

            # save pre-grasp frame
            if args.vis:
                rgb = env.render()
                _save_png(rgb, f"{args.out}/trial{trial+1:02d}_before.png")

            # execute grasp via env.step (uses built-in IK + gripper logic)
            obs, reward, done, info = env.step(g)

            after = make_after_state(env, robot)

            success = info["success"]
            successes += int(success)
            print(f"  result     : {'SUCCESS ✓' if success else 'FAIL ✗'}")
            print(f"  obj z after: {info['after_pos'][2]:.3f} m")

            # save post-grasp frame
            if args.vis:
                rgb = env.render()
                _save_png(rgb, f"{args.out}/trial{trial+1:02d}_after.png")

            logger.log(before, g, after, success, reward, info)

    print(f"\n{'='*40}")
    print(f"Success rate: {successes}/{args.n} = {successes/args.n:.1%}")
    print(f"Log saved to: {args.log}")
    env.close()


def _save_png(rgb_array, path: str):
    try:
        from PIL import Image
        Image.fromarray(rgb_array).save(path)
    except ImportError:
        import cv2
        cv2.imwrite(path, cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
