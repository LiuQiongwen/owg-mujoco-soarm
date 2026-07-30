#!/usr/bin/env python3
"""Closed-loop MuJoCo evaluation for a local LeRobot ACT action policy."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paperA_data" / "scripts"))
os.environ.setdefault("MUJOCO_GL", "egl")
import _lerobot_groot_patch  # noqa: E402,F401

from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from tango_robot.env_soarm import EnvironmentSoArm, ARM_JOINTS, TABLE_TOP_Z  # noqa: E402
from scripts.record_sim_lerobot_episodes import OBJECTS, get_state  # noqa: E402


def rollout(env, policy, pre, post, obj_key, seed, max_steps):
    rng = np.random.default_rng(seed)
    env.reset_robot(); env.remove_all_obj(); env._detach_obj()
    cx = float(rng.uniform(-0.06, 0.06))
    cy = -0.30 + float(rng.uniform(-0.04, 0.04))
    oid = env.load_obj(OBJECTS[obj_key], name=obj_key,
                       pos=[cx, cy, TABLE_TOP_Z + 0.12])
    env._steps(300)
    z0 = float(env.get_obj_pos(oid)[2])
    policy.reset()
    attached = False
    for _ in range(max_steps):
        image = env.get_obs(pointcloud=False)["image"]
        obs = {"observation.state": torch.from_numpy(get_state(env))[None],
               "observation.images.overhead":
                   torch.from_numpy(image).permute(2, 0, 1).float()[None] / 255.0,
               "task": [f"Pick up the {obj_key} and place it in the tray"]}
        with torch.no_grad():
            action = post(policy.select_action(pre(obs))).squeeze(0).cpu().numpy()
        for act_id, value in zip(env._arm_act_ids, action[:5]):
            lo, hi = env.model.actuator_ctrlrange[act_id]
            env.data.ctrl[act_id] = np.clip(value, lo, hi)
        lo, hi = env.model.actuator_ctrlrange[env._grip_act_id]
        env.data.ctrl[env._grip_act_id] = np.clip(action[5], lo, hi)
        for _ in range(25):  # 500 Hz physics / 20 Hz policy
            env.step_simulation()
            attached = env.update_policy_grasp_attachment(oid) or attached
    env._steps(40)
    z1 = float(env.get_obj_pos(oid)[2])
    success = bool(attached and z1 > z0 + 0.07)
    return {"object": obj_key, "seed": seed, "success": success,
            "attached": attached, "z_before": z0, "z_after": z1,
            "dz": z1 - z0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--objects", default="cracker,mustard,drill")
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=500)
    ap.add_argument("--max-policy-steps", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    policy = ACTPolicy.from_pretrained(args.checkpoint); policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cpu"}})
    objects = [x.strip() for x in args.objects.split(",")]
    env = EnvironmentSoArm(vis=False, debug=False)
    env.preload_pool([OBJECTS[k].replace("Ycb", "") for k in objects])
    rows = []
    try:
        for oi, obj in enumerate(objects):
            for si in range(args.scenes):
                row = rollout(env, policy, pre, post, obj,
                              args.base_seed * 10000 + oi * 100 + si,
                              args.max_policy_steps)
                rows.append(row); print(json.dumps(row), flush=True)
    finally:
        env.close()
    result = {"checkpoint": args.checkpoint, "rows": rows,
              "successes": sum(r["success"] for r in rows), "n": len(rows)}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"successes": result["successes"], "n": result["n"]}))


if __name__ == "__main__":
    main()
