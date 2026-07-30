#!/usr/bin/env python3
"""
Regenerate the RGB scene image for existing lggsn_candidates_v9.jsonl scenes.

The original collection script (collect_lggsn_data.py) rendered an RGB image
per episode to compute SAM visual_feat, but never saved the raw image to
disk -- only the resulting 256-dim per-candidate feature vector was kept.

Object spawn position is drawn from `np.random.default_rng(seed)` as the
VERY FIRST two draws (cx, cy), before any candidate sampling -- so replaying
just the object-load + settle steps (skipping candidate sampling entirely)
reproduces the exact same object pose / rendered image deterministically,
without needing to reproduce anything else about the episode.

Needed for: a real (not-toy) affordance-auxiliary VLA check (Stage 1 of
paperA_data/new_method_affordance_auxiliary_proposal.md) that requires
actual images, not just precomputed SAM feature vectors.

Usage:
  conda run -n owg-mujoco python scripts/regen_lggsn_scene_images.py \
    --objects all --seeds 1-30 --out grasp_6dof/dataset/lggsn_scene_images
"""
import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from PIL import Image as PILImage

from tango_robot.env_soarm import EnvironmentSoArm, TABLE_TOP_Z, GRASP_MODE_PHYSICS_WELD

OBJECTS = {
    "banana":   "YcbBanana",
    "pear":     "YcbPear",
    "mustard":  "YcbMustardBottle",
    "cracker":  "YcbCrackerBox",
    "drill":    "YcbPowerDrill",
    "can":      "YcbTomatoSoupCan",
    "cylinder": "YcbMediumClamp",
}
_CENTRE_Y     = -0.40
_SPREAD_XY    = 0.06
_DROP_Z       = TABLE_TOP_Z + 0.12
_SETTLE_STEPS = 300


@contextlib.contextmanager
def _suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def render_scene(env, obj_key, obj_class, seed):
    rng = np.random.default_rng(seed)
    with _suppress_stdout():
        env.reset_robot()
        env.remove_all_obj()
        cx = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
        cy = _CENTRE_Y + float(rng.uniform(-0.04, 0.04))
        env.load_obj(obj_class, name=obj_key, pos=[cx, cy, _DROP_Z])
        env._steps(_SETTLE_STEPS)
    obs = env.get_obs(pointcloud=False)
    return obs["image"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="all")
    ap.add_argument("--seeds", default="1-30")
    ap.add_argument("--out", default="grasp_6dof/dataset/lggsn_scene_images")
    args = ap.parse_args()

    obj_keys = list(OBJECTS.keys()) if args.objects == "all" else \
        [k.strip() for k in args.objects.split(",")]

    s = args.seeds
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x.strip()) for x in s.split(",")]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = EnvironmentSoArm(vis=False, debug=False, grasp_mode=GRASP_MODE_PHYSICS_WELD)
    logical_names = [cls.replace("Ycb", "") for cls in OBJECTS.values()]
    env.preload_pool(logical_names)

    n_done = 0
    try:
        for obj_key in obj_keys:
            obj_class = OBJECTS[obj_key]
            for seed in seeds:
                scene_id = f"{obj_key}_s{seed:03d}"
                out_path = out_dir / f"{scene_id}.png"
                if out_path.exists():
                    continue
                img = render_scene(env, obj_key, obj_class, seed)
                PILImage.fromarray(img).save(out_path)
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  ... {n_done} images rendered ({scene_id})")
    finally:
        env.close()

    print(f"Done. {n_done} new images written to {out_dir}")


if __name__ == "__main__":
    main()
