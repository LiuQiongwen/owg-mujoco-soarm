#!/usr/bin/env python3
"""
Cache one representative point cloud per object (deterministic replay of a
fixed seed, same technique as regen_lggsn_scene_images.py) for use as the
live-feature-computation reference during Geo-EBM hard-negative mining
(scripts/train_geo_ebm_grasp.py) -- analogous to how the existing EBM v2
already uses one mean visual feature per object for mining conditioning
(train_ebm_grasp.py's obj_mean_vis), just extended to point clouds.

Usage:
  conda run -n tango python scripts/cache_object_pointclouds.py
"""
import contextlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import open3d as o3d

from tango_robot.env_soarm import EnvironmentSoArm, TABLE_TOP_Z, GRASP_MODE_PHYSICS_WELD

OBJECTS = {
    "banana": "YcbBanana", "pear": "YcbPear", "mustard": "YcbMustardBottle",
    "cracker": "YcbCrackerBox", "drill": "YcbPowerDrill",
    "can": "YcbTomatoSoupCan", "cylinder": "YcbMediumClamp",
}
_CENTRE_Y, _SPREAD_XY = -0.40, 0.06
_DROP_Z = TABLE_TOP_Z + 0.12
_SETTLE_STEPS = 300
REF_SEED = 1
OUT_DIR = Path("grasp_6dof/dataset/lggsn_object_pointclouds")


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dev:
        old, sys.stdout = sys.stdout, dev
        try:
            yield
        finally:
            sys.stdout = old


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = EnvironmentSoArm(vis=False, debug=False, grasp_mode=GRASP_MODE_PHYSICS_WELD)
    logical_names = [cls.replace("Ycb", "") for cls in OBJECTS.values()]
    env.preload_pool(logical_names)

    try:
        for obj_key, obj_class in OBJECTS.items():
            rng = np.random.default_rng(REF_SEED)
            with _quiet():
                env.reset_robot()
                env.remove_all_obj()
                cx = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
                cy = _CENTRE_Y + float(rng.uniform(-0.04, 0.04))
                env.load_obj(obj_class, name=obj_key, pos=[cx, cy, _DROP_Z])
                env._steps(_SETTLE_STEPS)
            obs = env.get_obs(pointcloud=True)
            pcd = obs["points"].reshape(-1, 3)

            # Precompute per-point normals ONCE for the whole cloud (the
            # expensive part -- KD-tree + consistent-orientation propagation
            # -- so live_geom_feats() during CEM mining can just index into
            # this array per candidate instead of re-running Open3D's normal
            # estimation pipeline from scratch for every one of thousands of
            # candidates evaluated during training.
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
            o3d_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            )
            try:
                o3d_pcd.orient_normals_consistent_tangent_plane(k=min(30, len(pcd) - 1))
            except RuntimeError:
                pass
            normals = np.asarray(o3d_pcd.normals)

            # Save the reference object's own (cx, cy) alongside the cloud so
            # any consumer can recentre this cloud onto a DIFFERENT live
            # object position later (translate by target_xy - ref_xy) --
            # this cloud's absolute coordinates are only valid for this one
            # specific replayed scene, not for other trials/seeds.
            np.savez(OUT_DIR / f"{obj_key}.npz", points=pcd, normals=normals,
                     ref_xy=np.array([cx, cy], dtype=np.float32))
            print(f"{obj_key}: {pcd.shape[0]} points + normals cached (ref_xy={cx:.4f},{cy:.4f})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
