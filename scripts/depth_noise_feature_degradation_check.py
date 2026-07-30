#!/usr/bin/env python3
"""
Phase 0 diagnostic (2026-07-13): does RealSense D435i-realistic depth noise
degrade the 3 hand-engineered local geometric features (local_point_density,
normal_consistency, contact_width_ratio) enough to matter for grasp
candidate scoring? Purely offline -- injects synthetic noise into the
already-cached clean simulated point clouds
(grasp_6dof/dataset/lggsn_object_pointclouds/*.npz), no camera or real
hardware needed. Motivated by literature (R2SGrasp's "Real-to-Sim Feature
Enhancer", DiffuDepGrasp, Camera Depth Models) confirming this is a real,
active problem class, not a hypothetical concern.

D435i noise parameters (from Intel's own published RMS-error guidance):
accuracy < 1% of distance at close range, error scales roughly with the
square of distance; ~2.5-5mm RMS at 1m. Table-top grasping distance here is
assumed ~0.4-0.7m (camera looking down/across at the workspace).

Corruption model (3 independently tunable severity knobs):
  1. Depth-dependent Gaussian jitter (per-point, scaled by distance^2)
  2. Edge/boundary voids (points near the local bbox edge dropped with
     probability p -- mimics structured-light dropout at depth discontinuities)
  3. Small-object sparsification (extra uniform point removal -- mimics the
     IR pattern not fully covering small/thin objects, per Intel's own
     documentation and the D415 metrological characterization literature)

Usage:
  python3 scripts/depth_noise_feature_degradation_check.py --object pear
"""
import argparse
import json
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grasp_6dof.grasp_sampler import rpy_to_R, local_point_density, normal_consistency, contact_width_ratio
from tango_robot.env_soarm import TABLE_TOP_Z

JSONL_PATH = "grasp_6dof/dataset/lggsn_candidates_v9.jsonl"
PCD_DIR = "grasp_6dof/dataset/lggsn_object_pointclouds"
CAM_DIST_M = 0.55  # assumed camera-to-workspace distance, mid-range for a desk-mounted D435i

SEVERITY_LEVELS = {
    "clean":  dict(gauss_scale=0.0,  edge_void_p=0.0,  sparsify_p=0.0),
    "mild":   dict(gauss_scale=0.5,  edge_void_p=0.10, sparsify_p=0.10),
    "medium": dict(gauss_scale=1.0,  edge_void_p=0.25, sparsify_p=0.25),
    "severe": dict(gauss_scale=2.0,  edge_void_p=0.45, sparsify_p=0.45),
}


def d435i_rms_error(dist_m: float) -> float:
    """RMS depth error per Intel's published guidance: ~2.5-5mm at 1m,
    scaling with distance^2. Using the mid-point (3.5mm @ 1m) as the base."""
    return 0.0035 * (dist_m ** 2) / (1.0 ** 2) if dist_m > 0 else 0.0


def corrupt_pointcloud(pcd: np.ndarray, ref_xy: np.ndarray, rng: np.random.Generator,
                        gauss_scale: float, edge_void_p: float, sparsify_p: float) -> np.ndarray:
    if gauss_scale == 0 and edge_void_p == 0 and sparsify_p == 0:
        return pcd.copy()

    pcd = pcd.copy()
    n = len(pcd)

    # 1. depth-dependent Gaussian jitter (z dominated, matches stereo depth noise)
    base_err = d435i_rms_error(CAM_DIST_M) * gauss_scale
    pcd[:, 2] += rng.normal(0, base_err, size=n)
    pcd[:, :2] += rng.normal(0, base_err * 0.5, size=(n, 2))  # lateral jitter, smaller

    # 2. edge/boundary voids: drop points near the object's own local boundary
    #    (approximate "edge" as points far from the object's own centroid --
    #    i.e. the object's silhouette boundary, where structured-light
    #    dropout is worst per the literature)
    obj_dist = np.linalg.norm(pcd[:, :2] - ref_xy, axis=1)
    near_edge = obj_dist > np.percentile(obj_dist, 70)  # outer 30% = "edge" region
    drop_edge = near_edge & (rng.uniform(size=n) < edge_void_p)

    # 3. small-object sparsification: extra uniform dropout on top
    drop_sparse = rng.uniform(size=n) < sparsify_p

    keep = ~(drop_edge | drop_sparse)
    return pcd[keep]


def live_geom_feats(x, y, yaw, width, H, pe_ik, pcd):
    roll, pitch = math.pi, 0.0
    R = rpy_to_R(roll, pitch, float(yaw))
    pos_g = np.array([x, y, TABLE_TOP_Z + 0.02])
    ld = local_point_density(pos_g, R, width, pcd)
    nc = normal_consistency(pos_g, R, width, pcd)
    cwr = contact_width_ratio(pos_g, R, width, pcd)
    return ld, nc, cwr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="pear")
    ap.add_argument("--n-trials", type=int, default=3, help="repeat corruption with different seeds")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(JSONL_PATH)]
    obj_rows = [r for r in rows if r["query"] == args.object]
    groups = np.array([r["scene_id"] for r in obj_rows])
    y = np.array([r["label"] for r in obj_rows])

    d = np.load(f"{PCD_DIR}/{args.object}.npz")
    clean_pcd, ref_xy = d["points"], d["ref_xy"]

    print(f"Object: {args.object}  |  clean point cloud: {len(clean_pcd)} points")
    print(f"Assumed camera distance: {CAM_DIST_M}m  |  D435i RMS error @ this range: "
          f"{d435i_rms_error(CAM_DIST_M)*1000:.2f}mm\n")

    print(f"{'severity':<10} {'geom3_AUC (mean±std over trials)':<35} {'n_points_after_corruption'}")
    for level, params in SEVERITY_LEVELS.items():
        aucs, n_pts_list = [], []
        for trial in range(args.n_trials):
            rng = np.random.default_rng(trial)
            noisy_pcd = corrupt_pointcloud(clean_pcd, ref_xy, rng, **params)
            n_pts_list.append(len(noisy_pcd))

            X = np.array([
                live_geom_feats(r["x"], r["y"], r["yaw"], r["width"], r["H"], r["pe_ik"], noisy_pcd)
                for r in obj_rows
            ])

            gkf = GroupKFold(n_splits=5)
            fold_aucs = []
            for tr_idx, te_idx in gkf.split(X, y, groups):
                sc = StandardScaler().fit(X[tr_idx])
                clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr_idx]), y[tr_idx])
                fold_aucs.append(roc_auc_score(y[te_idx], clf.predict_proba(sc.transform(X[te_idx]))[:, 1]))
            aucs.append(np.mean(fold_aucs))

        print(f"{level:<10} {np.mean(aucs):.3f} ± {np.std(aucs):.3f}{'':<20} "
              f"{np.mean(n_pts_list):.0f}/{len(clean_pcd)}")


if __name__ == "__main__":
    main()
