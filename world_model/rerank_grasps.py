#!/usr/bin/env python3
"""
Rerank grasp candidates using the trained MLP world-model predictor.

World-model composite score (per grasp):
    wm_score = success_prob * (1 - fell_off_prob) * (1 + clip(dz_pred, -0.1, 0.3))

The reranker can be used as a drop-in replacement for LGGSN:
    ranked = rerank(grasps, obj_pos, obj_quat, pc_stats, model)
    best_grasp = ranked[0]["grasp"]

Standalone comparison (geometry score vs world-model score):
    MUJOCO_GL=egl conda run -n bridge python world_model/rerank_grasps.py \\
        --obj banana --n-grasps 20
    MUJOCO_GL=egl conda run -n bridge python world_model/rerank_grasps.py \\
        --obj cylinder --n-grasps 20
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.transition_logger import build_feature, compute_pc_stats, FEATURE_DIM
from world_model.train_mlp_predictor import load_model, MODEL_PATH


# ── Core scoring API ──────────────────────────────────────────────────────────

def score_grasps(
    grasp_poses: np.ndarray,     # (N, 6)
    obj_pos:     np.ndarray,     # (3,)
    obj_quat:    np.ndarray,     # (4,)
    pc_stats:    np.ndarray,     # (9,)
    model:       dict,
) -> tuple:
    """
    Score a batch of grasp candidates with the world-model.

    Returns
    -------
    wm_scores : (N,) composite score
    preds     : dict with per-head predictions
    """
    N = len(grasp_poses)
    X = np.stack([
        build_feature(grasp_poses[i], obj_pos, obj_quat, pc_stats)
        for i in range(N)
    ])                                          # (N, 22)
    X_sc = model["scaler"].transform(X)

    p_success = model["clf_success"].predict_prob(X_sc)   # (N,)
    dz_pred   = model["reg_dz"].predict(X_sc)              # (N,)
    p_fell    = model["clf_fell"].predict_prob(X_sc)       # (N,)

    dz_bonus  = 1.0 + np.clip(dz_pred, -0.10, 0.30)
    wm_scores = p_success * (1.0 - p_fell) * dz_bonus

    return wm_scores, {
        "success_prob": p_success,
        "dz_pred":      dz_pred,
        "fell_prob":    p_fell,
    }


def rerank(
    grasp_list: list,
    obj_pos:    np.ndarray,
    obj_quat:   np.ndarray,
    pc_stats:   np.ndarray,
    model:      dict,
    geo_weight: float = 0.30,
    wm_weight:  float = 0.70,
) -> list:
    """
    Rerank grasp candidates by blending geometry score and world-model score.

    Parameters
    ----------
    grasp_list : list of array-like length ≥ 6.
                 Optional 7th element is the geometry score (e.g., GR-ConvNet quality).
    geo_weight / wm_weight : blending weights (must sum to 1.0).

    Returns
    -------
    List of dicts sorted by combined_score descending. Each dict contains:
        grasp, geo_score, geo_score_norm, wm_score, wm_score_norm,
        success_prob, dz_pred, fell_prob, combined_score, rank
    """
    if not grasp_list:
        return []

    poses      = np.array([np.asarray(g[:6], dtype=np.float32) for g in grasp_list])
    geo_scores = np.array([float(g[6]) if len(g) > 6 else 0.0 for g in grasp_list],
                          dtype=np.float32)

    wm_scores, preds = score_grasps(poses, obj_pos, obj_quat, pc_stats, model)

    def _norm(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-8)

    geo_norm = _norm(geo_scores)
    wm_norm  = _norm(wm_scores)
    combined = geo_weight * geo_norm + wm_weight * wm_norm

    results = []
    for i, g in enumerate(grasp_list):
        results.append({
            "grasp":           g,
            "geo_score":       float(geo_scores[i]),
            "geo_score_norm":  float(geo_norm[i]),
            "wm_score":        float(wm_scores[i]),
            "wm_score_norm":   float(wm_norm[i]),
            "success_prob":    float(preds["success_prob"][i]),
            "dz_pred":         float(preds["dz_pred"][i]),
            "fell_prob":       float(preds["fell_prob"][i]),
            "combined_score":  float(combined[i]),
        })

    results.sort(key=lambda r: r["combined_score"], reverse=True)
    for rank, r in enumerate(results):
        r["rank"] = rank
    return results


# ── Standalone comparison ─────────────────────────────────────────────────────

def compare_geo_vs_wm(
    env,
    obj_key:   str,
    n_grasps:  int,
    model:     dict,
    rng:       np.random.Generator,
):
    """
    Load an object, sample N random grasps with synthetic geo scores,
    then print a side-by-side WM-rank vs geo-rank table.
    """
    from owg_robot.env_soarm import TABLE_TOP_Z
    from scripts.collect_mujoco_transitions import (
        OBJECTS, _CENTRE_Y, _SETTLE_STEPS, sample_grasp,
    )

    obj_name = OBJECTS[obj_key]
    env.reset_robot()
    env.remove_all_obj()
    oid = env.load_obj(obj_name, name=obj_key,
                       pos=[0.0, _CENTRE_Y, TABLE_TOP_Z + 0.12])
    env._steps(_SETTLE_STEPS)

    obj_pos  = env.get_obj_pos(oid).copy()
    obj_quat = env.get_obj_pose(oid)["quaternion"].copy()
    obs      = env.get_obs(pointcloud=True)
    pc_stats = compute_pc_stats(obs, oid)

    # Sample grasps with random geometry scores (simulating GR-ConvNet output)
    grasps = []
    for _ in range(n_grasps):
        g         = sample_grasp(obj_pos, rng)
        geo_score = float(rng.uniform(0.1, 1.0))
        grasps.append(np.append(g, geo_score))

    ranked    = rerank(grasps, obj_pos, obj_quat, pc_stats, model)
    geo_order = sorted(range(n_grasps), key=lambda i: grasps[i][6], reverse=True)
    geo_rank_of = {i: r for r, i in enumerate(geo_order)}

    # Map ranked result back to original grasp index
    pose_to_idx = {tuple(grasps[i][:6].tolist()): i for i in range(n_grasps)}

    print(f"\n{'='*72}")
    print(f"  Object: {obj_key}   n_grasps={n_grasps}   "
          f"pc_stats_npts={pc_stats[8]:.2f}")
    print(f"{'='*72}")
    hdr = (f"{'WM':>4}  {'GEO':>4}  "
           f"{'wm_score':>9}  {'geo_score':>9}  "
           f"{'success_p':>9}  {'fell_p':>7}  {'dz_pred':>7}  "
           f"{'combined':>8}")
    print(hdr)
    print("-" * 72)

    for r_wm, res in enumerate(ranked):
        key     = tuple(np.asarray(res["grasp"][:6]).tolist())
        orig_i  = pose_to_idx.get(key, -1)
        r_geo   = geo_rank_of.get(orig_i, -1)
        moved   = "↑" if r_geo > r_wm else ("↓" if r_geo < r_wm else " ")
        print(f"  {r_wm:>2}    {r_geo:>2}  "
              f"{res['wm_score']:>9.3f}  {res['geo_score']:>9.3f}  "
              f"{res['success_prob']:>9.3f}  {res['fell_prob']:>7.3f}  "
              f"{res['dz_pred']:>7.3f}  "
              f"{res['combined_score']:>8.3f}  {moved}")

    print(f"\n  WM vs GEO rank correlation (Spearman approx):", end=" ")
    wm_ranks  = list(range(n_grasps))
    geo_ranks = [geo_rank_of.get(
                     pose_to_idx.get(tuple(r["grasp"][:6].tolist()
                                           if hasattr(r["grasp"], "tolist")
                                           else list(r["grasp"][:6])), -1), n_grasps)
                 for r in ranked]
    d2 = sum((w - g) ** 2 for w, g in zip(wm_ranks, geo_ranks))
    n  = n_grasps
    rho = 1 - 6 * d2 / (n * (n ** 2 - 1))
    print(f"{rho:.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--obj", default="banana",
                    choices=["banana", "cylinder", "cracker", "mustard", "drill"],
                    help="Object key (default: banana)")
    ap.add_argument("--n-grasps", type=int, default=20,
                    help="Grasp candidates to rank (default: 20)")
    ap.add_argument("--model-path", default=str(MODEL_PATH))
    ap.add_argument("--geo-weight", type=float, default=0.30)
    ap.add_argument("--wm-weight",  type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("  Run: python world_model/train_mlp_predictor.py")
        sys.exit(1)

    os.environ.setdefault("MUJOCO_GL", "egl")
    from owg_robot.env_soarm import EnvironmentSoArm

    model = load_model(model_path)
    rng   = np.random.default_rng(args.seed)
    env   = EnvironmentSoArm(vis=False, debug=False)
    try:
        compare_geo_vs_wm(env, args.obj, args.n_grasps, model, rng)
    finally:
        env.close()


if __name__ == "__main__":
    main()
