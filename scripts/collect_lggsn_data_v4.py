#!/usr/bin/env python3
"""
Collect LGGSN v4 training data: per-candidate success labels.

Key difference from v3: each candidate gets its own label from an individual
execution attempt, enabling within-episode pairwise supervision.

For each (object, seed):
  1. Spawn object, settle, record initial_state
  2. Sample N_CANDIDATES=30 random grasp candidates near object CoM
  3. Compute 17-dim features for ALL candidates from the initial pointcloud
  4. For candidates[0..N_EVAL-1]: reset object to initial_state, execute,
     record per-candidate label (1=success, 0=fail)
  5. Remaining candidates (not executed): label=0 (conservative)
  6. Append one JSONL row per candidate with its own label

Output: grasp_6dof/dataset/lggsn_candidates_v4.jsonl
  Same columns as v3 plus:
    evaluated: 1 if this candidate was actually executed, else 0

Usage:
  conda run -n owg-mujoco python scripts/collect_lggsn_data_v4.py
  conda run -n owg-mujoco python scripts/collect_lggsn_data_v4.py \\
    --objects banana,drill --seeds 1-20 --n-eval 5
"""

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from tango_robot.env_soarm import (
    EnvironmentSoArm,
    TABLE_TOP_Z,
    GRASP_Z_TABLE_MARGIN,
    GRASP_MODE_PHYSICS_WELD,
)
from grasp_6dof.grasp_sampler import (
    rpy_to_R,
    local_point_density,
    normal_consistency,
    contact_width_ratio,
)

OBJECTS: dict[str, str] = {
    "banana":   "YcbBanana",
    "pear":     "YcbPear",
    "mustard":  "YcbMustardBottle",
    "cracker":  "YcbCrackerBox",
    "drill":    "YcbPowerDrill",
    "can":      "YcbTomatoSoupCan",
    "cylinder": "YcbMediumClamp",
}

_CENTRE_Y      = -0.40
_SPREAD_XY     = 0.06
_DROP_Z        = TABLE_TOP_Z + 0.12
_SETTLE_STEPS  = 300
N_CANDIDATES   = 30
N_EVAL_DEFAULT = 5       # candidates actually executed per episode
DEFAULT_OUT    = Path("grasp_6dof/dataset/lggsn_candidates_v4.jsonl")


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dev:
        old, sys.stdout = sys.stdout, dev
        try:
            yield
        finally:
            sys.stdout = old


def _compute_H(obs: dict, obj_id: int) -> float:
    seg    = obs.get("seg")
    points = obs.get("points")
    if seg is None or points is None:
        return 0.05
    flat_pts = points.reshape(-1, 3) if points.ndim == 3 else points
    flat_seg = seg.ravel()
    n        = min(len(flat_seg), len(flat_pts))
    obj_pts  = flat_pts[:n][flat_seg[:n] == obj_id]
    if len(obj_pts) < 5:
        return 0.05
    return float(max(0.005, obj_pts[:, 2].max() - TABLE_TOP_Z))


def _sample_candidates(com: np.ndarray, H: float,
                        rng: np.random.Generator, n: int) -> np.ndarray:
    rows = []
    for _ in range(n):
        x       = float(com[0] + rng.uniform(-_SPREAD_XY, _SPREAD_XY))
        y       = float(com[1] + rng.uniform(-_SPREAD_XY, _SPREAD_XY))
        z       = float(com[2] + GRASP_Z_TABLE_MARGIN)
        yaw     = float(rng.uniform(-math.pi / 2, math.pi / 2))
        opening = float(rng.uniform(0.04, 0.09))
        rows.append([x, y, z, yaw, opening, H])
    return np.array(rows, dtype=np.float64)


def _featurize_candidates(cands: np.ndarray,
                           episode_pcd: np.ndarray) -> list[dict]:
    N  = len(cands)
    xs = cands[:, 0]; ys = cands[:, 1]; zs = cands[:, 2]
    cx, cy = xs.mean(), ys.mean()
    dist_c = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    z_min, z_max = zs.min(), zs.max()
    z_rel  = (zs - z_min) / (z_max - z_min + 1e-8)

    rows = []
    for i in range(N):
        x, y, z, yaw, opening, H = cands[i]
        roll, pitch = math.pi, 0.0
        pos_g = np.array([x, y, z])
        R     = rpy_to_R(roll, pitch, float(yaw))
        ld    = local_point_density(pos_g, R, float(opening), episode_pcd)
        nc    = normal_consistency(pos_g, R, float(opening), episode_pcd)
        cwr   = contact_width_ratio(pos_g, R, float(opening), episode_pcd)
        rows.append({
            "x":                   round(float(x),       6),
            "y":                   round(float(y),       6),
            "z":                   round(float(z),       6),
            "roll":                round(roll,           6),
            "pitch":               round(pitch,          6),
            "yaw":                 round(float(yaw),     6),
            "width":               round(float(opening), 6),
            "score":               0.0,
            "dz":                  0.0, "dz_lift": 0.0, "need_dz": 0.0,
            "H":                   round(float(H),       6),
            "dist_to_centroid":    round(float(dist_c[i]), 6),
            "z_rel":               round(float(z_rel[i]),  6),
            "local_point_density": round(float(ld),  6),
            "normal_consistency":  round(float(nc),  6),
            "contact_width_ratio": round(float(cwr), 6),
        })
    return rows


def collect_episode(
    env:       EnvironmentSoArm,
    obj_key:   str,
    obj_class: str,
    seed:      int,
    n_eval:    int,
    out_path:  Path,
    quiet:     bool = False,
) -> tuple[int, int]:
    """Run one v4 episode. Returns (n_success, n_evaluated)."""
    scene_id = f"{obj_key}_s{seed:03d}"
    rng = np.random.default_rng(seed)

    # ── spawn + settle ────────────────────────────────────────────────────────
    with _quiet():
        env.reset_robot()
        env.remove_all_obj()
        cx     = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
        cy     = _CENTRE_Y + float(rng.uniform(-0.04, 0.04))
        obj_id = env.load_obj(obj_class, name=obj_key, pos=[cx, cy, _DROP_Z])
        env._steps(_SETTLE_STEPS)

    # ── snapshot initial object state for resets ──────────────────────────────
    initial_state = env.get_obj_states()

    # ── observation + featurize (done once from initial pose) ─────────────────
    obs         = env.get_obs(pointcloud=True)
    episode_pcd = obs["points"].reshape(-1, 3)
    H           = _compute_H(obs, obj_id)
    try:
        com = env.get_obj_com_pos(obj_id)
    except Exception:
        com = env.get_obj_pos(obj_id)

    cands     = _sample_candidates(com, H, rng, N_CANDIDATES)
    feat_rows = _featurize_candidates(cands, episode_pcd)

    # ── per-candidate execution ───────────────────────────────────────────────
    labels    = [0] * N_CANDIDATES
    evaluated = [0] * N_CANDIDATES
    n_success = 0

    for i in range(min(n_eval, N_CANDIDATES)):
        # reset object to initial settled pose before each attempt
        with _quiet():
            env.set_obj_state(initial_state)
            env.reset_robot()
            env.set_obj_grasps(obj_id,
                               [np.array(cands[i], dtype=np.float32)],
                               grasp_rects=[])
            success, _, _ = env.pick_obj_by_id(obj_id, grasp_indices=[0])

        labels[i]    = 1 if success else 0
        evaluated[i] = 1
        n_success   += int(success)

    # ── write JSONL ───────────────────────────────────────────────────────────
    with open(out_path, "a") as f:
        for feat, lbl, evl in zip(feat_rows, labels, evaluated):
            row = {
                "scene_id":  scene_id,
                "query":     obj_key,
                "label":     lbl,
                "evaluated": evl,
                **feat,
            }
            f.write(json.dumps(row) + "\n")

    if not quiet:
        sr_ep = f"{n_success}/{n_eval}"
        print(f"  {obj_key:<10} s{seed:03d}  sr={sr_ep}  "
              f"H={H:.3f}  eval={n_eval}")

    return n_success, n_eval


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--objects", default="all")
    ap.add_argument("--seeds",   default="1-100")
    ap.add_argument("--n-eval",  type=int, default=N_EVAL_DEFAULT,
                    help="Candidates to execute per episode (default 5)")
    ap.add_argument("--out",     default=str(DEFAULT_OUT))
    ap.add_argument("--quiet",   action="store_true")
    ap.add_argument("--append",  action="store_true",
                    help="Append to existing file instead of overwriting")
    args = ap.parse_args()

    obj_keys = list(OBJECTS.keys()) if args.objects == "all" else \
               [k.strip() for k in args.objects.split(",")]
    for k in obj_keys:
        if k not in OBJECTS:
            ap.error(f"Unknown object '{k}'. Valid: {list(OBJECTS.keys())}")

    s = args.seeds
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x.strip()) for x in s.split(",")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.append and out_path.exists():
        out_path.unlink()

    env = EnvironmentSoArm(vis=False, debug=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD)
    logical_names = [cls.replace("Ycb", "") for cls in OBJECTS.values()]
    env.preload_pool(logical_names)

    n_ep = len(obj_keys) * len(seeds)
    print(f"V4 collection: {len(obj_keys)} objects × {len(seeds)} seeds "
          f"= {n_ep} episodes  (n_eval={args.n_eval} candidates each)")
    print(f"Output → {out_path}\n")

    total_ep = total_ok = total_eval = 0
    try:
        for obj_key in obj_keys:
            print(f"── {obj_key} ({OBJECTS[obj_key]}) ──")
            for seed in seeds:
                n_ok, n_ev = collect_episode(
                    env, obj_key, OBJECTS[obj_key],
                    seed, args.n_eval, out_path, quiet=args.quiet,
                )
                total_ep   += 1
                total_ok   += n_ok
                total_eval += n_ev
    finally:
        env.close()

    rows   = total_ep * N_CANDIDATES
    sr     = total_ok / total_eval if total_eval else 0.0
    kb     = out_path.stat().st_size // 1024 if out_path.exists() else 0
    print(f"\nDone  {total_ep} episodes  "
          f"exec SR={total_ok}/{total_eval} ({sr:.1%})")
    print(f"Rows written: {rows}  |  File: {out_path}  ({kb} KB)")


if __name__ == "__main__":
    main()
