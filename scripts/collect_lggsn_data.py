#!/usr/bin/env python3
"""
Collect LGGSN training data: 18-dim features (v6 = v3 17-dim + pe_ik) for all
grasp candidates per episode, with an episode-level success label from
physics_weld execution.

For each (object, seed):
  1. Spawn object at random tabletop position, settle physics (300 steps)
  2. Sample N_CANDIDATES=30 random grasp candidates near object CoM
  3. Compute pe_ik (IK position error) for the CoM target — episode-level
     constant, matches inference where all candidates share same CoM target
  4. Execute candidates[0] under physics_weld_after_bilateral (stage-3 order)
  5. Record episode label = 1 (success) or 0 (fail)
  6. Compute 17-dim geom features + pe_ik for ALL candidates
  7. Append one JSONL row per candidate

Output: grasp_6dof/dataset/lggsn_candidates_v6.jsonl
  Columns (18 + meta):
    scene_id, query, label                           (meta)
    x, y, z, roll, pitch, yaw, width, score,         (base 8)
    dz, dz_lift, need_dz, H,                         (base 12)
    dist_to_centroid, z_rel,                          (+2, episode-context)
    local_point_density, normal_consistency,          (+3, PC features)
    contact_width_ratio,
    pe_ik                                             (+1, IK reachability)

Usage:
  # 5 objects x 100 seeds = 500 episodes
  conda run -n owg-mujoco python scripts/collect_lggsn_data.py \\
    --objects Banana,TomatoSoupCan,Pear,MustardBottle,PowerDrill --seeds 1-100

  # Quick test — 2 objects x 3 seeds = 6 episodes
  conda run -n owg-mujoco python scripts/collect_lggsn_data.py \\
    --objects banana,pear --seeds 1-3

  # Custom output path
  conda run -n owg-mujoco python scripts/collect_lggsn_data.py \\
    --objects all --seeds 1-50 --out grasp_6dof/dataset/lggsn_candidates_v6.jsonl
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

from owg_robot.env_soarm import (
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

# ── Object catalogue (same as CLAUDE.md / benchmark/runner.py) ────────────────
OBJECTS: dict[str, str] = {
    "banana":   "YcbBanana",
    "pear":     "YcbPear",
    "mustard":  "YcbMustardBottle",
    "cracker":  "YcbCrackerBox",
    "drill":    "YcbPowerDrill",
    "can":      "YcbTomatoSoupCan",
    "cylinder": "YcbMediumClamp",
}

_CENTRE_Y     = -0.40     # reachable table centre (y-axis)
_SPREAD_XY    = 0.06      # max random XY offset from centre
_DROP_Z       = TABLE_TOP_Z + 0.12   # initial drop height
_SETTLE_STEPS = 300
N_CANDIDATES  = 30        # grasp candidates per episode
DEFAULT_OUT   = Path("grasp_6dof/dataset/lggsn_candidates_v6.jsonl")


# ── Helpers ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _suppress_stdout():
    """Redirect stdout to /dev/null for noisy physics calls."""
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def _compute_H(obs: dict, obj_id: int) -> float:
    """Object height estimate: max_z of segmented object points - TABLE_TOP_Z."""
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
    """
    Sample n top-down grasp candidates near the object CoM.
    Returns (n, 6): [x, y, z, yaw, opening_len, obj_height]
    """
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
                           episode_pcd: np.ndarray,
                           pe_ik: float) -> list[dict]:
    """
    Build 18-dim feature dict for every candidate.

    cands: (N, 6)  [x, y, z, yaw, opening_len, obj_height]
    episode_pcd: (M, 3) full scene pointcloud in robot-base frame
    pe_ik: IK position error for the CoM target (same for all candidates in
           episode — mirrors inference where all candidates share CoM target)

    dist_to_centroid and z_rel are episode-level (computed across all N
    candidates) and stored directly so the training script needs no recompute.
    """
    N  = len(cands)
    xs = cands[:, 0]
    ys = cands[:, 1]
    zs = cands[:, 2]

    cx, cy = xs.mean(), ys.mean()
    dist_c = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    z_min, z_max = zs.min(), zs.max()
    z_rel = (zs - z_min) / (z_max - z_min + 1e-8)

    rows = []
    for i in range(N):
        x, y, z, yaw, opening, H = cands[i]
        roll  = math.pi
        pitch = 0.0
        pos_g = np.array([x, y, z])
        R     = rpy_to_R(roll, pitch, float(yaw))

        ld  = local_point_density(pos_g, R, float(opening), episode_pcd)
        nc  = normal_consistency(pos_g, R, float(opening), episode_pcd)
        cwr = contact_width_ratio(pos_g, R, float(opening), episode_pcd)

        rows.append({
            "x":                   round(float(x),       6),
            "y":                   round(float(y),       6),
            "z":                   round(float(z),       6),
            "roll":                round(roll,           6),
            "pitch":               round(pitch,          6),
            "yaw":                 round(float(yaw),     6),
            "width":               round(float(opening), 6),
            "score":               0.0,
            "dz":                  0.0,
            "dz_lift":             0.0,
            "need_dz":             0.0,
            "H":                   round(float(H),       6),
            "dist_to_centroid":    round(float(dist_c[i]), 6),
            "z_rel":               round(float(z_rel[i]),  6),
            "local_point_density": round(float(ld),  6),
            "normal_consistency":  round(float(nc),  6),
            "contact_width_ratio": round(float(cwr), 6),
            "pe_ik":               round(float(pe_ik),   6),
        })
    return rows


# ── Episode runner ────────────────────────────────────────────────────────────

def collect_episode(
    env:       EnvironmentSoArm,
    obj_key:   str,
    obj_class: str,
    seed:      int,
    out_path:  Path,
    quiet:     bool = False,
) -> bool:
    """Run one collection episode. Returns success label."""
    scene_id = f"{obj_key}_s{seed:03d}"
    rng = np.random.default_rng(seed)

    # ── spawn + settle ────────────────────────────────────────────────────────
    with _suppress_stdout():
        env.reset_robot()
        env.remove_all_obj()
        cx     = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
        cy     = _CENTRE_Y + float(rng.uniform(-0.04, 0.04))
        obj_id = env.load_obj(obj_class, name=obj_key, pos=[cx, cy, _DROP_Z])
        env._steps(_SETTLE_STEPS)

    # ── pre-grasp observation + pointcloud ────────────────────────────────────
    obs         = env.get_obs(pointcloud=True)
    episode_pcd = obs["points"].reshape(-1, 3)
    H           = _compute_H(obs, obj_id)

    # ── sample candidates ─────────────────────────────────────────────────────
    try:
        com = env.get_obj_com_pos(obj_id)
    except Exception:
        com = env.get_obj_pos(obj_id)
    cands = _sample_candidates(com, H, rng, N_CANDIDATES)

    # ── pe_ik: IK error for the CoM target (episode-level constant) ──────────
    # Matches inference: _setup_grasps_mujoco targets the same CoM for all
    # candidates, so pe_ik is the same across all grasps in an episode.
    com_target = np.array([float(com[0]), float(com[1]),
                           float(com[2]) + GRASP_Z_TABLE_MARGIN])
    pe_iks = env.compute_ik_reachability([com_target])
    pe_ik  = pe_iks[0]

    # ── register + execute candidates[0] (stage-3: no reranking) ─────────────
    env.set_obj_grasps(obj_id, [np.array(c, dtype=np.float32) for c in cands],
                       grasp_rects=[])
    with _suppress_stdout():
        success, _, _ = env.pick_obj_by_id(obj_id, grasp_indices=[0])
    label = 1 if success else 0

    # ── compute 18-dim features for ALL candidates ────────────────────────────
    feat_rows = _featurize_candidates(cands, episode_pcd, pe_ik)

    # ── append to JSONL ───────────────────────────────────────────────────────
    with open(out_path, "a") as f:
        for feat in feat_rows:
            row = {"scene_id": scene_id, "query": obj_key, "label": label, **feat}
            f.write(json.dumps(row) + "\n")

    if not quiet:
        sym      = "✓" if success else "✗"
        ld_mean  = float(np.mean([r["local_point_density"] for r in feat_rows]))
        nc_mean  = float(np.mean([r["normal_consistency"]  for r in feat_rows]))
        cwr_mean = float(np.mean([r["contact_width_ratio"] for r in feat_rows]))
        print(f"  [{sym}] {obj_key:<10} s{seed:03d}  H={H:.3f}  "
              f"ld={ld_mean:.3f}  nc={nc_mean:.3f}  cwr={cwr_mean:.3f}  "
              f"pe_ik={pe_ik*1000:.1f}mm")

    return bool(success)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--objects", default="all",
                    help="Comma-separated object keys, or 'all' (default: all)")
    ap.add_argument("--seeds", default="1-50",
                    help="Seed range 'start-end' or comma list (default: 1-50)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"Output JSONL (default: {DEFAULT_OUT})")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-episode progress lines")
    args = ap.parse_args()

    # parse objects (accept both lowercase keys and display names like "Banana")
    _DISPLAY_TO_KEY = {
        "banana": "banana", "Banana": "banana",
        "tomatosoupcan": "can", "TomatoSoupCan": "can", "can": "can",
        "pear": "pear", "Pear": "pear",
        "mustardbottle": "mustard", "MustardBottle": "mustard", "mustard": "mustard",
        "powerdrill": "drill", "PowerDrill": "drill", "drill": "drill",
        "crackerbox": "cracker", "CrackerBox": "cracker", "cracker": "cracker",
        "mediumclamp": "cylinder", "MediumClamp": "cylinder",
        "scissors": "cylinder", "Scissors": "cylinder", "cylinder": "cylinder",
    }
    if args.objects == "all":
        obj_keys = list(OBJECTS.keys())
    else:
        raw_keys = [k.strip() for k in args.objects.split(",")]
        obj_keys = []
        for k in raw_keys:
            mapped = _DISPLAY_TO_KEY.get(k, k)
            if mapped not in OBJECTS:
                ap.error(f"Unknown object '{k}'. Valid: {list(OBJECTS.keys())}")
            obj_keys.append(mapped)

    # parse seeds
    s = args.seeds
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x.strip()) for x in s.split(",")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # init env — pre-register all 7 object types so model is built once
    env = EnvironmentSoArm(vis=False, debug=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD)
    logical_names = [cls.replace("Ycb", "") for cls in OBJECTS.values()]
    env.preload_pool(logical_names)

    total, n_ok = 0, 0
    n_ep = len(obj_keys) * len(seeds)
    print(f"Collecting {len(obj_keys)} objects x {len(seeds)} seeds "
          f"= {n_ep} episodes  ({N_CANDIDATES} candidates each)")
    print(f"Output -> {out_path}\n")

    try:
        for obj_key in obj_keys:
            print(f"-- {obj_key} ({OBJECTS[obj_key]}) --")
            for seed in seeds:
                ok = collect_episode(
                    env, obj_key, OBJECTS[obj_key],
                    seed, out_path, quiet=args.quiet,
                )
                total += 1
                n_ok  += int(ok)
    finally:
        env.close()

    sr = n_ok / total if total else 0.0
    n_rows = total * N_CANDIDATES
    kb     = out_path.stat().st_size // 1024 if out_path.exists() else 0
    print(f"\nDone  {n_ok}/{total} episodes succeeded ({sr:.1%})")
    print(f"Rows written: {n_rows}  |  File: {out_path}  ({kb} KB)")

    # ── spot-check first row ──────────────────────────────────────────────────
    with open(out_path) as f:
        sample = json.loads(f.readline())
    feat_keys = [k for k in sample if k not in ("scene_id", "query", "label")]
    print(f"\nRow sample ({len(feat_keys)}-dim features):")
    for k in feat_keys:
        print(f"  {k:<24} = {sample[k]}")


if __name__ == "__main__":
    main()
