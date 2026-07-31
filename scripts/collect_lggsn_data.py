#!/usr/bin/env python3
"""
Collect LGGSN training data: per-candidate labels (v7).

Unlike v6 (episode-level label from candidates[0]), v7 executes each of the
top --max-candidates individually, respawning the object at the same pose
between each execution, and records an independent success/fail per candidate.

For each (object, seed):
  1. Spawn object at random tabletop position, settle physics (300 steps)
  2. Sample N_CANDIDATES=30 random grasp candidates near object CoM
  3. Compute pe_ik (IK position error) for the CoM target — episode-level
     constant, matches inference where all candidates share same CoM target
  4. Featurize all 30 candidates upfront (episode-context stats use full set)
  5. For each of the top --max-candidates (default 5):
       a. Respawn object at stored spawn position, settle physics
       b. Execute that specific candidate under physics_weld_after_bilateral
       c. Record per-candidate label = 1 (success) or 0 (fail)
       d. Append one JSONL row with candidate_idx + per-candidate label

Output: grasp_6dof/dataset/lggsn_candidates_v7.jsonl
  Columns (18 + meta):
    scene_id, query, candidate_idx, label            (meta)
    x, y, z, roll, pitch, yaw, width, score,         (base 8)
    dz, dz_lift, need_dz, H,                         (base 12)
    dist_to_centroid, z_rel,                          (+2, episode-context)
    local_point_density, normal_consistency,          (+3, PC features)
    contact_width_ratio,
    pe_ik                                             (+1, IK reachability)

Usage:
  # 7 objects x 100 seeds x 5 candidates = 3500 executions (~10-15 hrs)
  conda run -n owg-mujoco python scripts/collect_lggsn_data.py \
    --objects all --seeds 1-100 --max-candidates 5 \
    --out grasp_6dof/dataset/lggsn_candidates_v7.jsonl

  # Quick test — 2 objects x 3 seeds x 5 candidates = 30 executions
  conda run -n owg-mujoco python scripts/collect_lggsn_data.py \
    --objects banana,pear --seeds 1-3 --max-candidates 5

  # Custom output path
  conda run -n owg-mujoco python scripts/collect_lggsn_data.py \
    --objects all --seeds 1-50 --out grasp_6dof/dataset/lggsn_candidates_v7.jsonl
"""

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from tango_robot.env_soarm import (
    EnvironmentSoArm,
    TABLE_TOP_Z,
    GRASP_Z_TABLE_MARGIN,
    GRASP_MODE_PHYSICS_WELD,
    IMG_SIZE,
    FOVY,
)
from tango_robot.pointcloud import compute_intrinsics
from grasp_6dof.grasp_sampler import (
    rpy_to_R,
    local_point_density,
    normal_consistency,
    contact_width_ratio,
)

try:
    from tango.markers.segmentor import SegmentAnythingMarkGenerator
    _HAS_SEGMENTOR = True
except ImportError:
    _HAS_SEGMENTOR = False

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
DEFAULT_OUT   = Path("grasp_6dof/dataset/lggsn_candidates_v7.jsonl")


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


def _get_obj_bbox(obs: dict, obj_id: int) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, x2, y2) pixel bbox of obj_id in the seg map, or None."""
    seg = obs.get("seg")
    if seg is None:
        return None
    seg2d = seg[:, :, 0] if seg.ndim == 3 else seg
    ys, xs = np.where(seg2d == obj_id)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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


def _compute_per_candidate_vis_feats(
    cands: np.ndarray,
    pil_img: "PILImage.Image",
    K: np.ndarray,
    cam_to_world: np.ndarray,
    segmentor,
) -> list[list]:
    """
    Project each candidate's (x,y,z) into the SAM feature map and return
    a list of 256-dim vectors (one per candidate).

    cands: (N, 6)  [x, y, z, yaw, opening, H]
    Returns list of N lists, each of length 256.
    """
    N = len(cands)
    zeros = [0.0] * 256

    try:
        # Encode image once — populates segmentor._cached_embeddings
        segmentor._cached_embeddings = None
        segmentor.get_roi_feature((0, 0, 1, 1), image=pil_img)
    except Exception as e:
        print(f"  [WARN] SAM encode failed: {e}")
        return [zeros] * N

    feat_map = segmentor._cached_embeddings          # [1, 256, 64, 64]
    if feat_map is None:
        return [zeros] * N

    _, C, H_f, W_f = feat_map.shape                 # 256, 64, 64
    H_img, W_img   = IMG_SIZE, IMG_SIZE

    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    world_to_cam = np.linalg.inv(cam_to_world)

    result = []
    for i in range(N):
        x, y, z = float(cands[i, 0]), float(cands[i, 1]), float(cands[i, 2])
        xyz_w = np.array([x, y, z, 1.0])
        xyz_c = world_to_cam @ xyz_w
        depth = float(-xyz_c[2])
        if depth <= 1e-4:
            result.append(zeros)
            continue

        u = fx * float(xyz_c[0]) / depth + cx       # column
        v = -fy * float(xyz_c[1]) / depth + cy      # row

        fu = u * W_f / W_img
        fv = v * H_f / H_img

        fu_i = int(round(fu))
        fv_i = int(round(fv))

        u0 = max(0, fu_i - 1); u1 = min(W_f, fu_i + 2)
        v0 = max(0, fv_i - 1); v1 = min(H_f, fv_i + 2)

        if u0 >= u1 or v0 >= v1:
            result.append(zeros)
            continue

        import torch
        roi = feat_map[0, :, v0:v1, u0:u1]          # [256, h, w]
        vec = roi.mean(dim=(-2, -1)).cpu().tolist()  # list[float] len=256
        result.append(vec)

    return result


def _featurize_candidates(cands: np.ndarray,
                           episode_pcd: np.ndarray,
                           pe_ik: float,
                           vis_feats: list | None = None,
                           per_cand_ik: list | None = None) -> list[dict]:
    """
    Build 18-dim feature dict for every candidate (21-dim when per_cand_ik
    is given — see below).

    cands: (N, 6)  [x, y, z, yaw, opening_len, obj_height]
    episode_pcd: (M, 3) full scene pointcloud in robot-base frame
    pe_ik: IK position error for the CoM target (same for all candidates in
           episode — mirrors inference where all candidates share CoM target)
    per_cand_ik: optional list of N dicts from
        EnvironmentSoArm.compute_ik_reachability_per_candidate (Tier-3),
        each {"ik_converged", "ik_residual", "max_joint_delta"} solved
        independently for that candidate's own (x, y, z, yaw) — unlike
        pe_ik, this genuinely varies within an episode. When given, these
        three keys are added to every row; when None (default), they're
        omitted and rows stay 18-dim, unchanged from before.

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

        row = {
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
            # 256-dim SAM feature centred on this candidate's projected position.
            # Per-candidate (each grasp gets its own local patch). Zeros when SAM unavailable.
            "visual_feat":         vis_feats[i] if vis_feats is not None
                                   else [0.0] * 256,
        }
        if per_cand_ik is not None:
            row["ik_converged"]   = bool(per_cand_ik[i]["ik_converged"])
            row["ik_residual"]    = round(float(per_cand_ik[i]["ik_residual"]), 6)
            row["max_joint_delta"] = round(float(per_cand_ik[i]["max_joint_delta"]), 6)
        rows.append(row)
    return rows


# ── Episode runner ────────────────────────────────────────────────────────────

def collect_episode(
    env:            EnvironmentSoArm,
    obj_key:        str,
    obj_class:      str,
    seed:           int,
    out_path:       Path,
    max_candidates: int = 5,
    reset_seed:     int | None = None,
    quiet:          bool = False,
    segmentor=None,
) -> tuple[int, int]:
    """Run one episode: execute top-max_candidates individually.
    Returns (n_success, n_executed)."""
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

    spawn_pos = (cx, cy)  # fixed for all candidate re-spawns this episode

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
    com_target = np.array([float(com[0]), float(com[1]),
                           float(com[2]) + GRASP_Z_TABLE_MARGIN])
    pe_iks = env.compute_ik_reachability([com_target])
    pe_ik  = pe_iks[0]

    # ── Tier-3: per-candidate IK reachability (genuinely varies within the
    #    episode, unlike pe_ik above which reuses one shared CoM target) ──────
    per_cand_ik = env.compute_ik_reachability_per_candidate(
        [(float(cands[i, 0]), float(cands[i, 1]), float(cands[i, 2]), float(cands[i, 3]))
         for i in range(len(cands))]
    )

    # ── SAM visual features: per-candidate projection onto feature map ────────
    vis_feats = None
    if segmentor is not None:
        K            = compute_intrinsics(IMG_SIZE, IMG_SIZE, FOVY)
        cam_to_world = env.cam_to_robot_base
        pil_img      = PILImage.fromarray(obs["image"])
        vis_feats    = _compute_per_candidate_vis_feats(
            cands, pil_img, K, cam_to_world, segmentor
        )

    # ── featurize ALL candidates upfront (episode-context stats use full set) ─
    feat_rows = _featurize_candidates(cands, episode_pcd, pe_ik, vis_feats=vis_feats,
                                       per_cand_ik=per_cand_ik)

    # ── execute top-max_candidates individually, respawning each time ─────────
    n_exec = min(max_candidates, len(cands))
    n_success = 0

    for cand_idx in range(n_exec):
        # respawn at the same position to ensure consistent object pose
        with _suppress_stdout():
            env.reset_robot()
            env.remove_all_obj()
            obj_id = env.load_obj(obj_class, name=obj_key,
                                  pos=[spawn_pos[0], spawn_pos[1], _DROP_Z])
            env._steps(_SETTLE_STEPS)

        # execute this candidate only
        env.set_obj_grasps(
            obj_id,
            [np.array(cands[cand_idx], dtype=np.float32)],
            grasp_rects=[],
        )
        with _suppress_stdout():
            success, _, _ = env.pick_obj_by_id(obj_id, grasp_indices=[0])
        label = 1 if success else 0
        n_success += label

        row = {
            "scene_id":      scene_id,
            "query":         obj_key,
            "candidate_idx": cand_idx,
            "label":         label,
            **feat_rows[cand_idx],
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if not quiet:
            sym = "✓" if success else "✗"
            r   = feat_rows[cand_idx]
            ik_tag = (f"  ik_res={r['ik_residual']*1000:.1f}mm"
                      f" conv={int(r['ik_converged'])}" if "ik_residual" in r else "")
            print(f"  [{sym}] {obj_key:<10} s{seed:03d}  c={cand_idx}  "
                  f"H={H:.3f}  "
                  f"ld={r['local_point_density']:.3f}  "
                  f"nc={r['normal_consistency']:.3f}  "
                  f"pe_ik={pe_ik*1000:.1f}mm{ik_tag}")

    return n_success, n_exec


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
                    help="Suppress per-candidate progress lines")
    ap.add_argument("--max-candidates", type=int, default=5, dest="max_candidates",
                    help="Number of candidates to execute per episode (default: 5)")
    ap.add_argument("--reset-seed", type=int, default=None, dest="reset_seed",
                    help="Seed stored in output rows to document pose reproducibility "
                         "(default: None = use episode seed). Does not change spawn "
                         "position, which is always re-used from the initial spawn.")
    ap.add_argument("--no-vis-feat", action="store_true", dest="no_vis_feat",
                    help="Disable SAM visual features (store zero vectors). "
                         "Use when SAM is not installed or for a quick dry run.")
    ap.add_argument("--sam-model", default="facebook/sam-vit-base",
                    dest="sam_model",
                    help="HuggingFace SAM model id (default: facebook/sam-vit-base). "
                         "sam-vit-base uses ~300MB VRAM; sam-vit-huge needs ~2.5GB.")
    ap.add_argument("--sam-device", default=None, dest="sam_device",
                    help="Device for SAM ('cuda' or 'cpu'). "
                         "Default: cuda if available, else cpu.")
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

    # ── SAM segmentor (optional, loaded once and reused across episodes) ────────
    segmentor = None
    if not args.no_vis_feat:
        if not _HAS_SEGMENTOR:
            print("[WARN] owg.markers.segmentor not importable; "
                  "visual_feat will be all zeros. Pass --no-vis-feat to silence.")
        else:
            import torch
            if args.sam_device:
                _dev = args.sam_device
            else:
                _dev = "cuda" if torch.cuda.is_available() else "cpu"
            _model_id = args.sam_model
            print(f"[INFO] Loading SAM ({_model_id}) on {_dev} ...")
            segmentor = SegmentAnythingMarkGenerator(device=_dev, model_name=_model_id)
            print("[INFO] SAM ready.")

    # init env — pre-register all 7 object types so model is built once
    env = EnvironmentSoArm(vis=False, debug=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD)
    logical_names = [cls.replace("Ycb", "") for cls in OBJECTS.values()]
    env.preload_pool(logical_names)

    total_exec, n_ok = 0, 0
    n_ep = len(obj_keys) * len(seeds)
    n_total_exec = n_ep * args.max_candidates
    print(f"Collecting {len(obj_keys)} objects x {len(seeds)} seeds "
          f"= {n_ep} episodes x {args.max_candidates} candidates "
          f"= {n_total_exec} executions")
    print(f"Output -> {out_path}\n")

    try:
        for obj_key in obj_keys:
            print(f"-- {obj_key} ({OBJECTS[obj_key]}) --")
            for seed in seeds:
                n_s, n_e = collect_episode(
                    env, obj_key, OBJECTS[obj_key],
                    seed, out_path,
                    max_candidates=args.max_candidates,
                    reset_seed=args.reset_seed,
                    quiet=args.quiet,
                    segmentor=segmentor,
                )
                total_exec += n_e
                n_ok       += n_s
    finally:
        env.close()

    sr = n_ok / total_exec if total_exec else 0.0
    kb = out_path.stat().st_size // 1024 if out_path.exists() else 0
    print(f"\nDone  {n_ok}/{total_exec} candidates succeeded ({sr:.1%})")
    print(f"Rows written: {total_exec}  |  File: {out_path}  ({kb} KB)")

    # ── spot-check first row ──────────────────────────────────────────────────
    with open(out_path) as f:
        sample = json.loads(f.readline())
    feat_keys = [k for k in sample if k not in ("scene_id", "query", "candidate_idx", "label")]
    print(f"\nRow sample ({len(feat_keys)}-dim features):")
    for k in feat_keys:
        print(f"  {k:<24} = {sample[k]}")


if __name__ == "__main__":
    main()
