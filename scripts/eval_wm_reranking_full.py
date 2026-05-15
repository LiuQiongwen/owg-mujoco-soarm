#!/usr/bin/env python3
"""
Full evaluation benchmark: geometry-score ranking vs world-model reranking.

Improvements over eval_wm_reranking.py:
  - Resume-safe: reads existing CSV, skips completed (object, method, trial_idx)
  - Per-trial deterministic seeds: seed = f(base_seed, object, method, trial_idx)
  - Failed-episode log: failed.jsonl — one JSON line per crashed trial
  - 95 % Wilson confidence intervals on success rate
  - Cohen's h effect size + two-proportion z-test vs geometry baseline
  - dz distribution histogram SVG (dz_hist.svg, pure-Python, no matplotlib)
  - Comparison chart with CI error bars (comparison.svg)
  - Rolling 10-trial SR shown per progress line

Output (in --out-dir):
  results.csv       — one row per completed trial
  failed.jsonl      — one JSON line per crashed trial
  summary.txt       — statistics table with 95 % Wilson CIs
  comparison.svg    — 3-panel chart: per-obj SR + CI, overall rates, mean dz
  dz_hist.svg       — dz histograms side-by-side by method

Usage:
  # Full run (500 trials ≈ 90 min)
  MUJOCO_GL=egl conda run -n bridge \\
      python scripts/eval_wm_reranking_full.py --out-dir results/run_full_01

  # Resume an interrupted run — same --out-dir, skips completed trials
  MUJOCO_GL=egl conda run -n bridge \\
      python scripts/eval_wm_reranking_full.py --out-dir results/run_full_01

  # Quick smoke-test (5 trials per combo = 50 total)
  MUJOCO_GL=egl conda run -n bridge \\
      python scripts/eval_wm_reranking_full.py --quick \\
          --out-dir results/run_quick_01
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from owg_robot.env_soarm import EnvironmentSoArm, TABLE_TOP_Z, ARM_JOINTS
from data.transition_logger import compute_pc_stats
from world_model.train_mlp_predictor import load_model, MODEL_PATH
from world_model.rerank_grasps import score_grasps

# ── Object catalogue ──────────────────────────────────────────────────────────

OBJECTS = {
    "banana":   "YcbBanana",
    "cylinder": "YcbTomatoSoupCan",
    "cracker":  "YcbCrackerBox",
    "mustard":  "YcbMustardBottle",
    "drill":    "YcbPowerDrill",
}
ALL_METHODS = ["geometry", "world_model"]

_OBJ_IDX  = {k: i for i, k in enumerate(OBJECTS)}
_METH_IDX = {m: i for i, m in enumerate(ALL_METHODS)}

# ── Physics / workspace constants ─────────────────────────────────────────────

_CENTRE_Y       = -0.40
_SPREAD_XY      = 0.06
_DROP_Z         = TABLE_TOP_Z + 0.12
_SETTLE_STEPS   = 300
_GRASP_Z_OFFSET = 0.025
_LIFT_STEPS     = 80
_FELL_OFF_Z     = TABLE_TOP_Z - 0.10
_LIFT_THRESHOLD = 0.07

DEFAULT_K = 10

# ── CSV schema ────────────────────────────────────────────────────────────────

CSV_COLS = [
    "trial_id", "trial_idx", "trial_seed",
    "object", "method",
    "success", "dz", "fell_off",
    "geo_score_top1", "wm_score_top1",
    "success_prob_top1", "dz_pred_top1", "fell_prob_top1",
    "geo_score_mean", "wm_score_mean",
    "grasp_x", "grasp_y", "grasp_z", "grasp_yaw", "opening_len",
    "k_grasps", "timestamp",
]


# ── Deterministic per-trial seeding ──────────────────────────────────────────

def trial_seed(base_seed: int, obj_key: str, method: str, idx: int) -> int:
    """Unique, reproducible uint32 seed for (base, object, method, trial_idx)."""
    return (base_seed * 10_000_000
            + _OBJ_IDX.get(obj_key, 0) * 100_000
            + _METH_IDX.get(method, 0) * 1_000
            + idx) % (2 ** 32)


def trial_rng(base_seed: int, obj_key: str, method: str,
              idx: int) -> np.random.Generator:
    return np.random.default_rng(trial_seed(base_seed, obj_key, method, idx))


# ── Resume: load completed trials ────────────────────────────────────────────

def load_completed(csv_path: Path) -> tuple:
    """
    Read existing CSV.

    Returns
    -------
    done_set   : set of (object, method, trial_idx:int) already complete
    rows       : list of row dicts (values are strings, as read from CSV)
    next_id    : int — next trial_id to assign
    """
    done_set, rows = set(), []
    if not csv_path.exists():
        return done_set, rows, 0

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            try:
                done_set.add((row["object"], row["method"],
                               int(row["trial_idx"])))
            except (KeyError, ValueError):
                pass

    next_id = max((int(r["trial_id"]) for r in rows
                   if r.get("trial_id", "").isdigit()), default=-1) + 1
    return done_set, rows, next_id


# ── Failed episode logger ─────────────────────────────────────────────────────

class FailedLogger:
    def __init__(self, path: Path):
        self.path = path

    def log(self, trial_id: int, trial_idx: int, seed: int,
            obj_key: str, method: str, exc: Exception, tb: str):
        entry = {
            "trial_id":   trial_id,
            "trial_idx":  trial_idx,
            "trial_seed": seed,
            "object":     obj_key,
            "method":     method,
            "error":      type(exc).__name__,
            "message":    str(exc),
            "traceback":  tb,
            "timestamp":  time.time(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ── Geometry scoring heuristic ────────────────────────────────────────────────

def geo_score(grasp: np.ndarray, obj_pos: np.ndarray,
              pc_stats: np.ndarray) -> float:
    """
    Geometry-only quality heuristic (no learned model).
    XY centering 60 %, Z alignment 30 %, opening width 10 %.
    """
    x, y, z  = float(grasp[0]), float(grasp[1]), float(grasp[2])
    opening  = float(grasp[4])
    cx, cy   = float(pc_stats[0]), float(pc_stats[1])
    min_z, max_z = float(pc_stats[6]), float(pc_stats[7])
    obj_top  = max_z if max_z > TABLE_TOP_Z else float(obj_pos[2]) + 0.03
    obj_h    = max(max_z - min_z, 0.02)

    xy_err = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    xy_s   = math.exp(-30  * xy_err  ** 2)
    z_err  = abs(z - (obj_top + _GRASP_Z_OFFSET))
    z_s    = math.exp(-200 * z_err   ** 2)
    t_op   = float(np.clip(obj_h * 0.8 + 0.02, 0.03, 0.09))
    op_s   = math.exp(-200 * (opening - t_op) ** 2)

    return float(0.60 * xy_s + 0.30 * z_s + 0.10 * op_s)


# ── Grasp sampling ────────────────────────────────────────────────────────────

def _sample_grasp(obj_pos: np.ndarray, rng: np.random.Generator,
                  spread: float = 0.04) -> np.ndarray:
    return np.array([
        float(obj_pos[0] + rng.uniform(-spread, spread)),
        float(obj_pos[1] + rng.uniform(-spread, spread)),
        float(obj_pos[2] + _GRASP_Z_OFFSET),
        float(rng.uniform(-math.pi / 2, math.pi / 2)),
        float(rng.uniform(0.04, 0.09)),
        0.05,
    ], dtype=np.float32)


# ── Grasp execution ───────────────────────────────────────────────────────────

def _apply_top_down_yaw(env: EnvironmentSoArm, yaw: float):
    try:
        idx = ARM_JOINTS.index("wrist_roll")
        lo  = env.model.jnt_range[env._arm_jnt_ids[idx], 0]
        hi  = env.model.jnt_range[env._arm_jnt_ids[idx], 1]
        cmd = float(np.clip(yaw, lo, hi))
        env.data.ctrl[env._arm_act_ids[idx]] = cmd
        env.data.qpos[env._arm_qpos_adr[idx]] = cmd
        env._steps(80)
    except Exception:
        pass


def _execute(env: EnvironmentSoArm, obj_id: int,
             grasp: np.ndarray, yaw_mode: str = "top_down_yaw") -> bool:
    x, y, z = float(grasp[0]), float(grasp[1]), float(grasp[2])
    yaw     = float(grasp[3])
    opening = float(grasp[4])
    try:
        env.reset_robot()
        env.move_gripper(opening)
        env.move_ee([x, y, env.GRIPPER_MOVING_HEIGHT, None])
        if yaw_mode == "top_down_yaw":
            _apply_top_down_yaw(env, yaw)
        env.move_ee([x, y, z, None])
        env.auto_close_gripper(check_contact=False)
        env._steps(60)
        contact = obj_id in env.check_grasped_id()
        env.move_ee([x, y, env.GRIPPER_MOVING_HEIGHT, None])
        env._steps(_LIFT_STEPS)
        grasped = obj_id in env.check_grasped_id()
        obj_z   = env.get_obj_pos(obj_id)[2]
        return contact or grasped or (obj_z > TABLE_TOP_Z + _LIFT_THRESHOLD)
    except Exception:
        return False


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(env: EnvironmentSoArm, obj_key: str, method: str,
              rng: np.random.Generator, model: dict,
              k_grasps: int = DEFAULT_K) -> dict:
    """
    Execute one evaluation trial.

    Both geo_score and wm_score are computed for every trial (regardless of
    which method is active) so the CSV captures the counterfactual ranker.
    """
    obj_name = OBJECTS[obj_key]

    env.reset_robot()
    env.remove_all_obj()
    cx  = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
    cy  = _CENTRE_Y + float(rng.uniform(-0.04, 0.04))
    oid = env.load_obj(obj_name, name=obj_key, pos=[cx, cy, _DROP_Z])
    env._steps(_SETTLE_STEPS)

    obj_pos  = env.get_obj_pos(oid).copy()
    obj_quat = env.get_obj_pose(oid)["quaternion"].copy()
    obs      = env.get_obs(pointcloud=True)
    pc_stats = compute_pc_stats(obs, oid)

    candidates = np.stack([_sample_grasp(obj_pos, rng) for _ in range(k_grasps)])

    geo_scores             = np.array([geo_score(g, obj_pos, pc_stats)
                                       for g in candidates])
    wm_scores, preds       = score_grasps(candidates, obj_pos, obj_quat,
                                          pc_stats, model)

    best   = int(np.argmax(geo_scores if method == "geometry" else wm_scores))
    chosen = candidates[best]

    pos_before = obj_pos.copy()
    success    = _execute(env, oid, chosen)
    env._steps(40)
    pos_after  = env.get_obj_pos(oid).copy()
    dz         = float(pos_after[2] - pos_before[2])
    fell_off   = bool(pos_after[2] < _FELL_OFF_Z)

    return {
        "success":            bool(success),
        "dz":                 dz,
        "fell_off":           fell_off,
        "geo_score_top1":     float(geo_scores[best]),
        "wm_score_top1":      float(wm_scores[best]),
        "success_prob_top1":  float(preds["success_prob"][best]),
        "dz_pred_top1":       float(preds["dz_pred"][best]),
        "fell_prob_top1":     float(preds["fell_prob"][best]),
        "geo_score_mean":     float(geo_scores.mean()),
        "wm_score_mean":      float(wm_scores.mean()),
        "grasp_x":            float(chosen[0]),
        "grasp_y":            float(chosen[1]),
        "grasp_z":            float(chosen[2]),
        "grasp_yaw":          float(chosen[3]),
        "opening_len":        float(chosen[4]),
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """95 % Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p     = k / n
    z2n   = z * z / n
    denom = 1.0 + z2n
    center = (p + z2n / 2.0) / denom
    half   = z * math.sqrt(max(0.0, p * (1 - p) / n + z2n / (4 * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size (|h| ≥ 0.2 small, 0.5 medium, 0.8 large)."""
    return (2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
            - 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2)))))


def ztest_2prop(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-proportion z-test. Positive z → p1 > p2."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p1  = k1 / n1
    p2  = k2 / n2
    p   = (k1 + k2) / (n1 + n2)
    se  = math.sqrt(max(1e-12, p * (1 - p) * (1 / n1 + 1 / n2)))
    return (p1 - p2) / se


def _is_success(r) -> bool:
    v = r.get("success", "false")
    return str(v).lower() in ("true", "1")


def _is_fell(r) -> bool:
    v = r.get("fell_off", "false")
    return str(v).lower() in ("true", "1")


# ── Summary computation ───────────────────────────────────────────────────────

def compute_summary(rows: list, obj_keys: list, methods: list) -> dict:
    """
    Return nested summary[method][obj_or_OVERALL] = stats dict.

    Stats keys: n, k_success, sr, ci_lo, ci_hi, fell_off, dz_mean, dz_std, dz_vals
    """
    by_mo: dict = defaultdict(list)
    for r in rows:
        key = (r.get("method", ""), r.get("object", ""))
        by_mo[key].append(r)

    summary: dict = defaultdict(dict)

    for (method, obj), rs in by_mo.items():
        k_s     = sum(1 for r in rs if _is_success(r))
        n       = len(rs)
        dz_vals = [float(r["dz"]) for r in rs]
        fo      = sum(1 for r in rs if _is_fell(r))
        ci      = wilson_ci(k_s, n)
        summary[method][obj] = {
            "n":         n,
            "k_success": k_s,
            "sr":        k_s / n if n else float("nan"),
            "ci_lo":     ci[0],
            "ci_hi":     ci[1],
            "fell_off":  fo / n if n else float("nan"),
            "dz_mean":   float(np.mean(dz_vals)) if dz_vals else float("nan"),
            "dz_std":    float(np.std(dz_vals))  if dz_vals else float("nan"),
            "dz_vals":   dz_vals,
        }

    for method in methods:
        rs = [r for r in rows if r.get("method") == method]
        if not rs:
            continue
        k_s     = sum(1 for r in rs if _is_success(r))
        n       = len(rs)
        dz_vals = [float(r["dz"]) for r in rs]
        fo      = sum(1 for r in rs if _is_fell(r))
        ci      = wilson_ci(k_s, n)
        summary[method]["OVERALL"] = {
            "n":         n,
            "k_success": k_s,
            "sr":        k_s / n if n else float("nan"),
            "ci_lo":     ci[0],
            "ci_hi":     ci[1],
            "fell_off":  fo / n if n else float("nan"),
            "dz_mean":   float(np.mean(dz_vals)) if dz_vals else float("nan"),
            "dz_std":    float(np.std(dz_vals))  if dz_vals else float("nan"),
            "dz_vals":   dz_vals,
        }

    return summary


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(summary: dict, obj_keys: list, methods: list,
                  file=None) -> None:
    def _p(*a, **kw):
        print(*a, **kw, file=file)

    W = 96
    _p("\n" + "=" * W)
    _p(f"  {'Object':<12} {'Method':<14} {'N':>5}  "
       f"{'SR':>6}  {'95 % CI':^17}  {'FO':>5}  "
       f"{'dz_mean':>8}  {'dz_std':>7}  {'Δ':>7}  {'h':>5}  {'sig':>3}")
    _p("-" * W)

    geo_overall = summary.get("geometry", {}).get("OVERALL", {})

    for obj in obj_keys + ["OVERALL"]:
        sg = summary.get("geometry",    {}).get(obj, {})
        sw = summary.get("world_model", {}).get(obj, {})

        for method in methods:
            s = summary.get(method, {}).get(obj)
            if not s:
                continue
            sr   = s["sr"]
            ci   = f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
            fo   = f"{s['fell_off']:.3f}" if math.isfinite(s["fell_off"]) else "  nan"
            dzm  = f"{s['dz_mean']:+.4f}" if math.isfinite(s.get("dz_mean", float("nan"))) else "    nan"
            dzs  = f"{s['dz_std']:.4f}"   if math.isfinite(s.get("dz_std",  float("nan"))) else "   nan"

            delta_s, h_s, sig_s = "", "", ""
            if method == "world_model" and sg:
                delta   = sr - sg.get("sr", float("nan"))
                tag     = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "~")
                delta_s = f"{delta:+.3f}{tag}"
                h_val   = cohens_h(sr, sg.get("sr", sr))
                h_s     = f"{h_val:+.3f}"
                z       = ztest_2prop(s["k_success"], s["n"],
                                      sg.get("k_success", 0), sg.get("n", 1))
                sig_s   = "**" if abs(z) > 2.576 else ("*" if abs(z) > 1.96 else "")

            _p(f"  {obj:<12} {method:<14} {s['n']:>5}  "
               f"{sr:>6.3f}  {ci:^17}  {fo:>5}  "
               f"{dzm:>8}  {dzs:>7}  {delta_s:>7}  {h_s:>5}  {sig_s:>3}")

        if obj != "OVERALL":
            _p()

    _p("=" * W)
    _p("\n  sig: * p<0.05, ** p<0.01  (two-proportion z-test, WM vs geometry baseline)")
    _p("  h: Cohen's h  (0.2=small, 0.5=medium, 0.8=large)\n")


# ── Progress display ──────────────────────────────────────────────────────────

class Progress:
    def __init__(self, n_pending: int, quiet: bool = False):
        self.n_pending = n_pending
        self.done      = 0
        self.t_start   = time.time()
        self.quiet     = quiet
        self._recent   = deque(maxlen=20)

    def tick(self, obj_key: str, method: str, result: dict,
             elapsed: float, grp_k: int, grp_n: int, trial_id: int):
        self.done += 1
        self._recent.append(elapsed)
        if self.quiet:
            return
        sym    = "✓" if result.get("success") else "✗"
        ff     = " [fell]" if result.get("fell_off") else ""
        pct    = 100 * self.done / max(self.n_pending, 1)
        avg_t  = sum(self._recent) / len(self._recent)
        eta_s  = avg_t * (self.n_pending - self.done)
        run_sr = grp_k / grp_n if grp_n else 0.0
        print(f"  [{self.done:04d}/{self.n_pending}] {pct:4.0f}%  "
              f"{obj_key:<10} {method:<14} {sym}{ff}  "
              f"dz={result['dz']:+.3f}  "
              f"geo={result['geo_score_top1']:.3f}  "
              f"wm={result['wm_score_top1']:.3f}  "
              f"SR={run_sr:.2f}({grp_k}/{grp_n})  "
              f"ETA {eta_s/60:.0f}m")

    def group_done(self, obj_key: str, method: str, k: int, n: int):
        if self.quiet or n == 0:
            return
        ci = wilson_ci(k, n)
        print(f"\n  ✔ {obj_key}/{method}: "
              f"SR={k/n:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  ({k}/{n})\n")


# ── SVG helpers (pure Python) ─────────────────────────────────────────────────

_C = {"geometry": "#4878CF", "world_model": "#D65F5F"}


def _rect(x, y, w, h, fill, rx=2, op=0.85):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" opacity="{op}" rx="{rx}"/>')


def _text(x, y, t, anchor="middle", sz=9, fill="#333", bold=False):
    fw = ' font-weight="bold"' if bold else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-size="{sz}" fill="{fill}"{fw}>{t}</text>')


def _line(x1, y1, x2, y2, stroke="#aaa", w=1, dash=""):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{da}/>')


def _errorbar(cx, y_lo, y_hi, cap=3, stroke="#333", w=1.5):
    elems = [_line(cx, y_hi, cx, y_lo, stroke=stroke, w=w)]
    for ey in (y_hi, y_lo):
        elems.append(_line(cx - cap, ey, cx + cap, ey, stroke=stroke, w=w))
    return elems


# ── SVG: comparison chart ─────────────────────────────────────────────────────

def make_comparison_svg(summary: dict, obj_keys: list, methods: list,
                         out_path: Path) -> None:
    """
    Three-panel SVG:
      1. Per-object success rate with 95 % Wilson CI error bars
      2. Overall success rate + fell-off rate
      3. Mean dz ± std per object
    """
    W, H = 960, 420
    MT, MB, ML, MR = 58, 80, 62, 22
    PW   = (W - ML - MR) // 3
    CH   = H - MT - MB
    YB   = MT + CH

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" fill="#f8f8f8"/>',
           _text(W // 2, 24, "Geometry Score vs World-Model Reranking",
                 sz=14, bold=True, fill="#222")]

    colors = [_C.get(m, "#888") for m in methods]

    # Legend
    lx = W - MR - 180
    for i, (m, c) in enumerate(zip(methods, colors)):
        svg.append(_rect(lx, 8 + i * 18, 12, 11, c, op=1.0))
        svg.append(_text(lx + 16, 19 + i * 18, m.replace("_", " "),
                         anchor="start", sz=10, fill="#333"))

    def _px(panel): return ML + panel * PW

    def _axes(px0, title, y_ticks, y_max):
        x1 = px0 + PW - 22
        svg.append(_line(px0, MT, px0, YB))
        svg.append(_line(px0, YB, x1, YB))
        svg.append(_text(px0 + (PW - 22) // 2, MT - 12, title,
                         sz=11, bold=True, fill="#333"))
        ys = CH / y_max
        for t in y_ticks:
            ty = YB - t * ys
            svg.append(_line(px0, ty, x1, ty, stroke="#e0e0e0", w=0.8))
            svg.append(_text(px0 - 5, ty + 4, f"{t:.2f}",
                             anchor="end", sz=8, fill="#666"))
        return ys

    bw = 14

    # ── Panel 1: per-object SR with Wilson CI ─────────────────────────────────
    px1 = _px(0)
    ys1 = _axes(px1, "Per-Object Success Rate",
                [0.0, 0.25, 0.50, 0.75, 1.0], 1.0)
    svg.append(_text(px1 - 42, MT + CH // 2, "Success Rate",
                     anchor="middle", sz=10, fill="#555"))

    n_obj  = len(obj_keys)
    gspan1 = PW - 44
    ggap1  = gspan1 / max(n_obj, 1)

    for oi, obj in enumerate(obj_keys):
        cx   = px1 + 22 + oi * ggap1 + ggap1 / 2
        nm   = len(methods)
        tw   = nm * bw + (nm - 1) * 4
        x0   = cx - tw / 2
        for i, (m, c) in enumerate(zip(methods, colors)):
            s = summary.get(m, {}).get(obj)
            if not s:
                continue
            sr, ci_lo, ci_hi = s["sr"], s["ci_lo"], s["ci_hi"]
            bx = x0 + i * (bw + 4)
            bh = sr * ys1
            by = YB - bh
            svg.append(_rect(bx, by, bw, bh, c))
            svg.append(_text(bx + bw / 2, by - 4, f"{sr:.2f}", sz=8))
            ecx = bx + bw / 2
            svg.extend(_errorbar(ecx, YB - ci_hi * ys1, YB - ci_lo * ys1))
        svg.append(_text(cx, YB + 16, obj, sz=9, fill="#555"))

    # ── Panel 2: overall SR + fell-off ───────────────────────────────────────
    px2 = _px(1)
    ys2 = _axes(px2, "Overall Rates",
                [0.0, 0.25, 0.50, 0.75, 1.0], 1.0)

    metrics2 = [("sr", "Success Rate"), ("fell_off", "Fell-Off")]
    mg2      = (PW - 44) / max(len(metrics2), 1)
    for mi, (mkey, mlabel) in enumerate(metrics2):
        cx  = px2 + 22 + mi * mg2 + mg2 / 2
        nm  = len(methods)
        bw2 = bw + 2
        tw  = nm * bw2 + (nm - 1) * 4
        x0  = cx - tw / 2
        for i, (m, c) in enumerate(zip(methods, colors)):
            s  = summary.get(m, {}).get("OVERALL")
            if not s:
                continue
            v  = s.get(mkey, 0.0)
            if not math.isfinite(v):
                continue
            bx = x0 + i * (bw2 + 4)
            bh = v * ys2
            by = YB - bh
            svg.append(_rect(bx, by, bw2, bh, c))
            svg.append(_text(bx + bw2 / 2, by - 4, f"{v:.3f}", sz=8))
            if mkey == "sr":
                ecx = bx + bw2 / 2
                svg.extend(_errorbar(ecx,
                                     YB - s["ci_hi"] * ys2,
                                     YB - s["ci_lo"] * ys2))
        svg.append(_text(cx, YB + 16, mlabel, sz=9, fill="#555"))

    # ── Panel 3: mean dz per object ───────────────────────────────────────────
    px3 = _px(2)
    pw3 = PW - 22

    all_bounds = []
    for m in methods:
        for obj in obj_keys:
            s = summary.get(m, {}).get(obj, {})
            dz, sd = s.get("dz_mean", float("nan")), s.get("dz_std", 0.0)
            if math.isfinite(dz):
                all_bounds.extend([dz - sd, dz + sd])

    dz_lo = min(all_bounds + [-0.01]) - 0.025
    dz_hi = max(all_bounds + [ 0.01]) + 0.025
    dz_rng = dz_hi - dz_lo

    def _dzy(v): return YB - (v - dz_lo) / dz_rng * CH

    # Axes
    x13 = px3 + pw3
    svg.append(_line(px3, MT, px3, YB))
    svg.append(_line(px3, YB, x13, YB))
    svg.append(_text(px3 + pw3 // 2, MT - 12, "Mean dz (± std)",
                     sz=11, bold=True, fill="#333"))
    svg.append(_text(px3 - 42, MT + CH // 2, "dz (m)",
                     anchor="middle", sz=10, fill="#555"))

    tick_step = 0.05
    t = math.ceil(dz_lo / tick_step) * tick_step
    while t <= dz_hi + 0.001:
        ty = _dzy(t)
        svg.append(_line(px3, ty, x13, ty, stroke="#e0e0e0", w=0.8))
        svg.append(_text(px3 - 5, ty + 4, f"{t:+.2f}",
                         anchor="end", sz=8, fill="#666"))
        t = round(t + tick_step, 3)

    # Zero line
    svg.append(_line(px3, _dzy(0), x13, _dzy(0),
                     stroke="#999", w=1, dash="4,3"))

    bw3 = 13
    gw3 = (PW - 44) / max(n_obj, 1)
    for oi, obj in enumerate(obj_keys):
        cx  = px3 + 22 + oi * gw3 + gw3 / 2
        nm  = len(methods)
        tw  = nm * bw3 + (nm - 1) * 4
        x0  = cx - tw / 2
        for i, (m, c) in enumerate(zip(methods, colors)):
            s  = summary.get(m, {}).get(obj, {})
            dz = s.get("dz_mean", float("nan"))
            sd = s.get("dz_std",  0.0)
            if not math.isfinite(dz):
                continue
            bx  = x0 + i * (bw3 + 4)
            by0 = _dzy(0)
            by1 = _dzy(dz)
            byt = min(by0, by1)
            bh  = abs(by1 - by0)
            svg.append(_rect(bx, byt, bw3, bh, c))
            ecx = bx + bw3 / 2
            svg.extend(_errorbar(ecx, _dzy(dz + sd), _dzy(dz - sd),
                                  cap=3, stroke="#444", w=1.2))
        svg.append(_text(cx, YB + 16, obj, sz=9, fill="#555"))

    # X-axis label row 2: Δ annotation for OVERALL
    sg_ov = summary.get("geometry",    {}).get("OVERALL", {})
    sw_ov = summary.get("world_model", {}).get("OVERALL", {})
    if sg_ov and sw_ov:
        for m, c in zip(methods, colors):
            s = summary.get(m, {}).get("OVERALL", {})
            n_s  = s.get("n", 0)
            sr_s = s.get("sr", float("nan"))
            ci   = s.get("ci_lo", sr_s), s.get("ci_hi", sr_s)
            ann  = f"n={n_s}  SR={sr_s:.3f}"
            pass  # appended below the summary table via text file

    svg.append("</svg>")
    out_path.write_text("\n".join(svg))
    print(f"  comparison.svg → {out_path}")


# ── SVG: dz histogram ────────────────────────────────────────────────────────

def make_dz_hist_svg(summary: dict, methods: list,
                     out_path: Path, n_bins: int = 25) -> None:
    """
    Side-by-side dz histogram for each method (all objects combined).
    Includes mean marker and zero reference line.
    """
    W, H = 820, 370
    MT, MB, ML, MR = 52, 65, 58, 20
    GAP  = 24
    PW   = (W - ML - MR - GAP) // 2
    CH   = H - MT - MB

    # Collect dz_vals from OVERALL per method
    dz_by_m = {}
    for m in methods:
        s = summary.get(m, {}).get("OVERALL", {})
        dz_by_m[m] = s.get("dz_vals", [])

    all_dz = [v for vs in dz_by_m.values() for v in vs]
    if not all_dz:
        out_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<text x="10" y="20">No data</text></svg>')
        return

    dz_lo  = min(all_dz) - 0.005
    dz_hi  = max(all_dz) + 0.005
    bw_bin = (dz_hi - dz_lo) / n_bins

    def _hist(vals):
        counts = [0] * n_bins
        for v in vals:
            idx = min(int((v - dz_lo) / bw_bin), n_bins - 1)
            counts[idx] += 1
        return counts

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" fill="#f8f8f8"/>',
           _text(W // 2, 24, "dz Distribution by Method (all objects combined)",
                 sz=13, bold=True, fill="#222")]

    for pi, method in enumerate(methods):
        vals   = dz_by_m.get(method, [])
        counts = _hist(vals)
        max_c  = max(counts) if any(counts) else 1

        px0 = ML + pi * (PW + GAP)
        YB_ = MT + CH
        x1_ = px0 + PW
        ys  = CH / max(max_c, 1)
        c   = _C.get(method, "#888")

        # Axes
        svg.append(_line(px0, MT, px0, YB_))
        svg.append(_line(px0, YB_, x1_, YB_))
        svg.append(_text(px0 + PW // 2, MT - 14,
                         f"{method.replace('_', ' ')}  (n={len(vals)})",
                         sz=11, bold=True, fill="#333"))
        svg.append(_text(px0 - 42, MT + CH // 2, "Count",
                         anchor="middle", sz=10, fill="#555"))
        svg.append(_text(px0 + PW // 2, YB_ + 42, "dz  (m)",
                         sz=10, fill="#555"))

        # Y ticks
        y_step = max(1, max_c // 5)
        for t in range(0, max_c + y_step + 1, y_step):
            ty = YB_ - t * ys
            svg.append(_line(px0, ty, x1_, ty, stroke="#e0e0e0", w=0.8))
            svg.append(_text(px0 - 5, ty + 4, str(t),
                             anchor="end", sz=8, fill="#666"))

        # Histogram bars
        bar_pw = PW / n_bins
        for i, cnt in enumerate(counts):
            bx = px0 + i * bar_pw
            bh = cnt * ys
            svg.append(_rect(bx, YB_ - bh, bar_pw - 1, bh, c, rx=1, op=0.75))

        # X ticks
        n_xticks = 6
        for j in range(n_xticks + 1):
            fi  = j * n_bins / n_xticks
            tx  = px0 + fi * bar_pw
            val = dz_lo + fi * bw_bin
            svg.append(_line(tx, YB_, tx, YB_ + 5, stroke="#aaa", w=0.8))
            svg.append(_text(tx, YB_ + 17, f"{val:+.3f}",
                             anchor="middle", sz=8, fill="#666"))

        # Zero reference line
        if dz_lo < 0 < dz_hi:
            zx = px0 + (0 - dz_lo) / (dz_hi - dz_lo) * PW
            svg.append(_line(zx, MT + 10, zx, YB_,
                             stroke="#c33", w=1.2, dash="4,3"))
            svg.append(_text(zx + 3, MT + 22, "0", anchor="start",
                             sz=9, fill="#c33"))

        # Mean marker
        if vals:
            mn  = float(np.mean(vals))
            mnx = px0 + (mn - dz_lo) / (dz_hi - dz_lo) * PW
            if px0 <= mnx <= x1_:
                svg.append(_line(mnx, MT + 28, mnx, YB_,
                                 stroke="#222", w=1.8, dash="6,3"))
                svg.append(_text(mnx + 4, MT + 42, f"μ={mn:+.3f}",
                                 anchor="start", sz=8, fill="#222"))

        # Std shading — a thin rect around ±1 std
        if vals:
            sd  = float(np.std(vals))
            mn  = float(np.mean(vals))
            x_l = px0 + max(0, (mn - sd - dz_lo) / (dz_hi - dz_lo)) * PW
            x_r = px0 + min(PW, (mn + sd - dz_lo) / (dz_hi - dz_lo)) * PW
            svg.append(
                f'<rect x="{x_l:.1f}" y="{MT:.1f}" '
                f'width="{max(0, x_r - x_l):.1f}" height="{CH:.1f}" '
                f'fill="{c}" opacity="0.08"/>'
            )

    svg.append("</svg>")
    out_path.write_text("\n".join(svg))
    print(f"  dz_hist.svg    → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trials",     type=int, default=50,
                    help="Trials per (object × method) (default: 50)")
    ap.add_argument("--objects",    default="all",
                    help="Comma-separated keys or 'all' (default: all)")
    ap.add_argument("--methods",    default="geometry,world_model",
                    help="Comma-separated methods (default: geometry,world_model)")
    ap.add_argument("--k-grasps",   type=int, default=DEFAULT_K,
                    help=f"Grasp candidates per trial (default: {DEFAULT_K})")
    ap.add_argument("--model-path", default=str(MODEL_PATH),
                    help="Path to world-model pickle")
    ap.add_argument("--out-dir",    default="results/eval_full",
                    help="Output directory — reuse to resume (default: results/eval_full)")
    ap.add_argument("--seed",       type=int, default=42,
                    help="Base seed for per-trial RNG (default: 42)")
    ap.add_argument("--quick",      action="store_true",
                    help="5 trials per combo (smoke test, 50 total)")
    ap.add_argument("--quiet",      action="store_true",
                    help="Suppress per-trial output")
    args = ap.parse_args()

    if args.quick:
        args.trials = 5

    # Parse + validate object / method lists
    obj_keys = (list(OBJECTS.keys()) if args.objects.strip() == "all"
                else [k.strip() for k in args.objects.split(",")])
    methods  = [m.strip() for m in args.methods.split(",")]
    for k in obj_keys:
        if k not in OBJECTS:
            ap.error(f"Unknown object '{k}'. Valid: {sorted(OBJECTS)}")
    for m in methods:
        if m not in ALL_METHODS:
            ap.error(f"Unknown method '{m}'. Valid: {ALL_METHODS}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path     = out_dir / "results.csv"
    failed_path  = out_dir / "failed.jsonl"
    summary_path = out_dir / "summary.txt"
    chart_path   = out_dir / "comparison.svg"
    hist_path    = out_dir / "dz_hist.svg"

    # ── Resume: load previously completed trials ──────────────────────────────
    done_set, existing_rows, next_trial_id = load_completed(csv_path)

    n_total   = args.trials * len(obj_keys) * len(methods)
    n_resumed = sum(
        1 for obj in obj_keys for m in methods
        for ti in range(args.trials)
        if (obj, m, ti) in done_set
    )
    n_pending = n_total - n_resumed

    if n_resumed:
        print(f"\nResuming: {n_resumed}/{n_total} trials already done in {out_dir}")

    print(f"\nEval: {out_dir}")
    print(f"  objects  : {obj_keys}")
    print(f"  methods  : {methods}")
    print(f"  trials   : {args.trials}/combo  →  {n_total} total"
          f"  ({n_pending} pending)")
    print(f"  k_grasps : {args.k_grasps}")
    print(f"  base_seed: {args.seed}\n")

    # ── Load world model ──────────────────────────────────────────────────────
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("  Run: python world_model/train_mlp_predictor.py")
        sys.exit(1)
    model = load_model(model_path)
    print(f"Model: {model_path}\n")

    # ── Environment + helpers ─────────────────────────────────────────────────
    env    = EnvironmentSoArm(vis=False, debug=False)
    failed = FailedLogger(failed_path)
    prog   = Progress(n_pending, quiet=args.quiet)

    trial_id = next_trial_id

    with open(csv_path, "a", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLS)
        if n_resumed == 0:
            writer.writeheader()

        for obj_key in obj_keys:
            for method in methods:
                grp_n = 0
                grp_k = 0

                # Seed running totals from already-done rows in this group
                for r in existing_rows:
                    if r.get("object") == obj_key and r.get("method") == method:
                        grp_n += 1
                        if _is_success(r):
                            grp_k += 1

                for ti in range(args.trials):
                    if (obj_key, method, ti) in done_set:
                        continue  # resume: skip completed trial

                    seed = trial_seed(args.seed, obj_key, method, ti)
                    rng  = trial_rng(args.seed, obj_key, method, ti)

                    t0 = time.time()
                    try:
                        result  = run_trial(env, obj_key, method, rng, model,
                                            args.k_grasps)
                        elapsed = time.time() - t0

                        grp_n += 1
                        grp_k += int(result["success"])

                        row = {
                            "trial_id":   trial_id,
                            "trial_idx":  ti,
                            "trial_seed": seed,
                            "object":     obj_key,
                            "method":     method,
                            "k_grasps":   args.k_grasps,
                            "timestamp":  time.time(),
                            **result,
                        }
                        writer.writerow(row)
                        fcsv.flush()
                        existing_rows.append({k: str(v) for k, v in row.items()})

                        prog.tick(obj_key, method, result, elapsed,
                                  grp_k, grp_n, trial_id)

                    except Exception as exc:
                        elapsed = time.time() - t0
                        tb      = traceback.format_exc()
                        failed.log(trial_id, ti, seed, obj_key, method, exc, tb)
                        if not args.quiet:
                            print(f"  [FAIL] {obj_key}/{method} trial {ti}: "
                                  f"{type(exc).__name__}: {exc}")

                    trial_id += 1

                prog.group_done(obj_key, method, grp_k, grp_n)

    env.close()

    # ── Final summary (reload from disk for correctness) ──────────────────────
    _, all_rows, _ = load_completed(csv_path)
    summary        = compute_summary(all_rows, obj_keys, methods)

    with open(summary_path, "w") as f:
        print(f"Eval: {out_dir}", file=f)
        print(f"trials={args.trials}  k={args.k_grasps}  "
              f"objects={obj_keys}  methods={methods}", file=f)
        print_summary(summary, obj_keys, methods, file=f)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print_summary(summary, obj_keys, methods)

    print("Generating charts …")
    make_comparison_svg(summary, obj_keys, methods, chart_path)
    make_dz_hist_svg(summary, methods, hist_path)

    elapsed_total = time.time() - prog.t_start
    n_ran         = prog.done
    spt           = elapsed_total / max(n_ran, 1)

    # Count failed
    n_failed = 0
    if failed_path.exists():
        n_failed = sum(1 for _ in open(failed_path))

    print(f"\n  CSV        → {csv_path}  ({len(all_rows)} rows)")
    print(f"  Failed     → {failed_path}  ({n_failed} episodes)")
    print(f"  Summary    → {summary_path}")
    print(f"  Charts     → {chart_path}")
    print(f"             → {hist_path}")
    print(f"\n  Time this run: {elapsed_total/60:.1f} min  ({spt:.1f} s/trial)")


if __name__ == "__main__":
    main()
