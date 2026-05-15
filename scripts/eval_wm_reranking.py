#!/usr/bin/env python3
"""
Benchmark: geometry-score ranking vs world-model reranking.

For each (object, method, trial):
  1. Load a YCB object at a random tabletop position
  2. Wait for physics to settle
  3. Sample K grasp candidates around the settled centroid
  4. Rank candidates by method (geometry heuristic  OR  world-model MLP)
  5. Execute the top-1 ranked grasp
  6. Log: success, dz, fell_off, scores, grasp pose, timestamp

Geometry score — no model, pure heuristic:
  - XY centering over object pointcloud centroid  (60 %)
  - Z alignment with object top surface           (30 %)
  - Opening length matched to object size         (10 %)

World-model score — MLP predictor from train_mlp_predictor.py:
  wm_score = success_prob * (1 - fell_off_prob) * (1 + clip(dz_pred, -0.1, 0.3))

Output files (in --out-dir/eval_YYYYMMDD_HHMMSS/):
  results.csv          — one row per trial
  summary.txt          — per-object and overall table
  comparison.svg       — bar charts (pure-Python SVG, no matplotlib)

Usage:
  # Full benchmark (500 trials, ~90 min)
  MUJOCO_GL=egl conda run -n bridge \\
      python scripts/eval_wm_reranking.py

  # Quick smoke-test (50 trials, ~8 min)
  MUJOCO_GL=egl conda run -n bridge \\
      python scripts/eval_wm_reranking.py --quick

  # Custom objects / methods
  MUJOCO_GL=egl conda run -n bridge \\
      python scripts/eval_wm_reranking.py \\
          --objects banana,cylinder --methods world_model --trials 20
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from owg_robot.env_soarm import EnvironmentSoArm, TABLE_TOP_Z, ARM_JOINTS
from data.transition_logger import compute_pc_stats
from world_model.train_mlp_predictor import load_model, MODEL_PATH
from world_model.rerank_grasps import score_grasps

# ── Catalogue ─────────────────────────────────────────────────────────────────

OBJECTS = {
    "banana":   "YcbBanana",
    "cylinder": "YcbTomatoSoupCan",
    "cracker":  "YcbCrackerBox",
    "mustard":  "YcbMustardBottle",
    "drill":    "YcbPowerDrill",
}
ALL_METHODS = ["geometry", "world_model"]

# ── Physics / workspace constants ─────────────────────────────────────────────

_CENTRE_Y       = -0.40
_SPREAD_XY      = 0.06
_DROP_Z         = TABLE_TOP_Z + 0.12
_SETTLE_STEPS   = 300
_GRASP_Z_OFFSET = 0.025   # m above settled centroid
_LIFT_STEPS     = 80
_FELL_OFF_Z     = TABLE_TOP_Z - 0.10
_LIFT_THRESHOLD = 0.07    # m above table → successfully lifted

DEFAULT_K       = 10      # candidates per trial

# ── CSV schema ────────────────────────────────────────────────────────────────

CSV_COLS = [
    "trial_id", "object", "method",
    "success", "dz", "fell_off",
    "geo_score_top1", "wm_score_top1",
    "geo_score_mean", "wm_score_mean",
    "grasp_x", "grasp_y", "grasp_z", "grasp_yaw", "opening_len",
    "k_grasps", "timestamp",
]


# ── Geometry scoring heuristic ────────────────────────────────────────────────

def geo_score(grasp: np.ndarray, obj_pos: np.ndarray,
              pc_stats: np.ndarray) -> float:
    """
    Geometry-only grasp quality heuristic (no learned model).

    Rewards grasps that are:
      - Centered over the object pointcloud centroid (XY)
      - At the right height above the object top surface (Z)
      - With an opening width matched to the object's estimated size
    """
    x, y, z   = float(grasp[0]), float(grasp[1]), float(grasp[2])
    opening   = float(grasp[4])

    # Centroid from pc_stats[0:3]
    cx, cy    = float(pc_stats[0]), float(pc_stats[1])
    min_z     = float(pc_stats[6])
    max_z     = float(pc_stats[7])
    obj_top   = max_z if max_z > TABLE_TOP_Z else float(obj_pos[2]) + 0.03
    obj_height = max(max_z - min_z, 0.02)

    # XY centering — Gaussian, σ ≈ 0.057 m
    xy_err   = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    xy_s     = math.exp(-30 * xy_err ** 2)

    # Z alignment — should be _GRASP_Z_OFFSET above object top
    z_err    = abs(z - (obj_top + _GRASP_Z_OFFSET))
    z_s      = math.exp(-200 * z_err ** 2)

    # Opening width — proportional to object height as width proxy
    target_op = float(np.clip(obj_height * 0.8 + 0.02, 0.03, 0.09))
    op_s     = math.exp(-200 * (opening - target_op) ** 2)

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
    x, y, z    = float(grasp[0]), float(grasp[1]), float(grasp[2])
    yaw        = float(grasp[3])
    opening    = float(grasp[4])
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
    One evaluation trial. Returns a dict of logged metrics.
    Both geo_score and wm_score are computed for every trial regardless
    of method, so the CSV records what the other ranker would have chosen.
    """
    obj_name = OBJECTS[obj_key]

    # ── load + settle ─────────────────────────────────────────────────────────
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

    # ── sample K candidates ───────────────────────────────────────────────────
    candidates = np.stack([_sample_grasp(obj_pos, rng) for _ in range(k_grasps)])

    # ── score all candidates with both methods ────────────────────────────────
    geo_scores = np.array([geo_score(g, obj_pos, pc_stats) for g in candidates])
    wm_scores, _preds = score_grasps(candidates, obj_pos, obj_quat, pc_stats, model)

    # ── select top-1 by the active method ────────────────────────────────────
    if method == "geometry":
        best = int(np.argmax(geo_scores))
    else:  # world_model
        best = int(np.argmax(wm_scores))
    chosen = candidates[best]

    # ── execute ───────────────────────────────────────────────────────────────
    pos_before = obj_pos.copy()
    success    = _execute(env, oid, chosen)

    env._steps(40)
    pos_after = env.get_obj_pos(oid).copy()
    dz        = float(pos_after[2] - pos_before[2])
    fell_off  = bool(pos_after[2] < _FELL_OFF_Z)

    return {
        "success":        bool(success),
        "dz":             dz,
        "fell_off":       fell_off,
        "geo_score_top1": float(geo_scores[best]),
        "wm_score_top1":  float(wm_scores[best]),
        "geo_score_mean": float(geo_scores.mean()),
        "wm_score_mean":  float(float(wm_scores.mean())),
        "grasp_x":        float(chosen[0]),
        "grasp_y":        float(chosen[1]),
        "grasp_z":        float(chosen[2]),
        "grasp_yaw":      float(chosen[3]),
        "opening_len":    float(chosen[4]),
    }


# ── Statistics helpers ────────────────────────────────────────────────────────

def _stat(rows: list, key: str) -> tuple:
    """Return (mean, std, n) for a key across a list of row dicts."""
    vals = [float(r[key]) for r in rows]
    if not vals:
        return (float("nan"), float("nan"), 0)
    return (float(np.mean(vals)), float(np.std(vals)), len(vals))


def compute_summary(rows: list) -> dict:
    """
    Build nested summary: summary[method][obj] = {n, sr, fo, dz_mean, dz_std}
    Also populates summary[method]["OVERALL"].
    """
    by_mo = defaultdict(list)
    for r in rows:
        by_mo[(r["method"], r["object"])].append(r)

    summary = defaultdict(dict)
    for (method, obj), rs in by_mo.items():
        m_sr,  _, n = _stat(rs, "success")
        m_fo,  _, _ = _stat(rs, "fell_off")
        m_dz,  s_dz, _ = _stat(rs, "dz")
        summary[method][obj] = {
            "n":          n,
            "sr":         m_sr,
            "fell_off":   m_fo,
            "dz_mean":    m_dz,
            "dz_std":     s_dz,
        }

    for method in ALL_METHODS:
        rs = [r for r in rows if r["method"] == method]
        if rs:
            m_sr,  _, n = _stat(rs, "success")
            m_fo,  _, _ = _stat(rs, "fell_off")
            m_dz, s_dz, _ = _stat(rs, "dz")
            summary[method]["OVERALL"] = {
                "n":       n,
                "sr":      m_sr,
                "fell_off": m_fo,
                "dz_mean": m_dz,
                "dz_std":  s_dz,
            }

    return summary


def print_summary(summary: dict, obj_keys: list, methods: list,
                  file=None) -> None:
    W = 76
    def _p(*args, **kw):
        print(*args, **kw, file=file)

    _p("\n" + "=" * W)
    _p(f"  {'Object':<12} {'Method':<14} {'N':>5}  "
       f"{'SR':>6}  {'FellOff':>7}  {'dz_mean':>8}  {'dz_std':>7}")
    _p("-" * W)

    for obj in obj_keys + ["OVERALL"]:
        for method in methods:
            s = summary.get(method, {}).get(obj)
            if s is None:
                continue
            sr  = f"{s['sr']:.3f}"
            fo  = f"{s['fell_off']:.3f}"
            dzm = f"{s['dz_mean']:+.4f}"
            dzs = f"{s['dz_std']:.4f}"
            _p(f"  {obj:<12} {method:<14} {s['n']:>5}  "
               f"{sr:>6}  {fo:>7}  {dzm:>8}  {dzs:>7}")
        if obj != "OVERALL":
            _p()

    _p("=" * W)

    # Delta row
    _p("\n  Δ success_rate (world_model − geometry):")
    for obj in obj_keys + ["OVERALL"]:
        sg = summary.get("geometry",    {}).get(obj, {}).get("sr", float("nan"))
        sw = summary.get("world_model", {}).get(obj, {}).get("sr", float("nan"))
        delta = sw - sg
        tag   = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "~")
        _p(f"    {obj:<12}  {delta:+.3f}  {tag}")
    _p()


# ── SVG chart (pure Python, no matplotlib) ───────────────────────────────────

_SVG_COLORS = {"geometry": "#4878CF", "world_model": "#D65F5F"}


def _svg_bar_group(cx, cy_base, bar_w, values, colors, labels,
                   y_scale, y_offset, group_gap=8) -> list:
    """Return list of SVG <rect> and <text> elements for one group of bars."""
    elems = []
    n     = len(values)
    total_w = n * bar_w + (n - 1) * group_gap
    x0    = cx - total_w / 2

    for i, (v, c, lbl) in enumerate(zip(values, colors, labels)):
        if not math.isfinite(v):
            continue
        bx = x0 + i * (bar_w + group_gap)
        bh = max(0.0, v) * y_scale
        by = cy_base - bh - y_offset
        elems.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" '
            f'width="{bar_w:.1f}" height="{bh:.1f}" '
            f'fill="{c}" opacity="0.85" rx="2"/>'
        )
        # value label on top
        elems.append(
            f'<text x="{bx + bar_w/2:.1f}" y="{by - 3:.1f}" '
            f'text-anchor="middle" font-size="9" fill="#333">{v:.2f}</text>'
        )
    return elems


def make_svg_chart(summary: dict, obj_keys: list, methods: list,
                   out_path: Path) -> None:
    """
    Generate a three-panel comparison bar chart as an SVG file.

    Panel 1 — per-object success rate (side-by-side bars)
    Panel 2 — overall success rate + fell-off rate
    Panel 3 — mean dz with std error bars
    """
    W, H    = 900, 380
    margin  = {"top": 50, "bottom": 70, "left": 55, "right": 20}
    panel_w = (W - margin["left"] - margin["right"]) // 3
    chart_h = H - margin["top"] - margin["bottom"]
    y_base  = margin["top"] + chart_h   # SVG y increases downward

    svgs = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">']

    # ── background ────────────────────────────────────────────────────────────
    svgs.append(f'<rect width="{W}" height="{H}" fill="#f8f8f8"/>')

    # ── title ─────────────────────────────────────────────────────────────────
    svgs.append(
        f'<text x="{W//2}" y="22" text-anchor="middle" '
        f'font-size="14" font-weight="bold" fill="#222">'
        f'Geometry Score vs World-Model Reranking</text>'
    )

    colors = [_SVG_COLORS.get(m, "#888") for m in methods]

    # ── Legend ────────────────────────────────────────────────────────────────
    lx = W - margin["right"] - 160
    for i, (m, c) in enumerate(zip(methods, colors)):
        svgs.append(
            f'<rect x="{lx}" y="{8 + i*16}" '
            f'width="12" height="10" fill="{c}" rx="2"/>'
        )
        svgs.append(
            f'<text x="{lx + 16}" y="{18 + i*16}" '
            f'font-size="10" fill="#333">{m.replace("_", " ")}</text>'
        )

    def panel_x(p: int) -> int:
        return margin["left"] + p * panel_w

    def draw_axes(px: int, label: str, y_max: float, y_ticks: list):
        """Draw Y axis, gridlines, and title for one panel."""
        y_scale = chart_h / y_max
        x0 = px
        x1 = px + panel_w - 20
        svgs.append(
            f'<line x1="{x0}" y1="{margin["top"]}" '
            f'x2="{x0}" y2="{y_base}" stroke="#aaa" stroke-width="1"/>'
        )
        svgs.append(
            f'<line x1="{x0}" y1="{y_base}" '
            f'x2="{x1}" y2="{y_base}" stroke="#aaa" stroke-width="1"/>'
        )
        for t in y_ticks:
            ty = y_base - t * y_scale
            svgs.append(
                f'<line x1="{x0}" y1="{ty:.1f}" '
                f'x2="{x1}" y2="{ty:.1f}" stroke="#ddd" stroke-width="0.8"/>'
            )
            svgs.append(
                f'<text x="{x0-4}" y="{ty+4:.1f}" '
                f'text-anchor="end" font-size="9" fill="#666">{t:.1f}</text>'
            )
        # Y axis label
        svgs.append(
            f'<text x="{x0-38}" y="{margin["top"] + chart_h//2}" '
            f'text-anchor="middle" font-size="10" fill="#555" '
            f'transform="rotate(-90 {x0-38} {margin["top"] + chart_h//2})">'
            f'{label}</text>'
        )
        return y_scale

    # ════════════════════════════════════════════════════════════════════════
    # Panel 1 — per-object success rate
    # ════════════════════════════════════════════════════════════════════════
    px1    = panel_x(0)
    y_max1 = 1.0
    y_sc1  = draw_axes(px1, "Success Rate", y_max1,
                       [0.0, 0.25, 0.50, 0.75, 1.0])

    svgs.append(
        f'<text x="{px1 + (panel_w-20)//2}" y="{margin["top"]-8}" '
        f'text-anchor="middle" font-size="11" font-weight="bold" fill="#333">'
        f'Per-Object Success Rate</text>'
    )

    bar_w = 12
    n_obj = len(obj_keys)
    group_span = panel_w - 40
    group_gap  = group_span / max(n_obj, 1)

    for oi, obj in enumerate(obj_keys):
        cx = px1 + 20 + oi * group_gap + group_gap / 2
        vals = [summary.get(m, {}).get(obj, {}).get("sr", 0.0) for m in methods]
        elems = _svg_bar_group(cx, y_base, bar_w, vals, colors,
                               methods, y_sc1, 0)
        svgs.extend(elems)
        # X tick
        svgs.append(
            f'<text x="{cx:.1f}" y="{y_base + 14}" '
            f'text-anchor="middle" font-size="9" fill="#555">{obj}</text>'
        )

    # ════════════════════════════════════════════════════════════════════════
    # Panel 2 — overall success + fell-off rates
    # ════════════════════════════════════════════════════════════════════════
    px2   = panel_x(1)
    y_max2 = 1.0
    y_sc2  = draw_axes(px2, "", y_max2, [0.0, 0.25, 0.50, 0.75, 1.0])

    svgs.append(
        f'<text x="{px2 + (panel_w-20)//2}" y="{margin["top"]-8}" '
        f'text-anchor="middle" font-size="11" font-weight="bold" fill="#333">'
        f'Overall Rates</text>'
    )

    metrics_p2 = [("sr", "Success"), ("fell_off", "Fell-Off")]
    mg2        = (panel_w - 40) / max(len(metrics_p2), 1)

    for mi, (mkey, mlabel) in enumerate(metrics_p2):
        cx = px2 + 20 + mi * mg2 + mg2 / 2
        vals = [summary.get(m, {}).get("OVERALL", {}).get(mkey, 0.0)
                for m in methods]
        svgs.extend(_svg_bar_group(cx, y_base, bar_w + 4, vals, colors,
                                   methods, y_sc2, 0))
        svgs.append(
            f'<text x="{cx:.1f}" y="{y_base + 14}" '
            f'text-anchor="middle" font-size="9" fill="#555">{mlabel}</text>'
        )

    # ════════════════════════════════════════════════════════════════════════
    # Panel 3 — mean dz per object (with ± std error bars)
    # ════════════════════════════════════════════════════════════════════════
    px3 = panel_x(2)

    # Collect dz values to set y range
    all_dz_means = []
    all_dz_stds  = []
    for m in methods:
        for obj in obj_keys:
            s = summary.get(m, {}).get(obj, {})
            if s:
                all_dz_means.append(s.get("dz_mean", 0.0))
                all_dz_stds.append(s.get("dz_std", 0.0))

    dz_min = min((v - s for v, s in zip(all_dz_means, all_dz_stds)),
                 default=-0.15)
    dz_max = max((v + s for v, s in zip(all_dz_means, all_dz_stds)),
                 default=0.15)
    dz_min = min(dz_min, -0.01)
    dz_max = max(dz_max,  0.01)
    dz_pad  = 0.02
    dz_min -= dz_pad
    dz_max += dz_pad
    dz_range = dz_max - dz_min

    # Map dz → SVG y
    def dz_to_y(v): return y_base - (v - dz_min) / dz_range * chart_h

    y_ticks_dz = [round(t, 2)
                  for t in np.arange(
                      math.ceil(dz_min * 20) / 20,
                      dz_max + 0.001, 0.05)]

    draw_axes(px3, "Mean dz (m)", dz_range, [])   # draw axes frame only

    # Gridlines for dz
    x0_3 = px3
    x1_3 = px3 + panel_w - 20
    for t in y_ticks_dz:
        ty = dz_to_y(t)
        svgs.append(
            f'<line x1="{x0_3}" y1="{ty:.1f}" '
            f'x2="{x1_3}" y2="{ty:.1f}" stroke="#ddd" stroke-width="0.8"/>'
        )
        svgs.append(
            f'<text x="{x0_3-4}" y="{ty+4:.1f}" '
            f'text-anchor="end" font-size="9" fill="#666">{t:+.2f}</text>'
        )

    # Zero line
    zy = dz_to_y(0)
    svgs.append(
        f'<line x1="{x0_3}" y1="{zy:.1f}" '
        f'x2="{x1_3}" y2="{zy:.1f}" stroke="#999" stroke-width="1" '
        f'stroke-dasharray="4,3"/>'
    )

    svgs.append(
        f'<text x="{px3 + (panel_w-20)//2}" y="{margin["top"]-8}" '
        f'text-anchor="middle" font-size="11" font-weight="bold" fill="#333">'
        f'Mean dz (± std)</text>'
    )

    gapw3 = (panel_w - 40) / max(n_obj, 1)
    for oi, obj in enumerate(obj_keys):
        cx = px3 + 20 + oi * gapw3 + gapw3 / 2
        n_m = len(methods)
        bw  = 12
        total_w = n_m * bw + (n_m - 1) * 4
        x0  = cx - total_w / 2
        for i, (m, c) in enumerate(zip(methods, colors)):
            s  = summary.get(m, {}).get(obj, {})
            dz = s.get("dz_mean", float("nan"))
            sd = s.get("dz_std",  0.0)
            if not math.isfinite(dz):
                continue
            bx  = x0 + i * (bw + 4)
            by0 = dz_to_y(0)
            by1 = dz_to_y(dz)
            bh  = abs(by1 - by0)
            by_top = min(by0, by1)
            svgs.append(
                f'<rect x="{bx:.1f}" y="{by_top:.1f}" '
                f'width="{bw:.1f}" height="{bh:.1f}" '
                f'fill="{c}" opacity="0.85" rx="2"/>'
            )
            # Error bar
            bcx = bx + bw / 2
            err_top = dz_to_y(dz + sd)
            err_bot = dz_to_y(dz - sd)
            svgs.append(
                f'<line x1="{bcx:.1f}" y1="{err_top:.1f}" '
                f'x2="{bcx:.1f}" y2="{err_bot:.1f}" '
                f'stroke="#333" stroke-width="1.2"/>'
            )
            for ey in (err_top, err_bot):
                svgs.append(
                    f'<line x1="{bcx-3:.1f}" y1="{ey:.1f}" '
                    f'x2="{bcx+3:.1f}" y2="{ey:.1f}" '
                    f'stroke="#333" stroke-width="1.2"/>'
                )

        svgs.append(
            f'<text x="{cx:.1f}" y="{y_base + 14}" '
            f'text-anchor="middle" font-size="9" fill="#555">{obj}</text>'
        )

    # ── close SVG ─────────────────────────────────────────────────────────────
    svgs.append("</svg>")
    out_path.write_text("\n".join(svgs))
    print(f"  Chart  → {out_path}")


# ── CLI / main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trials",      type=int, default=50,
                    help="Trials per (object, method) (default: 50)")
    ap.add_argument("--objects",     default="all",
                    help="Comma-separated object keys, or 'all'")
    ap.add_argument("--methods",     default="geometry,world_model",
                    help="Comma-separated methods")
    ap.add_argument("--k-grasps",    type=int, default=DEFAULT_K,
                    help=f"Candidates per trial (default: {DEFAULT_K})")
    ap.add_argument("--model-path",  default=str(MODEL_PATH))
    ap.add_argument("--out-dir",     default="results",
                    help="Parent output directory (default: results/)")
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--quick",       action="store_true",
                    help="5 trials per (object, method) — for smoke testing")
    ap.add_argument("--quiet",       action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.trials = 5

    # ── parse objects / methods ───────────────────────────────────────────────
    obj_keys = list(OBJECTS.keys()) if args.objects == "all" \
               else [k.strip() for k in args.objects.split(",")]
    methods  = [m.strip() for m in args.methods.split(",")]
    for k in obj_keys:
        if k not in OBJECTS:
            ap.error(f"Unknown object '{k}'. Valid: {list(OBJECTS.keys())}")
    for m in methods:
        if m not in ALL_METHODS:
            ap.error(f"Unknown method '{m}'. Valid: {ALL_METHODS}")

    # ── output directory ──────────────────────────────────────────────────────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"eval_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path     = out_dir / "results.csv"
    summary_path = out_dir / "summary.txt"
    chart_path   = out_dir / "comparison.svg"

    n_total = args.trials * len(obj_keys) * len(methods)
    print(f"\nEval run: {run_id}")
    print(f"  objects  : {obj_keys}")
    print(f"  methods  : {methods}")
    print(f"  trials   : {args.trials} per (object, method)  →  {n_total} total")
    print(f"  k_grasps : {args.k_grasps}")
    print(f"  seed     : {args.seed}")
    print(f"  out_dir  : {out_dir}\n")

    # ── load model ────────────────────────────────────────────────────────────
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("  Run: python world_model/train_mlp_predictor.py")
        sys.exit(1)
    model = load_model(model_path)
    print(f"Loaded model from {model_path}\n")

    # ── env + rng ─────────────────────────────────────────────────────────────
    env = EnvironmentSoArm(vis=False, debug=False)
    rng = np.random.default_rng(args.seed)

    # ── run trials ────────────────────────────────────────────────────────────
    rows     = []
    trial_id = 0
    t0_total = time.time()

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLS)
        writer.writeheader()

        for obj_key in obj_keys:
            for method in methods:
                n_success = 0
                for ti in range(args.trials):
                    t0 = time.time()
                    result = run_trial(env, obj_key, method, rng, model,
                                       args.k_grasps)
                    elapsed = time.time() - t0

                    row = {
                        "trial_id":  trial_id,
                        "object":    obj_key,
                        "method":    method,
                        "k_grasps":  args.k_grasps,
                        "timestamp": time.time(),
                        **result,
                    }
                    rows.append(row)
                    writer.writerow(row)
                    fcsv.flush()

                    trial_id  += 1
                    n_success += int(result["success"])

                    if not args.quiet:
                        sym  = "✓" if result["success"] else "✗"
                        ff   = " [fell]" if result["fell_off"] else ""
                        done = trial_id
                        pct  = 100 * done / n_total
                        eta_s = (time.time() - t0_total) / done * (n_total - done)
                        print(f"  [{done:04d}/{n_total}] {pct:4.0f}%  "
                              f"{obj_key:<10} {method:<14} {sym}{ff}  "
                              f"dz={result['dz']:+.3f}  "
                              f"geo={result['geo_score_top1']:.3f}  "
                              f"wm={result['wm_score_top1']:.3f}  "
                              f"({elapsed:.0f}s  ETA {eta_s/60:.0f}m)")

                print(f"\n  → {obj_key}/{method}: "
                      f"SR={n_success/args.trials:.2f} "
                      f"({n_success}/{args.trials})\n")

    # ── summary ───────────────────────────────────────────────────────────────
    summary = compute_summary(rows)

    with open(summary_path, "w") as f:
        print(f"Eval run: {run_id}", file=f)
        print(f"trials={args.trials}  k={args.k_grasps}  "
              f"objects={obj_keys}  methods={methods}", file=f)
        print_summary(summary, obj_keys, methods, file=f)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print_summary(summary, obj_keys, methods)

    # ── chart ─────────────────────────────────────────────────────────────────
    print("\nGenerating chart …")
    make_svg_chart(summary, obj_keys, methods, chart_path)

    # ── file manifest ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0_total
    print(f"\n  CSV     → {csv_path}")
    print(f"  Summary → {summary_path}")
    print(f"  Total time: {elapsed_total/60:.1f} min  "
          f"({elapsed_total/trial_id:.1f} s/trial)")

    env.close()


if __name__ == "__main__":
    main()
