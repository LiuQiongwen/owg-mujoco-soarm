#!/usr/bin/env python3
"""Failure analysis heatmaps for Panda grasp benchmark.

Reads failure_summary.csv and generates:
  1. yaw_error_deg    vs success rate  (histogram overlay)
  2. obj_height       vs success rate  (binned bar chart)
  3. finger_gap       vs success rate  (histogram overlay)
  4. failure_reason   breakdown        (pie + bar)
  5. contact_ratio    vs success rate
  6. orn_error_deg    vs success rate

Usage
-----
    python grasp_6dof/failure_heatmap.py
    python grasp_6dof/failure_heatmap.py --csv grasp_6dof/out/failure_summary.csv
    python grasp_6dof/failure_heatmap.py --csv grasp_6dof/out/failure_summary.csv --out plots/
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# matplotlib Agg (no display needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── colour palette ────────────────────────────────────────────
_SUCCESS_COLOR = "#2ecc71"
_FAIL_COLOR    = "#e74c3c"
_REASON_COLORS = [
    "#e74c3c","#e67e22","#f1c40f","#1abc9c",
    "#3498db","#9b59b6","#2ecc71","#95a5a6","#34495e",
]


# ── data loading ──────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # cast numeric fields
    float_fields = [
        "yaw_error_deg","pitch_error_deg","roll_error_deg","orn_error_deg",
        "contact_ratio","finger_gap","finger_force_proxy","lift_dz","obj_height",
        "ik_pos_err","world_yaw_deg","grasp_score",
    ]
    int_fields = [
        "contact_steps","bilateral_contact_steps","total_lift_polls",
    ]
    for row in rows:
        for f in float_fields:
            try:   row[f] = float(row[f]) if row.get(f) not in (None, "", "None") else None
            except ValueError: row[f] = None
        for f in int_fields:
            try:   row[f] = int(row[f]) if row.get(f) not in (None, "", "None") else None
            except ValueError: row[f] = None
        row["success"] = row.get("success", "False").strip().lower() in ("true","1","yes")
    return rows


# ── helpers ───────────────────────────────────────────────────
def _pairs(rows, field):
    """Return (values, successes) for rows where field is not None."""
    xs, ys = [], []
    for r in rows:
        v = r.get(field)
        if v is not None:
            xs.append(float(v))
            ys.append(1 if r["success"] else 0)
    return np.array(xs), np.array(ys)


def _binned_success_rate(xs, ys, n_bins=10, x_range=None):
    """Return (bin_centres, success_rates, counts)."""
    lo = x_range[0] if x_range else xs.min()
    hi = x_range[1] if x_range else xs.max()
    edges = np.linspace(lo, hi, n_bins + 1)
    centres, rates, counts = [], [], []
    for i in range(n_bins):
        mask = (xs >= edges[i]) & (xs < edges[i + 1])
        n = mask.sum()
        counts.append(int(n))
        centres.append(float(0.5 * (edges[i] + edges[i + 1])))
        rates.append(float(ys[mask].mean()) if n > 0 else np.nan)
    return np.array(centres), np.array(rates), np.array(counts)


def _scalar_panel(ax, xs, ys, xlabel, n_bins=10, x_range=None):
    """Draw histogram + success-rate overlay on a single Axes."""
    if len(xs) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    centres, rates, counts = _binned_success_rate(xs, ys, n_bins=n_bins, x_range=x_range)
    width = centres[1] - centres[0] if len(centres) > 1 else 1.0

    # count bars on left y-axis
    ax.bar(centres, counts, width=width * 0.8, color="#bdc3c7", alpha=0.7,
           label="count", zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count", color="#7f8c8d")

    # success rate on right y-axis
    ax2 = ax.twinx()
    valid = ~np.isnan(rates)
    ax2.plot(centres[valid], rates[valid], "o-", color=_SUCCESS_COLOR,
             linewidth=2, markersize=5, label="success rate", zorder=3)
    ax2.axhline(ys.mean(), color=_FAIL_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Success rate", color=_SUCCESS_COLOR)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)


# ── individual plot functions ─────────────────────────────────
def plot_reason_breakdown(rows, out_dir: Path, prefix="") -> list[Path]:
    """Pie + bar chart of failure reasons."""
    reasons = [r.get("failure_reason") or "SUCCESS" for r in rows]
    cnt = Counter(reasons)
    labels = list(cnt.keys())
    sizes  = [cnt[l] for l in labels]
    colors = (_REASON_COLORS * 4)[:len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Failure Reason Breakdown", fontsize=13)

    # pie
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.8)
    for t in autotexts:
        t.set_fontsize(8)

    # bar
    axes[1].barh(labels, sizes, color=colors)
    axes[1].set_xlabel("Count")
    for i, (l, s) in enumerate(zip(labels, sizes)):
        axes[1].text(s + 0.2, i, str(s), va="center", fontsize=8)
    axes[1].invert_yaxis()

    saved = []
    for ext in ("png", "pdf"):
        p_ = out_dir / f"{prefix}failure_reasons.{ext}"
        fig.savefig(p_, dpi=150, bbox_inches="tight")
        saved.append(p_)
    plt.close(fig)
    return saved


def plot_scalar_heatmaps(rows, out_dir: Path, prefix="") -> list[Path]:
    """6-panel figure: yaw_err, obj_height, finger_gap, contact_ratio, orn_err, ik_pos_err."""
    panels = [
        ("yaw_error_deg",    "Yaw error (°)",         (0, 90),   10),
        ("obj_height",       "Object height (m)",      None,       8),
        ("finger_gap",       "Finger gap (m)",          (0, 0.08), 8),
        ("contact_ratio",    "Contact ratio (lift)",   (0, 1.0),  10),
        ("orn_error_deg",    "Orientation error (°)",  (0, 90),   10),
        ("ik_pos_err",       "IK pos error (m)",        None,       8),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Grasp Diagnostics vs Success Rate", fontsize=13)

    for ax, (field, xlabel, x_range, n_bins) in zip(axes.flat, panels):
        xs, ys = _pairs(rows, field)
        ax.set_title(xlabel, fontsize=9)
        _scalar_panel(ax, xs, ys, xlabel, n_bins=n_bins, x_range=x_range)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    saved = []
    for ext in ("png", "pdf"):
        p_ = out_dir / f"{prefix}scalar_heatmaps.{ext}"
        fig.savefig(p_, dpi=150, bbox_inches="tight")
        saved.append(p_)
    plt.close(fig)
    return saved


def plot_yaw_polar(rows, out_dir: Path, prefix="") -> list[Path]:
    """Polar histogram: world_yaw_deg coloured by success/fail."""
    succ_yaws = [np.radians(r["world_yaw_deg"]) for r in rows
                 if r["success"] and r.get("world_yaw_deg") is not None]
    fail_yaws = [np.radians(r["world_yaw_deg"]) for r in rows
                 if not r["success"] and r.get("world_yaw_deg") is not None]

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="polar")
    n_bins = 24
    edges  = np.linspace(-np.pi, np.pi, n_bins + 1)
    width  = 2 * np.pi / n_bins

    sc, _ = np.histogram(succ_yaws, bins=edges)
    fc, _ = np.histogram(fail_yaws, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    ax.bar(centres, sc, width=width, bottom=0, alpha=0.7,
           color=_SUCCESS_COLOR, label="success")
    ax.bar(centres, fc, width=width, bottom=sc, alpha=0.7,
           color=_FAIL_COLOR, label="fail")
    ax.set_title("World Yaw Distribution\n(success vs fail)", pad=20, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)

    saved = []
    for ext in ("png", "pdf"):
        p_ = out_dir / f"{prefix}yaw_polar.{ext}"
        fig.savefig(p_, dpi=150, bbox_inches="tight")
        saved.append(p_)
    plt.close(fig)
    return saved


def plot_contact_persistence(rows, out_dir: Path, prefix="") -> list[Path]:
    """Scatter: contact_ratio vs lift_dz, coloured success/fail."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sx = [r["contact_ratio"] for r in rows
          if r["success"] and r.get("contact_ratio") is not None]
    sy = [r["lift_dz"]       for r in rows
          if r["success"] and r.get("lift_dz") is not None]
    fx = [r["contact_ratio"] for r in rows
          if not r["success"] and r.get("contact_ratio") is not None]
    fy = [r["lift_dz"]       for r in rows
          if not r["success"] and r.get("lift_dz") is not None]

    ax.scatter(fx, fy, alpha=0.5, s=20, c=_FAIL_COLOR,    label=f"fail  (n={len(fx)})")
    ax.scatter(sx, sy, alpha=0.7, s=25, c=_SUCCESS_COLOR, label=f"success (n={len(sx)})")
    ax.axhline(0.05, color="k", linestyle="--", linewidth=1, label="lift threshold")
    ax.set_xlabel("Bilateral contact ratio during lift")
    ax.set_ylabel("Lift Δz (m)")
    ax.set_title("Contact Persistence vs Lift Height")
    ax.legend(fontsize=8)
    fig.tight_layout()

    saved = []
    for ext in ("png", "pdf"):
        p_ = out_dir / f"{prefix}contact_persistence.{ext}"
        fig.savefig(p_, dpi=150, bbox_inches="tight")
        saved.append(p_)
    plt.close(fig)
    return saved


# ── main ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Failure heatmaps for Panda grasp benchmark")
    ap.add_argument("--csv",    default="grasp_6dof/out/failure_summary.csv",
                    help="Path to failure_summary.csv")
    ap.add_argument("--out",    default="grasp_6dof/out/heatmaps",
                    help="Output directory for plots")
    ap.add_argument("--prefix", default="",
                    help="Optional filename prefix")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        print("Run validate_grasps_panda.py first to generate failure_summary.csv")
        sys.exit(1)

    rows = load_csv(csv_path)
    print(f"[INFO] Loaded {len(rows)} trials  "
          f"({sum(r['success'] for r in rows)} success / {sum(not r['success'] for r in rows)} fail)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    saved += plot_reason_breakdown(rows, out_dir, prefix=args.prefix)
    saved += plot_scalar_heatmaps( rows, out_dir, prefix=args.prefix)
    saved += plot_yaw_polar(       rows, out_dir, prefix=args.prefix)
    saved += plot_contact_persistence(rows, out_dir, prefix=args.prefix)

    print(f"\n[INFO] {len(saved)} plots saved to {out_dir}/")
    for p_ in saved:
        print(f"  {p_.name}")


if __name__ == "__main__":
    main()
