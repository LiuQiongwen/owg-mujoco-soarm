#!/usr/bin/env python3
"""
Generate three paper figures from EXPERIMENTS.md data.
Saves to figures/ as PDF.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

os.makedirs("figures", exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "pdf.fonttype":      42,   # embed fonts (required by IEEE)
    "ps.fonttype":       42,
})

BLUE   = "#2166ac"
RED    = "#d6604d"
GRAY   = "#888888"
GREEN  = "#4dac26"
ORANGE = "#f4a582"

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: flat_frac vs S4 net improvement  (scatter + regression line)
# Data from EXPERIMENTS.md Table VI
# ══════════════════════════════════════════════════════════════════════════════
objects = ["CrackerBox", "MustardBottle", "Banana", "TomatoSoupCan", "PowerDrill", "Scissors"]
flat_frac = np.array([0.02,  0.03,  0.05,  0.05,  0.10,  0.457])
s4_net    = np.array([+4,    +2,     0,    -1,    -1,    -4])
sigma_yaw = np.array([0.06,  0.05,  0.04,  0.05,  0.195, 0.08])

# colour-code: positive net = blue, negative = red, zero = grey
point_colors = [BLUE if n > 0 else (RED if n < 0 else GRAY) for n in s4_net]

fig, ax = plt.subplots(figsize=(3.5, 3.0))

# Spearman rho
rho, pval = stats.spearmanr(flat_frac, s4_net)

# OLS trend line
m, b = np.polyfit(flat_frac, s4_net, 1)
x_line = np.linspace(-0.01, 0.48, 200)
ax.plot(x_line, m * x_line + b, color=GRAY, lw=1.2, ls="--", zorder=1, label=f"OLS fit")

# Scatter
for i, obj in enumerate(objects):
    # marker size encodes sigma_yaw
    ms = 60 + sigma_yaw[i] * 400
    ax.scatter(flat_frac[i], s4_net[i],
               s=ms, color=point_colors[i], zorder=3,
               edgecolors="white", linewidths=0.5)
    # label offset: nudge to avoid overlap
    offsets = {
        "CrackerBox":    ( 0.005, -0.7),
        "MustardBottle": ( 0.005,  0.4),
        "Banana":        ( 0.005,  0.4),
        "TomatoSoupCan": (-0.015, -0.8),
        "PowerDrill":    ( 0.005,  0.4),
        "Scissors":      (-0.025, -0.7),
    }
    dx, dy = offsets[obj]
    ax.annotate(obj, (flat_frac[i] + dx, s4_net[i] + dy),
                fontsize=6.5, color="#222222")

ax.axhline(0, color="black", lw=0.7, ls="-")
ax.set_xlabel("Flat fraction  (frac. candidates with $H < 0.001$)")
ax.set_ylabel("Stage 4 net improvement  (trials, $N=25$ seeds)")
ax.set_title(f"Geometric degeneracy predicts reranking outcome\n"
             f"Spearman $\\rho$ = {rho:+.2f},  $p$ = {pval:.3f}")

# legend for marker size
for yaw_val, label in [(0.05, r"$\sigma_\mathrm{yaw}=0.05$"), (0.20, r"$\sigma_\mathrm{yaw}=0.20$")]:
    ax.scatter([], [], s=60 + yaw_val * 400, color=GRAY, edgecolors="white",
               linewidths=0.5, label=label)
ax.legend(frameon=False, loc="upper right", handletextpad=0.3)

ax.set_xlim(-0.02, 0.52)
ax.set_ylim(-5.5, 5.5)
ax.set_yticks([-4, -2, 0, 2, 4])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.5)
fig.savefig("figures/fig1_flat_frac_vs_net.pdf", bbox_inches="tight")
print("Saved figures/fig1_flat_frac_vs_net.pdf")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: per-object success rate — Stage 3 vs Stage 4 (grouped bar chart)
# Data from EXPERIMENTS.md Table I  (25 seeds)
# ══════════════════════════════════════════════════════════════════════════════
obj_labels = ["Banana", "Cracker\nBox", "Mustard\nBottle", "Power\nDrill", "Scissors", "Tomato\nSoupCan"]
s3_counts = np.array([18, 13, 21, 22, 20, 21])
s4_counts = np.array([18, 17, 23, 21, 16, 20])
N = 25
s3_rate = s3_counts / N
s4_rate = s4_counts / N
net      = s4_counts - s3_counts

x = np.arange(len(obj_labels))
width = 0.36

fig, ax = plt.subplots(figsize=(5.5, 3.0))

bars3 = ax.bar(x - width/2, s3_rate * 100, width, label="Stage 3 (baseline)",
               color=GRAY, alpha=0.85, edgecolor="white", linewidth=0.5)
bars4 = ax.bar(x + width/2, s4_rate * 100, width, label="Stage 4 (LGGSN v2)",
               edgecolor="white", linewidth=0.5,
               color=[BLUE if n > 0 else (RED if n < 0 else GRAY) for n in net],
               alpha=0.92)

# Net annotation above bar pairs
for i, n in enumerate(net):
    top = max(s3_rate[i], s4_rate[i]) * 100
    color = BLUE if n > 0 else (RED if n < 0 else GRAY)
    sign  = "+" if n >= 0 else ""
    ax.text(x[i], top + 1.5, f"net{sign}{n}", ha="center", va="bottom",
            fontsize=7.5, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(obj_labels, fontsize=8)
ax.set_ylabel("Success rate (%)")
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_title("Per-object grasp success rate: Stage 3 vs Stage 4  ($N = 25$ seeds)")
ax.legend(frameon=False, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Horizontal reference line at 76.7% (overall average)
overall = (s3_counts.sum()) / (N * len(obj_labels)) * 100
ax.axhline(overall, color=GRAY, lw=0.8, ls=":", zorder=0)
ax.text(len(obj_labels) - 0.05, overall + 1.0, f"avg {overall:.0f}%",
        ha="right", va="bottom", fontsize=7, color=GRAY)

fig.tight_layout(pad=0.5)
fig.savefig("figures/fig2_success_rate_bar.pdf", bbox_inches="tight")
print("Saved figures/fig2_success_rate_bar.pdf")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Ablation study visualisation
# Data from EXPERIMENTS.md Table II  (10 seeds, 60 trials each condition)
# Two panels: (a) val pair_acc  (b) task net improvement
# ══════════════════════════════════════════════════════════════════════════════
conditions    = ["D\n(12-dim\nbase)", "C\n(+z_rel)", "B\n(+dist)", "A\n(v2 both)"]
val_pair_acc  = np.array([0.579, 0.576, 0.558, 0.664])
task_net      = np.array([-1, -2, +7, +4])
# S3 success totals vary slightly by condition (different random seeds for S3)
s3_totals     = np.array([46, 41, 46, 46])
s4_totals     = np.array([45, 39, 53, 50])
s3_rate_ab    = s3_totals / 60 * 100
s4_rate_ab    = s4_totals / 60 * 100

colors_net = [RED if n < 0 else BLUE for n in task_net]
x = np.arange(len(conditions))

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))

# ── Panel (a): val pair_acc ────────────────────────────────────────────────
ax = axes[0]
bars = ax.bar(x, val_pair_acc, color=[GRAY, GRAY, GRAY, BLUE],
              edgecolor="white", linewidth=0.5, width=0.55, alpha=0.88)
ax.axhline(0.5, color="black", lw=0.8, ls="--", zorder=0)
ax.text(len(conditions) - 1 + 0.35, 0.503, "random", fontsize=7, color="black",
        va="bottom", ha="right")
for bar, val in zip(bars, val_pair_acc):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.004,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
            color=BLUE if val == max(val_pair_acc) else "#333333",
            fontweight="bold" if val == max(val_pair_acc) else "normal")
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=8)
ax.set_ylabel("Validation pair accuracy")
ax.set_ylim(0.45, 0.72)
ax.set_title("(a) Offline ranking accuracy")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Panel (b): task net improvement ───────────────────────────────────────
ax = axes[1]
bars = ax.bar(x, task_net, color=colors_net,
              edgecolor="white", linewidth=0.5, width=0.55, alpha=0.88)
ax.axhline(0, color="black", lw=0.8)
for bar, val in zip(bars, task_net):
    ypos = val + 0.15 if val >= 0 else val - 0.4
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{val:+d}", ha="center", va="bottom" if val >= 0 else "top",
            fontsize=8, fontweight="bold",
            color=BLUE if val > 0 else (RED if val < 0 else GRAY))
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=8)
ax.set_ylabel("Net improvement  (trials, $N=10$ seeds)")
ax.set_ylim(-4.5, 10)
ax.set_yticks([-2, 0, 2, 4, 6, 8])
ax.set_title("(b) Task-level net improvement")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# shared subtitle note
fig.suptitle("Feature ablation study  (10 seeds × 6 objects = 60 paired trials per condition)",
             fontsize=9, y=1.01)
fig.tight_layout(pad=0.6)
fig.savefig("figures/fig3_ablation.pdf", bbox_inches="tight")
print("Saved figures/fig3_ablation.pdf")
plt.close(fig)

print("\nAll figures written to figures/")
