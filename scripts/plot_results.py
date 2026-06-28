#!/usr/bin/env python3
"""
Generate publication figures for the OWG paper (RAS/Robotica submission).
Outputs:
  figures/fig_fiveway.pdf   — five-way overall SR bar chart with 95% CI
  figures/fig_perobject.pdf — per-object grouped bar chart
  figures/fig_combined.pdf  — two-panel combined figure
"""

import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm

# ── Data ─────────────────────────────────────────────────────────────────────

OBJECTS = ["Banana", "TomatoSoupCan", "Pear", "MustardBottle",
           "Scissors", "CrackerBox", "PowerDrill"]
OBJ_LABELS = ["Banana", "Tomato\nSoup Can", "Pear", "Mustard\nBottle",
              "Scissors", "Cracker\nBox", "Power\nDrill"]

LOGS = {
    "Random\n(Stage 2)":  "eval_baseline_nosem_v2.log",
    "GRC-6DoF":                "eval_grc6dof_nosem_v2.log",
    "CFM\n(no OT)":       "eval_cfm_noOT_nosem.log",
    "DDPM\n(DDIM-50)":         "eval_ddpm_ddim50_nosem_175.log",
    "OT-CFM\n(ours)":          "eval_cfm_ot_nosem_current.log",
}
SCISSORS_FIX = {}  # v2 logs use post-fix Scissors config; no correction needed

# Colorblind-safe (Wong 2011) + distinct B&W hatching
COLORS = ["#999999", "#0072B2", "#E69F00", "#D55E00", "#009E73"]
HATCH  = ["//", "..", "xx", "\\\\", ""]

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def parse_log(path):
    obj_data = {}
    with open(path) as f:
        for line in f:
            m = re.search(r"---\s+(\w+):\s+(\d+)/(\d+)", line)
            if m:
                obj, s, n = m.group(1), int(m.group(2)), int(m.group(3))
                obj_data[obj] = (s, n)
    return obj_data


def wilson_ci(s, n, alpha=0.05):
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - alpha / 2)
    p = s / n
    center = (p + z**2 / (2*n)) / (1 + z**2 / n)
    half = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return max(0.0, center - half), min(1.0, center + half)


# ── Load data ─────────────────────────────────────────────────────────────────

results = {}
for label, fname in LOGS.items():
    path = os.path.join(LOG_DIR, fname)
    obj_data = parse_log(path)
    if label in SCISSORS_FIX and "Scissors" in obj_data:
        s, n = obj_data["Scissors"]
        obj_data["Scissors"] = (s + SCISSORS_FIX[label], n)
    results[label] = obj_data

labels = list(LOGS.keys())
n_methods = len(labels)
n_obj     = len(OBJECTS)

# Overall totals
totals = {lb: (sum(v[0] for v in d.values()), sum(v[1] for v in d.values()))
          for lb, d in results.items()}

# Per-object SR matrix  [n_methods × n_obj]
sr_matrix = np.zeros((n_methods, n_obj))
for i, lb in enumerate(labels):
    for j, obj in enumerate(OBJECTS):
        s, n = results[lb].get(obj, (0, 25))
        sr_matrix[i, j] = s / n

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "figures"), exist_ok=True)
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


# ── Figure 1: five-way overall SR ─────────────────────────────────────────────

def plot_fiveway(ax, annotate_sig=True):
    srs  = [totals[lb][0] / totals[lb][1] for lb in labels]
    lows = []
    highs = []
    for lb in labels:
        s, n = totals[lb]
        lo, hi = wilson_ci(s, n)
        lows.append(srs[labels.index(lb)] - lo)
        highs.append(hi - srs[labels.index(lb)])

    x = np.arange(n_methods)
    bars = ax.bar(x, [s * 100 for s in srs],
                  color=COLORS, edgecolor="white", linewidth=0.8,
                  hatch=[h for h in HATCH], alpha=0.88,
                  yerr=[[l*100 for l in lows], [h*100 for h in highs]],
                  capsize=3, error_kw=dict(elinewidth=1.0, ecolor="#333333"))

    # Annotate bar tops
    for xi, (s, lo, hi) in enumerate(zip(srs, lows, highs)):
        ax.text(xi, s*100 + hi*100 + 1.2, f"{s*100:.1f}%",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if xi == n_methods-1 else "normal")

    # Significance bracket: OT-CFM vs DDPM (best competitor)
    if annotate_sig:
        ours_sr = srs[-1] * 100
        best_alt = srs[-2] * 100
        y_bracket = ours_sr + highs[-1]*100 + 5
        ax.annotate("", xy=(n_methods-1, y_bracket), xytext=(n_methods-2, y_bracket),
                    arrowprops=dict(arrowstyle="-", lw=1.0, color="#555"))
        ax.annotate("", xy=(n_methods-2, y_bracket), xytext=(n_methods-2, best_alt + highs[-2]*100 + 1),
                    arrowprops=dict(arrowstyle="-", lw=1.0, color="#555"))
        ax.annotate("", xy=(n_methods-1, y_bracket), xytext=(n_methods-1, ours_sr + highs[-1]*100 + 1),
                    arrowprops=dict(arrowstyle="-", lw=1.0, color="#555"))
        ax.text((n_methods-1 + n_methods-2)/2, y_bracket + 0.5,
                "$p{=}0.0004$", ha="center", va="bottom", fontsize=7.5, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Success rate (%)", fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(a) Overall success rate (175 trials)", fontsize=9, pad=6)


# ── Figure 2: per-object grouped bars ─────────────────────────────────────────

def plot_perobject(ax):
    x = np.arange(n_obj)
    width = 0.14
    offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * width

    for i, (lb, color, hatch) in enumerate(zip(labels, COLORS, HATCH)):
        srs = sr_matrix[i] * 100
        ax.bar(x + offsets[i], srs, width,
               color=color, edgecolor="white", linewidth=0.5,
               hatch=hatch, alpha=0.88,
               label=lb.replace("\n", " "))

    # Highlight CrackerBox — OT coupling critical
    ax.axvspan(5 - 0.45, 5 + 0.45, alpha=0.06, color="#D55E00", zorder=0)
    ax.text(5, 104, "OT coupling\ncritical", ha="center", va="bottom",
            fontsize=6.5, color="#b34400", linespacing=1.2)
    # Per-bar numbers for CrackerBox (no-OT, DDPM, OT-CFM)
    cracker_x = 5
    cracker_vals = [(2, "0/25"), (3, "9/25"), (4, "20/25")]  # (method_idx, label)
    for mi, txt in cracker_vals:
        bx = cracker_x + offsets[mi]
        by = sr_matrix[mi, 5] * 100
        ax.text(bx, by + 1.5, txt, ha="center", va="bottom",
                fontsize=5.5, color="#333", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(OBJ_LABELS, fontsize=8)
    ax.set_ylabel("Success rate (%)", fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(b) Per-object success rate", fontsize=9, pad=6)

    legend_labels = [lb.replace("\n", " ") for lb in labels]
    handles = [mpatches.Patch(facecolor=c, hatch=h, edgecolor="white", alpha=0.88,
                               label=l)
               for c, h, l in zip(COLORS, HATCH, legend_labels)]
    ax.legend(handles=handles, fontsize=7.5, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.14),
              frameon=False, handlelength=1.6, handletextpad=0.5,
              columnspacing=1.0)


# ── Render combined figure ────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                         gridspec_kw={"width_ratios": [1, 1.75]})
fig.subplots_adjust(wspace=0.30, bottom=0.26)

plot_fiveway(axes[0])
plot_perobject(axes[1])

out_combined = os.path.join(FIG_DIR, "fig_results.pdf")
fig.savefig(out_combined, bbox_inches="tight", dpi=300)
print(f"Saved: {out_combined}")

# Also save individual panels
fig1, ax1 = plt.subplots(figsize=(3.2, 3.1))
fig1.subplots_adjust(bottom=0.18)
plot_fiveway(ax1)
out1 = os.path.join(FIG_DIR, "fig_fiveway.pdf")
fig1.savefig(out1, bbox_inches="tight", dpi=300)
print(f"Saved: {out1}")

fig2, ax2 = plt.subplots(figsize=(5.5, 3.3))
fig2.subplots_adjust(bottom=0.28)
plot_perobject(ax2)
out2 = os.path.join(FIG_DIR, "fig_perobject.pdf")
fig2.savefig(out2, bbox_inches="tight", dpi=300)
print(f"Saved: {out2}")

plt.close("all")
print("Done.")
