#!/usr/bin/env python3
"""
Generate F1 and F4 for paper_tro.tex sec:critic / sec:multihead (the two
remaining "nice-to-have" visual-polish items from
results/risk_gated_vla/FIGURE_CHECKLIST_sec4.5_4.6.md -- all essential
tables are already in the paper as LaTeX tables; these are the plots).

F1: geometry vs. corrected critic, dev-test and confirmatory batches
    (tab:critic_results's numbers, as a bar chart).
F4: live-executed (sec:critic) vs. offline re-scored (sec:multihead)
    consistency on the confirmatory batch -- visually reinforces the
    paper's own repeated warning that these are two different,
    complementary measurements, not to be pooled.

All numbers below are sourced directly from paper_tro.tex's own already-
verified tab:critic_results and tab:multihead_vs_geom (cross-checked
against results/risk_gated_vla/final_report.md and
phase1/multitask_outcome_critic/C3_RESULT.md) -- no new computation, just
plotting confirmed numbers.

Saves to figures/ as PDF (IEEE-compliant embedded fonts).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#2166ac"
RED = "#d6604d"
GRAY = "#888888"
GREEN = "#4dac26"

# ═══════════════════════════════════════════════════════════════════════
# F1: geometry vs. corrected critic, live-executed, both batches
# (tab:critic_results)
# ═══════════════════════════════════════════════════════════════════════
batches = ["Dev-test\n(n=90)", "Confirmatory\n(n=150)"]
geometry = [30 / 90 * 100, 54 / 150 * 100]
critic = [44 / 90 * 100, 75 / 150 * 100]
pvals = ["p=0.00258", "p=3.24e-4"]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
x = np.arange(len(batches))
width = 0.32
b1 = ax.bar(x - width / 2, geometry, width, label="Geometry", color=GRAY)
b2 = ax.bar(x + width / 2, critic, width, label="Corrected critic", color=BLUE)

for i, (g, c, p) in enumerate(zip(geometry, critic, pvals)):
    ax.text(x[i] - width / 2, g + 1.5, f"{g:.1f}%", ha="center", fontsize=7)
    ax.text(x[i] + width / 2, c + 1.5, f"{c:.1f}%", ha="center", fontsize=7)
    ax.text(x[i], max(g, c) + 8, p, ha="center", fontsize=7, style="italic")

ax.set_ylabel("Grasp success rate")
ax.set_xticks(x)
ax.set_xticklabels(batches)
ax.set_ylim(0, 68)
ax.yaxis.set_major_formatter(lambda v, _: f"{int(v)}%")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/fig_critic_results.pdf")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# F4: live-executed vs. offline re-scored, confirmatory batch only
# (tab:critic_results row 2 vs. tab:multihead_vs_geom pooled row) --
# same confirmatory scene population, two different, complementary
# measurement methods. NOT pooled/averaged -- shown side by side.
# ═══════════════════════════════════════════════════════════════════════
methods = ["Live-executed\n(sec. critic)", "Offline re-scored\n(sec. multihead)"]
geometry2 = [54 / 150 * 100, 35.3]
critic2 = [75 / 150 * 100, 46.0]
deltas = ["+14.0pp", "+10.7pp"]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
x = np.arange(len(methods))
width = 0.32
ax.bar(x - width / 2, geometry2, width, label="Geometry", color=GRAY)
ax.bar(x + width / 2, critic2, width, label="Critic", color=GREEN)

for i, (g, c, d) in enumerate(zip(geometry2, critic2, deltas)):
    ax.text(x[i] - width / 2, g + 1.5, f"{g:.1f}%", ha="center", fontsize=7)
    ax.text(x[i] + width / 2, c + 1.5, f"{c:.1f}%", ha="center", fontsize=7)
    ax.text(x[i], max(g, c) + 8, d, ha="center", fontsize=7, style="italic", color=RED)

ax.set_ylabel("Grasp success / top-1 rate")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 65)
ax.yaxis.set_major_formatter(lambda v, _: f"{int(v)}%")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/fig_offline_vs_live_consistency.pdf")
plt.close(fig)

print("Wrote figures/fig_critic_results.pdf")
print("Wrote figures/fig_offline_vs_live_consistency.pdf")
