#!/usr/bin/env python3
"""
Experiment 1 — Semantic Alignment Analysis
Correlate per-object geometric properties (from candidate log) with
Stage 4 net improvement (from 25-seed paired eval).

Inputs:
  logs/lggsn_live_candidates.jsonl   — per-candidate features
  logs/batch_s3s4_v2_25seed.jsonl    — 25-seed Stage 3 vs Stage 4 results

Outputs (printed):
  - Geometric property table (per object)
  - Spearman correlations vs Stage 4 net improvement
"""

import json
import numpy as np
from collections import defaultdict
from scipy import stats

CANDIDATE_LOG = "logs/lggsn_live_candidates.jsonl"
EVAL_LOG      = "logs/batch_s3s4_v2_25seed.jsonl"
OBJECTS       = ["Banana", "CrackerBox", "MustardBottle",
                 "PowerDrill", "Scissors", "TomatoSoupCan"]

# ── 1. Load candidate log ────────────────────────────────────────────────────
cands = [json.loads(l) for l in open(CANDIDATE_LOG)]

# Group into episodes: (query, scene_id)
episodes = defaultdict(list)
for c in cands:
    episodes[(c["query"], c["scene_id"])].append(c)

# ── 2. Per-object geometric properties ──────────────────────────────────────
geom = {}
for obj in OBJECTS:
    obj_eps = [v for (q, _), v in episodes.items() if q == obj]
    all_H   = [c["H"]   for ep in obj_eps for c in ep]
    all_yaw = [c["yaw"] for ep in obj_eps for c in ep]
    all_w   = [c["width"] for ep in obj_eps for c in ep]

    # Within-episode standard deviations (mean over episodes with ≥2 candidates)
    h_stds, yaw_stds, z_stds = [], [], []
    for ep in obj_eps:
        if len(ep) < 2:
            continue
        h_stds.append(np.std([c["H"]   for c in ep]))
        yaw_stds.append(np.std([c["yaw"] for c in ep]))
        z_stds.append(np.std([c["z"]   for c in ep]))

    geom[obj] = {
        "n_episodes":       len(obj_eps),
        "n_candidates":     len(all_H),
        "H_mean":           np.mean(all_H),
        "H_std_within":     np.mean(h_stds)   if h_stds   else 0.0,
        "yaw_std_within":   np.mean(yaw_stds) if yaw_stds else 0.0,
        "z_std_within":     np.mean(z_stds)   if z_stds   else 0.0,
        "flat_frac":        np.mean([1 if h < 0.001 else 0 for h in all_H]),
        "width_mean":       np.mean(all_w),
    }

# ── 3. Per-object Stage 4 net improvement from 25-seed log ──────────────────
rows = [json.loads(l) for l in open(EVAL_LOG)]
seeds   = sorted(set(r["seed"]   for r in rows))
prompts = sorted(set(r["prompt"] for r in rows))

net_by_obj = {}
for obj in OBJECTS:
    imp = reg = 0
    for seed in seeds:
        r3 = next((r["success"] for r in rows
                   if r["stage"]==3 and r["seed"]==seed and r["prompt"]==obj), None)
        r4 = next((r["success"] for r in rows
                   if r["stage"]==4 and r["seed"]==seed and r["prompt"]==obj), None)
        if r3 is None or r4 is None:
            continue
        if r4 and not r3: imp += 1
        elif r3 and not r4: reg += 1
    net_by_obj[obj] = imp - reg

# ── 4. Print geometric property table ───────────────────────────────────────
print("=" * 90)
print("TABLE: Geometric properties vs Stage 4 net improvement (25 seeds)")
print("=" * 90)
hdr = f"{'Object':20s}  {'H_mean':>8}  {'σ_H_within':>10}  {'σ_yaw_within':>12}  "
hdr += f"{'flat_frac':>9}  {'width_mean':>10}  {'S4_net':>6}"
print(hdr)
print("-" * 90)
for obj in OBJECTS:
    g = geom[obj]
    print(f"{obj:20s}  {g['H_mean']:>8.4f}  {g['H_std_within']:>10.4f}  "
          f"{g['yaw_std_within']:>12.4f}  {g['flat_frac']:>9.3f}  "
          f"{g['width_mean']:>10.4f}  {net_by_obj[obj]:>+6d}")
print()

# ── 5. Spearman correlations ─────────────────────────────────────────────────
net_vals = np.array([net_by_obj[o] for o in OBJECTS])
props = {
    "H_std_within":   np.array([geom[o]["H_std_within"]   for o in OBJECTS]),
    "flat_frac":      np.array([geom[o]["flat_frac"]       for o in OBJECTS]),
    "yaw_std_within": np.array([geom[o]["yaw_std_within"]  for o in OBJECTS]),
    "H_mean":         np.array([geom[o]["H_mean"]          for o in OBJECTS]),
    "width_mean":     np.array([geom[o]["width_mean"]      for o in OBJECTS]),
}

print("=" * 60)
print("SPEARMAN CORRELATIONS vs Stage 4 net improvement")
print("=" * 60)
print(f"{'Property':20s}  {'rho':>8}  {'p-value':>10}  {'interpretation':s}")
print("-" * 60)
for name, vals in props.items():
    rho, pval = stats.spearmanr(vals, net_vals)
    interp = ""
    if pval < 0.05:  interp = "* significant"
    elif pval < 0.10: interp = "+ marginal"
    else:             interp = "  ns"
    print(f"{name:20s}  {rho:>+8.3f}  {pval:>10.4f}  {interp}")
print()
print(f"n = {len(OBJECTS)} objects (Spearman exact permutation-based p-values)")
print()

# ── 6. Ranked object list by H_std_within ───────────────────────────────────
print("Objects ranked by σ_H_within (ascending = worst for reranking):")
ranked = sorted(OBJECTS, key=lambda o: geom[o]["H_std_within"])
for obj in ranked:
    print(f"  {obj:20s}  σ_H={geom[obj]['H_std_within']:.4f}  "
          f"flat_frac={geom[obj]['flat_frac']:.3f}  net={net_by_obj[obj]:+d}")
