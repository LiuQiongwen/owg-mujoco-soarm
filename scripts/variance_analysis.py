#!/usr/bin/env python3
"""
Experiment 2 — Repeated-Run Variance Analysis
Characterises how stable per-object net-improvement estimates are across
three independent 25-seed runs, and computes the minimum N needed for
80% power to detect each object's observed mean effect.

Inputs (all existing):
  logs/batch_s3s4_v2_25seed.jsonl         — run 1: v2, no gate
  logs/batch_s3s4_v2_gate_25seed.jsonl    — run 2: v2 + H_std gate
  logs/batch_s3s4_conditional_25seed.jsonl — run 3: v2 + class-conditional

Outputs (printed):
  - Cross-run stability table
  - N-stability curve (80% CI width vs N for each object)
  - Minimum N for 80% power per object
"""

import json
import numpy as np
import random
from collections import defaultdict

random.seed(0)
np.random.seed(0)

LOGS = {
    "v2_no_gate":    "logs/batch_s3s4_v2_25seed.jsonl",
    "v2_gate":       "logs/batch_s3s4_v2_gate_25seed.jsonl",
    "conditional":   "logs/batch_s3s4_conditional_25seed.jsonl",
}
OBJECTS = ["Banana", "CrackerBox", "MustardBottle",
           "PowerDrill", "Scissors", "TomatoSoupCan"]
N_BOOT  = 10_000


def load_pairs(path):
    """Return dict: object -> list of (s3_success, s4_success) pairs, one per seed."""
    rows  = [json.loads(l) for l in open(path)]
    seeds = sorted(set(r["seed"] for r in rows))
    out   = defaultdict(list)
    for obj in OBJECTS:
        for seed in seeds:
            r3 = next((r["success"] for r in rows
                       if r["stage"]==3 and r["seed"]==seed and r["prompt"]==obj), None)
            r4 = next((r["success"] for r in rows
                       if r["stage"]==4 and r["seed"]==seed and r["prompt"]==obj), None)
            if r3 is not None and r4 is not None:
                out[obj].append((int(r3), int(r4)))
    return out


def net(pairs):
    return sum(1 for a, b in pairs if b and not a) - \
           sum(1 for a, b in pairs if a and not b)


def bootstrap_ci_width(pairs, n_sub, n_boot=N_BOOT, ci=0.80):
    """Bootstrap 80% CI width for net improvement on a sample of n_sub pairs."""
    if n_sub > len(pairs):
        n_sub = len(pairs)
    boots = sorted(
        net(random.choices(pairs, k=n_sub))
        for _ in range(n_boot)
    )
    lo = boots[int((1 - ci) / 2 * n_boot)]
    hi = boots[int((1 + ci) / 2 * n_boot)]
    return hi - lo


def power_at_n(pairs, true_effect, n_sub, n_boot=N_BOOT):
    """Fraction of bootstrap resamples that detect net > 0 (one-sided)."""
    count = 0
    for _ in range(n_boot):
        sample = random.choices(pairs, k=n_sub)
        if net(sample) > 0:
            count += 1
    return count / n_boot


# ── 1. Load all three runs ───────────────────────────────────────────────────
all_pairs = {label: load_pairs(path) for label, path in LOGS.items()}

# ── 2. Cross-run stability table ─────────────────────────────────────────────
print("=" * 80)
print("TABLE: Cross-run stability — per-object net improvement across 3 runs")
print("=" * 80)
hdr = f"{'Object':20s}  {'Run1':>6}  {'Run2':>6}  {'Run3':>6}  {'Mean':>6}  {'σ':>6}  {'Range':>8}"
print(hdr)
print("-" * 80)
mean_net_by_obj = {}
all_run_nets = {}
for obj in OBJECTS:
    nets = [net(all_pairs[lbl][obj]) for lbl in LOGS]
    mn   = np.mean(nets)
    sd   = np.std(nets, ddof=1) if len(nets) > 1 else 0.0
    rng  = max(nets) - min(nets)
    mean_net_by_obj[obj] = mn
    all_run_nets[obj]    = nets
    print(f"{obj:20s}  {nets[0]:>+6d}  {nets[1]:>+6d}  {nets[2]:>+6d}  "
          f"{mn:>+6.1f}  {sd:>6.2f}  {rng:>8d}")
print()

# ── 3. Reliability classification ───────────────────────────────────────────
print("Reliability classification (same sign across all 3 runs):")
for obj in OBJECTS:
    nets = all_run_nets[obj]
    if all(n > 0 for n in nets):   label = "RELIABLE POSITIVE"
    elif all(n < 0 for n in nets): label = "RELIABLE NEGATIVE"
    elif all(n == 0 for n in nets): label = "reliably neutral"
    else:                           label = "inconsistent / noise"
    print(f"  {obj:20s}: {label}  (nets = {nets})")
print()

# ── 4. N-stability curve (80% CI width vs N) ─────────────────────────────────
# Use run 1 pairs (v2_no_gate) as the reference distribution
print("=" * 70)
print("N-STABILITY: 80% CI width of net improvement estimate vs N seeds")
print("(reference distribution: run 1 / v2_no_gate)")
print("=" * 70)
N_VALS = [5, 10, 15, 20, 25]
ref_pairs = all_pairs["v2_no_gate"]

# Header
obj_cols = "  ".join(f"{o[:10]:>10}" for o in OBJECTS)
print(f"{'N':>4}  {obj_cols}")
print("-" * 70)
ci_widths_by_obj = defaultdict(list)
for n in N_VALS:
    row = f"{n:>4}"
    for obj in OBJECTS:
        w = bootstrap_ci_width(ref_pairs[obj], n_sub=n)
        ci_widths_by_obj[obj].append((n, w))
        row += f"  {w:>10.1f}"
    print(row)
print()
print("Values = full width of 80% CI (e.g. 8 means CI spans [-4, +4])")
print()

# ── 5. Minimum N for 80% power ───────────────────────────────────────────────
print("=" * 70)
print("POWER ANALYSIS: minimum N for 80% power to detect observed mean effect")
print("(one-sided test: P(net > 0 | true effect = mean across runs))")
print("=" * 70)
# Pool all pairs across runs for a larger reference distribution
pooled = {obj: [] for obj in OBJECTS}
for lbl in LOGS:
    for obj in OBJECTS:
        pooled[obj].extend(all_pairs[lbl][obj])

N_SEARCH = [5, 10, 15, 20, 25, 35, 50, 75, 100]
print(f"{'Object':20s}  {'Mean net':>9}  {'P@25':>6}  {'N_80pct':>8}  {'note'}")
print("-" * 70)
for obj in OBJECTS:
    mn = mean_net_by_obj[obj]
    p25 = power_at_n(pooled[obj], mn, 25)
    n80 = None
    for n in N_SEARCH:
        p = power_at_n(pooled[obj], mn, n)
        if p >= 0.80:
            n80 = n
            break
    note = ""
    if mn <= 0:
        note = "(negative effect — N_80 for detection of harm)"
    n80_str = str(n80) if n80 else f">={N_SEARCH[-1]}"
    print(f"{obj:20s}  {mn:>+9.1f}  {p25:>6.3f}  {n80_str:>8}  {note}")
print()

# ── 6. Summary ───────────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY FOR PAPER")
print("=" * 70)
print()
print("Strong findings (consistent direction across all 3 runs):")
for obj in OBJECTS:
    nets = all_run_nets[obj]
    if all(n < 0 for n in nets) or all(n > 0 for n in nets):
        p25 = power_at_n(pooled[obj], mean_net_by_obj[obj], 25)
        print(f"  {obj:20s}: mean={mean_net_by_obj[obj]:+.1f}  P(detect|N=25)={p25:.2f}")
print()
print("Inconclusive findings (sign varies across runs):")
for obj in OBJECTS:
    nets = all_run_nets[obj]
    if not (all(n < 0 for n in nets) or all(n > 0 for n in nets)):
        p25 = power_at_n(pooled[obj], mean_net_by_obj[obj], 25)
        print(f"  {obj:20s}: mean={mean_net_by_obj[obj]:+.1f}  P(detect|N=25)={p25:.2f}  nets={nets}")
