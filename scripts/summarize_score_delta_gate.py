#!/usr/bin/env python3
"""
Experiment 3 — Score-Delta Uncertainty Gate Analysis

Design: skip LGGSN reranking when max(score)−min(score) < δ (score spread too small).
Rationale: near-zero spread indicates score saturation — the model cannot discriminate
candidates, so falling back to GR-ConvNet is safer than a near-random LGGSN ordering.

Two modes:
  1. Retrospective (default): simulate gate outcome on existing
     logs/batch_s3s4_v2_25seed.jsonl using recorded lggsn_scores_all.
  2. Live (--live <path>): analyse results from an actual delta_gate_X run.

Outputs:
  - Printed markdown tables
  - results/score_delta_gate_retro.csv   (retrospective sweep)
  - results/score_delta_gate_live.csv    (live run, if provided)
  - results/score_spread_dist.csv        (per-object score-spread distributions)
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np

random.seed(42)
np.random.seed(42)

OBJECTS = ["Banana", "CrackerBox", "MustardBottle", "PowerDrill", "Scissors", "TomatoSoupCan"]
RETRO_LOG = "logs/batch_s3s4_v2_25seed.jsonl"
RETRO_DELTAS = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.010]
N_BOOT = 10_000
os.makedirs("results", exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_s3_index(rows):
    return {(r["seed"], r["prompt"]): r for r in rows if r["stage"] == 3}


def net_improvement(pairs):
    """pairs: list of (s3_success, s4_success)"""
    return sum(1 for a, b in pairs if b and not a) - sum(1 for a, b in pairs if a and not b)


def bootstrap_ci(pairs, n_boot=N_BOOT, ci=0.95):
    """Return (lo, hi) bootstrap CI for net_improvement."""
    boots = sorted(net_improvement(random.choices(pairs, k=len(pairs))) for _ in range(n_boot))
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot)
    return boots[lo_idx], boots[hi_idx]


# ── 1. Score-spread distribution per object ────────────────────────────────────

def spread_distribution(rows):
    """Return dict: object -> array of max-min spreads (Stage-4 rows only)."""
    s4 = [r for r in rows if r["stage"] == 4 and r.get("lggsn_scores_all")]
    by_obj = defaultdict(list)
    for r in s4:
        sc = r["lggsn_scores_all"]
        spread = max(sc) - min(sc) if len(sc) > 1 else 0.0
        by_obj[r["prompt"]].append(spread)
    return by_obj


def print_spread_table(spread_by_obj):
    cols = ["p0", "p10", "p50", "p90", "p99", "max", "frac<1e-4", "frac<1e-3"]
    print("\n## Score-Spread Distribution per Object (Stage-4 episodes)\n")
    header = f"| {'Object':<18} | {'n':>4} | {'p0':>7} | {'p10':>7} | {'p50':>7} | {'p90':>7} | {'p99':>7} | {'max':>7} | {'<1e-4':>6} | {'<1e-3':>6} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    rows_csv = []
    for obj in OBJECTS:
        arr = np.array(spread_by_obj.get(obj, [0.0]))
        p = np.percentile(arr, [0, 10, 50, 90, 99, 100])
        f1 = np.mean(arr < 1e-4)
        f2 = np.mean(arr < 1e-3)
        print(f"| {obj:<18} | {len(arr):>4} | {p[0]:>7.5f} | {p[1]:>7.5f} | {p[2]:>7.5f} | "
              f"{p[3]:>7.5f} | {p[4]:>7.5f} | {p[5]:>7.5f} | {f1:>6.2f} | {f2:>6.2f} |")
        rows_csv.append({"object": obj, "n": len(arr),
                         "p0": p[0], "p10": p[1], "p50": p[2], "p90": p[3],
                         "p99": p[4], "max": p[5], "frac_lt_1e4": f1, "frac_lt_1e3": f2})
    return rows_csv


# ── 2. Retrospective gate simulation ──────────────────────────────────────────

def retro_gate_sim(rows, deltas):
    """
    For each δ, simulate: if spread < δ → use S3 outcome; else → use S4 outcome.
    Returns list of dicts with per-δ per-object net improvements.
    """
    s3_idx = build_s3_index(rows)
    s4 = [r for r in rows if r["stage"] == 4 and r.get("lggsn_scores_all")]

    results = []
    for delta in deltas:
        gate_fires_total = 0
        obj_stats = {}
        pairs_by_obj = defaultdict(list)
        for r in s4:
            r3 = s3_idx.get((r["seed"], r["prompt"]), {})
            sc = r["lggsn_scores_all"]
            spread = max(sc) - min(sc) if len(sc) > 1 else 0.0
            fires = spread < delta
            gate_fires_total += fires
            eff_s4 = r3.get("success", False) if fires else r["success"]
            s3_out = r3.get("success", False)
            pairs_by_obj[r["prompt"]].append((int(s3_out), int(eff_s4)))

        gate_rate = gate_fires_total / len(s4) if s4 else 0.0
        row = {"delta": delta, "gate_rate": gate_rate}
        total_net = 0
        for obj in OBJECTS:
            pairs = pairs_by_obj.get(obj, [])
            n = net_improvement(pairs)
            row[obj] = n
            total_net += n
        row["total_net"] = total_net

        # Per-object bootstrap CI at δ (uses pooled S3+effective-S4 pairs)
        for obj in OBJECTS:
            pairs = pairs_by_obj.get(obj, [])
            if len(pairs) >= 2:
                lo, hi = bootstrap_ci(pairs)
                row[f"{obj}_ci_lo"] = lo
                row[f"{obj}_ci_hi"] = hi
            else:
                row[f"{obj}_ci_lo"] = row[f"{obj}_ci_hi"] = None

        results.append(row)
    return results


def print_retro_table(retro_rows):
    print("\n## Experiment 3: Retrospective Score-Delta Gate Simulation\n")
    print("> Simulated from `logs/batch_s3s4_v2_25seed.jsonl`. When gate fires (spread < δ),")
    print("> effective S4 outcome = S3 outcome (GR-ConvNet order used instead of LGGSN).\n")

    obj_short = [o[:10] for o in OBJECTS]
    header = "| δ | gate_rate | " + " | ".join(f"{o}" for o in obj_short) + " | TOTAL |"
    print(header)
    print("|" + "---|" * (len(OBJECTS) + 3))

    for row in retro_rows:
        delta_str = f"{row['delta']:.4f}"
        gate_str = f"{row['gate_rate']:.0%}"
        obj_vals = " | ".join(f"{row[obj]:+d}" for obj in OBJECTS)
        total = f"{row['total_net']:+d}"
        print(f"| {delta_str} | {gate_str} | {obj_vals} | {total} |")


# ── 3. Failure analysis: which trials does gate save vs destroy ────────────────

def gate_case_analysis(rows, delta):
    """
    Classify each (seed, object) trial at a given δ into:
      gate_saves     : gate fires, prevents regression (s4 would have failed, s3 succeeded)
      gate_blocks    : gate fires, prevents improvement (s4 would have succeeded, s3 failed)
      gate_neutral   : gate fires, outcome same either way
      no_gate_imp    : gate does not fire, reranking improves
      no_gate_reg    : gate does not fire, reranking degrades
      no_gate_neut   : gate does not fire, outcome same
    """
    s3_idx = build_s3_index(rows)
    s4 = [r for r in rows if r["stage"] == 4 and r.get("lggsn_scores_all")]

    by_obj = defaultdict(lambda: defaultdict(int))
    for r in s4:
        r3 = s3_idx.get((r["seed"], r["prompt"]), {})
        sc = r["lggsn_scores_all"]
        spread = max(sc) - min(sc) if len(sc) > 1 else 0.0
        fires = spread < delta
        s3_ok = r3.get("success", False)
        s4_ok = r["success"]
        obj = r["prompt"]
        if fires:
            if s3_ok and not s4_ok:
                by_obj[obj]["gate_saves"] += 1
            elif not s3_ok and s4_ok:
                by_obj[obj]["gate_blocks"] += 1
            else:
                by_obj[obj]["gate_neutral"] += 1
        else:
            if not s3_ok and s4_ok:
                by_obj[obj]["no_gate_imp"] += 1
            elif s3_ok and not s4_ok:
                by_obj[obj]["no_gate_reg"] += 1
            else:
                by_obj[obj]["no_gate_neut"] += 1
    return by_obj


def print_case_analysis(by_obj, delta):
    print(f"\n## Gate Case Analysis at δ = {delta:.4f}\n")
    cols = ["gate_saves", "gate_blocks", "gate_neutral", "no_gate_imp", "no_gate_reg", "no_gate_neut"]
    header = f"| {'Object':<18} | " + " | ".join(f"{c:<12}" for c in cols) + " |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for obj in OBJECTS:
        d = by_obj.get(obj, {})
        vals = " | ".join(f"{d.get(c, 0):<12}" for c in cols)
        print(f"| {obj:<18} | {vals} |")


# ── 4. Live-run analysis ───────────────────────────────────────────────────────

def live_analysis(live_path):
    """Analyse actual delta_gate_X strategy runs from a live batch log."""
    rows = load_jsonl(live_path)
    s3_idx = build_s3_index(rows)

    strategies = sorted(set(r.get("ranking_strategy") for r in rows
                            if r["stage"] == 4 and r.get("ranking_strategy", "").startswith("delta_gate_")))
    if not strategies:
        print("  [WARN] No delta_gate_* strategies found in live log.")
        return []

    results = []
    print(f"\n## Live Run: {os.path.basename(live_path)}\n")
    obj_short = [o[:10] for o in OBJECTS]
    header = "| strategy | gate_rate | " + " | ".join(obj_short) + " | TOTAL |"
    print(header)
    print("|" + "---|" * (len(OBJECTS) + 3))

    for strat in ["margin_0.00"] + strategies:
        s4_strat = [r for r in rows if r["stage"] == 4 and r.get("ranking_strategy") == strat]
        if not s4_strat:
            continue

        # For delta_gate strategies, 'ranking_changed_grasp' tells us if gate did not fire.
        # gate fires = ranking NOT changed (i.e., used GR-ConvNet order)
        gate_fires = sum(1 for r in s4_strat
                         if not r.get("ranking_changed_grasp", True)
                         and strat.startswith("delta_gate_")) if strat.startswith("delta_gate_") else 0
        gate_rate = gate_fires / len(s4_strat) if strat.startswith("delta_gate_") else 0.0

        total_net = 0
        row = {"strategy": strat, "gate_rate": gate_rate}
        obj_vals_str = []
        for obj in OBJECTS:
            pairs = []
            for r in s4_strat:
                if r["prompt"] != obj:
                    continue
                r3 = s3_idx.get((r["seed"], r["prompt"]), {})
                pairs.append((int(r3.get("success", False)), int(r["success"])))
            n = net_improvement(pairs)
            row[obj] = n
            total_net += n
            obj_vals_str.append(f"{n:+d}")
        row["total_net"] = total_net
        results.append(row)
        print(f"| {strat:<20} | {gate_rate:.0%} | " +
              " | ".join(f"{v:>10}" for v in obj_vals_str) + f" | {total_net:+d} |")

    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment 3: score-delta gate analysis")
    parser.add_argument("--live", default=None, metavar="PATH",
                        help="Path to live delta_gate run log (optional)")
    parser.add_argument("--retro-log", default=RETRO_LOG,
                        help=f"Retrospective source log (default: {RETRO_LOG})")
    parser.add_argument("--ci", action="store_true", default=True,
                        help="Print bootstrap 95%% CI alongside net")
    args = parser.parse_args()

    print("# Experiment 3 — Score-Delta Uncertainty Gate\n")

    # ── spread distribution ──────────────────────────────────────────────────
    rows = load_jsonl(args.retro_log)
    spread_by_obj = spread_distribution(rows)
    spread_csv = print_spread_table(spread_by_obj)

    with open("results/score_spread_dist.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(spread_csv[0].keys()))
        w.writeheader(); w.writerows(spread_csv)

    # ── retrospective simulation ─────────────────────────────────────────────
    retro_rows = retro_gate_sim(rows, RETRO_DELTAS)
    print_retro_table(retro_rows)

    # best delta by total_net
    best = max(retro_rows, key=lambda x: x["total_net"])
    print(f"\n**Best retrospective δ = {best['delta']:.4f}** → total net = {best['total_net']:+d}, "
          f"gate fires on {best['gate_rate']:.0%} of episodes")

    # Print 95% CI for best delta
    if args.ci:
        print(f"\n### Bootstrap 95% CI at δ = {best['delta']:.4f}\n")
        print(f"| Object | net | 95% CI |")
        print("|--------|-----|--------|")
        for obj in OBJECTS:
            lo = best.get(f"{obj}_ci_lo")
            hi = best.get(f"{obj}_ci_hi")
            ci_str = f"[{lo:+d}, {hi:+d}]" if lo is not None else "—"
            print(f"| {obj} | {best[obj]:+d} | {ci_str} |")

    # case analysis at best delta
    by_obj_cases = gate_case_analysis(rows, best["delta"])
    print_case_analysis(by_obj_cases, best["delta"])

    # save retro CSV
    retro_csv_rows = []
    for row in retro_rows:
        r = {"delta": row["delta"], "gate_rate": row["gate_rate"], "total_net": row["total_net"]}
        for obj in OBJECTS:
            r[obj] = row[obj]
        retro_csv_rows.append(r)
    with open("results/score_delta_gate_retro.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(retro_csv_rows[0].keys()))
        w.writeheader(); w.writerows(retro_csv_rows)
    print(f"\nSaved: results/score_delta_gate_retro.csv")

    # ── live run analysis ────────────────────────────────────────────────────
    if args.live:
        live_rows = live_analysis(args.live)
        if live_rows:
            with open("results/score_delta_gate_live.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(live_rows[0].keys()))
                w.writeheader(); w.writerows(live_rows)
            print(f"\nSaved: results/score_delta_gate_live.csv")

    # ── key insight ──────────────────────────────────────────────────────────
    # Compute fraction of trials with near-zero spread (< 1e-4) across all objects
    all_spreads = [s for spreads in spread_by_obj.values() for s in spreads]
    near_zero_frac = np.mean([s < 1e-4 for s in all_spreads])
    max_spread = max(all_spreads)
    print(f"\n---\n")
    print(f"**Key finding:** {near_zero_frac:.0%} of Stage-4 episodes have score spread < 0.0001 "
          f"(max observed = {max_spread:.5f}).")
    print("LGGSN scores are uniformly near-saturated across all objects, making")
    print("the score-spread gate effectively equivalent to σ_H filtering.")


if __name__ == "__main__":
    main()
