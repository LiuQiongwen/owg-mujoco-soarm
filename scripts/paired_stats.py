#!/usr/bin/env python3
"""
Paired Statistical Analysis — OWG Stage-3 vs Stage-4 experiments

Computes per-object and per-condition:
  1. Paired bootstrap 95% CI for net improvement (N=10_000 resamples)
  2. McNemar's test (exact binomial) for significance of paired binary outcomes
  3. Effect size: Cohen's h for paired proportions
  4. Minimum N for 80% power per object (pooled across conditions)

Inputs: all existing paired-eval logs in logs/
Outputs:
  - Printed markdown tables
  - results/paired_stats.csv         (per condition × object)
  - results/power_analysis.csv       (per object)
  stats_summary.md                   (top-level markdown report, written to repo root)
"""

import csv
import json
import math
import os
import random
from collections import defaultdict
from scipy import stats as scipy_stats
import numpy as np

random.seed(42)
np.random.seed(42)

OBJECTS = ["Banana", "CrackerBox", "MustardBottle", "PowerDrill", "Scissors", "TomatoSoupCan"]
N_BOOT = 10_000
os.makedirs("results", exist_ok=True)

# All conditions with available logs
CONDITIONS = {
    "v2_baseline":  "logs/batch_s3s4_v2_25seed.jsonl",
    "v2_gate":      "logs/batch_s3s4_v2_gate_25seed.jsonl",
    "conditional":  "logs/batch_s3s4_conditional_25seed.jsonl",
    "ablation_D":   "logs/batch_s3s4_ablation_D.jsonl",
    "ablation_B":   "logs/batch_s3s4_ablation_B.jsonl",
    "ablation_C":   "logs/batch_s3s4_ablation_C.jsonl",
    "gc_phase2":    "logs/batch_s3s4_gc_phase2.jsonl",
    "scissors_gate":"logs/batch_s3s4_scissors_gate.jsonl",
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_pairs(path, objects=None):
    """
    Load paired (s3, s4) outcomes per (seed, object).
    Returns dict: object -> list of (s3_success: int, s4_success: int)
    """
    if not os.path.exists(path):
        return {}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    seeds = sorted(set(r["seed"] for r in rows if "seed" in r))
    objects = objects or OBJECTS
    out = defaultdict(list)
    for obj in objects:
        for seed in seeds:
            r3 = next((r for r in rows if r["stage"] == 3 and r.get("seed") == seed
                       and r["prompt"] == obj), None)
            r4 = next((r for r in rows if r["stage"] == 4 and r.get("seed") == seed
                       and r["prompt"] == obj), None)
            if r3 is not None and r4 is not None:
                out[obj].append((int(r3["success"]), int(r4["success"])))
    return dict(out)


# ── statistics ────────────────────────────────────────────────────────────────

def net_improvement(pairs):
    return sum(1 for a, b in pairs if b > a) - sum(1 for a, b in pairs if a > b)


def bootstrap_ci_net(pairs, n_boot=N_BOOT, ci=0.95):
    """Bootstrap percentile CI for net improvement."""
    if len(pairs) < 2:
        return (None, None)
    boots = sorted(net_improvement(random.choices(pairs, k=len(pairs))) for _ in range(n_boot))
    lo = boots[int((1 - ci) / 2 * n_boot)]
    hi = boots[int((1 + ci) / 2 * n_boot)]
    return lo, hi


def mcnemar_test(pairs):
    """
    McNemar's exact test (binomial) for paired binary outcomes.
    H0: P(s3=0,s4=1) == P(s3=1,s4=0)
    Returns (n01, n10, p_value, statistic)
    n01 = improvements, n10 = regressions
    Uses exact binomial p-value (two-tailed) when n01+n10 < 25, otherwise chi-sq.
    """
    n01 = sum(1 for a, b in pairs if a == 0 and b == 1)
    n10 = sum(1 for a, b in pairs if a == 1 and b == 0)
    n_disc = n01 + n10
    if n_disc == 0:
        return n01, n10, 1.0, 0.0
    if n_disc < 25:
        # Exact binomial: P(X >= n01 | n_disc, 0.5) * 2 (two-tailed)
        p = 2 * scipy_stats.binom.sf(max(n01, n10) - 1, n_disc, 0.5)
        p = min(p, 1.0)
        stat = float(n01 - n10)
    else:
        # McNemar chi-squared with continuity correction
        stat = (abs(n01 - n10) - 1.0) ** 2 / n_disc
        p = scipy_stats.chi2.sf(stat, df=1)
    return n01, n10, p, stat


def cohen_h(pairs):
    """
    Effect size: Cohen's h = 2*arcsin(sqrt(p4)) - 2*arcsin(sqrt(p3))
    where p3 = S3 success rate, p4 = S4 success rate.
    """
    n = len(pairs)
    if n == 0:
        return None
    p3 = sum(a for a, b in pairs) / n
    p4 = sum(b for a, b in pairs) / n
    h = 2 * math.asin(math.sqrt(p4)) - 2 * math.asin(math.sqrt(p3))
    return round(h, 4)


def power_at_n(ref_pairs, n_sub, n_boot=N_BOOT):
    """Fraction of bootstrap resamples detecting net > 0 at sample size n_sub."""
    if n_sub > len(ref_pairs):
        return power_at_n(ref_pairs, len(ref_pairs), n_boot)
    count = sum(1 for _ in range(n_boot) if net_improvement(random.choices(ref_pairs, k=n_sub)) > 0)
    return count / n_boot


# ── per-condition analysis ─────────────────────────────────────────────────────

def analyse_condition(name, pairs_by_obj):
    rows = []
    for obj in OBJECTS:
        pairs = pairs_by_obj.get(obj, [])
        if not pairs:
            continue
        n = len(pairs)
        s3_rate = sum(a for a, b in pairs) / n
        s4_rate = sum(b for a, b in pairs) / n
        n_imp = sum(1 for a, b in pairs if b > a)
        n_reg = sum(1 for a, b in pairs if a > b)
        net = n_imp - n_reg
        ci_lo, ci_hi = bootstrap_ci_net(pairs)
        n01, n10, p_val, stat = mcnemar_test(pairs)
        h = cohen_h(pairs)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else
              ("+" if p_val < 0.10 else "ns")))
        rows.append({
            "condition": name, "object": obj, "n": n,
            "s3_rate": round(s3_rate, 3), "s4_rate": round(s4_rate, 3),
            "n_imp": n_imp, "n_reg": n_reg, "net": net,
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
            "mcnemar_p": round(p_val, 4), "mcnemar_sig": sig,
            "cohen_h": h,
        })
    return rows


# ── power analysis ────────────────────────────────────────────────────────────

def power_analysis(v2_pairs, pooled_pairs):
    N_SEARCH = [5, 10, 15, 20, 25, 35, 50, 75, 100, 150]
    rows = []
    for obj in OBJECTS:
        ref = pooled_pairs.get(obj, [])
        v2 = v2_pairs.get(obj, [])
        if not ref:
            continue
        mean_net = net_improvement(v2) if v2 else 0
        p25 = power_at_n(ref, 25)
        n80 = None
        for n in N_SEARCH:
            if power_at_n(ref, n) >= 0.80:
                n80 = n
                break
        rows.append({
            "object": obj,
            "mean_net_v2_25seed": mean_net,
            "power_at_N25": round(p25, 3),
            "N_for_80pct_power": n80 if n80 else f">={N_SEARCH[-1]}",
        })
    return rows


# ── markdown writers ──────────────────────────────────────────────────────────

def md_condition_table(all_rows, condition):
    """Print markdown table for a single condition."""
    rows = [r for r in all_rows if r["condition"] == condition]
    if not rows:
        return
    print(f"\n### {condition}\n")
    print("| Object | N | S3% | S4% | imp | reg | net | 95% CI | McNemar p | sig | Cohen h |")
    print("|--------|---|-----|-----|-----|-----|-----|--------|-----------|-----|---------|")
    for r in rows:
        ci = f"[{r['ci95_lo']:+d},{r['ci95_hi']:+d}]" if r["ci95_lo"] is not None else "—"
        print(f"| {r['object']:<18} | {r['n']} | {r['s3_rate']:.1%} | {r['s4_rate']:.1%} | "
              f"{r['n_imp']} | {r['n_reg']} | {r['net']:+d} | {ci} | "
              f"{r['mcnemar_p']:.4f} | {r['mcnemar_sig']} | {r['cohen_h'] or '—'} |")


def md_summary_table(all_rows):
    """Compact summary across all conditions, net improvement only."""
    # pivot: condition × object
    net_by = {}
    for r in all_rows:
        net_by.setdefault(r["condition"], {})[r["object"]] = r["net"]

    conds = list(CONDITIONS.keys())
    obj_cols = " | ".join(f"{o[:8]}" for o in OBJECTS)
    header = f"| Condition | {obj_cols} | Total |"
    print(header)
    print("|" + "---|" * (len(OBJECTS) + 2))
    for cond in conds:
        if cond not in net_by:
            continue
        nets = [net_by[cond].get(obj, "—") for obj in OBJECTS]
        total = sum(v for v in nets if isinstance(v, int))
        vals = " | ".join(f"{v:+d}" if isinstance(v, int) else "—" for v in nets)
        print(f"| {cond:<14} | {vals} | {total:+d} |")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    all_rows = []
    pairs_all = {}  # condition -> {object -> pairs}

    for name, path in CONDITIONS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        # gc_phase2 only has CrackerBox + Scissors
        obj_subset = ["CrackerBox", "Scissors"] if name == "gc_phase2" else None
        obj_subset = ["Scissors"] if name == "scissors_gate" else obj_subset
        pboj = load_pairs(path, obj_subset)
        pairs_all[name] = pboj
        rows = analyse_condition(name, pboj)
        all_rows.extend(rows)

    # ── per-condition tables ────────────────────────────────────────────────
    print("# Paired Statistical Analysis — OWG Stage-3 vs Stage-4\n")
    print("## 1. Per-Condition Tables\n")
    for name in CONDITIONS:
        md_condition_table(all_rows, name)

    # ── summary net table ───────────────────────────────────────────────────
    print("\n## 2. Net Improvement Summary (all conditions)\n")
    md_summary_table(all_rows)

    # ── pooled power analysis ───────────────────────────────────────────────
    print("\n## 3. Power Analysis\n")
    pooled = defaultdict(list)
    for name in ["v2_baseline", "v2_gate", "conditional"]:
        for obj, pairs in pairs_all.get(name, {}).items():
            pooled[obj].extend(pairs)

    pwr_rows = power_analysis(pairs_all.get("v2_baseline", {}), dict(pooled))
    print("| Object | v2 net (N=25) | Power@N=25 | N for 80% power |")
    print("|--------|--------------|------------|----------------|")
    for r in pwr_rows:
        print(f"| {r['object']:<18} | {r['mean_net_v2_25seed']:>+13} | "
              f"{r['power_at_N25']:>10.3f} | {r['N_for_80pct_power']:>16} |")

    # ── significance summary ────────────────────────────────────────────────
    print("\n## 4. Objects with statistically significant effects\n")
    print("| Condition | Object | net | McNemar p | sig |")
    print("|-----------|--------|-----|-----------|-----|")
    for r in all_rows:
        if r["mcnemar_sig"] not in ("ns",):
            print(f"| {r['condition']} | {r['object']} | {r['net']:+d} | {r['mcnemar_p']:.4f} | {r['mcnemar_sig']} |")

    # ── save CSVs ───────────────────────────────────────────────────────────
    if all_rows:
        with open("results/paired_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        print(f"\nSaved: results/paired_stats.csv")

    if pwr_rows:
        with open("results/power_analysis.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pwr_rows[0].keys()))
            w.writeheader(); w.writerows(pwr_rows)
        print(f"Saved: results/power_analysis.csv")

    # ── write stats_summary.md ──────────────────────────────────────────────
    write_stats_summary(all_rows, pwr_rows, pairs_all)
    print(f"Saved: stats_summary.md")


def write_stats_summary(all_rows, pwr_rows, pairs_all):
    from io import StringIO
    import sys

    md = []
    md.append("# OWG Stage-4 LGGSN — Statistical Summary\n")
    md.append("> Auto-generated by `scripts/paired_stats.py`. Paired bootstrap 95% CI, N=10,000 resamples.\n")

    # v2 baseline full table
    md.append("\n## Primary Result: v2 Model, 25 Seeds × 6 Objects\n")
    md.append("| Object | N | S3% | S4% | Δ% | net | 95% CI | McNemar p | sig | Cohen h |\n")
    md.append("|--------|---|-----|-----|----|-----|--------|-----------|-----|---------|")
    v2_rows = [r for r in all_rows if r["condition"] == "v2_baseline"]
    total_n_imp = total_n_reg = 0
    total_s3 = total_s4 = 0
    for r in v2_rows:
        ci = f"[{r['ci95_lo']:+d}, {r['ci95_hi']:+d}]" if r["ci95_lo"] is not None else "—"
        delta_pct = (r["s4_rate"] - r["s3_rate"]) * 100
        md.append(f"\n| {r['object']:<18} | {r['n']} | {r['s3_rate']:.1%} | {r['s4_rate']:.1%} | "
                  f"{delta_pct:+.1f}pp | {r['net']:+d} | {ci} | {r['mcnemar_p']:.4f} | "
                  f"{r['mcnemar_sig']} | {r['cohen_h'] or '—'} |")
        total_n_imp += r["n_imp"]; total_n_reg += r["n_reg"]
        total_s3 += r["s3_rate"] * r["n"]; total_s4 += r["s4_rate"] * r["n"]
    total_n = sum(r["n"] for r in v2_rows) or 1
    md.append(f"\n| **TOTAL** | {total_n} | {total_s3/total_n:.1%} | {total_s4/total_n:.1%} | "
              f"{(total_s4-total_s3)/total_n*100:+.1f}pp | "
              f"{total_n_imp-total_n_reg:+d} | — | — | — | — |")

    md.append("\n\n**Key:** sig codes: `***` p<0.001, `**` p<0.01, `*` p<0.05, `+` p<0.10, `ns` not significant.\n")

    # ablation table
    md.append("\n## Ablation Study: Feature Contributions (10 Seeds)\n")
    md.append("| Condition | Features | Val acc | S3 | S4 | net | 95% CI | McNemar p |\n")
    md.append("|-----------|----------|---------|----|----|-----|--------|-----------|")
    ablation_info = {
        "ablation_B": ("dist_to_centroid only", "0.558"),
        "ablation_C": ("z_rel only",            "0.576"),
        "ablation_D": ("12-dim base",            "0.579"),
        "v2_baseline": ("dist + z_rel (v2)",     "0.664"),
    }
    for cname, (feats, vacc) in ablation_info.items():
        crows = [r for r in all_rows if r["condition"] == cname]
        if not crows: continue
        s3_total = sum(int(r["s3_rate"]*r["n"]) for r in crows)
        s4_total = sum(int(r["s4_rate"]*r["n"]) for r in crows)
        net_total = sum(r["net"] for r in crows)
        # pool pairs for CI
        pairs = []
        for r in crows:
            pairs.extend(pairs_all.get(cname, {}).get(r["object"], []))
        ci_lo, ci_hi = bootstrap_ci_net(pairs) if pairs else (None, None)
        n01, n10, pval, _ = mcnemar_test(pairs) if pairs else (0, 0, 1.0, 0)
        ci_str = f"[{ci_lo:+d}, {ci_hi:+d}]" if ci_lo is not None else "—"
        md.append(f"\n| {cname:<14} | {feats:<22} | {vacc} | {s3_total} | {s4_total} | "
                  f"{net_total:+d} | {ci_str} | {pval:.4f} |")

    # power table
    md.append("\n\n## Power Analysis\n")
    md.append("| Object | Mean net (25-seed) | Power @ N=25 | N for 80% power |\n")
    md.append("|--------|-------------------|--------------|----------------|")
    for r in pwr_rows:
        md.append(f"\n| {r['object']:<18} | {r['mean_net_v2_25seed']:>+18} | "
                  f"{r['power_at_N25']:>12.3f} | {r['N_for_80pct_power']:>16} |")

    # per-object grouped failure analysis
    md.append("\n\n## Per-Object Failure Analysis (from v2 baseline)\n")
    v2_pairs = pairs_all.get("v2_baseline", {})
    md.append("\n| Object | n_imp | n_reg | net | changed_imp | changed_reg | "
              "non_det | interpretation |\n")
    md.append("|--------|-------|-------|-----|-------------|-------------|---------|----------------|")

    # load raw rows for classification
    raw = [json.loads(l) for l in open(CONDITIONS["v2_baseline"]) if l.strip()]
    s3_idx = {(r["seed"], r["prompt"]): r for r in raw if r["stage"] == 3}
    for obj in OBJECTS:
        s4_obj = [r for r in raw if r["stage"] == 4 and r["prompt"] == obj]
        n_imp = n_reg = changed_imp = changed_reg = nondet = 0
        for r in s4_obj:
            r3 = s3_idx.get((r["seed"], r["prompt"]), {})
            s3, s4 = r3.get("success", False), r["success"]
            ch = r.get("ranking_changed_grasp", False)
            if s4 and not s3: n_imp += 1; changed_imp += ch
            if s3 and not s4: n_reg += 1; changed_reg += ch
            if s3 != s4 and not ch: nondet += 1
        net = n_imp - n_reg

        # interpretation
        if net > 2:       interp = "reliable gain"
        elif net >= 0:    interp = "marginal / neutral"
        elif net == -1:   interp = "slight regression"
        else:             interp = "clear regression"
        md.append(f"\n| {obj:<18} | {n_imp:>5} | {n_reg:>5} | {net:>+3} | "
                  f"{changed_imp:>11} | {changed_reg:>11} | {nondet:>7} | {interp} |")

    with open("stats_summary.md", "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
