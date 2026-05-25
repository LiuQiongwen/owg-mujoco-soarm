#!/usr/bin/env python3
"""Statistical analysis of diverse benchmark results.

Computes per-difficulty and per-object statistics including:
  - Wilson 95% CI for success rates
  - Fisher's exact test for method comparisons
  - Cohen's h effect size for proportions
  - Mann-Whitney U + rank-biserial r for continuous metrics (dz, final_z)
  - Bootstrap CI for mean dz

Writes:
  results/<run_id>/overall_summary.csv
  results/<run_id>/difficulty_breakdown.csv
  results/<run_id>/per_object_stats.csv
  results/<run_id>/significance_tests.csv
  results/<run_id>/effect_sizes.csv

Usage
-----
    python scripts/analyze_diverse_results.py --run-dir results/diverse_medium
    python scripts/analyze_diverse_results.py --all   # analyse all diverse_* runs
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.logger import wilson_ci


# ── statistics ────────────────────────────────────────────────────────────────

def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.
    |h| < 0.2 = small, 0.2–0.5 = medium, > 0.5 = large.
    """
    if p1 < 0 or p1 > 1 or p2 < 0 or p2 > 1:
        return float("nan")
    return 2 * math.asin(math.sqrt(max(0, min(1, p1)))) - \
           2 * math.asin(math.sqrt(max(0, min(1, p2))))


def _hypergeometric_pmf(k: int, n: int, K: int, N: int) -> float:
    """Hypergeometric PMF: P(X=k) for sampling k successes from N with K total successes."""
    from math import comb
    if k < max(0, n + K - N) or k > min(n, K):
        return 0.0
    return comb(K, k) * comb(N - K, n - k) / comb(N, n)


def fisher_test(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    """Fisher's exact test for 2×2 contingency table (pure Python, no scipy).
    Returns (odds_ratio, p_value).
    """
    from math import comb
    # 2×2 table: [[k1, n1-k1], [k2, n2-k2]]
    # Hypergeometric: fix row/col marginals, compute tail probability
    N = n1 + n2
    K = k1 + k2
    n = n1
    if n1 == 0 or n2 == 0 or N == 0:
        return float("nan"), float("nan")

    # odds ratio
    a, b, c, d = k1, n1 - k1, k2, n2 - k2
    if b == 0 or c == 0:
        odds_ratio = float("inf") if (a > 0 and d > 0) else float("nan")
    else:
        odds_ratio = (a * d) / (b * c) if (b * c) != 0 else float("nan")

    # two-sided p-value via exact hypergeometric sum
    p_obs = _hypergeometric_pmf(k1, n1, K, N)
    p_val = sum(
        _hypergeometric_pmf(k, n, K, N)
        for k in range(max(0, n + K - N), min(n, K) + 1)
        if _hypergeometric_pmf(k, n, K, N) <= p_obs + 1e-10
    )
    return float(odds_ratio), float(min(1.0, p_val))


def mannwhitney(a: List[float], b: List[float]) -> Tuple[float, float, float]:
    """Mann-Whitney U test + rank-biserial r (pure numpy, no scipy).
    Returns (U_stat, p_value, rank_biserial_r).
    r > 0 means group a tends to be larger than group b.
    p-value approximated via normal approximation (valid for n > ~20).
    """
    if not a or not b:
        return float("nan"), float("nan"), float("nan")
    n1, n2 = len(a), len(b)
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, ties = [], []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank
        for _ in range(j - i):
            ranks.append(avg_rank)
        if j - i > 1:
            ties.append(j - i)
        i = j
    U1 = sum(r for r, (_, g) in zip(ranks, combined) if g == 0) - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    U  = min(U1, U2)
    r  = 1.0 - 2 * U1 / (n1 * n2)  # rank-biserial, positive if a > b

    # normal approximation for p-value
    mu  = n1 * n2 / 2.0
    N   = n1 + n2
    tie_corr = sum(t**3 - t for t in ties) / (N * (N - 1)) if ties else 0
    var = (n1 * n2 / 12) * (N + 1 - tie_corr)
    if var <= 0:
        return float(U), float("nan"), float(r)
    z = (U - mu) / math.sqrt(var)
    # normal CDF approximation (Abramowitz & Stegun)
    def _norm_cdf(x: float) -> float:
        t = 1 / (1 + 0.2316419 * abs(x))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
        base = 1 - math.exp(-x**2 / 2) / math.sqrt(2 * math.pi) * poly
        return base if x >= 0 else 1 - base
    p = 2 * min(_norm_cdf(z), 1 - _norm_cdf(z))
    return float(U), float(p), float(r)


def bootstrap_ci(
    vals: List[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI.  Returns (mean, lo, hi)."""
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng   = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
             for _ in range(n_boot)]
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return float(np.mean(vals)), lo, hi


# ── data loading ──────────────────────────────────────────────────────────────

def load_trials(run_dir: Path) -> List[dict]:
    path = run_dir / "trials.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


# ── aggregation helpers ───────────────────────────────────────────────────────

def _group_key(rec: dict, keys: List[str]) -> tuple:
    return tuple(rec.get(k) for k in keys)


def aggregate_success(
    records: List[dict],
    group_keys: List[str],
) -> Dict[tuple, dict]:
    """Return {group_key: {k, n, rate, ci_lo, ci_hi, dz_vals, final_z_vals}} per group."""
    rows = defaultdict(lambda: {"k": 0, "n": 0, "dz_vals": [], "final_z_vals": []})
    for rec in records:
        if not rec.get("stability_valid"):
            continue
        key = _group_key(rec, group_keys)
        rows[key]["n"] += 1
        if rec.get("success"):
            rows[key]["k"] += 1
        if rec.get("dz") is not None:
            rows[key]["dz_vals"].append(rec["dz"])
        if rec.get("final_z") is not None:
            rows[key]["final_z_vals"].append(rec["final_z"])
    result = {}
    for key, d in rows.items():
        rate, lo, hi = wilson_ci(d["k"], d["n"])
        result[key] = {**d, "rate": rate, "ci_lo": lo, "ci_hi": hi}
    return result


# ── writers ───────────────────────────────────────────────────────────────────

def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.name}  ({len(rows)} rows)")


def _fmt(v, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{round(float(v), decimals)}"


# ── analysis passes ───────────────────────────────────────────────────────────

def write_overall_summary(records: List[dict], out_dir: Path) -> None:
    agg = aggregate_success(records, ["method", "object"])

    rows = []
    for (method, obj), d in sorted(agg.items()):
        dz_mean, dz_lo, dz_hi = bootstrap_ci(d["dz_vals"])
        fz_mean, fz_lo, fz_hi = bootstrap_ci(d["final_z_vals"])
        rows.append({
            "method":       method or "",
            "object":       obj    or "",
            "n_valid":      d["n"],
            "n_success":    d["k"],
            "success_rate": _fmt(d["rate"]),
            "ci_lo":        _fmt(d["ci_lo"]),
            "ci_hi":        _fmt(d["ci_hi"]),
            "dz_mean":      _fmt(dz_mean),
            "dz_ci_lo":     _fmt(dz_lo),
            "dz_ci_hi":     _fmt(dz_hi),
            "final_z_mean": _fmt(fz_mean),
        })

    _write_csv(out_dir / "overall_summary.csv",
               ["method","object","n_valid","n_success","success_rate",
                "ci_lo","ci_hi","dz_mean","dz_ci_lo","dz_ci_hi","final_z_mean"],
               rows)


def write_difficulty_breakdown(records: List[dict], out_dir: Path) -> None:
    agg = aggregate_success(records, ["difficulty", "method"])

    rows = []
    for (diff, method), d in sorted(agg.items()):
        dz_mean, dz_lo, dz_hi = bootstrap_ci(d["dz_vals"])
        rows.append({
            "difficulty":   diff   or "",
            "method":       method or "",
            "n_valid":      d["n"],
            "n_success":    d["k"],
            "success_rate": _fmt(d["rate"]),
            "ci_lo":        _fmt(d["ci_lo"]),
            "ci_hi":        _fmt(d["ci_hi"]),
            "dz_mean":      _fmt(dz_mean),
            "dz_ci_lo":     _fmt(dz_lo),
            "dz_ci_hi":     _fmt(dz_hi),
        })

    _write_csv(out_dir / "difficulty_breakdown.csv",
               ["difficulty","method","n_valid","n_success","success_rate",
                "ci_lo","ci_hi","dz_mean","dz_ci_lo","dz_ci_hi"],
               rows)


def write_per_object_stats(records: List[dict], out_dir: Path) -> None:
    agg = aggregate_success(records, ["object", "difficulty", "method"])

    rows = []
    for (obj, diff, method), d in sorted(agg.items()):
        bilateral_n = sum(1 for r in records
                          if r.get("object") == obj
                          and r.get("difficulty") == diff
                          and r.get("method") == method
                          and r.get("stability_valid")
                          and r.get("bilateral_contact") is not None)
        bilateral_k = sum(1 for r in records
                          if r.get("object") == obj
                          and r.get("difficulty") == diff
                          and r.get("method") == method
                          and r.get("stability_valid")
                          and r.get("bilateral_contact"))
        bi_rate, bi_lo, bi_hi = wilson_ci(bilateral_k, bilateral_n)

        weld_k = sum(1 for r in records
                     if r.get("object") == obj
                     and r.get("difficulty") == diff
                     and r.get("method") == method
                     and r.get("stability_valid")
                     and r.get("weld_triggered"))
        weld_rate = weld_k / bilateral_n if bilateral_n else float("nan")

        rows.append({
            "object":          obj    or "",
            "difficulty":      diff   or "",
            "method":          method or "",
            "n_valid":         d["n"],
            "n_success":       d["k"],
            "success_rate":    _fmt(d["rate"]),
            "ci_lo":           _fmt(d["ci_lo"]),
            "ci_hi":           _fmt(d["ci_hi"]),
            "bilateral_rate":  _fmt(bi_rate),
            "bilateral_ci_lo": _fmt(bi_lo),
            "bilateral_ci_hi": _fmt(bi_hi),
            "weld_rate":       _fmt(weld_rate),
        })

    _write_csv(out_dir / "per_object_stats.csv",
               ["object","difficulty","method","n_valid","n_success","success_rate",
                "ci_lo","ci_hi","bilateral_rate","bilateral_ci_lo","bilateral_ci_hi",
                "weld_rate"],
               rows)


def write_significance_tests(records: List[dict], out_dir: Path) -> None:
    """Pairwise Fisher's exact test for each (difficulty × object) condition."""
    methods = sorted({r["method"] for r in records if r.get("method")})
    if len(methods) < 2:
        print("  [skip] significance tests require ≥2 methods")
        return

    agg = aggregate_success(records, ["difficulty", "object", "method"])

    rows = []
    diffs   = sorted({r.get("difficulty") for r in records if r.get("difficulty")},
                     key=lambda x: {"easy": 0, "medium": 1, "hard": 2}.get(x, 9))
    objects = sorted({r.get("object") for r in records if r.get("object")})

    for diff in diffs:
        for obj in objects:
            for i, m1 in enumerate(methods):
                for m2 in methods[i+1:]:
                    key1 = (diff, obj, m1)
                    key2 = (diff, obj, m2)
                    d1 = agg.get(key1)
                    d2 = agg.get(key2)
                    if not d1 or not d2 or not d1["n"] or not d2["n"]:
                        continue

                    or_val, p = fisher_test(d1["k"], d1["n"], d2["k"], d2["n"])
                    h = cohen_h(d1["rate"], d2["rate"])
                    u, p_cts, r = mannwhitney(d1["dz_vals"], d2["dz_vals"])

                    rows.append({
                        "difficulty": diff  or "all",
                        "object":     obj   or "all",
                        "method_a":   m1,
                        "method_b":   m2,
                        "n_a":        d1["n"],
                        "n_b":        d2["n"],
                        "rate_a":     _fmt(d1["rate"]),
                        "rate_b":     _fmt(d2["rate"]),
                        "odds_ratio": _fmt(or_val),
                        "p_fisher":   _fmt(p, 6),
                        "sig_005":    "yes" if (isinstance(p, float) and p < 0.05) else "no",
                        "cohen_h":    _fmt(h),
                        "h_size":     ("large" if abs(h) > 0.5 else
                                       "medium" if abs(h) > 0.2 else "small")
                                      if not math.isnan(h) else "",
                        "mw_p":       _fmt(p_cts, 6),
                        "rankbis_r":  _fmt(r),
                    })

    # also add overall (pooled across objects/difficulties)
    m_agg = aggregate_success(records, ["method"])
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            d1 = m_agg.get((m1,))
            d2 = m_agg.get((m2,))
            if not d1 or not d2:
                continue
            or_val, p = fisher_test(d1["k"], d1["n"], d2["k"], d2["n"])
            h = cohen_h(d1["rate"], d2["rate"])
            u, p_cts, r = mannwhitney(d1["dz_vals"], d2["dz_vals"])
            rows.append({
                "difficulty": "OVERALL",
                "object":     "OVERALL",
                "method_a":   m1,
                "method_b":   m2,
                "n_a":        d1["n"],
                "n_b":        d2["n"],
                "rate_a":     _fmt(d1["rate"]),
                "rate_b":     _fmt(d2["rate"]),
                "odds_ratio": _fmt(or_val),
                "p_fisher":   _fmt(p, 6),
                "sig_005":    "yes" if (isinstance(p, float) and p < 0.05) else "no",
                "cohen_h":    _fmt(h),
                "h_size":     ("large" if abs(h) > 0.5 else
                               "medium" if abs(h) > 0.2 else "small")
                              if not math.isnan(h) else "",
                "mw_p":       _fmt(p_cts, 6),
                "rankbis_r":  _fmt(r),
            })

    _write_csv(out_dir / "significance_tests.csv",
               ["difficulty","object","method_a","method_b","n_a","n_b",
                "rate_a","rate_b","odds_ratio","p_fisher","sig_005",
                "cohen_h","h_size","mw_p","rankbis_r"],
               rows)


def write_failure_breakdown(records: List[dict], out_dir: Path) -> None:
    """Count failure reasons per (method, difficulty)."""
    counts: dict = defaultdict(lambda: defaultdict(int))
    for rec in records:
        if rec.get("stability_valid") and not rec.get("success"):
            key = (rec.get("method", "?"), rec.get("difficulty", "?"))
            reason = rec.get("failure_reason") or "unknown"
            counts[key][reason] += 1

    rows = []
    for (method, diff), reasons in sorted(counts.items()):
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            rows.append({
                "method":    method,
                "difficulty": diff,
                "reason":    reason,
                "count":     count,
            })
    if rows:
        _write_csv(out_dir / "failure_breakdown.csv",
                   ["method","difficulty","reason","count"], rows)


# ── main ──────────────────────────────────────────────────────────────────────

def analyse_run(run_dir: Path) -> None:
    print(f"\n[analyse] {run_dir.name}")
    records = load_trials(run_dir)
    if not records:
        print(f"  [skip] no trials.jsonl found in {run_dir}")
        return

    valid = [r for r in records if r.get("stability_valid")]
    print(f"  {len(records)} total records, {len(valid)} stability-valid")
    if not valid:
        return

    out_dir = run_dir
    write_overall_summary(records, out_dir)
    write_difficulty_breakdown(records, out_dir)
    write_per_object_stats(records, out_dir)
    write_significance_tests(records, out_dir)
    write_failure_breakdown(records, out_dir)
    print(f"  analysis complete → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Statistical analysis of diverse benchmark results")
    ap.add_argument("--run-dir",  default=None, help="Single run directory to analyse")
    ap.add_argument("--all",      action="store_true",
                    help="Analyse all results/diverse_* directories")
    ap.add_argument("--combined", action="store_true",
                    help="Also combine all diverse_* runs into one pooled analysis")
    args = ap.parse_args()

    results_root = ROOT / "results"

    if args.all:
        run_dirs = sorted(results_root.glob("diverse_*"))
        if not run_dirs:
            print("[analyse] No diverse_* directories found in results/")
            return
        for rd in run_dirs:
            if (rd / "trials.jsonl").exists():
                analyse_run(rd)
        if args.combined:
            all_records = []
            for rd in run_dirs:
                all_records += load_trials(rd)
            if all_records:
                combined_dir = results_root / "diverse_combined"
                combined_dir.mkdir(exist_ok=True)
                print(f"\n[analyse] combined  ({len(all_records)} total records)")
                write_overall_summary(all_records, combined_dir)
                write_difficulty_breakdown(all_records, combined_dir)
                write_per_object_stats(all_records, combined_dir)
                write_significance_tests(all_records, combined_dir)
                write_failure_breakdown(all_records, combined_dir)
    elif args.run_dir:
        analyse_run(Path(args.run_dir))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
