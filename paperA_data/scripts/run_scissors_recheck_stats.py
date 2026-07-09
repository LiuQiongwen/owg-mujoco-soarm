"""
Formalizes the 2026-07-09 Scissors re-measurement and its downstream effect
on paper_final.tex's Table I/II/III.

Background: paper_final.tex's headline numbers (Baseline 82.3%, OT-CFM+LGGSN
94.3%) were built by adding a separately-measured 25-trial Scissors block to
an existing 150-trial (6-object) total, following the physics-config fix
documented in logs/eval_scissors_fix_summary.log. That block was measured
25/25=100% on 2026-06-26. A later commit (cf58a7d, 2026-06-28) claimed a
re-run found 23/25=92% and called the 06-26 figure "imputed" -- which is
inaccurate (eval_scissors_baseline.log/eval_scissors_cfm.log show real
per-seed [Y] entries, not an assumption), but the discrepancy itself is real:
two actual measurements of the same nominal condition gave different counts.
A fresh from-scratch re-run here (run_scissors_recheck_2026-07-09.sh) found a
*third* value, 22/25=88%, and disagreed with the 06-26 log at the exact same
seed (seed=1) in a live smoke test -- confirming this specific object's
collision geometry (a 4cm box proxy, barely above the gripper's 4cm minimum
opening) is sensitive to run-to-run variation, not that any one run was
"wrong."

This script takes the freshest measurement (2026-07-09, 22/25 both
conditions) as authoritative per user decision, and recomputes every
downstream number: the 6-object (non-Scissors) baseline/OT-CFM totals are
held fixed at their existing values (119/150 and 140/150 respectively, taken
from logs/eval_scissors_fix_summary.log's "old" pre-Scissors-fix totals,
which by construction already exclude Scissors) since those were not
re-verified in this pass; only the Scissors block (25->22) is corrected.

Output: paperA_data/formal_results/scissors_recheck_corrected_totals.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

BASE = Path(__file__).resolve().parent.parent
DIAG = BASE / "phase0_diag_extended"

NON_SCISSORS_BASELINE = 119  # 6-object total, unchanged, from eval_scissors_fix_summary.log
NON_SCISSORS_OTCFM = 140     # 6-object total, unchanged
N_NONSCISSORS = 150
N_SCISSORS = 25
N_TOTAL = 175


def load_recheck(cond):
    rows = [json.loads(l) for l in open(DIAG / f"scissors_recheck_{cond}.jsonl")]
    succ = sum(1 for r in rows if r["success"] == "true")
    return succ, len(rows)


def two_prop_z(s1, n1, s2, n2):
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def wilson_ci(s, n, z=1.96):
    p = s / n
    denom = 1 + z ** 2 / n
    center = p + z ** 2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (center - margin) / denom * 100, (center + margin) / denom * 100


def main():
    scissors_base, n_base_check = load_recheck("baseline")
    scissors_otcfm, n_otcfm_check = load_recheck("otcfm")
    assert n_base_check == N_SCISSORS and n_otcfm_check == N_SCISSORS

    baseline_total = NON_SCISSORS_BASELINE + scissors_base
    otcfm_total = NON_SCISSORS_OTCFM + scissors_otcfm

    z, p = two_prop_z(otcfm_total, N_TOTAL, baseline_total, N_TOTAL)
    lo_b, hi_b = wilson_ci(baseline_total, N_TOTAL)
    lo_o, hi_o = wilson_ci(otcfm_total, N_TOTAL)

    rows = [
        {
            "metric": "Scissors baseline (2026-07-09 recheck)",
            "value": f"{scissors_base}/{N_SCISSORS}",
            "pct": round(scissors_base / N_SCISSORS * 100, 1),
        },
        {
            "metric": "Scissors OT-CFM (2026-07-09 recheck)",
            "value": f"{scissors_otcfm}/{N_SCISSORS}",
            "pct": round(scissors_otcfm / N_SCISSORS * 100, 1),
        },
        {
            "metric": "Table I/II/III Baseline (random CoM + LGGSN), corrected",
            "value": f"{baseline_total}/{N_TOTAL}",
            "pct": round(baseline_total / N_TOTAL * 100, 1),
        },
        {
            "metric": "Table I/II/III OT-CFM + LGGSN (full), corrected",
            "value": f"{otcfm_total}/{N_TOTAL}",
            "pct": round(otcfm_total / N_TOTAL * 100, 1),
        },
        {
            "metric": "Baseline vs OT-CFM: z, p",
            "value": f"z={z:.2f}",
            "pct": round(p, 6),
        },
        {
            "metric": "Baseline Wilson 95% CI",
            "value": f"[{lo_b:.1f}, {hi_b:.1f}]",
            "pct": "",
        },
        {
            "metric": "OT-CFM Wilson 95% CI",
            "value": f"[{lo_o:.1f}, {hi_o:.1f}]",
            "pct": "",
        },
    ]

    out_path = BASE / "formal_results/scissors_recheck_corrected_totals.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")
    for r in rows:
        print(f"  {r['metric']}: {r['value']}  {r['pct']}")


if __name__ == "__main__":
    main()
