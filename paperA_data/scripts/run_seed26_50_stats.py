"""
Formalizes the 2026-07-09 TomatoSoupCan/PowerDrill seed extension
(25 -> 50 seeds) for paper_final.tex's Table II Baseline/OT-CFM columns.

Motivation: both objects were non-significant at 25 seeds
(TomatoSoupCan +8pp p=0.490, PowerDrill +12pp p=0.463) and the contact-
feature diagnostic suggested TomatoSoupCan might have real, seed-limited
headroom. Extending to 50 seeds (25 new, run_seed26_50_tomatosoupcan_
powerdrill.sh, same harness reverse-engineered from the original source
logs' own headers) tests this directly.

Finding: TomatoSoupCan's direction REVERSES at n=50 (baseline 94.0% vs
OT-CFM 88.0%, -6.0pp) -- the original +8pp was a favorable draw, not an
underpowered real effect. PowerDrill's direction holds (+12.0pp) but
remains non-significant. Per user decision, Table II is NOT overwritten
with the n=50 figures (that would make its per-object methodology
inconsistent -- 2 of 7 objects at n=50, the rest at n=25); this is
reported as a disclosed robustness caution in the Per-Object Analysis
text instead.

Output: paperA_data/formal_results/seed26_50_tomatosoupcan_powerdrill.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, fisher_exact

BASE = Path(__file__).resolve().parent.parent
DIAG = BASE / "phase0_diag_extended"

# Original seed 1-25 counts, confirmed to exactly match paper_final.tex's
# Table II by cross-checking against the original source logs' own printed
# headers (logs/eval_baseline_nosem.log, logs/eval_cfm_ot_nosem_current.log).
ORIG = {
    ("TomatoSoupCan", "baseline"): 23,
    ("TomatoSoupCan", "otcfm"): 25,
    ("PowerDrill", "baseline"): 19,
    ("PowerDrill", "otcfm"): 22,
}


def two_prop_z(s1, n1, s2, n2):
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def main():
    merged = {}
    rows_out = []
    for obj in ["TomatoSoupCan", "PowerDrill"]:
        for cond in ["baseline", "otcfm"]:
            new_rows = [json.loads(l) for l in open(DIAG / f"seed26_50_{obj}_{cond}.jsonl")]
            new_succ = sum(1 for r in new_rows if r["success"] == "true")
            assert len(new_rows) == 25
            total = ORIG[(obj, cond)] + new_succ
            merged[(obj, cond)] = total
            rows_out.append({
                "object": obj, "condition": cond,
                "orig_1_25": ORIG[(obj, cond)],
                "new_26_50": new_succ,
                "merged_1_50": total,
                "merged_pct": round(total / 50 * 100, 1),
            })

    for obj in ["TomatoSoupCan", "PowerDrill"]:
        b, o = merged[(obj, "baseline")], merged[(obj, "otcfm")]
        z, p = two_prop_z(o, 50, b, 50)
        _, p_fisher = fisher_exact([[o, 50 - o], [b, 50 - b]])
        rows_out.append({
            "object": obj, "condition": "OT-CFM_vs_baseline_n50",
            "orig_1_25": "", "new_26_50": "",
            "merged_1_50": f"delta={o/50*100-b/50*100:+.1f}pp",
            "merged_pct": f"z={z:.3f} p={p:.4f} fisher_p={p_fisher:.4f}",
        })

    out_path = BASE / "formal_results/seed26_50_tomatosoupcan_powerdrill.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out_path}")
    for r in rows_out:
        print(" ", r)


if __name__ == "__main__":
    main()
