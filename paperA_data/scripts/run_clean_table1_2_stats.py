"""
Rebuilds paper_final.tex's Table I (Main Result) and Table II (Per-Object
Analysis) from scratch, using only trials collected on the fixed code
(post-merge of worktree-fix-eval-seeding into main, 2026-07-09/10) -- the
original data (scripts/quick_eval.sh's pre-fix runs) is discarded entirely,
not blended with it, since the pre-fix OT-CFM condition was measuring an
artifact of unseeded torch.randn(...) rather than the intended experiment.

n=50 per object per condition (7 objects x 2 conditions x 50 seeds =
700 trials), built from two 25-seed halves each:

  Banana/TomatoSoupCan/Pear/MustardBottle/CrackerBox/PowerDrill:
    seeds 1-25:  phase0_diag_extended/clean_seed1_25_{obj}_{cond}.jsonl
    seeds 26-50: phase0_diag_extended/seed26_50_{obj}_{cond}.jsonl
      (collected earlier in the same audit, already on the fixed code by
      coincidence of following the Tier-1 script pattern -- verified
      identical harness: same flags, same checkpoint, same env)

  Scissors (CFM path irrelevant -- always falls back to the same
  random-CoM sampler in both conditions, per the separate name-matching
  bug -- but included at n=50 for methodological consistency and because
  demo.py's spawn-orientation seeding fix still applies to it):
    seeds 1-25:  phase0_diag_extended/scissors_recheck_{baseline,otcfm}.jsonl
    seeds 26-50: phase0_diag_extended/clean_scissors_seed26_50_{baseline,otcfm}.jsonl

Output: paperA_data/formal_results/clean_table1_2.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

BASE = Path(__file__).resolve().parent.parent
DIAG = BASE / "phase0_diag_extended"

SOURCES = {
    "Banana": {
        "baseline": ["clean_seed1_25_Banana_baseline.jsonl", "seed26_50_Banana_baseline.jsonl"],
        "otcfm": ["clean_seed1_25_Banana_otcfm.jsonl", "seed26_50_Banana_otcfm.jsonl"],
    },
    "TomatoSoupCan": {
        "baseline": ["clean_seed1_25_TomatoSoupCan_baseline.jsonl", "seed26_50_TomatoSoupCan_baseline.jsonl"],
        "otcfm": ["clean_seed1_25_TomatoSoupCan_otcfm.jsonl", "seed26_50_TomatoSoupCan_otcfm.jsonl"],
    },
    "Pear": {
        "baseline": ["clean_seed1_25_Pear_baseline.jsonl", "seed26_50_Pear_baseline.jsonl"],
        "otcfm": ["clean_seed1_25_Pear_otcfm.jsonl", "seed26_50_Pear_otcfm.jsonl"],
    },
    "MustardBottle": {
        "baseline": ["clean_seed1_25_MustardBottle_baseline.jsonl", "seed26_50_MustardBottle_baseline.jsonl"],
        "otcfm": ["clean_seed1_25_MustardBottle_otcfm.jsonl", "seed26_50_MustardBottle_otcfm.jsonl"],
    },
    "Scissors": {
        "baseline": ["scissors_recheck_baseline.jsonl", "clean_scissors_seed26_50_baseline.jsonl"],
        "otcfm": ["scissors_recheck_otcfm.jsonl", "clean_scissors_seed26_50_otcfm.jsonl"],
    },
    "CrackerBox": {
        "baseline": ["clean_seed1_25_CrackerBox_baseline.jsonl", "seed26_50_CrackerBox_baseline.jsonl"],
        "otcfm": ["clean_seed1_25_CrackerBox_otcfm.jsonl", "seed26_50_CrackerBox_otcfm.jsonl"],
    },
    "PowerDrill": {
        "baseline": ["clean_seed1_25_PowerDrill_baseline.jsonl", "seed26_50_PowerDrill_baseline.jsonl"],
        "otcfm": ["clean_seed1_25_PowerDrill_otcfm.jsonl", "seed26_50_PowerDrill_otcfm.jsonl"],
    },
}

OBJECT_ORDER = ["Banana", "TomatoSoupCan", "Pear", "MustardBottle", "Scissors", "CrackerBox", "PowerDrill"]


def load_count(files):
    succ, n = 0, 0
    for fname in files:
        rows = [json.loads(l) for l in open(DIAG / fname)]
        succ += sum(1 for r in rows if r["success"] == "true")
        n += len(rows)
    return succ, n


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
    rows = []
    tot_base, tot_otcfm, tot_n = 0, 0, 0
    for obj in OBJECT_ORDER:
        b_succ, b_n = load_count(SOURCES[obj]["baseline"])
        o_succ, o_n = load_count(SOURCES[obj]["otcfm"])
        assert b_n == 50 and o_n == 50, f"{obj}: expected n=50, got base={b_n} otcfm={o_n}"
        tot_base += b_succ
        tot_otcfm += o_succ
        tot_n += b_n
        z, p = two_prop_z(o_succ, o_n, b_succ, b_n)
        rows.append({
            "object": obj,
            "base_succ": b_succ, "base_n": b_n, "base_pct": round(b_succ / b_n * 100, 1),
            "otcfm_succ": o_succ, "otcfm_n": o_n, "otcfm_pct": round(o_succ / o_n * 100, 1),
            "delta_pp": round(o_succ / o_n * 100 - b_succ / b_n * 100, 1),
            "z": round(z, 3), "p": round(p, 4),
        })

    z, p = two_prop_z(tot_otcfm, tot_n, tot_base, tot_n)
    lo_b, hi_b = wilson_ci(tot_base, tot_n)
    lo_o, hi_o = wilson_ci(tot_otcfm, tot_n)
    rows.append({
        "object": "ALL",
        "base_succ": tot_base, "base_n": tot_n, "base_pct": round(tot_base / tot_n * 100, 1),
        "otcfm_succ": tot_otcfm, "otcfm_n": tot_n, "otcfm_pct": round(tot_otcfm / tot_n * 100, 1),
        "delta_pp": round(tot_otcfm / tot_n * 100 - tot_base / tot_n * 100, 1),
        "z": round(z, 3), "p": round(p, 6),
    })

    out_path = BASE / "formal_results/clean_table1_2.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}\n")
    for r in rows:
        print(f"  {r['object']:<15} base={r['base_succ']}/{r['base_n']}={r['base_pct']}%  "
              f"otcfm={r['otcfm_succ']}/{r['otcfm_n']}={r['otcfm_pct']}%  "
              f"delta={r['delta_pp']:+.1f}pp  z={r['z']}  p={r['p']}")
    print(f"\nBaseline Wilson 95% CI: [{lo_b:.1f}, {hi_b:.1f}]")
    print(f"OT-CFM   Wilson 95% CI: [{lo_o:.1f}, {hi_o:.1f}]")


if __name__ == "__main__":
    main()
