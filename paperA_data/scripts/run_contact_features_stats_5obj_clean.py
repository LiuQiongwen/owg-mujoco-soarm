"""
**Authoritative for Paper A submission (2026-07-09).** Supersedes
run_contact_features_stats_extended.py's 6-object file for citation purposes.

Scissors is excluded entirely (not just flagged) from both the per-test
p-values and the multiple-comparison correction family. Root cause (see
paperA_data/README.md, "CRITICAL" section, confirmed 2026-07-08): all 50
Scissors trials in phase0_diag_extended/ silently ran ui.py's random-CoM
fallback sampler, not the OT-CFM checkpoint, because
_cfm_sample_candidates()'s fuzzy string match never matches "scissors" against
the trained "cylinder" conditioning key. Scissors' 3 contact-feature p-values
were also structurally degenerate (all-zero features, p_raw=1.0 by
construction for a thin/flat object) -- a second, independent reason to
exclude it, but the fallback-bug is the primary one.

Same Mann-Whitney + rank-biserial + Bonferroni + Benjamini-Hochberg procedure
as run_contact_features_stats_extended.py, over the 5 valid objects only
(3 features x 5 objects = 15 tests, alpha = 0.05/15 = 0.00333).

Output: paperA_data/formal_results/contact_features_bonferroni_bh_5obj_clean.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, false_discovery_control

BASE = Path(__file__).resolve().parent.parent

OBJECTS = [
    ("Pear", BASE / "phase0_diag/data_with_contact_feats.json"),
    ("MustardBottle", BASE / "phase0_diag/data_with_contact_feats.json"),
    ("CrackerBox", BASE / "phase0_diag/data_with_contact_feats.json"),
    ("TomatoSoupCan", BASE / "phase0_diag_extended/data_with_contact_feats_new3.json"),
    ("PowerDrill", BASE / "phase0_diag_extended/data_with_contact_feats_new3.json"),
]
FEATURES = ["local_point_density", "normal_consistency", "contact_width_ratio"]


def main():
    cache = {}
    rows = []
    for obj, path in OBJECTS:
        if path not in cache:
            cache[path] = json.load(open(path))
        d = cache[path]
        obj_data = [x for x in d if x["object"] == obj]
        succ = [x for x in obj_data if x["success"]]
        fail = [x for x in obj_data if not x["success"]]
        for feat in FEATURES:
            s = np.array([x[feat] for x in succ])
            f = np.array([x[feat] for x in fail])
            degenerate = np.all(s == s[0]) and np.all(f == f[0]) and (len(set(s.tolist() + f.tolist())) <= 1)
            if degenerate:
                u, p, rank_biserial = float("nan"), 1.0, 0.0
            else:
                u, p = mannwhitneyu(s, f, alternative="two-sided")
                rank_biserial = 1 - 2 * u / (len(s) * len(f))
            rows.append({
                "object": obj,
                "feature": feat,
                "n_succ": len(s),
                "n_fail": len(f),
                "succ_mean": round(float(s.mean()), 5),
                "succ_std": round(float(s.std()), 5),
                "fail_mean": round(float(f.mean()), 5),
                "fail_std": round(float(f.std()), 5),
                "mannwhitney_U": u,
                "p_raw": p,
                "rank_biserial_effect_size": round(rank_biserial, 4),
                "degenerate_all_zero": degenerate,
            })

    n_tests = len(rows)
    alpha = 0.05
    bonf_alpha = alpha / n_tests
    p_raw = np.array([r["p_raw"] for r in rows])
    p_bh = false_discovery_control(p_raw, method="bh")

    for i, r in enumerate(rows):
        r["bonferroni_alpha"] = round(bonf_alpha, 6)
        r["sig_bonferroni"] = "SIG" if r["p_raw"] < bonf_alpha else "ns"
        r["p_bh_adjusted"] = round(float(p_bh[i]), 6)
        r["sig_bh_0.05"] = "SIG" if p_bh[i] < alpha else "ns"

    rows.sort(key=lambda r: r["p_raw"])

    out_path = BASE / "formal_results/contact_features_bonferroni_bh_5obj_clean.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows ({n_tests} tests, Bonferroni alpha={bonf_alpha:.5f}) to {out_path}")


if __name__ == "__main__":
    main()
