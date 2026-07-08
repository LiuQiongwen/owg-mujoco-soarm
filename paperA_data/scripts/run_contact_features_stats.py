"""
Formalizes the contact-feature discriminability analysis (source of the
"Bonferroni table"): Mann-Whitney U + rank-biserial effect size for 3 contact
features x 3 objects (9 tests), with Bonferroni and Benjamini-Hochberg
multiple-comparison correction.

Input: paperA_data/phase0_diag/data_with_contact_feats.json (150 trials:
Pear/MustardBottle/CrackerBox x 50 each), produced by phase2_contact_features.py.

Output: paperA_data/formal_results/contact_features_bonferroni_bh.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, false_discovery_control

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "phase0_diag/data_with_contact_feats.json"

OBJECTS = ["Pear", "MustardBottle", "CrackerBox"]
FEATURES = ["local_point_density", "normal_consistency", "contact_width_ratio"]


def main():
    d = json.load(open(DATA))

    rows = []
    for obj in OBJECTS:
        obj_data = [x for x in d if x["object"] == obj]
        succ = [x for x in obj_data if x["success"]]
        fail = [x for x in obj_data if not x["success"]]
        for feat in FEATURES:
            s = np.array([x[feat] for x in succ])
            f = np.array([x[feat] for x in fail])
            u, p = mannwhitneyu(s, f, alternative="two-sided")
            rank_biserial = 1 - 2 * u / (len(s) * len(f))
            rows.append({
                "object": obj,
                "feature": feat,
                "n_succ": len(s),
                "n_fail": len(f),
                "succ_mean": round(s.mean(), 5),
                "succ_std": round(s.std(), 5),
                "fail_mean": round(f.mean(), 5),
                "fail_std": round(f.std(), 5),
                "mannwhitney_U": u,
                "p_raw": p,
                "rank_biserial_effect_size": round(rank_biserial, 4),
            })

    n_tests = len(rows)
    alpha = 0.05
    bonf_alpha = alpha / n_tests
    p_raw = np.array([r["p_raw"] for r in rows])
    p_bh = false_discovery_control(p_raw, method="bh")

    order = np.argsort(p_raw)
    for i, r in enumerate(rows):
        r["bonferroni_alpha"] = round(bonf_alpha, 6)
        r["sig_bonferroni"] = "SIG" if r["p_raw"] < bonf_alpha else "ns"
        r["p_bh_adjusted"] = round(float(p_bh[i]), 6)
        r["sig_bh_0.05"] = "SIG" if p_bh[i] < alpha else "ns"

    rows.sort(key=lambda r: r["p_raw"])

    out_path = BASE / "formal_results/contact_features_bonferroni_bh.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows ({n_tests} tests, Bonferroni alpha={bonf_alpha:.5f}) to {out_path}")


if __name__ == "__main__":
    main()
