"""
Pairwise significance tests for the exp1_variance sampler-variance data
(OT-CFM vs CFM-noOT vs DDPM sampled with 50-step DDIM).

Design note: all three methods were run on the identical grid of
(object, orient_seed, gen_seed) triples (7 objects x 5 orient_seeds x 10 gen_seeds
= 350 trials each), so trials are matched by triple across methods -> McNemar's
exact test (paired) is the primary test. Mann-Whitney U and Welch's t-test
(both unpaired, treating each method's 350 trials as an independent sample)
are also reported because they are what was informally quoted in an old,
unsaved session transcript -- included here for continuity, not because they
are the statistically preferred test for this matched design.

Scope note: the "DDPM" column is the DDPM-trained checkpoint sampled via
50-step DDIM (env var DDIM_STEPS=50), not an independently trained "DDIM
model". Conclusions here are about three *sampling procedures*, not an
ODE-vs-SDE comparison -- no ODE/SDE or AUC analysis exists in this repo.

Output: paperA_data/formal_results/exp1_variance_significance.csv

**2026-07-09 update**: Scissors rows are annotated `excluded_reason` =
fallback-bug-invalid (see paperA_data/README.md CRITICAL section) -- kept for
record/provenance, not for citation. An additional pooled scope,
"ALL_excl_scissors" (6 valid objects: Banana/Pear/MustardBottle/CrackerBox/
PowerDrill/TomatoSoupCan, n=300/method), is added alongside the original
"ALL" (7 objects incl. the invalid Scissors, n=350/method, kept for
comparison) -- cite ALL_excl_scissors, not ALL, in Paper A.
"""
SCISSORS_EXCLUDED_REASON = (
    "FALLBACK_BUG_INVALID (excluded 2026-07-09) -- ui.py's _cfm_sample_"
    "candidates() never matches 'scissors' against the trained 'cylinder' "
    "key, so all 50 trials for every method silently used the same "
    "random-CoM fallback sampler, not OT-CFM/CFM-noOT/DDPM. See "
    "paperA_data/README.md CRITICAL section. Do not cite this scope's rows."
)
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind, binomtest

BASE = Path(__file__).resolve().parent.parent
FILES = {
    "OT-CFM": BASE / "exp1_variance/raw_results.jsonl",
    "CFM-noOT": BASE / "exp1_variance/raw_results_CFM-noOT.jsonl",
    "DDPM(DDIM-50)": BASE / "exp1_variance/raw_results_DDPM.jsonl",
}


def load(path):
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        r["success"] = 1 if r["success"] == "true" else 0
    return rows


def mcnemar_exact(a_success, b_success):
    """Paired binary outcomes, matched by (object, orient_seed, gen_seed).
    b = A-success/B-fail count, c = A-fail/B-success count.
    Exact two-sided binomial test on the discordant pairs."""
    b = sum(1 for a, bb in zip(a_success, b_success) if a == 1 and bb == 0)
    c = sum(1 for a, bb in zip(a_success, b_success) if a == 0 and bb == 1)
    n_discordant = b + c
    if n_discordant == 0:
        return b, c, float("nan")
    res = binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided")
    return b, c, res.pvalue


def run_tests(rows_a, rows_b):
    """rows_a/rows_b already aligned (same order of (object,orient_seed,gen_seed))."""
    a = np.array([r["success"] for r in rows_a])
    b = np.array([r["success"] for r in rows_b])
    out = {}
    out["n"] = len(a)
    out["rate_a"] = a.mean()
    out["rate_b"] = b.mean()
    if a.std() == 0 and b.std() == 0:
        out["mannwhitney_p"] = float("nan")
        out["ttest_p"] = float("nan")
        out["note"] = "zero variance in both groups (all success or all fail) - test undefined"
    else:
        try:
            _, out["mannwhitney_p"] = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            out["mannwhitney_p"] = float("nan")
        try:
            _, out["ttest_p"] = ttest_ind(a, b, equal_var=False)
        except Exception:
            out["ttest_p"] = float("nan")
        out["note"] = ""
    b_disc, c_disc, mcnemar_p = mcnemar_exact(a, b)
    out["mcnemar_b_Asucc_Bfail"] = b_disc
    out["mcnemar_c_Afail_Bsucc"] = c_disc
    out["mcnemar_p"] = mcnemar_p
    return out


def main():
    data = {name: load(path) for name, path in FILES.items()}
    key = lambda r: (r["object"], r["orient_seed"], r["gen_seed"])
    for name, rows in data.items():
        rows.sort(key=key)
    keys_ref = [key(r) for r in data["OT-CFM"]]
    for name, rows in data.items():
        assert [key(r) for r in rows] == keys_ref, f"{name} grid does not match OT-CFM grid"

    objects = sorted(set(r["object"] for r in data["OT-CFM"]))
    pairs = list(itertools.combinations(FILES.keys(), 2))

    out_rows = []
    for scope in ["ALL", "ALL_excl_scissors"] + objects:
        for m1, m2 in pairs:
            if scope == "ALL":
                rows1, rows2 = data[m1], data[m2]
            elif scope == "ALL_excl_scissors":
                rows1 = [r for r in data[m1] if r["object"] != "Scissors"]
                rows2 = [r for r in data[m2] if r["object"] != "Scissors"]
            else:
                rows1 = [r for r in data[m1] if r["object"] == scope]
                rows2 = [r for r in data[m2] if r["object"] == scope]
            res = run_tests(rows1, rows2)
            out_rows.append({
                "scope": scope,
                "method_a": m1,
                "method_b": m2,
                "n": res["n"],
                "success_rate_a": round(res["rate_a"], 4),
                "success_rate_b": round(res["rate_b"], 4),
                "mannwhitney_p": res["mannwhitney_p"],
                "welch_ttest_p": res["ttest_p"],
                "mcnemar_exact_p_paired": res["mcnemar_p"],
                "mcnemar_discordant_Asucc_Bfail": res["mcnemar_b_Asucc_Bfail"],
                "mcnemar_discordant_Afail_Bsucc": res["mcnemar_c_Afail_Bsucc"],
                "note": res["note"],
                "excluded_reason": SCISSORS_EXCLUDED_REASON if scope == "Scissors" else "",
            })

    out_path = BASE / "formal_results/exp1_variance_significance.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
