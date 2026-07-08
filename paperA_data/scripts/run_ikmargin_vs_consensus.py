"""
IK-margin vs consensus candidate-selection strategy comparison, for
Pear / MustardBottle / CrackerBox.

IMPORTANT CONFOUND, found while writing this script (not previously flagged
in this repo): the two strategies were NOT run with a matched ensemble size.
- ikmargin_*.jsonl: --ikmargin-n 10 (pick lowest-IK-error candidate out of 10)
- consensus data (both phase1_v2/consensus_MustardBottle.jsonl and
  phase1_pilot/consensus_trials_n10.jsonl): --consensus-n 5 (pick the
  candidate closest to the median of 5) -- the "n10" in the pilot filename
  refers to 10 *repetitions* of a 5-candidate ensemble, not ensemble size 10.
So this script compares "ikmargin over a 10-candidate pool" against
"consensus over a 5-candidate pool". Any difference found below is
confounded with ensemble size and should NOT be reported as an apples-to-apples
test of selection rule alone without saying so.

Each object's two samples (ikmargin n=50, consensus n=50) are independent
(no shared candidate pool / seed reuse between the two strategies for the
same object), so Fisher's exact test on the 2x2 success/fail table is used.

Output: paperA_data/formal_results/ikmargin_vs_consensus.csv
"""
import csv
import json
from pathlib import Path

from scipy.stats import fisher_exact

BASE = Path(__file__).resolve().parent.parent

IKMARGIN_FILES = {
    "Pear": BASE / "phase1_v2/ikmargin_Pear.jsonl",
    "MustardBottle": BASE / "phase1_v2/ikmargin_MustardBottle.jsonl",
    "CrackerBox": BASE / "phase1_v2/ikmargin_CrackerBox.jsonl",
}


def load(path):
    return [json.loads(l) for l in open(path)]


def consensus_rows_for(obj):
    if obj == "MustardBottle":
        return load(BASE / "phase1_v2/consensus_MustardBottle.jsonl")
    rows = load(BASE / "phase1_pilot/consensus_trials_n10.jsonl")
    return [r for r in rows if r["object"] == obj]


def counts(rows):
    s = sum(1 for r in rows if r["success"] == "true")
    return s, len(rows) - s, len(rows)


def main():
    out_rows = []
    for obj, ik_path in IKMARGIN_FILES.items():
        ik_rows = load(ik_path)
        cons_rows = consensus_rows_for(obj)

        ik_s, ik_f, ik_n = counts(ik_rows)
        co_s, co_f, co_n = counts(cons_rows)

        table = [[ik_s, ik_f], [co_s, co_f]]
        odds_ratio, p = fisher_exact(table, alternative="two-sided")

        out_rows.append({
            "object": obj,
            "ikmargin_ensemble_n": 10,
            "ikmargin_success": ik_s,
            "ikmargin_n": ik_n,
            "ikmargin_rate": round(ik_s / ik_n, 4),
            "consensus_ensemble_n": 5,
            "consensus_success": co_s,
            "consensus_n": co_n,
            "consensus_rate": round(co_s / co_n, 4),
            "rate_diff_ikmargin_minus_consensus": round(ik_s / ik_n - co_s / co_n, 4),
            "fisher_exact_odds_ratio": round(odds_ratio, 4),
            "fisher_exact_p": p,
            "confound_note": "ensemble sizes differ (10 vs 5) -- not a controlled test of selection rule alone",
        })

    out_path = BASE / "formal_results/ikmargin_vs_consensus.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
