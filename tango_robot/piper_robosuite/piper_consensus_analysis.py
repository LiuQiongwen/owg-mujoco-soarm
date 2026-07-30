"""
Analyze piper_consensus_experiment_runner.py's results: success rate per
(object, candidate_selection), plus McNemar's exact test comparing
consensus vs best per object. Mirrors piper_experiment_analysis.py's
statistical approach (same paired-design reasoning applies here: trial_id
draws the same candidate pool for both selection strategies).

Usage:
  conda run -n tango python3 -m tango_robot.piper_robosuite.piper_consensus_analysis
"""
import glob
import json
from collections import defaultdict

from scipy.stats import binomtest

RESULTS_GLOB = "/lena/projects/OWG-main/tango_robot/piper_robosuite/consensus_results_*.json"


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def main():
    results = []
    for path in sorted(glob.glob(RESULTS_GLOB)):
        with open(path) as f:
            results.extend(json.load(f))
    print(f"loaded {len(results)} trials from {len(glob.glob(RESULTS_GLOB))} files")

    # Old (pre-noise_model-param) result files don't have this key -- they
    # were all generated under the original direct-jitter approach.
    for r in results:
        r.setdefault("noise_model", "kinematic")

    by_cell = defaultdict(list)
    for r in results:
        by_cell[(r["noise_model"], r["object"], r["candidate_selection"])].append(r)

    noise_models = sorted({r["noise_model"] for r in results})
    objects = sorted({r["object"] for r in results})

    for noise_model in noise_models:
        print(f"\n=== noise_model={noise_model} ===")
        print(f"{'object':10s} {'selection':10s} {'success':>8s} {'n':>4s} {'rate':>6s}")
        for obj in objects:
            for sel in ["consensus", "best"]:
                trials = by_cell[(noise_model, obj, sel)]
                n = len(trials)
                if n == 0:
                    continue
                s = sum(t["success"] for t in trials)
                rate = s / n if n else float("nan")
                print(f"{obj:10s} {sel:10s} {s:8d} {n:4d} {rate:6.2f}")

        print("Paired comparison (McNemar's exact test, consensus vs best):")
        for obj in objects:
            consensus = {t["trial_id"]: t["success"] for t in by_cell[(noise_model, obj, "consensus")]}
            best = {t["trial_id"]: t["success"] for t in by_cell[(noise_model, obj, "best")]}
            if not consensus or not best:
                continue
            common_ids = sorted(set(consensus) & set(best))
            b = sum(1 for i in common_ids if consensus[i] and not best[i])   # consensus wins
            c = sum(1 for i in common_ids if best[i] and not consensus[i])   # best wins
            both = sum(1 for i in common_ids if consensus[i] and best[i])
            neither = sum(1 for i in common_ids if not consensus[i] and not best[i])
            p = mcnemar_exact_p(b, c)
            print(f"  {obj:10s} both_succeed={both} both_fail={neither} "
                  f"consensus_only={b} best_only={c}  p={p:.4f}"
                  + ("  (discordant pairs=0, test undefined)" if b + c == 0 else ""))


if __name__ == "__main__":
    main()
