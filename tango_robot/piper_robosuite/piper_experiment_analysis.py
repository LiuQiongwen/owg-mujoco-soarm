"""
Analyze piper_experiment_runner.py's results.json: success rate per
(object, strategy), plus McNemar's exact test comparing oriented vs fixed
per object.

NOTE on test choice: trials are PAIRED, not independent -- run_one_trial()
reseeds numpy with the same trial_id for both strategies, so oriented/trial 7
and fixed/trial 7 see the identical object spawn position and yaw. A
two-proportion z-test or Fisher's exact test assumes independent samples and
would be the wrong tool here; McNemar's test is the correct one for paired
binary outcomes (it only uses the discordant pairs: trials where the two
strategies disagreed).

Usage:
  conda run -n tango python3 -m tango_robot.piper_robosuite.piper_experiment_analysis
"""
import glob
import json
from collections import defaultdict

from scipy.stats import binomtest

RESULTS_GLOB = "/lena/projects/OWG-main/tango_robot/piper_robosuite/experiment_results_*.json"


def mcnemar_exact_p(b, c):
    """Exact McNemar's test p-value: b = oriented-succeeds/fixed-fails count,
    c = oriented-fails/fixed-succeeds count (the two discordant-pair types).
    Two-sided exact binomial test on the discordant pairs against p=0.5."""
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

    by_cell = defaultdict(list)
    for r in results:
        by_cell[(r["object"], r["strategy"])].append(r)

    objects = sorted({r["object"] for r in results})

    print(f"{'object':10s} {'strategy':10s} {'success':>8s} {'n':>4s} {'rate':>6s}")
    for obj in objects:
        for strat in ["oriented", "fixed"]:
            trials = by_cell[(obj, strat)]
            n = len(trials)
            s = sum(t["success"] for t in trials)
            rate = s / n if n else float("nan")
            print(f"{obj:10s} {strat:10s} {s:8d} {n:4d} {rate:6.2f}")

    print("\nPaired comparison (McNemar's exact test, oriented vs fixed):")
    for obj in objects:
        oriented = {t["trial_id"]: t["success"] for t in by_cell[(obj, "oriented")]}
        fixed = {t["trial_id"]: t["success"] for t in by_cell[(obj, "fixed")]}
        common_ids = sorted(set(oriented) & set(fixed))
        b = sum(1 for i in common_ids if oriented[i] and not fixed[i])   # oriented wins
        c = sum(1 for i in common_ids if fixed[i] and not oriented[i])   # fixed wins
        both = sum(1 for i in common_ids if oriented[i] and fixed[i])
        neither = sum(1 for i in common_ids if not oriented[i] and not fixed[i])
        p = mcnemar_exact_p(b, c)
        print(f"  {obj:10s} both_succeed={both} both_fail={neither} "
              f"oriented_only={b} fixed_only={c}  p={p:.4f}"
              + ("  (discordant pairs=0, test undefined)" if b + c == 0 else ""))


if __name__ == "__main__":
    main()
