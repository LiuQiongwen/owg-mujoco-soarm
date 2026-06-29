#!/usr/bin/env python3
"""
Statistical comparison: DDPM + LGGSN  vs  OT-CFM + LGGSN  (175-trial evals).

Usage:
  python scripts/analyze_diffusion_vs_cfm.py \
    --cfm  logs/eval_v5d_175.log \          # OT-CFM baseline (165/175)
    --ddpm logs/eval_ddpm_175.log           # DDPM result

Outputs:
  - Per-object SR table
  - Two-proportion z-test (global)
  - Fisher's exact test (global)
  - Chi-square test per object
"""

import argparse
import math
import re
from scipy.stats import fisher_exact, chi2_contingency


OBJECTS = ["Banana", "TomatoSoupCan", "Pear", "MustardBottle",
           "Scissors", "CrackerBox", "PowerDrill"]


def parse_log(path: str) -> dict:
    """Parse quick_eval.sh log.  Returns {obj: (success, total), 'total': (S, N)}."""
    results = {}
    with open(path) as f:
        for line in f:
            m = re.search(r"---\s+(\w+):\s+(\d+)/(\d+)", line)
            if m:
                obj, s, n = m.group(1), int(m.group(2)), int(m.group(3))
                results[obj] = (s, n)
    total_s = sum(v[0] for v in results.values())
    total_n = sum(v[1] for v in results.values())
    results["__total__"] = (total_s, total_n)
    return results


def ztest_two_proportions(s1, n1, s2, n2):
    """Two-proportion z-test (two-tailed).  Returns (z, p)."""
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # two-tailed p via normal CDF approximation
    from scipy.stats import norm
    p = 2 * norm.sf(abs(z))
    return z, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfm",  default="logs/eval_cfm_175.log",
                        help="OT-CFM eval log (default: looks for eval_cfm_175.log)")
    parser.add_argument("--ddpm", default="logs/eval_ddpm_175.log",
                        help="DDPM eval log")
    # Try alternate CFM log names automatically
    import os
    args = parser.parse_args()

    # Auto-find CFM baseline log
    cfm_candidates = [
        args.cfm,
        "logs/eval_cfm_175.log",
        "logs/eval_v5d_175.log",
        "logs/eval_stage4_175.log",
    ]
    cfm_log = next((p for p in cfm_candidates if os.path.isfile(p)), None)
    if cfm_log is None:
        print("[WARN] CFM baseline log not found; showing DDPM results only.")
        print(f"  Tried: {cfm_candidates}")
        cfm_results = None
    else:
        cfm_results = parse_log(cfm_log)
        print(f"CFM  log : {cfm_log}")

    if not os.path.isfile(args.ddpm):
        print(f"[ERROR] DDPM log not found: {args.ddpm}")
        return

    ddpm_results = parse_log(args.ddpm)
    print(f"DDPM log : {args.ddpm}")
    print()

    # ── Per-object table ──────────────────────────────────────────────────────
    header = f"{'Object':<18}  {'DDPM SR':>8}"
    if cfm_results:
        header += f"  {'CFM SR':>7}  {'Δ SR':>7}  {'p (χ²)':>9}"
    print(header)
    print("-" * len(header))

    for obj in OBJECTS:
        ds, dn = ddpm_results.get(obj, (0, 0))
        line = f"  {obj:<16}  {ds}/{dn} ({ds/dn:.0%})"
        if cfm_results:
            cs, cn = cfm_results.get(obj, (0, 0))
            delta = ds/dn - cs/cn if dn and cn else float("nan")
            # chi-square for this object
            ct = [[ds, dn-ds], [cs, cn-cs]]
            try:
                _, p_chi, _, _ = chi2_contingency(ct)
                p_str = f"{p_chi:.3f}"
            except Exception:
                p_str = "n/a"
            line += f"  {cs}/{cn} ({cs/cn:.0%})  {delta:+.0%}  {p_str}"
        print(line)

    # ── Global stats ─────────────────────────────────────────────────────────
    print()
    ds, dn = ddpm_results["__total__"]
    print(f"DDPM total : {ds}/{dn} = {ds/dn:.1%}")

    if cfm_results:
        cs, cn = cfm_results["__total__"]
        print(f"CFM  total : {cs}/{cn} = {cs/cn:.1%}")
        delta = ds/dn - cs/cn
        print(f"Δ SR       : {delta:+.1%}")
        print()

        z, p_z = ztest_two_proportions(ds, dn, cs, cn)
        print(f"Two-proportion z-test : z={z:.3f}  p={p_z:.4f}")

        table = [[ds, dn-ds], [cs, cn-cs]]
        _, p_fisher = fisher_exact(table)
        print(f"Fisher exact test     : p={p_fisher:.4f}")

        sig = "***" if p_fisher < 0.001 else "**" if p_fisher < 0.01 else "*" if p_fisher < 0.05 else "n.s."
        print(f"Significance          : {sig}  (α=0.05)")


if __name__ == "__main__":
    main()
