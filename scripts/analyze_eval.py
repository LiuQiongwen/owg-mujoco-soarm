#!/usr/bin/env python3
"""
Parse quick_eval.sh output logs and compute:
  - per-object and overall SR
  - 95% Wilson CI
  - two-proportion z-test p-value between two conditions
  - per-object breakdown table

Usage:
  python scripts/analyze_eval.py baseline.log cfm.log
"""

import re, sys
from math import sqrt
from scipy import stats as scipy_stats

OBJECTS = ["Banana","TomatoSoupCan","Pear","MustardBottle","Scissors","CrackerBox","PowerDrill"]

def parse_log(path: str) -> dict:
    """Return {object: {"ok": int, "total": int}} and overall."""
    results = {o: {"ok": 0, "total": 0} for o in OBJECTS}
    current_obj = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            for obj in OBJECTS:
                if f"] {obj} seed=" in line:
                    current_obj = obj
                    results[obj]["total"] += 1
                    if line.startswith("[✓]"):
                        results[obj]["ok"] += 1
                    break
    total_ok    = sum(v["ok"]    for v in results.values())
    total_total = sum(v["total"] for v in results.values())
    results["_ALL"] = {"ok": total_ok, "total": total_total}
    return results

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI."""
    if n == 0:
        return 0.0, 0.0
    p  = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half   = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, centre - half), min(1, centre + half)

def two_prop_z(k1, n1, k2, n2):
    """Two-sided z-test for difference in proportions. Returns (z, p)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1 = k1 / n1;  p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return float("nan"), float("nan")
    z_stat = (p1 - p2) / se
    p_val  = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
    return z_stat, p_val

def main():
    if len(sys.argv) < 3:
        print("Usage: analyze_eval.py <baseline.log> <cfm.log>"); sys.exit(1)
    base_log, cfm_log = sys.argv[1], sys.argv[2]
    base = parse_log(base_log)
    cfm  = parse_log(cfm_log)

    print(f"\n{'='*78}")
    print(f"{'Object':<14} {'Base SR':>8} {'CFM SR':>8} {'Δpp':>7} "
          f"{'z':>7} {'p':>8}  {'Base CI':>16}  {'CFM CI':>16}")
    print(f"{'-'*78}")

    for obj in OBJECTS + ["_ALL"]:
        bk, bn = base[obj]["ok"], base[obj]["total"]
        ck, cn = cfm[obj]["ok"],  cfm[obj]["total"]
        if bn == 0 or cn == 0:
            print(f"  {obj:<12}  no data"); continue

        bp = bk / bn;  cp = ck / cn
        delta_pp = (cp - bp) * 100
        z_stat, p_val = two_prop_z(ck, cn, bk, bn)
        blo, bhi = wilson_ci(bk, bn)
        clo, chi = wilson_ci(ck, cn)
        sig = "**" if p_val < 0.05 else ("*" if p_val < 0.10 else "")

        label = f"{'ALL':>14}" if obj == "_ALL" else f"  {obj:<12}"
        print(f"{label}  {bp:>7.1%}  {cp:>7.1%}  {delta_pp:>+6.1f}pp  "
              f"{z_stat:>6.2f}  {p_val:>7.4f}{sig:2s}  "
              f"[{blo:.3f},{bhi:.3f}]  [{clo:.3f},{chi:.3f}]")

    bk, bn = base["_ALL"]["ok"], base["_ALL"]["total"]
    ck, cn = cfm["_ALL"]["ok"],  cfm["_ALL"]["total"]
    z_stat, p_val = two_prop_z(ck, cn, bk, bn)
    print(f"{'='*78}")
    print(f"\nBaseline : {bk}/{bn} = {bk/bn:.1%}")
    print(f"OT-CFM   : {ck}/{cn} = {ck/cn:.1%}  Δ={((ck/cn)-(bk/bn))*100:+.1f}pp")
    print(f"z = {z_stat:.3f},  p = {p_val:.4f}  "
          f"({'SIGNIFICANT at α=0.05' if p_val<0.05 else 'not significant at α=0.05'})")
    print(f"\n** p<0.05   * p<0.10")

if __name__ == "__main__":
    main()
