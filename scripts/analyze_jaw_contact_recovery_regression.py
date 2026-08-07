"""Step 3E input: does the recovery result survive the corrected jaw geometry?

Reads the paired step-3D runs (same checkpoint, seeds, templates, ranker and
protocol; only `--jaw-contact-model` differs) and reports what actually decides
whether the recovery work needs regenerating:

  1. does the variant ORDER hold -- baseline < r0_regrasp_only < r1_plus_attached_lift
  2. per-scene label flip rates, paired by (object, seed)
  3. where the flips sit: policy behaviour before intervention, or recovery after

Trials are paired by construction: `--base-seed` and scene index determine both
the spawn and the RNG, so row i of one arm is the same scene as row i of the other.

Usage:
  conda run -n tango python scripts/analyze_jaw_contact_recovery_regression.py \
      --dir outputs/step3d
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ARMS = ["proxy_spheres", "measured_pads_aimed"]
VARIANT_ORDER = ["r0_regrasp_only", "r1_plus_attached_lift"]


def load(dirpath: Path) -> dict:
    runs = {}
    for f in sorted(dirpath.glob("recovery_*.json")):
        d = json.loads(f.read_text())
        cfg = d["config"]
        runs[(cfg["object"], cfg["jaw_contact_model"])] = d
    return runs


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on discordant pairs."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/step3d")
    args = ap.parse_args()
    runs = load(Path(args.dir))

    objects = sorted({o for o, _ in runs})
    missing = [(o, a) for o in objects for a in ARMS if (o, a) not in runs]
    if missing:
        raise SystemExit(f"missing runs: {missing}")

    print("=" * 78)
    print("system success rate by variant and jaw contact model")
    print("=" * 78)
    header = f"{'object':10s} {'arm':20s} {'baseline':>9}"
    for v in VARIANT_ORDER:
        header += f" {v:>22}"
    print(header)

    pooled = {a: {"n": 0, "baseline": 0,
                  **{v: 0 for v in VARIANT_ORDER}} for a in ARMS}
    for o in objects:
        for a in ARMS:
            s = runs[(o, a)]["summary"]
            n = s["n"]
            pooled[a]["n"] += n
            pooled[a]["baseline"] += s["baseline_act_successes"]
            row = (f"{o:10s} {a:20s} "
                   f"{s['baseline_act_successes']:4d}/{n:<4d}")
            for v in VARIANT_ORDER:
                c = s["variants"][v]["successes"]
                pooled[a][v] += c
                row += f" {c:17d}/{n:<4d}"
            print(row)

    print("\npooled")
    for a in ARMS:
        p = pooled[a]
        row = f"{'':10s} {a:20s} {p['baseline']:4d}/{p['n']:<4d}"
        for v in VARIANT_ORDER:
            row += f" {p[v]:17d}/{p['n']:<4d}"
        print(row)

    print("\n" + "=" * 78)
    print("does the variant ordering hold?  (baseline <= r0 <= r1)")
    print("=" * 78)
    for a in ARMS:
        p = pooled[a]
        seq = [p["baseline"]] + [p[v] for v in VARIANT_ORDER]
        ok = all(x <= y for x, y in zip(seq, seq[1:]))
        print(f"  {a:20s} {seq}  ->  {'HOLDS' if ok else 'BROKEN'}")

    print("\n" + "=" * 78)
    print("per-scene label flips, paired by (object, scene index)")
    print("=" * 78)
    flips = Counter()
    total = 0
    transitions = Counter()
    for o in objects:
        ra = runs[(o, ARMS[0])]["rows"]
        rb = runs[(o, ARMS[1])]["rows"]
        if len(ra) != len(rb):
            raise SystemExit(f"{o}: row count differs {len(ra)} vs {len(rb)}")
        for x, y in zip(ra, rb):
            total += 1
            if x["baseline_act_success"] != y["baseline_act_success"]:
                flips["baseline_act_success"] += 1
            if x["intervention_reason"] != y["intervention_reason"]:
                flips["intervention_reason"] += 1
                transitions[f"{x['intervention_reason']} -> {y['intervention_reason']}"] += 1
            for v in VARIANT_ORDER:
                if x["variants"][v]["system_success"] != y["variants"][v]["system_success"]:
                    flips[v] += 1
    for k in ["baseline_act_success", "intervention_reason"] + VARIANT_ORDER:
        f = flips[k]
        print(f"  {k:24s} {f:3d}/{total}  = {100*f/max(total,1):5.1f}%")

    print("\nintervention-reason transitions (legacy -> corrected)")
    for t, c in transitions.most_common():
        print(f"  {c:3d}  {t}")

    print("\n" + "=" * 78)
    print("paired McNemar on the selected variant (r1_plus_attached_lift)")
    print("=" * 78)
    b = c = 0
    for o in objects:
        for x, y in zip(runs[(o, ARMS[0])]["rows"], runs[(o, ARMS[1])]["rows"]):
            xs = x["variants"]["r1_plus_attached_lift"]["system_success"]
            ys = y["variants"]["r1_plus_attached_lift"]["system_success"]
            if xs and not ys:
                b += 1
            elif ys and not xs:
                c += 1
    print(f"  legacy-only successes  b = {b}")
    print(f"  corrected-only successes c = {c}")
    print(f"  exact two-sided p = {mcnemar_exact(b, c):.4f}")

    # The frozen spec's PRIMARY comparison is r1 vs r0 within an arm -- i.e. what
    # the attached-lift rule adds on top of regrasp.  Recomputing it per arm says
    # whether the paper's own headline survives the corrected geometry.
    print("\n" + "=" * 78)
    print("the frozen spec's primary comparison, recomputed within each arm")
    print("  r1_plus_attached_lift vs r0_regrasp_only, exact two-sided McNemar")
    print("=" * 78)
    for a in ARMS:
        b = c = 0
        for o in objects:
            for r in runs[(o, a)]["rows"]:
                r0 = r["variants"]["r0_regrasp_only"]["system_success"]
                r1 = r["variants"]["r1_plus_attached_lift"]["system_success"]
                if r1 and not r0:
                    b += 1
                elif r0 and not r1:
                    c += 1
        p = pooled[a]
        gain = (p["r1_plus_attached_lift"] - p["r0_regrasp_only"]) / max(p["n"], 1)
        print(f"  {a:20s} r0={p['r0_regrasp_only']:2d}/{p['n']}  "
              f"r1={p['r1_plus_attached_lift']:2d}/{p['n']}  "
              f"gain={100*gain:+5.1f}pp  discordant b={b} c={c}  "
              f"p={mcnemar_exact(b, c):.4f}")

    print("\n" + "=" * 78)
    print("intervention / recovery counts")
    print("=" * 78)
    print(f"{'object':10s} {'arm':20s} {'attached_fail':>13} {'lift_att':>9} "
          f"{'lift_succ':>10} {'regrasp_n':>10}")
    for o in objects:
        for a in ARMS:
            s = runs[(o, a)]["summary"]
            print(f"{o:10s} {a:20s} {s['attached_insufficient_lift']:13d} "
                  f"{s['attached_lift_attempts']:9d} "
                  f"{s['attached_lift_successes']:10d} "
                  f"{s.get('fallback_attempts', 0):10d}")


if __name__ == "__main__":
    main()
