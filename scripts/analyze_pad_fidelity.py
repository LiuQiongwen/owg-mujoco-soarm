"""Analyze a pad-fidelity diagnostic collection (scripts/collect_pad_fidelity_diagnostic.py).

Produces exactly the six outputs specified for this diagnostic pass:

  1. per-trial minimum/median/final pad distances
  2. duration of bilateral geometric engagement
  3. excessive-penetration duration
  4. legacy-contact vs geometric-state confusion table
  5. per-object distributions
  6. reclassification output -- WITHOUT modifying success labels

Item 6 is a report, not a mutation: every row prints the legacy label and the
geometric_verdict side by side. Nothing in this script, or in
tango_robot/pad_fidelity.py, writes to any result file or changes any
`success`/`bilateral_contact`/`weld_triggered` value.

Usage:
  conda run -n tango python scripts/analyze_pad_fidelity.py --in outputs/pad_fidelity.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def mm(v):
    return "   n/a" if v is None else f"{v*1000:6.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/pad_fidelity.jsonl")
    args = ap.parse_args()
    rows = load(Path(args.inp))
    if not rows:
        raise SystemExit(f"no rows in {args.inp}")
    n = len(rows)
    objects = sorted({r["object"] for r in rows})

    # ── 1. per-trial minimum/median/final pad distances ─────────────────────
    print("=" * 100)
    print("1. per-trial minimum / median / final pad distances (mm)")
    print("=" * 100)
    print(f"{'scene':22s} {'min_fix':>7} {'med_fix':>7} {'fin_fix':>7} | "
          f"{'min_mov':>7} {'med_mov':>7} {'fin_mov':>7}  verdict")
    for r in rows:
        tag = f"{r['object']}/s{r['seed']}"
        print(f"{tag:22s} {mm(r['min_pad_dist_fixed_m'])} {mm(r['median_pad_dist_fixed_m'])} "
              f"{mm(r['final_pad_dist_fixed_m'])} | {mm(r['min_pad_dist_moving_m'])} "
              f"{mm(r['median_pad_dist_moving_m'])} {mm(r['final_pad_dist_moving_m'])}  "
              f"{r['geometric_verdict']}")

    # ── 2 & 3. engagement / excessive-penetration duration ──────────────────
    print("\n" + "=" * 100)
    print("2-3. duration of bilateral geometric engagement vs excessive penetration "
          "(samples; ~4 physics steps each)")
    print("=" * 100)
    print(f"{'scene':22s} {'n_samples':>9} {'engaged':>8} {'excessive':>10} "
          f"{'no_bilateral':>13} {'ambiguous':>10}")
    tot_engaged = tot_excessive = tot_no_bi = tot_ambig = tot_samples = 0
    for r in rows:
        tag = f"{r['object']}/s{r['seed']}"
        print(f"{tag:22s} {r['n_samples']:9d} {r['bilateral_engagement_samples']:8d} "
              f"{r['excessive_penetration_samples']:10d} {r['no_bilateral_samples']:13d} "
              f"{r['ambiguous_samples']:10d}")
        tot_samples += r["n_samples"]
        tot_engaged += r["bilateral_engagement_samples"]
        tot_excessive += r["excessive_penetration_samples"]
        tot_no_bi += r["no_bilateral_samples"]
        tot_ambig += r["ambiguous_samples"]
    print(f"{'TOTAL':22s} {tot_samples:9d} {tot_engaged:8d} {tot_excessive:10d} "
          f"{tot_no_bi:13d} {tot_ambig:10d}")

    # ── 4. legacy-contact vs geometric-state confusion table (trial level) ──
    # (fine-grained per-STEP confusion needs the raw sample sequence, which
    #  this collection stores only as a summary; the trial-level table below
    #  cross-checks the legacy trial-level decision against the geometric
    #  trial-level verdict, which is what item 6's reclassification uses too.)
    print("\n" + "=" * 100)
    print("4. legacy bilateral_contact vs geometric_verdict — confusion table (trial level)")
    print("=" * 100)
    conf = Counter((bool(r["legacy_bilateral_contact"]), r["geometric_verdict"]) for r in rows)
    verdicts = ["NO_ENGAGEMENT", "PLAUSIBLE_ENGAGEMENT",
               "EXCESSIVE_PENETRATION_DOMINANT", "AMBIGUOUS"]
    print(f"{'legacy_bilateral':17s} " + " ".join(f"{v:>26}" for v in verdicts))
    for legacy in (False, True):
        row = " ".join(f"{conf[(legacy, v)]:26d}" for v in verdicts)
        print(f"{str(legacy):17s} {row}")

    print("\nlegacy success vs geometric_verdict")
    conf2 = Counter((bool(r["legacy_success"]), r["geometric_verdict"]) for r in rows)
    print(f"{'legacy_success':17s} " + " ".join(f"{v:>26}" for v in verdicts))
    for legacy in (False, True):
        row = " ".join(f"{conf2[(legacy, v)]:26d}" for v in verdicts)
        print(f"{str(legacy):17s} {row}")

    n_success_excessive = sum(1 for r in rows if r["legacy_success"]
                              and r["geometric_verdict"] == "EXCESSIVE_PENETRATION_DOMINANT")
    n_success = sum(1 for r in rows if r["legacy_success"])
    print(f"\nof {n_success} legacy successes, {n_success_excessive} "
          f"({100*n_success_excessive/max(n_success,1):.1f}%) have a persistent "
          f"excessive-penetration run")

    # ── 5. per-object distributions ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("5. per-object distributions")
    print("=" * 100)
    print(f"{'object':16s} {'n':>3} {'legacy_succ':>11} | " +
          " ".join(f"{v.split('_')[0]:>10}" for v in verdicts) +
          f" | {'med_min_fix_mm':>14} {'med_min_mov_mm':>14}")
    for o in objects:
        rs = [r for r in rows if r["object"] == o]
        counts = Counter(r["geometric_verdict"] for r in rs)
        med_fix = statistics.median(r["min_pad_dist_fixed_m"] for r in rs
                                    if r["min_pad_dist_fixed_m"] is not None)
        med_mov = statistics.median(r["min_pad_dist_moving_m"] for r in rs
                                    if r["min_pad_dist_moving_m"] is not None)
        print(f"{o:16s} {len(rs):3d} {sum(r['legacy_success'] for r in rs):11d} | " +
              " ".join(f"{counts.get(v, 0):10d}" for v in verdicts) +
              f" | {med_fix*1000:14.1f} {med_mov*1000:14.1f}")

    # ── 6. reclassification output — legacy labels untouched ────────────────
    print("\n" + "=" * 100)
    print("6. reclassification (report only — legacy fields are echoed, never modified)")
    print("=" * 100)
    disagree = [r for r in rows
               if bool(r["legacy_success"]) and r["geometric_verdict"] != "PLAUSIBLE_ENGAGEMENT"]
    print(f"{len(disagree)}/{n} legacy-success trials do NOT have a clean "
          f"PLAUSIBLE_ENGAGEMENT geometric verdict:")
    print(f"{'scene':22s} {'legacy_success':>14} {'legacy_bilateral':>17} "
          f"{'geometric_verdict':>32}")
    for r in disagree:
        tag = f"{r['object']}/s{r['seed']}"
        print(f"{tag:22s} {str(r['legacy_success']):>14} "
              f"{str(r['legacy_bilateral_contact']):>17} {r['geometric_verdict']:>32}")

    print(f"\n(no result file was modified; this table is the entire effect of "
          f"reclassification -- {n} trials read, 0 written)")


if __name__ == "__main__":
    main()
