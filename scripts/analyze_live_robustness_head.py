#!/usr/bin/env python3
"""
Paired significance test for the live robustness-head candidate-selection
comparison (results/risk_gated_vla/object_extension/live_robustness_head_eval_base1901).

LIVE_ROBUSTNESS_HEAD_RESULT.md reports raw per-object success counts for
geometry/point/neighborhood-mean/robustness-head but explicitly flags itself
as incomplete: "should be analyzed with paired tests before being promoted
to a headline claim." This script closes that gap, reusing this project's
canonical McNemar implementation (scripts/paired_stats.py::mcnemar_test) so
the test matches the methodology used everywhere else in this project
(C4_4OBJ_RESULT.md, PEAR_OOD_RESULT.md, ROBUST05_RESULT.md).

Pairing unit: each (scene, perturbation_type) is one paired trial -- the
same scene/perturbation, but each method's own selected candidate is
executed independently under it. This matches how ROBUST05_RESULT.md pairs
its five perturbations per scene.

Usage:
    conda run -n tango python scripts/analyze_live_robustness_head.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_stats import mcnemar_test  # noqa: E402

SCENES_PATH = Path("results/risk_gated_vla/object_extension/live_robustness_head_eval_base1901/scenes.jsonl")
OUT_PATH = Path("results/risk_gated_vla/object_extension/LIVE_ROBUSTNESS_HEAD_PAIRED_RESULT.md")

OBJECTS = ["cracker", "mustard", "drill", "tomato_soup_can"]
COMPARATORS = ["geometry", "point", "mean"]
TARGET = "robustness_head"


def load_pairs():
    """Returns pairs[object][comparator] = list of (comparator_success, robustness_head_success)."""
    pairs = {obj: {c: [] for c in COMPARATORS} for obj in OBJECTS}
    for line in SCENES_PATH.read_text().splitlines():
        scene = json.loads(line)
        obj = scene["object"]
        sel = scene["selected"]
        by_method_pert = {}
        for o in scene["outcomes"]:
            if o["candidate_idx"] == sel.get(o["method"]):
                by_method_pert[(o["method"], o["perturbation_type"])] = int(o["success"])
        pert_types = sorted(set(pt for (_, pt) in by_method_pert.keys()))
        for pt in pert_types:
            head_val = by_method_pert.get((TARGET, pt))
            if head_val is None:
                continue
            for comp in COMPARATORS:
                comp_val = by_method_pert.get((comp, pt))
                if comp_val is not None:
                    pairs[obj][comp].append((comp_val, head_val))
    return pairs


def main():
    pairs = load_pairs()
    lines = [
        "# Live Robustness-Head Paired Significance Test",
        "",
        "Paired exact McNemar test (this project's canonical implementation, "
        "`scripts/paired_stats.py::mcnemar_test`) applied to the raw counts in "
        "`LIVE_ROBUSTNESS_HEAD_RESULT.md`. Pairing unit: (scene, perturbation) "
        "-- 30 scenes x 5 perturbations = 150 paired trials per object per "
        "comparator, matching the n=150 already reported there.",
        "",
        "| Object | Comparator | n01 (head wins) | n10 (comparator wins) | McNemar p | Direction |",
        "|---|---|---:|---:|---:|---|",
    ]
    pooled = {c: [] for c in COMPARATORS}
    for obj in OBJECTS:
        for comp in COMPARATORS:
            obj_pairs = pairs[obj][comp]
            pooled[comp].extend(obj_pairs)
            n01, n10, p, _ = mcnemar_test(obj_pairs)
            direction = ("head" if n01 > n10 else "comparator" if n10 > n01 else "tie")
            sig = "*" if p < 0.05 else ""
            lines.append(f"| {obj} | {comp} | {n01} | {n10} | {p:.4f}{sig} | {direction} |")
    lines.append("")
    lines.append("**Pooled (all four objects, n=600 per comparator):**")
    lines.append("")
    lines.append("| Comparator | n01 (head wins) | n10 (comparator wins) | McNemar p | Direction |")
    lines.append("|---|---:|---:|---:|---|")
    for comp in COMPARATORS:
        n01, n10, p, _ = mcnemar_test(pooled[comp])
        direction = ("head" if n01 > n10 else "comparator" if n10 > n01 else "tie")
        sig = "*" if p < 0.05 else ""
        lines.append(f"| {comp} | {n01} | {n10} | {p:.4f}{sig} | {direction} |")
    lines.append("")
    lines.append("`*` = p < 0.05, uncorrected. Apply Holm correction across the "
                  "3 comparators x 4 objects (+pooled) family before citing as "
                  "significant in the paper, matching this project's existing "
                  "practice (C4_4OBJ_RESULT.md).")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
