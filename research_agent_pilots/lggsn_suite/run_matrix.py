#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runs the LGGSN feature-ablation matrix: the four verified checkpoints
(base / nodist / nozrel / full_v2) through the standalone evaluator, on the
same dataset split and pairing logic, plus the two provenance-incomplete
checkpoints (clearly labeled, not part of the core ablation comparison),
plus structured BLOCKED records for every checkpoint this evaluator cannot
score -- nothing is silently omitted.

Runs OUTSIDE MVP4, same as evaluator.py itself (see docs/LGGSN_EVAL_SUITE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT_DEFAULT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import checkpoint_registry as reg  # noqa: E402
import evaluator  # noqa: E402

CORE_MATRIX = reg.MATRIX_NAMES  # ("base", "nodist", "nozrel", "full_v2")
BONUS_PROVENANCE_INCOMPLETE = ("v1_live", "v2_phase1")
ALWAYS_BLOCKED = tuple(
    name for name in reg.CHECKPOINTS_BY_NAME
    if name not in CORE_MATRIX and name not in BONUS_PROVENANCE_INCOMPLETE
)


def run_matrix(*, repo_root: str, output_root: str) -> dict:
    os.makedirs(output_root, exist_ok=True)
    records = []

    for name in CORE_MATRIX + BONUS_PROVENANCE_INCOMPLETE:
        entry = reg.CHECKPOINTS_BY_NAME[name]
        out_dir = os.path.join(output_root, name)
        try:
            metrics = evaluator.evaluate_checkpoint(name, repo_root=repo_root, output_dir=out_dir)
            records.append({
                "checkpoint_name": name,
                "status": "EVALUATED",
                "provenance_status": entry.provenance_status.value,
                "in_core_matrix": name in CORE_MATRIX,
                "eligible_group_count": metrics["eligible_group_count"],
                "eligible_pair_count": metrics["eligible_pair_count"],
                "pair_accuracy": metrics["pair_accuracy"],
                "deterministic_digest": metrics["deterministic_digest"],
                "artifacts_dir": out_dir,
            })
        except Exception as e:
            # Even a checkpoint classified VERIFIED that fails at runtime
            # (e.g. a hash mismatch discovered only now) must be reported
            # structurally, not silently dropped from the matrix.
            records.append({
                "checkpoint_name": name,
                "status": "ERROR",
                "provenance_status": entry.provenance_status.value,
                "in_core_matrix": name in CORE_MATRIX,
                "reason": f"{type(e).__name__}: {e}",
                "artifacts_dir": out_dir,
            })

    for name in ALWAYS_BLOCKED:
        entry = reg.CHECKPOINTS_BY_NAME[name]
        record = evaluator.describe_blocked(name)
        record["in_core_matrix"] = False
        records.append(record)

    same_dataset_sha = {
        r["checkpoint_name"]: json.load(open(os.path.join(r["artifacts_dir"], "metrics.json")))["dataset_sha256"]
        for r in records if r["status"] == "EVALUATED"
    }
    distinct_dataset_shas = set(same_dataset_sha.values())

    summary = {
        "core_matrix": list(CORE_MATRIX),
        "bonus_provenance_incomplete": list(BONUS_PROVENANCE_INCOMPLETE),
        "always_blocked": list(ALWAYS_BLOCKED),
        "records": records,
        "same_dataset_split_verified": len(distinct_dataset_shas) <= 1,
        "distinct_dataset_shas_used": sorted(distinct_dataset_shas),
    }
    with open(os.path.join(output_root, "matrix_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=_REPO_ROOT_DEFAULT)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    summary = run_matrix(repo_root=args.repo_root, output_root=args.output_root)
    for r in summary["records"]:
        core_tag = "[core]" if r.get("in_core_matrix") else "      "
        if r["status"] == "EVALUATED":
            print(f"{core_tag} {r['checkpoint_name']:<14} EVALUATED   "
                  f"pair_accuracy={r['pair_accuracy']:.4f} pairs={r['eligible_pair_count']}")
        elif r["status"] == "BLOCKED":
            print(f"{core_tag} {r['checkpoint_name']:<14} BLOCKED     {r['reason'][:80]}")
        else:
            print(f"{core_tag} {r['checkpoint_name']:<14} ERROR       {r.get('reason', '')[:80]}")
    print(f"\nsame_dataset_split_verified: {summary['same_dataset_split_verified']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
