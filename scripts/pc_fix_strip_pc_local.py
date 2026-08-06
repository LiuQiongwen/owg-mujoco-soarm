#!/usr/bin/env python3
"""Strip pc_stats_local from every candidate in a scenes.jsonl, producing a
data file where world_model.train_counterfactual_critic.feature()'s existing
fallback (cand.get("pc_stats_local", rec["pc_stats_before"])) reproduces the
OLD, pre-fix shared-scene-stat behavior exactly -- pure data transform, no
re-simulation. Used to train a directly comparable "old_shared" checkpoint
from the identical scenes/labels as the "corrected_local" checkpoint, with
the point-cloud-fix as the only variable.

Part of the pc_fix evidence chain backing paper_risk_gated_vla.tex's
sec:pcfix -- see results/risk_gated_vla/pc_fix_*/ for the data/checkpoints
this produced and pc_fix_compare_checkpoints.py for the comparison step.

Usage:
    conda run -n tango python scripts/pc_fix_strip_pc_local.py \\
        results/risk_gated_vla/pc_fix_train_base100/scenes.jsonl \\
        /tmp/train_base100_old_shared.jsonl
"""
import json
import sys

in_path, out_path = sys.argv[1], sys.argv[2]
n = 0
with open(in_path) as fin, open(out_path, "w") as fout:
    for line in fin:
        rec = json.loads(line)
        for cand in rec["oracle_per_candidate"]:
            cand.pop("pc_stats_local", None)
        fout.write(json.dumps(rec) + "\n")
        n += 1
print(f"Stripped pc_stats_local from {n} scenes: {in_path} -> {out_path}")
