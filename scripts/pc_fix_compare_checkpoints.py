#!/usr/bin/env python3
"""Compare old_shared vs corrected_local object_counterfactual checkpoints on
the SAME dev-test (base-seed 200) scenes, offline-re-scored against the real
oracle_per_candidate sweep (every candidate in every dev-test scene was
physically executed in MuJoCo during collection -- this is "offline
re-scored" in exactly the sense results/risk_gated_vla/final_report.md
already uses for its Global BCE / Object-relative BCE rows: a real,
physically recorded outcome for a candidate chosen post-hoc rather than
live at collection time). Also reports the live-executed geometry baseline
from the same collection, matching the paper's own table format. Uses exact
McNemar (scripts/risk_gated_vla_phase1_stats.py's own implementation) paired
at the scene level between the two checkpoint variants' picks -- perfectly
paired since both score the identical scenes/candidates, so MuJoCo's
~0.6-1% marginal non-determinism (final_report.md) cannot be a confound
between them (unlike a live-vs-live comparison would risk).

CONFIRMATORY BATCH (base-seed 300) IS NEVER TOUCHED BY THIS SCRIPT.

Part of the pc_fix evidence chain backing paper_risk_gated_vla.tex's
sec:pcfix. Produces results/risk_gated_vla/pc_fix_devtest_base200/
comparison_results.json's exact numbers (n=90, corrected_local 48/90 vs
old_shared 36/90, exact McNemar p=0.0227) when re-run against the
checkpoints and dev-test data already committed under results/risk_gated_vla/
pc_fix_*/.

Usage:
    conda run -n tango python scripts/pc_fix_compare_checkpoints.py \\
        results/risk_gated_vla/pc_fix_devtest_base200/scenes.jsonl \\
        results/risk_gated_vla/pc_fix_ckpts_old_shared \\
        results/risk_gated_vla/pc_fix_ckpts_corrected_local \\
        /tmp/comparison_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from world_model.train_counterfactual_critic import feature, load_ensemble, OBJECTS
from scripts.risk_gated_vla_phase1_stats import mcnemar_exact

DEVTEST_PATH = Path(sys.argv[1])          # results/.../pc_fix_devtest_base200/scenes.jsonl
OLD_SHARED_CKPT_DIR = Path(sys.argv[2])   # .../pc_fix_ckpts_old_shared
CORRECTED_CKPT_DIR = Path(sys.argv[3])    # .../pc_fix_ckpts_corrected_local
OUT_JSON = Path(sys.argv[4])

VARIANT = "object_counterfactual"


def score_and_pick(rec, ensemble, relative=True):
    x = torch.tensor(
        [feature(rec, c, relative) for c in rec["oracle_per_candidate"]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        scores = np.stack([
            model((x - mean) / std).sigmoid().numpy() for model, mean, std in ensemble
        ]).mean(0)
    return int(np.argmax(scores))


scenes = [json.loads(line) for line in DEVTEST_PATH.read_text().splitlines()]
print(f"Loaded {len(scenes)} dev-test scenes from {DEVTEST_PATH}")

old_ens = load_ensemble(OLD_SHARED_CKPT_DIR, VARIANT)
new_ens = load_ensemble(CORRECTED_CKPT_DIR, VARIANT)
print(f"old_shared ensemble: {len(old_ens)} members; corrected_local ensemble: {len(new_ens)} members")

rows = []
for rec in scenes:
    cands = rec["oracle_per_candidate"]
    idx_old = score_and_pick(rec, old_ens)
    idx_new = score_and_pick(rec, new_ens)
    success_old = bool(cands[idx_old]["success"])
    success_new = bool(cands[idx_new]["success"])
    success_geo_live = bool(rec["outcomes"]["geometry"]["success"])
    idx_geo_live = rec["outcomes"]["geometry"]["candidate_idx"]
    rows.append({
        "object": rec["object"], "seed": rec["seed"],
        "idx_old_shared": idx_old, "success_old_shared": success_old,
        "idx_corrected_local": idx_new, "success_corrected_local": success_new,
        "idx_geometry_live": idx_geo_live, "success_geometry_live": success_geo_live,
        "picks_differ_old_vs_new": idx_old != idx_new,
    })

n = len(rows)
n_old = sum(r["success_old_shared"] for r in rows)
n_new = sum(r["success_corrected_local"] for r in rows)
n_geo = sum(r["success_geometry_live"] for r in rows)
n_diff = sum(r["picks_differ_old_vs_new"] for r in rows)

succ_old = [r["success_old_shared"] for r in rows]
succ_new = [r["success_corrected_local"] for r in rows]
succ_geo = [r["success_geometry_live"] for r in rows]

m_old_vs_new = mcnemar_exact(succ_old, succ_new)
m_new_vs_geo = mcnemar_exact(succ_geo, succ_new)
m_old_vs_geo = mcnemar_exact(succ_geo, succ_old)

print()
print(f"n = {n} dev-test scenes (base-seed 200, IDENTICAL scenes to the paper's own published "
      f"dev-test row -- deterministic replay via scene_seed(base_seed, object, scene_idx))")
print()
print(f"Geometry (live-executed, paper's own reported baseline was 30/90=33.3%):"
      f"  {n_geo}/{n} ({100*n_geo/n:.1f}%)")
print(f"old_shared checkpoint (offline re-scored, the pre-fix bug):"
      f"      {n_old}/{n} ({100*n_old/n:.1f}%)")
print(f"corrected_local checkpoint (offline re-scored, the fix):"
      f"     {n_new}/{n} ({100*n_new/n:.1f}%)")
print()
print(f"Top-1 candidate CHANGED between old_shared and corrected_local in "
      f"{n_diff}/{n} scenes ({100*n_diff/n:.1f}%)")
print()
print(f"McNemar old_shared vs corrected_local: {m_old_vs_new}")
print(f"McNemar geometry(live) vs corrected_local: {m_new_vs_geo}")
print(f"McNemar geometry(live) vs old_shared:      {m_old_vs_geo}")

print()
print("Per-object breakdown:")
for obj in OBJECTS:
    obj_rows = [r for r in rows if r["object"] == obj]
    n_o = len(obj_rows)
    print(f"  {obj:>8}: geo={sum(r['success_geometry_live'] for r in obj_rows)}/{n_o}  "
          f"old_shared={sum(r['success_old_shared'] for r in obj_rows)}/{n_o}  "
          f"corrected_local={sum(r['success_corrected_local'] for r in obj_rows)}/{n_o}")

OUT_JSON.write_text(json.dumps({
    "n_scenes": n, "n_geometry_live": n_geo, "n_old_shared": n_old, "n_corrected_local": n_new,
    "n_picks_differ": n_diff,
    "mcnemar_old_vs_new": m_old_vs_new,
    "mcnemar_geometry_vs_new": m_new_vs_geo,
    "mcnemar_geometry_vs_old": m_old_vs_geo,
    "rows": rows,
}, indent=2))
print(f"\nWrote {OUT_JSON}")
