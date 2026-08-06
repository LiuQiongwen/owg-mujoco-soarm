# Point-cloud candidate-level feature fix: evidence chain

Backs `paper_risk_gated_vla.tex`'s `sec:pcfix` ("Post-hoc finding: a
candidate-level point-cloud feature defect"). Committed 2026-08-06,
force-added despite `results/` being gitignored (same precedent as
`results/risk_gated_vla/final_report.md`, `audit.md`, etc.) — this data
previously existed only on local disk in one session's environment, not
backed by git anywhere, which is a real reproducibility gap for a cited
result. Fixing that gap is the entire purpose of this commit.

## What's here

- `pc_fix_train_base100/` — training data, base-seed=100, 40 scenes/object
  x 3 objects (cracker/mustard/drill) = 120 scenes, collected via the
  unmodified `scripts/risk_gated_vla_phase1_eval.py`. Includes
  `pc_stats_local` automatically (the fix lives in `run_scene()`'s data
  path).
- `pc_fix_devtest_base200/` — dev-test data, base-seed=200, 30
  scenes/object x 3 = 90 scenes, same collection script. Same base seed as
  the paper's own originally-published dev-test row, but **not** a literal
  replay of it — the codebase has drifted since the original 2026-07-30
  collection (live geometry baseline here is 37/90=41.1%, vs. the
  originally published 30/90=33.3%). `comparison_results.json` is the exact
  computed output backing `sec:pcfix`'s numbers.
- `pc_fix_ckpts_corrected_local/`, `pc_fix_ckpts_old_shared/` — two
  `object_counterfactual` (+ `global_bce`/`object_bce`) ensembles, 5 seeds
  each, 400 epochs, trained via the real, unmodified
  `world_model/train_counterfactual_critic.py` on the identical
  `pc_fix_train_base100/scenes.jsonl`. The only difference: `old_shared`
  had `pc_stats_local` stripped before training (via
  `scripts/pc_fix_strip_pc_local.py`), so `feature()`'s existing fallback
  reproduces the pre-fix shared-stat behavior exactly. This isolates the
  point-cloud fix as the only variable between the two checkpoint sets.

## Reproducing `comparison_results.json` from scratch

```bash
# 1. Regenerate the old_shared training data (pure data transform, no re-simulation)
conda run -n tango python scripts/pc_fix_strip_pc_local.py \
    results/risk_gated_vla/pc_fix_train_base100/scenes.jsonl \
    /tmp/train_base100_old_shared.jsonl

# 2. Re-run the comparison against the already-trained checkpoints committed here
conda run -n tango python scripts/pc_fix_compare_checkpoints.py \
    results/risk_gated_vla/pc_fix_devtest_base200/scenes.jsonl \
    results/risk_gated_vla/pc_fix_ckpts_old_shared \
    results/risk_gated_vla/pc_fix_ckpts_corrected_local \
    /tmp/comparison_results.json
```

Verified bit-identical to the committed `comparison_results.json` before
this commit was made.

To retrain the checkpoints themselves from scratch (not just re-score them):

```bash
conda run -n tango python world_model/train_counterfactual_critic.py \
    --data results/risk_gated_vla/pc_fix_train_base100/scenes.jsonl \
    --out-dir /tmp/pc_fix_ckpts_corrected_local --epochs 400 --seeds 5

conda run -n tango python world_model/train_counterfactual_critic.py \
    --data /tmp/train_base100_old_shared.jsonl \
    --out-dir /tmp/pc_fix_ckpts_old_shared --epochs 400 --seeds 5
```

## Headline numbers (from `comparison_results.json`)

n=90 dev-test scenes: `corrected_local` 48/90 (53.3%) vs. `old_shared`
36/90 (40.0%), exact McNemar p=0.0227 (18 discordant wins for the fix vs.
6 against); `corrected_local` also beats live-executed geometry (37/90,
p=0.0347) while `old_shared` does not (p=1.0, statistically
indistinguishable from geometry).

## What this is not

Not the confirmatory batch (base-seed 300 was never touched, per the
paper's own single-read discipline — see `final_report.md`). Not a
replacement for the paper's Table I (`sec:positiveresult`), which was
collected before this fix existed. See `sec:pcfix` in
`paper_risk_gated_vla.tex` for the full, appropriately-hedged writeup.
