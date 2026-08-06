# Leave-one-object-out generalization test

Attempted to resolve the point-cloud/object-identity collinearity found in
`results/risk_gated_vla/onehot_ablation/` (point-cloud-only and one-hot-only
feature variants perform identically, p=1.0 between them, on the paper's
own 3-object in-distribution population) by breaking the collinearity
structurally: train on two objects only, evaluate on the third, entirely
unseen during training. If point-cloud geometry carries real
per-candidate signal independent of object identity, it should still beat
pose-only on a genuinely novel object, since no identity crutch is even
available in this setting (object one-hot is omitted, not zeroed, for
these variants).

**Data**: same real population as the onehot ablation and `sec:pcfix` --
`pc_fix_train_base100/scenes.jsonl` (120 scenes, base-seed=100, split
per fold into 80 training scenes from the 2 non-held-out objects) and
`pc_fix_devtest_base200/scenes.jsonl` (90 scenes, base-seed=200; each
fold evaluates only on its held-out object's 30 scenes). No new MuJoCo
collection. Confirmatory batch (base-seed 300) never touched.

## Method

Two feature variants (`pose_only`, `pose_pc` -- no one-hot variant, since
an identity feature is undefined/uninformative for a class never seen in
training), `object_counterfactual` architecture, 5 seeds x 400 epochs per
(held-out object, variant) combination = 3 folds x 2 variants x 5 seeds =
30 checkpoints. Pooled across all three leave-one-out folds for n=90,
matching the original dev-test population size, but now every prediction
comes from a model that never saw that scene's object during training.

## Result: the test did not resolve the collinearity question in the

hoped direction -- it surfaced a more fundamental limitation instead.

| Method | Pooled (n=90) | vs. geometry (exact McNemar) |
|---|---|---|
| Geometry (live-executed) | 37/90 (41.1%) | -- |
| `pose_only` (leave-one-out) | 24/90 (26.7%) | p=0.0533 (favoring geometry) |
| `pose_pc` (leave-one-out) | 27/90 (30.0%) | p=0.1102 (favoring geometry) |
| `pose_only` vs. `pose_pc` | -- | p=0.7283 (still indistinguishable) |

Per-object breakdown (held-out object / pose_only / pose_pc / geometry,
out of 30 each): cracker 6/4/2, mustard 4/15/22, drill 14/8/13.

## Honest reading

Both leave-one-out variants substantially *underperform* geometry when
generalizing to an unseen object -- the opposite of the in-distribution
result, and consistent with (an independent replication of, via a
different mechanism) `paper_tro.tex` `sec:fourobj`'s zero-shot pear
finding ("every critic variant loses to geometry"). Critically,
`pose_only` and `pose_pc` remain statistically indistinguishable from
each other even with object identity structurally eliminated (p=0.73,
nearly unchanged from the in-distribution p=1.0). This means the
in-distribution tie between point-cloud and object-identity features is
**not resolved by this test** -- we cannot conclude point-cloud carries
real, object-identity-independent generalizable signal, because neither
variant generalizes to unseen objects at all in this setting. The most
defensible reading: the critic's in-distribution gain over geometry is
substantially explained by something that does not transfer to a new
object (most plausibly some mixture of memorized per-object priors and
genuinely in-distribution-only geometric patterns), not by object-general
geometric reasoning. This is a real, non-clean negative result, reported
with the same rigor as the paper's other honest negatives -- not a
resolution of the original question, but a more precise characterization
of its limits.

## Reproducing

```bash
conda run -n tango python scripts/pc_fix_leave_one_object_out.py \
    --train results/risk_gated_vla/pc_fix_train_base100/scenes.jsonl \
    --devtest results/risk_gated_vla/pc_fix_devtest_base200/scenes.jsonl \
    --out-dir /tmp/loo_verify --epochs 400 --seeds 5
```

Full per-fold detail and pairwise McNemar in
`leave_one_object_out_summary.json`.
