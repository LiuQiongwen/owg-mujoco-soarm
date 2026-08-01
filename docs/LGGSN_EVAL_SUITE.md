# LGGSN standalone checkpoint evaluator — design notes

Companion to `research_agent_pilots/lggsn_suite/` (evaluator) and
`experiments/lggsn_suite/` (TANGO publish/verify specs). Written before any
code, per the task requirement to document exact semantics before
implementing.

## Why this exists, and why it does not go through MVP4's restricted execution

`torch` cannot run inside the MVP4 restricted-execution harness on this
machine: `libtorch_cpu.so` alone is 428MB, and just `import torch` fails to
mmap that shared library under `research_agent/restricted_subprocess.py`'s
fixed `RLIMIT_AS` of 1GB (verified directly: a bare `import torch`, no other
code, fails with `ImportError: libtorch_cpu.so: failed to map segment from
shared object` under rlimits matching the harness exactly). `research_agent`
is not modified in this task (explicit instruction), so this is a permanent
constraint for this suite, not a bug to fix here.

Per the approved architecture: the evaluator (`evaluator.py`) is a
**standalone, unrestricted** subprocess — it runs with the OS's normal
resource limits, outside MVP4 entirely. TANGO's role is downstream:
validating the experiment matrix/specification, and afterward verifying and
reporting on the artifacts the evaluator already produced. Every report and
log line from this suite says explicitly which stage a number came from —
**never** implying the torch computation happened inside MVP4's sandbox.

## Why the evaluator never imports `train_lggsn_pairwise.py`

`train_lggsn_pairwise.py:53` calls
`audit_feature_set(FEATURE_COLS, context=...)` at **module import time**,
using the default registry (`causal_validity_audit.provenance.ALL_FIELDS`).
That registry currently mis-classifies `dz` as `EXECUTION_DERIVED` — a
merge-order bug where `WORLD_MODEL_FIELDS["dz"]` (a different pipeline's
field, registered later) silently overwrites `SOARM_FIELDS["dz"]` (verified
`PRE_EXECUTION`, a hardcoded constant in the live LGGSN pipeline) in
`ALL_FIELDS = {**SOARM_FIELDS, **PIPER_FIELDS, **WORLD_MODEL_FIELDS}`. See
`docs/CAUSAL_VALIDITY_REGISTRY_BUG.md` for the full writeup. This task does
not fix, bypass, or monkeypatch that gate. Since fixing it is out of scope
and bypassing it is forbidden, the evaluator simply never imports the file
that calls it — importing `lggsn_model.LGGSN` (the model class, which has no
audit gate at all) is sufficient for scoring an already-trained checkpoint,
and the grouping/pairing logic below is reimplemented cleanly (it is
ordinary data orchestration, not the LGGSN algorithm — every training
script from v3 through v11 already reimplements a slightly different
version of it; this is not a departure from how the codebase already
treats that logic).

## Exact grouping/pairing semantics being replicated

Read (never imported) from `train_lggsn_pairwise.py:76-168`
(`load_episodes` / `build_pairs`), commit `2dc82b3` (unchanged since, see
provenance report). This is the semantic contract the evaluator's
`eval_core.py` must reproduce exactly for the four verified checkpoints to
be a faithful evaluation of what they were trained against.

1. **Episode key**: rows are grouped by `(query, scene_id)`. Each group is
   one "episode."
2. **Episode-level label (majority vote)**: for each episode, count
   `n_pos = sum(row["label"] for row in episode)`,
   `n_neg = len(episode) - n_pos`.
   - If `n_pos == n_neg`: the episode is **tied** and excluded from
     pairing entirely (`train_lggsn_pairwise.py:100-101`,
     `continue  # tied / ambiguous — skip`). The original script does this
     silently; this evaluator counts every excluded episode
     (`malformed_group_count`) and every row inside one
     (`skipped_row_count`) instead of dropping them unaccounted-for.
   - Otherwise `episode_label = 1 if n_pos > n_neg else 0`.
3. **Per-candidate feature rows, NOT filtered by individual label**: every
   row in the episode contributes one feature vector to that episode's
   bucket (`train_lggsn_pairwise.py:120`,
   `feats = [[c[f] for f in FEATURE_COLS] for c in cands]` — `cands` is
   every row in the episode, unconditionally). A candidate whose own
   `label` is 0 inside an `episode_label == 1` episode still contributes
   its feature row to the "pos" bucket. This looks surprising but is the
   verified, actual behavior of the script that trained these checkpoints
   — the evaluator must reproduce it exactly, not the (more intuitive but
   different) per-row-label filtering.
4. **Pairing (train_lggsn_pairwise.py:154-161, `cartesian`)**: for every
   query, and for every (positive episode, negative episode) pair within
   that query, form the full cross product of that positive episode's
   candidate rows × that negative episode's candidate rows. A query with
   $P$ positive episodes and $N$ negative episodes, each episode having on
   average $k$ candidates, produces on the order of $P \times N \times
   k^2$ pairs — not one pair per candidate.
5. **Deterministic train/val split, at the EPISODE level, per query**
   (`train_lggsn_pairwise.py:139-152`): one `random.Random(seed)` instance
   (seed = 42) is created once and reused, in order, across every query.
   For each query (iterated in the order queries were first encountered
   while scanning the file — see below): `rng.shuffle(pos_episodes)`, then
   `rng.shuffle(neg_episodes)`, then
   `n_val = max(1, round(len(episodes) * 0.2))` splits each list into
   train/val. This is deterministic given (a) the seed, (b) the exact
   query-iteration order, and (c) the exact per-episode candidate order —
   all three must be reproduced bit-for-bit, not just the random seed
   alone, for the split to match.
6. **Iteration order is insertion order, not sorted**: `ep_rows` and
   `ep_by_query` are both built by scanning the JSONL file top to bottom
   and inserting into a `dict`/`defaultdict` (Python 3.7+ preserves
   insertion order). A query's position in the iteration is determined by
   the **first row of the first episode of that query** encountered while
   scanning — not alphabetical, not by episode count. The evaluator's own
   grouping code must use the same "first-seen" insertion-order structure
   (plain `dict`, populated by a single top-to-bottom scan), not a sorted
   or hashed structure, or the split will silently diverge from what
   `train_lggsn_pairwise.py` would have produced.
7. **No filtering by which candidates are "eligible" beyond the tied-episode
   exclusion above** — every row in a non-tied episode is used.

## Metrics semantics chosen (evaluator-side decisions, not present in the
## original training script, added for this suite's reporting needs)

- **`positive_score_mean` / `negative_score_mean`**: mean model score over
  each **unique scored candidate row** (scored once, not once per pair)
  belonging to a positive/negative episode respectively — not weighted by
  how many pairs that row participates in. A per-pair mean would just be a
  multiplicity-weighted version of the same quantity and is less
  interpretable; not computed here.
- **`mean_score_margin` / `median_score_margin`**: computed over the full
  **pair** distribution (`score[pos] - score[neg]` for every eligible
  pair), matching `pair_accuracy`'s own denominator. This one *is*
  naturally weighted by pairing multiplicity, by design — it is a
  statistic of "the pair distribution actually evaluated," not of unique
  candidates.
- **`ties_count`**: pairs where `score[pos] == score[neg]` exactly. The
  original script's accuracy computation (`run_epoch`,
  `(logit_pos > logit_neg).sum()`) counts a tie as "not correct" (strict
  `>`); this evaluator preserves that exact rule for `pair_accuracy` but
  additionally reports `ties_count` for transparency (a reporting
  addition, not a semantic change to the accuracy figure itself).
- **Bootstrap CI for `pair_accuracy`**: resampled at the **episode** level
  (with replacement, same seed-derived RNG, fixed number of resamples),
  not at the pair level — pairs sharing an episode are not independent
  observations (a single episode's candidates appear in many pairs), so a
  naive pair-level bootstrap would understate the true interval width via
  pseudo-replication. Each resample reconstructs its own pair set from the
  resampled episodes and recomputes `pair_accuracy` from cached
  (already-computed) per-candidate scores — no re-scoring, so this is cheap
  and adds no additional model-forward-pass nondeterminism.
- **Per-query breakdown**: `pair_accuracy` computed the same way,
  restricted to pairs whose query matches.

## What is explicitly NOT computed

No AUC/F1/precision/recall: this dataset's "label" is an episode-outcome
majority vote turned into a pairwise preference target, not an independent
binary classification target with a meaningful single-example decision
threshold — those metrics would not have a well-defined sampling
interpretation here without inventing one, so they are omitted rather than
computed on a shaky basis.
