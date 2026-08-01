# Bug report: `ALL_FIELDS` registry collision silently mis-classifies `dz`

**Status:** documented only, not fixed in this task (per instruction —
`causal_validity_audit/` is untouched).
**Severity:** the audit gate currently fails closed (rejects) for a field it
should accept — a false positive, not a false negative. No causally-invalid
feature is being silently admitted anywhere as a result of this bug; the
practical effect is that `train_lggsn_pairwise.py` cannot currently be
imported/run at all (its own module-level `audit_feature_set` call raises).

## Root cause

`causal_validity_audit/provenance.py:239`:

```python
ALL_FIELDS = {**SOARM_FIELDS, **PIPER_FIELDS, **WORLD_MODEL_FIELDS}
```

Python dict-merge lets the last operand win on a key collision.
`SOARM_FIELDS["dz"]` (line 85) is `PRE_EXECUTION`:

```python
"dz": FieldSpec(
    Provenance.PRE_EXECUTION,
    "verified (2026-07-16): hardcoded constant 0.0 in both the live "
    "inference path and the training-log writer -- technically does "
    "not leak future information (it never varies with execution "
    "outcome at all), but is a DEAD/uninformative feature, a distinct "
    "problem from causal-validity leakage -- flag separately, do not "
    "conflate with genuine leakage",
),
```

`WORLD_MODEL_FIELDS["dz"]` (line 235, registered later — 2026-07-30, for
the unrelated Risk-Gated VLA world-model MLP critic pipeline,
`data/transition_logger.py`) is `EXECUTION_DERIVED`:

```python
"dz": FieldSpec(Provenance.EXECUTION_DERIVED, "obj_pos_after[2] - obj_pos_before[2] -- requires execution to have happened"),
```

Both entries are individually correct **for their own pipeline** — the two
pipelines happen to use the field name `dz` for two semantically different
quantities (a verified-constant placeholder in the SO-ARM101/LGGSN
pipeline; a genuine post-execution position delta in the world-model
pipeline). The merge silently picks the second, unconditionally, for every
caller that uses the default registry.

## Affected call sites

Every call site in the repository uses the default `registry=None -> ALL_FIELDS`
— none passes an explicit `registry=` argument (confirmed via
`grep -rn "audit_feature_set(" --include="*.py" .`):

- `train_lggsn_pairwise.py:53` — the LGGSN training script whose
  `FEATURE_COLS` this bug currently blocks entirely.
- `tango_robot/piper_robosuite/stage2_train_embodiment_lggsn.py:40-41`
- `tango_robot/piper_robosuite/stage2_train_embodiment_lggsn_v2.py:41-43`
- `tests/test_risk_gated_vla_phase1.py`, `scripts/risk_gated_vla_phase1_eval.py`,
  `causal_validity_audit/retrospective_audit.py` — these do not reference
  `dz` in their feature lists at the moment, so they are not currently
  affected by *this specific* collision, but they are equally exposed to
  the same class of bug for any other field name that collides across
  `SOARM_FIELDS`/`PIPER_FIELDS`/`WORLD_MODEL_FIELDS` in the future.

`CAUSAL_VALIDITY_METHOD.md`'s own checklist records
`train_lggsn_pairwise.py`'s gate as verified-passing at the time it was
wired in (2026-07-30, same date `WORLD_MODEL_FIELDS` was added) — "gate
passes (all 14 columns confirmed admissible ... dz ...)". Nothing in that
file documents this collision as a known, already-caught issue, which is
why this report treats it as a new finding rather than a rediscovery.

## Suggested scope for a separate fix (not implemented here)

Any of the following would resolve the class of bug, not just this one
instance:

1. **Namespaced registries** — key every field by `(pipeline, field_name)`
   instead of bare `field_name`, so `("soarm", "dz")` and
   `("world_model", "dz")` cannot collide. Requires updating every call
   site to pass its pipeline identifier, which is a larger, more invasive
   change but the most structurally sound.
2. **Require an explicit registry per caller** — remove the `ALL_FIELDS`
   default entirely (or keep it only for genuinely cross-pipeline callers
   that intentionally want the union), and require
   `train_lggsn_pairwise.py`, `stage2_train_embodiment_lggsn*.py`, etc. to
   each pass their own pipeline-specific registry (`SOARM_FIELDS`,
   `PIPER_FIELDS`, ...) explicitly. Smallest code change per call site, but
   relies on every future call site remembering to do this correctly — no
   structural guarantee.
3. **Collision detection at composition time** — keep `ALL_FIELDS` as a
   convenience union, but raise at import time (or in a dedicated test) if
   any two of `SOARM_FIELDS`/`PIPER_FIELDS`/`WORLD_MODEL_FIELDS` define the
   same key with different `Provenance` values, forcing a human to
   consciously resolve the collision (e.g. by renaming one field or
   explicitly namespacing it) rather than silently picking one. Cheapest to
   add, and would have caught this exact bug the moment
   `WORLD_MODEL_FIELDS` was registered.

A combination of (2) as the immediate fix (restore
`train_lggsn_pairwise.py`'s gate to `registry=SOARM_FIELDS`, matching what
`CAUSAL_VALIDITY_METHOD.md` already documents this call site as needing)
plus (3) as a permanent regression guard seems like the smallest change
that both fixes the immediate problem and prevents recurrence — but that
tradeoff call belongs to whoever owns `causal_validity_audit/`, not to this
task.
