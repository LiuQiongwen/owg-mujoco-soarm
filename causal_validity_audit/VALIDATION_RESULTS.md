# Auto-Tagger Stage 1 Validation Results

**Status: Stage 1 complete (2026-08-02), extended same day after Stage 2 contact with real code.** Per
`AUDIT_TOOL_VALIDATION_PLAN.md`, this is the prerequisite evidence needed before Stage 2 (auditing real
public codebases) can be interpreted at all — without it, a clean scan of public code is uninterpretable
(clean code vs. a detector that misses everything look identical).

**Update**: auditing a real public repo (Sim-Grasp) surfaced a field-defining pattern
(`container[...]["field"] = value`, subscript assignment) that the original 11-category suite did not
cover because `analyze_function` could not parse it at all. Extended the tool
(`_subscript_field_name()` + a third branch in `analyze_function`) and added category 12 to re-validate
the extension didn't regress anything already passing. This is the correct order or operations — Stage 1
evidence should be current with the tool being run against Stage 2 targets, not stale.

## Headline numbers

n = 30 labeled fields, 14 fixture functions, 12 categories (`causal_validity_audit/test_fixtures/`,
ground truth in `test_fixtures/ground_truth.json`, runner in `run_validation_suite.py`).

| Metric | Value |
|---|---:|
| Accuracy | 0.933 |
| Precision (EXECUTION_DERIVED as positive class) | 0.889 |
| Recall | 0.889 |
| F1 | 0.889 |
| False positive rate | 0.048 |

Confusion matrix:

|  | predicted EXECUTION_DERIVED | predicted PRE_EXECUTION |
|---|---:|---:|
| true EXECUTION_DERIVED | 8 | 1 |
| true PRE_EXECUTION | 1 | 20 |

Both disagreements are **documented, expected limitations, not surprises** — both were specifically
designed to characterize known boundaries of the tool's static heuristics, not accidents:

1. **Category 8 (expected false negative)**: the tool misses execution-derived fields when the
   physical-actuation handle is passed under any variable name other than exactly `env` (e.g.
   `sim`, `self.env`, a renamed parameter). Root cause, confirmed by reading the source directly:
   both `_reads_env_state()` and `_is_env_step_call()` hardcode `base.id == "env"` /
   `f.value.id == "env"` — there is no aliasing or type-based resolution, only an exact variable
   name match.
2. **Category 9 (expected false positive)**: the tool over-flags a static, construction-time
   configuration read (`env.config.gripper_width`) as execution-derived, because its heuristic
   ("any attribute chain rooted at a variable named `env` is a live-state read") cannot distinguish
   static config from live physical state — both are just `env.*` attribute access syntactically.

## A methodological note worth keeping: two of my own hypotheses about category 8 were wrong, and
only running the suite caught it

This is itself relevant evidence for how seriously to trust a clean scan later, so it's recorded
here rather than smoothed over. The original design intent for category 8 was "the tool misses
execution-derived fields when the entry METHOD NAME isn't in
`DEFAULT_EXECUTION_ENTRY_METHODS = {'step', 'put_obj_in_tray', 'step_simulation'}`" — e.g.
`env.advance_physics(...)`. Two fixture attempts using an unrecognized method name on a variable
still named `env` both **passed** (the tool correctly caught them), which on inspection turned out
to be for a reason unrelated to the intended test: `_reads_env_state()` walks an entire expression
looking for *any* attribute chain rooted at a variable named `env`, regardless of the specific
method or attribute name — so `env.advance_physics(...)` is already indistinguishable from
`env.step(...)` to this check, and `DEFAULT_EXECUTION_ENTRY_METHODS` never actually gets exercised
in the code path I assumed it did. Reading `_is_env_step_call()` and `_reads_env_state()` directly
(not just their docstrings/comments) revealed the real, narrower condition both checks share: the
hardcoded variable name `env`. The fixture was rewritten a third time around that actual mechanism
and, on that attempt, correctly reproduces the miss. **Lesson for Stage 2**: do not trust a
docstring's stated rationale for a heuristic without reading the exact AST condition it compiles
to — the same discipline this whole audit tool exists to enforce on other people's code turned out
to apply to characterizing this tool itself.

## Per-category results

| Category | Description | Result |
|---:|---|---|
| 1 | Clean pre-execution, direct assignment | 4/4 correct |
| 2 | Directly execution-derived | 2/2 correct |
| 3 | Multi-hop taint propagation | 2/2 correct |
| 4 | Post-commit reassignment (real `grasp_yaw` bug pattern) | 1/1 correct |
| 5 | Dead constant, misleading comment (real `dz`/`dz_lift`/`need_dz` pattern) | 2/2 correct |
| 6 | Pre-marker "settle" step using the same method name as real execution | 1/1 correct |
| 7 | Field-name collision (real `yaw`/`grasp_yaw` masking-bug pattern) | 2/2 correct |
| 8 | Aliased/renamed environment handle | 1/2 correct — **documented miss** |
| 9 | Static config read via `env.*` | 1/2 correct — **documented false positive** |
| 10 | Interprocedural taint via a known-entry-method helper | 2/2 correct |
| 11 | Baseline padding (mixed true positive/negative) | 8/8 correct |
| 12 | Subscript-assignment field writes (`d[a][b]["field"]=v`, found via Sim-Grasp) | 2/2 correct |

## What this licenses for Stage 2

The tool is sound on every mechanism this project's own real historical bugs actually exercised
(reassignment, comment-vs-code mismatch, name collision, interprocedural propagation, multi-hop
taint) — categories 4, 5, 7, 10 all pass, and these are not synthetic hypotheticals, they are
direct reconstructions of bugs already found and fixed in this project's own pipelines. It has two
narrow, now-precisely-characterized failure modes (variable-name aliasing false negatives,
static-config false positives) that should be stated as explicit limitations in the paper
regardless of what Stage 2 finds, and checked for specifically when auditing real code — e.g., if a
target codebase (LIBERO, CALVIN, a LeRobot training script) passes its simulator/environment handle
under a name other than `env` anywhere in the analyzed function or its helpers, this tool's silence
on that code is not evidence of cleanliness.

## Files

- `causal_validity_audit/test_fixtures/fixture_{01-11}_*.py` — labeled fixtures
- `causal_validity_audit/test_fixtures/ground_truth.json` — ground truth labels + documented
  expected outcomes
- `causal_validity_audit/run_validation_suite.py` — runner (re-run any time the tagger changes:
  `python3 causal_validity_audit/run_validation_suite.py`)
