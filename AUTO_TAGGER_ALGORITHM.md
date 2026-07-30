# Automated Causal-Validity Tagging — The New Algorithm

This is the concrete new algorithm requested to strengthen the T-RO submission beyond a pure
audit/methodology framing. It automates what `provenance.py`'s manual registry required a human
to do by hand for every field, and — while building and testing it against the real, live codebase
— it caught a real contamination bug the hand-built registry had missed and had already reported
as clean.

## The problem with the manual registry

`provenance.py` required a human to trace each field's computation against the live code and
assign `PRE_EXECUTION`/`EXECUTION_DERIVED` by hand. This does not scale, and — demonstrated
empirically in this project three separate times before this algorithm existed — humans doing this
by hand make mistakes: trusting a comment instead of re-tracing code, conflating "depends on
sibling candidates" with "depends on execution," and simply not noticing a variable gets
reassigned later in a long function.

## The algorithm

**Input**: a Python source file, a target function (assumed to build and return a dict of logged
fields), and one human-placed marker statement inside that function (`commit_marker.py`) at the
line after which the currently-selected candidate has begun physically executing.

**Why one marker is still required, not zero**: a purely syntactic analyzer cannot distinguish a
physics *settle* step (letting a just-spawned object's pose stabilize — legitimately part of
pre-execution scene observation) from a genuine *this-candidate-is-now-executing* step; both are
just calls to `env.step(...)`. Telling them apart requires one bit of domain knowledge a human
supplies once per function, not once per field. This reduces the manual burden from **O(number of
logged fields)** to **O(number of physical-commit points in the codebase)** — in this codebase,
from 13+ field judgments down to a single marker placement.

**Procedure** (`causal_validity_audit/auto_tagger.py`):

1. **Build the execution-touching function set.** Fixed-point over every function defined in the
   module: a function is execution-touching if it directly calls `env.step(...)`, or calls another
   function already known to be execution-touching.
2. **Forward taint analysis over the target function's body, in program order.** Before the marker,
   nothing is tainted — arbitrarily complex pre-execution setup and candidate-selection logic is
   fine. After the marker: an assignment is tainted if its right-hand side references an
   already-tainted name, calls an execution-touching function, or reads through an attribute chain
   rooted at a variable named `env` (a live physical-state read). `if`/`for` bodies are handled
   conservatively — taint is unioned across branches / one loop pass, so ambiguity fails toward
   `EXECUTION_DERIVED`, never away from it.
3. **In-place mutation is tracked too, not just rebinding.** `x[i] = ...` and `x.attr = ...`
   targets are resolved to their base variable and taint that base unconditionally if the
   assignment occurs after the marker — found necessary during development (see below), because a
   naive tracker that only follows simple `x = ...` rebinding silently misses container mutation,
   which is a common and dangerous blind spot for exactly the fields most likely to accumulate
   execution-derived state (a `phase_log` dict built up incrementally as physical phases execute).
4. **At the function's `return {...}`**, each key's provenance is read off the tainted-variable set
   at that point.

## Validated against real, live code — and it worked

Ran against `piper_pick_and_place.py::run_pick_and_place`'s actual return dict (12 fields). Result
matched the (already twice-corrected) hand-built registry on 11 of 12 fields — and **caught a
genuine error the hand-built registry still had**: it flagged `grasp_yaw` as `EXECUTION_DERIVED`,
while the manual registry had it marked `PRE_EXECUTION`.

Tracing why: `grasp_mat` is reassigned twice in `run_pick_and_place` — once pre-commit (candidate
selection, fine) and once post-commit, at the "pre-close refresh" step, from a post-descend
`env.sim.data.xquat` read (a deliberate mid-trial drift-correction mechanism, unrelated to this
bug). `grasp_yaw` is computed from whichever value `grasp_mat` holds by the time the function
returns — the second, post-descend one. A human reading the function once sees
`grasp_mat = compute_grasp_orientation(...)` early on and stops tracking; the automated tagger
mechanically follows every reassignment, and caught what the manual pass missed.

**This bug had real downstream consequences.** `grasp_yaw` was part of the "Stage 2 CORRECTED"
`[z, yaw, H]` feature set — the *specific result this project's whole causal-validity narrative had
been citing as the clean, trustworthy baseline* (§4.2's retrospective demonstration, IDEA_REPORT.md
Direction 2's correction section). It wasn't clean. Re-running Stage 2 with a genuinely clean
`[z, H]` feature set (§ below) confirmed the qualitative finding survives — but the number itself
changed, and had already been published in this project's own documentation as verified-clean
before the automated tool caught it.

## A second bug, found while validating the first

While re-checking `retrospective_audit.py`'s own test cases against the corrected registry,
found the Piper-side historical feature sets used the generic string `"yaw"` instead of the actual
field name `"grasp_yaw"`. Because `"yaw"` is *also* a separately, correctly registered
`PRE_EXECUTION` field on the SO-ARM101 side, every Piper-side lookup of `"yaw"` silently resolved
against the wrong platform's entry instead of failing as unregistered — masking `grasp_yaw`'s
contamination in every row, including the one the whole demonstration exists to vouch for. Fixed by
using the real field name everywhere.

## Re-verification: does the core finding survive?

Re-ran Stage 2 training with the truly clean `[z, H]` feature set (dropping `grasp_yaw` from both
platforms for a fair shared feature space):

| Condition | Mean pairwise accuracy (before, contaminated) | Mean pairwise accuracy (after, clean) |
|---|---|---|
| Zero-shot | 0.8236 | 0.1327 |
| Pooled, `none` | 0.8236 | 0.1327 |
| Pooled, `additive` | 0.8200 | 0.1327 |
| Pooled, `interaction` | 0.8198 | 0.1327 |

**The qualitative finding survives**: all four conditions are still exactly identical
(diff=0.0000) — pooling still adds nothing over zero-shot, with or without the contamination.
**The absolute number changed substantially and should be reported plainly, not smoothed over**:
accuracy dropped from 0.8236 to 0.1327 (now below the 0.50 majority baseline). Verified, rather than
assumed, why: `H` (object top-surface offset) is an *exact* dataset-wide constant across this
Cracker-only Piper collection (`sigma=0` over `n=250` rows — every row is the same object, so this
was structurally guaranteed once execution-derived features were removed). `z` (`spawn_pos[2]`) is
a near-constant *within each scene*: checked all 25 collected scenes directly, 14/25 show
within-scene spread under `2e-4` (consistent with floating-point/physics-settling noise, not a real
signal), because `spawn_pos` is the object's own spawn height, read once per scene before any
candidate-specific action — it cannot, by construction, distinguish which of the 10 pooled
candidates is later attempted. Together, `[z, H]` has almost no capacity to discriminate between
different candidates drawn from the *same* scene — exactly the comparison the pairwise BPR
objective is trained and evaluated on. This is a sharper, verified explanation than "H happens to be
constant, probably that's why": the corrected feature set isn't just weak, it is structurally
near-incapable of carrying per-candidate signal for this task, given what pre-execution,
per-candidate-varying quantities actually exist in the Piper pipeline today. The one candidate
feature that genuinely varies per-candidate and is not currently tested cleanly is the original,
pre-commit grasp orientation — distinct from the execution-derived `grasp_yaw` this section
disqualified — and re-collecting data with that field logged separately is flagged as concrete
future work, not attempted in this pass.

## Why this is a real algorithmic contribution, not just a bigger diagnostic

- It is a genuine **automated inference procedure** (fixed-point call-graph analysis + forward
  taint propagation with conservative branch handling), not a lookup table.
- It has a **precise, stated limitation** (needs one marker per function, cannot fully automate
  the settle-vs-execution distinction) rather than an implicit, undiscovered one — and the
  limitation itself is theoretically motivated (Algorithm step 3 in `CAUSAL_VALIDITY_METHOD.md`'s
  formal criterion already anticipated that provenance is a property of code, not field names).
- It **found a real bug a careful human missed**, on the first real codebase it was pointed at,
  with a **downstream empirical consequence** (changed a previously-published "clean" result) —
  this is a much stronger validation story than a synthetic benchmark would be.
- It generalizes beyond this codebase: the same procedure (execution-touching call-graph +
  marker-gated forward taint) applies to any Python-based robot-learning pipeline structured
  around a `step()`-driven simulator or hardware interface, not just this project's specific field
  names.

## Second validation: a structurally different function, and a real scope boundary found

Ran against `batch_s3s4.py::_emit_lggsn_candidates` (the SO-ARM101 training-data writer) — chosen
specifically because it doesn't match the first function's shape at all: no `return {...}` literal
(it writes rows to a file as a side effect and returns nothing), and it builds each row via
`dict(x=..., y=..., ...)` — a constructor call with keyword arguments, not a dict literal. Extended
`analyze_function` to also recognize this pattern (read provenance off a `dict(...)` call's keyword
arguments, not just a `return`'s dict-literal keys).

Result: all 15 fields (`x, y, z, roll, pitch, yaw, width, score, dz, dz_lift, need_dz, H, label,
query, scene_id, candidate_idx`) came back `PRE_EXECUTION` — matching the independently-verified
ground truth for this function (established earlier by direct trace: `x/y/z/yaw/width/H/score` come
from a raw candidate tuple, `dz/dz_lift/need_dz` are hardcoded `0.0` literals). Correct, but for a
reason worth stating honestly rather than presenting as a second clean win on equal footing with
the first: **this function has no `env.step` call and no marker in it at all** — every field traces
back to its two *parameters* (`obj_grasps`, `success`), not to anything computed within the
function's own body. With `committed` never set to `True`, the tagger's taint logic never actually
exercises its core mechanism here; it produces the right answer by finding nothing to flag, not by
successfully tracing an execution boundary the way it did for `grasp_yaw` in the first function.

**Honest scope statement**: this is a single-function-scoped analyzer. It can correctly determine
whether operations *within* the analyzed function contaminate a field. It cannot determine whether
an *input parameter* already carries execution-derived taint from the calling context — that would
require interprocedural analysis, tracing into whoever calls `_emit_lggsn_candidates` and produced
`obj_grasps`/`success`. For this specific function that's fine (the caller-side provenance was
already established separately, by the direct code trace this project did earlier), but a
general-purpose version of this tool would need to either (a) extend taint-tracking across call
boundaries, or (b) require a marker/parameter-provenance annotation at every function boundary, not
just at physical-commit points. Flagged as real future work below, not silently assumed away.

## Third validation: interprocedural resolution, closing the scope limitation above

Extended the tool with `resolve_parameter_provenance()`: for a target function's parameter, find
every real call site in the module and evaluate the taint of the argument bound to that parameter
*at the call site*, using the same taint rules applied to the **calling** function's own body up to
that point — rather than assuming a parameter is safe just because nothing inside the analyzed
function contaminates it.

Two things had to be fixed to make this work on a second, real codebase rather than just the
originally-instrumented one:

1. **The physical-actuation entry-point check was hardcoded to `env.step`.** `batch_s3s4.py` (the
   PyBullet-based SO-ARM101 pipeline) uses different method names for physical actuation
   (`env.put_obj_in_tray`, `env.step_simulation`) — a hardcoded `.step`-only check silently failed
   to recognize these as execution-touching, failing *open*, the wrong direction for this tool.
   Generalized to a configurable set of entry-point method names, documented as a manually curated
   (not automatically inferred) list — an honest limitation, not a hidden one.
2. **`run_trial` (the enclosing function of `_emit_lggsn_candidates`'s real call site) had no
   commit marker of its own.** Without one, `committed` never becomes `True` in that function
   either, and the tagger — correctly, by design — flags nothing as tainted, the same behavior
   already documented for `_emit_lggsn_candidates` itself. This generalizes the earlier finding:
   interprocedural resolution doesn't reduce the marker burden to "one marker in the whole
   codebase" — it requires one marker *per function that has a genuine execution boundary*, applied
   transitively up the call chain. Placed the marker in `run_trial`, immediately before the
   `env.put_obj_in_tray(...)` call (the actual grasp/place attempt), with the same "VERIFY:
   nothing above this line has executed" discipline as the original marker.

With both fixed, ran `resolve_parameter_provenance` on `_emit_lggsn_candidates`'s real call site
(`run_trial`, line 338) for all four of its parameters:

```
obj_grasps: PRE_EXECUTION   (call in run_trial() at line 338: PRE_EXECUTION)
success:    EXECUTION_DERIVED   (call in run_trial() at line 338: EXECUTION_DERIVED)
prompt:     PRE_EXECUTION   (call in run_trial() at line 338: PRE_EXECUTION)
scene_id:   PRE_EXECUTION   (call in run_trial() at line 338: PRE_EXECUTION)
```

This matches ground truth exactly: `obj_grasps` traces back to `env.get_obj_grasps(...)` (a
candidate-generation query, not physical actuation) called well before the marker; `success` is
computed from `env.put_obj_in_tray(...)` — the actual execution call — after it. This closes the
loop on `_emit_lggsn_candidates`'s own fields too: `label = 1 if success else 0` is confirmed
execution-derived, which is exactly correct because `label` is the training **label**, exempt from
the admissibility constraint per the Corollary (§ above) — while `x, y, z, yaw, width, H, score`,
all derived from the now-confirmed-`PRE_EXECUTION` `obj_grasps`, are correctly usable as live
selection input features. The tool no longer merely reproduces a fact established by a separate
manual trace here; it now derives it.

## Remaining work

- [x] Extend taint propagation across function-call boundaries (interprocedural analysis) — done,
      see above. Scoped to call sites within the same parsed module; a call from a different file
      is invisible to this analysis (see the next item).
- [ ] Extend interprocedural resolution across file boundaries (currently single-module).
- [x] Confidence flagging — done (2026-07-17). `expr_confidence()` distinguishes "provably
      PRE_EXECUTION" (`CERTAIN`) from "no execution-touching path found, but an unanalyzable call
      makes this uncertain" (`UNCERTAIN`, with the unresolved call name(s) reported). Fails toward
      uncertain, not toward assuming safe: a call is only treated as resolved if it's a known-safe
      builtin (`KNOWN_PURE_CALLS`, a manually curated allowlist — same "honest, not hidden"
      limitation as `DEFAULT_EXECUTION_ENTRY_METHODS`), a same-module function already proven pure
      by `_find_execution_touching`'s own fixed-point pass, a nested local function, or an
      `env.*` method call (already correctly handled by the taint check's own `_reads_env_state`
      logic, so re-flagging it here would be redundant, not extra caution). Uncertainty propagates
      through variable assignments the same way taint does — caught and fixed a real bug during
      development where the first version only checked the field expression's own calls, missing
      the common case where the unresolved call happens in an earlier assignment.

      Validated on the two existing target functions: `_emit_lggsn_candidates` comes back fully
      `CERTAIN` (0 uncertain fields). `run_pick_and_place` comes back `CERTAIN` on 10 of 13 fields;
      the 3 `UNCERTAIN` ones are legitimate, not false positives — `phases` traces through
      `ik.solve_multi_seed` (a method on an object this tool doesn't special-case the way it does
      `env`), and `candidates`/`candidate_grasp_yaw` trace through `sample_candidate_pool`/
      `sample_perception_noisy_candidates`, functions imported from a *different file*
      (`piper_candidate_selection.py`) — genuinely unresolvable without the cross-file analysis
      from the item above, and correctly reported as such rather than silently assumed safe. This
      is the confidence feature doing exactly its intended job: marking the honest boundary of
      single-file analysis instead of hiding it. Full provenance verdicts (the `PRE_EXECUTION`/
      `EXECUTION_DERIVED` labels themselves) are unchanged — confidence is a strictly additive
      signal, verified via regression check against the pre-existing results.
