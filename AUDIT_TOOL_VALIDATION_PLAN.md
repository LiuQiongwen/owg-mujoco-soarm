# Causal-Validity Auto-Tagger Validation Plan

**Status (2026-08-02): Stage 1 COMPLETE.** Results: `causal_validity_audit/VALIDATION_RESULTS.md` —
n=28 labeled fields, accuracy 0.929, precision/recall/F1 0.875/0.875/0.875, FPR 0.05, two documented
(not surprising) failure modes: variable-handle aliasing false negatives, static-config false positives.
**Stage 2 (public-codebase audit) may now proceed** — the prerequisite evidence this plan required exists.
Do not re-run Stage 1's design from scratch; extend `test_fixtures/` with new categories if a genuinely new
failure mode is hypothesized, don't redo the whole suite.

## Why this exists

The causal-validity audit tool (`causal_validity_audit/provenance.py` + `auto_tagger.py`) is the strongest
candidate contribution in this project (`results/risk_gated_vla/` counterfactual-critic paper,
`TRO_PAPER_OUTLINE.md` §4), but its one identified weakness is that it has only ever been validated on this
project's own pipelines — a reviewer will read that as a sophisticated internal postmortem, not evidence of
a general tool. This plan fixes that, in the order an external GPT-5.4 xhigh review insisted on: prove the
tool is sound and general on KNOWN cases first, before trusting what it reports on unknown public code.

## Stage 1: Labeled test suite (do this first, no public code yet)

### Goal

Measure `auto_tagger.py`'s per-field classification accuracy, precision/recall/F1 for `EXECUTION_DERIVED`
detection, and false-positive rate, against a small set of synthetic toy pipeline functions with
**human-assigned ground truth**, independent of the tool.

### What to build

1. **`causal_validity_audit/test_fixtures/`** (new directory): 15-20 small, self-contained Python functions,
   each written to mimic a realistic pattern from a robotics grasp-candidate pipeline (a marker call, a
   return dict of named fields), one function per file or grouped logically. Each fixture needs a companion
   ground-truth label (a `.json` or inline comment block mapping `field_name -> "PRE_EXECUTION" |
   "EXECUTION_DERIVED"`, decided by the person writing the fixture, NOT by running the tool).

2. **Test case categories to cover** (design each category from a DIFFERENT known failure pattern — do not
   write 20 variations of the same easy case):

   | # | Category | What it tests | Ground truth |
   |---|---|---|---|
   | 1 | Clean pre-execution feature, direct assignment | Baseline true negative | PRE_EXECUTION |
   | 2 | Directly execution-derived (reads env state after an `env.step(...)` call post-marker) | Baseline true positive | EXECUTION_DERIVED |
   | 3 | Multi-hop taint (feature computed via 2-3 intermediate variables chained from a tainted value) | Transitive taint propagation | EXECUTION_DERIVED |
   | 4 | Variable reassigned post-commit (assigned once before the marker admissibly, then REASSIGNED after, mirroring the real `grasp_yaw` bug found in this project — `CAUSAL_VALIDITY_METHOD.md`/`IDEA_REPORT.md` Direction 2's third addendum) | Does the tool track reassignment, not just first-assignment? | EXECUTION_DERIVED (post-reassignment value) |
   | 5 | Dead/constant feature with a misleading docstring (returns a hardcoded 0.0, but a comment nearby implies it's execution-derived — mirrors the real `dz`/`dz_lift`/`need_dz` case, which was initially misjudged by trusting a stale comment about a different dataset) | Does the tool correctly ignore prose and trust the actual code path? | PRE_EXECUTION (despite misleading comment) |
   | 6 | Physics "settle" step before the marker (calls the SAME method name as a real execution step, e.g. `env.step(...)`, but occurs BEFORE the marker) | Does the tool correctly respect marker placement rather than flagging any `env.step` call anywhere in the function? | PRE_EXECUTION |
   | 7 | Field-name collision (two different fields with similar/overlapping names, one admissible one not — mirrors the real `"yaw"` vs `"grasp_yaw"` masking bug in `retrospective_audit.py`) | Does the tool correctly disambiguate by exact name, not fuzzy/partial match? | Mixed — one PRE_EXECUTION, one EXECUTION_DERIVED |
   | 8 | Unknown execution-entry method name (calls something semantically equivalent to `env.step` but NOT in `DEFAULT_EXECUTION_ENTRY_METHODS = {"step", "put_obj_in_tray", "step_simulation"}`, e.g. `env.advance_physics(...)`) | Documents a KNOWN, expected miss (the tool's own honest limitation, per its source comments) — this should be reported as a limitation, not silently ignored | EXECUTION_DERIVED (tool will likely mis-tag PRE_EXECUTION — record this as a documented false negative, not a bug in the test) |
   | 9 | Attribute-chain read on a variable named `env` that is actually static config, not live physical state (e.g. `env.config.gripper_width`, not `env.data.qpos`) | False-positive stress test — the tool's heuristic is "any attribute chain rooted at `env`", which may over-flag | Ground truth PRE_EXECUTION; if tool flags EXECUTION_DERIVED, record as a false positive |
   | 10 | Nested function calling an execution-touching helper transitively (tests the "fixed point over module-level defs" interprocedural analysis, not just direct calls) | Interprocedural taint propagation | EXECUTION_DERIVED |

   Add 5-10 more fixtures as straightforward positive/negative baseline padding once the 10 categories above
   are covered, to get a reasonable sample size for the precision/recall numbers to mean something (aim for
   ~20-30 total labeled fields across all fixtures, not 10).

3. **Runner script** (`causal_validity_audit/run_validation_suite.py`): loads each fixture, calls
   `tag_file(path, function_name)` (or `resolve_parameter_provenance` where relevant), compares
   `TagResult.field_provenance` against the ground-truth label file, and outputs:
   - Overall accuracy, precision/recall/F1 for `EXECUTION_DERIVED` as the positive class
   - A confusion matrix
   - A per-category breakdown (which of the 10 categories above pass/fail)
   - Explicit list of every disagreement (fixture, field, tool's answer, ground truth, category)

### What "done" looks like for Stage 1

A report (`causal_validity_audit/VALIDATION_RESULTS.md` or similar) stating the measured precision/recall/F1
and confusion matrix, an honest discussion of every category the tool fails on (especially category 8,
which is an EXPECTED, already-documented limitation — report it as such, don't try to silently fix the
tool's entry-method list to make the number look better unless that fix is itself principled and general).
This report is what makes Stage 2's results interpretable at all — without it, a clean scan of public code
is uninterpretable (clean code vs. a tool that misses everything look identical).

## Stage 2: Public codebase audit (only after Stage 1's report exists)

### Candidates to audit — REVISED 2026-08-02, original list was mis-scoped

**Original list (LIBERO / CALVIN / LeRobot) does not fit this tool's target bug class and should not be
pursued as-is.** Investigated LeRobot directly (`/lena/projects/lerobot`, real HuggingFace clone) as the
first attempt: `envs/libero.py`'s `step()`/`_format_raw_obs()` and `rl/gym_manipulator.py`'s
`step_env_and_process_transition()` are both standard state→action RL/IL control-loop code — `env.step()`
called, THEN the resulting observation/transition assembled from it. That is normal RL semantics (the
policy's next decision correctly uses the post-action observation), not a causal-validity violation. The
underlying reason: **this tool's target bug class specifically requires a candidate-POOL-then-SCORE-then-
EXECUTE architecture** (a fixed set of candidates evaluated before one is chosen and executed — LGGSN,
this project's counterfactual critic, and GraspGen's discriminator all have this shape). LIBERO/CALVIN/
LeRobot are direct state→action RL/IL frameworks with no candidate pool at all — there is structurally no
place for this specific leakage pattern to occur, independent of whether the code is otherwise clean.
(Secondary, practical issue: LeRobot's `create_transition(...)` custom constructor isn't a pattern
`analyze_function` currently recognizes anyway — only `return {...}` dict literals and `dict(...)` calls
are, per its own docstring.)

**Corrected candidate class: grasp-candidate scoring/reranking/discriminator codebases specifically**, not
general RL/IL frameworks. Leading candidate: **GraspGen** (arXiv:2507.13097) — already cited in
`paper_tro.tex` §4.1 as an external system independently confirmed to follow this project's admissibility
criterion implicitly (discriminator inputs are point cloud + pose only, labels are execution-derived and
exempt) — check whether GraspGen's actual training code is public and, if so, whether that "compliant by
construction" claim (currently based on the paper's stated design, not a code-level static audit) holds up
under this tool's static analysis, which would be a stronger, code-verified version of an already-partially-
made claim rather than starting from zero. Other candidates: any public 6-DoF grasp-candidate discriminator/
reranker with released training code (search needed — GraspNet-1Billion's baseline scorer, Contact-GraspNet,
or similar pointwise/pairwise grasp-quality-prediction repos are architecturally the right shape; verify
public code availability before committing to one).

**GraspGen's code is confirmed public (2026-08-02): `https://github.com/NVlabs/GraspGen`, official NVLabs
release, cloned to `/lena/projects/GraspGen`.** Investigated `grasp_gen/grasp_server.py::
score_grasps_with_discriminator` — architecturally clean by direct code inspection (builds a `data = {...}`
dict from point cloud + grasp poses only, no physical-execution call of any kind in this function or,
searched across the whole repo, ANYWHERE in it — no MuJoCo/PyBullet/Isaac-style `.step()` calls found
outside renderer/USD-scene-creation scripts). **Conclusion: GraspGen's released code has no closed-loop
execution boundary at all — it is a pure grasp-pose generation+scoring library, with physical execution
(if any) happening in a downstream system not included in this repo.** This independently corroborates, at
the code level (not just from the paper's stated design), `paper_tro.tex` §4.1's existing citation of
GraspGen as compliant — but there is no marker to place and no automated tagger run to perform, since there
is no execution boundary within this codebase's scope to test against.

**Found a better target with a real closed-loop boundary: Sim-Grasp**
(`github.com/junchengli1/Sim-Grasp`, cloned to `/lena/projects/Sim-Grasp`) —
`grasp_sampling/grasp_simulation.py::main_simulation_loop` is a genuine candidate-pool → Isaac-Sim physical
execution → label-write loop (`new_candidates[obj]["grasp_samples"][idx]["simulation_quality"] = 1`, based
on `world.step(...)`-driven physics). Two real findings from this one function:
1. **Confirms Stage 1's category-8 aliasing limitation is real-world-relevant, not hypothetical**: uses
   `world.step(...)` (Isaac Sim naming convention), not `env.step(...)` — the tagger's hardcoded
   `base.id == "env"` check misses this on contact with real code, independent of the synthetic fixture.
2. **Surfaced a genuinely new limitation Stage 1 did not anticipate**: the label is written via subscript
   assignment (`d[a][b][c] = v`), a THIRD field-defining pattern beyond `return {...}` and `dict(k=v)` —
   `analyze_function` could not see it at all (zero fields extracted) before being extended.

**Tool extended same day (2026-08-02)**: added `_subscript_field_name()` + a third branch in
`analyze_function` recognizing `container[...]["field"] = value`, taking the innermost string-constant
subscript as the field name. Added fixture 12 to the Stage 1 suite to test this pattern in isolation (clean
`env`-named handle, so it doesn't conflate with category 8's aliasing issue) — **suite re-run, still
accuracy 0.933/precision-recall-F1 0.889 each, category 12 passes 2/2, no regressions on any of the
previously-passing 11 categories** (`VALIDATION_RESULTS.md` updated accordingly). Re-ran `tag_file` directly
against the real, unmodified Sim-Grasp file: `field_provenance` now correctly extracts
`{'simulation_quality': 'PRE_EXECUTION'}` (previously extracted nothing at all) — but `marker_found: False`,
because Sim-Grasp's code obviously has no `CAUSAL_VALIDITY_COMMIT_POINT()` marker (that is this project's
own convention, not something external code has) — **without a marker, `committed` never becomes True and
every field defaults to PRE_EXECUTION regardless of actual taint, so this specific run's PRE_EXECUTION
result is not yet meaningful evidence of anything** (it is NOT yet a confirmed false negative in the
Stage-2 sense — it's a demonstration that the extraction mechanism now works, nothing more).

**Next deliberate step, not yet done**: placing a marker in `main_simulation_loop` requires actually reading
that loop's control flow carefully to identify the correct "execution has now started" line (candidates for
the placement line: around the `world.step(render=...)` calls, or where the pose-check against
`pose_ori` begins) — this is exactly the "one human judgment call per codebase" `commit_marker.py`'s design
calls for, and should be done on a LOCAL, clearly-marked-as-modified copy for analysis purposes, never by
editing the actual cloned repo as if it were this project's own code. Once marked and manually verified, if
`simulation_quality` still resolves PRE_EXECUTION despite genuinely depending on `world.step`-driven
physics, THAT would be the real, reportable finding (the aliasing miss, demonstrated end-to-end on real
code) — worth writing up carefully and, per the plan's own Stage 2 rule, privately disclosed to Sim-Grasp's
maintainer before any publication, not published as a "gotcha."

**CORRECTION, same day, before any marker was placed — caught by manual verification exactly as this plan's
own rules require**: on closer reading, `main_simulation_loop` is NOT a candidate-scoring function at all.
Tracing the call graph, the actual physical grasp action (`articulation_controller.apply_action(actions)`,
computed via IK/pick-place `controller.forward(...)`) happens in a separate function,
`handle_action_for_env` (line 77), called BEFORE `main_simulation_loop` runs. By the time
`main_simulation_loop` executes, the candidate has already been selected and the grasp already commanded —
this function's job is purely to run a post-grasp stress test (lower the supporting surface, see if the
object stays attached) and write the resulting outcome as a LABEL. Per this project's own causal-validity
criterion, **labels are explicitly exempt from the PRE_EXECUTION requirement** (`provenance.py`'s own
labels-are-exempt corollary) — `simulation_quality` being execution-derived is correct and expected, not a
violation. Treating this as a "found bug" would have been a false, unsupportable claim. **Do not pursue a
marker in `main_simulation_loop` — it is a label-writing function, not a candidate-selection function, and
is out of scope for this audit by design, not by oversight.**

## Stage 2 status as of 2026-08-02 — consolidated summary

What was actually accomplished, stated plainly:

1. **GraspGen** (`/lena/projects/GraspGen`): code-level-verified clean (no execution boundary in the
   released repo at all — pure scoring/generation library). Strengthens the existing `paper_tro.tex` §4.1
   citation with direct code inspection rather than trusting the paper's stated design alone. No marker
   possible, no automated tagger run performed or needed.
2. **Sim-Grasp** (`/lena/projects/Sim-Grasp`): drove one real, validated tool extension (subscript-
   assignment field recognition, now part of the Stage 1 suite as category 12, no regressions). Its
   candidate-pool→execute→label loop (`main_simulation_loop`) was correctly ruled OUT as an audit target
   after tracing the call graph — it writes labels, which are exempt by this project's own criterion, not
   candidate-selection features. Its actual live-scoring/inference code (`inference_utils.py`,
   `sim_grasp_demo.py::evaluate_suction_model`) returns tuples/score-arrays, not a dict-based pattern the
   tagger currently recognizes — auditing it would require a further, more open-ended tool extension not
   undertaken this session.
3. **No confirmed real-world violation found yet** in either target — but that is not the main Stage 2
   result this session actually produced. The main result is methodological: (a) the subscript-assignment
   extension, validated end-to-end against real code that motivated it; (b) a caught near-miss where a
   label-writing function was almost misreported as a candidate-scoring violation, corrected by exactly the
   manual-verification discipline this plan's own rules require — itself a legitimate, honest finding worth
   a sentence in the paper's external-validation section (demonstrates the audit *process*, not just the
   tool, catching a mistake before publication).

**If Stage 2 is resumed later**: the next concrete, well-scoped step is extending `analyze_function` to
recognize tuple-returning score functions (or picking a different target whose live-scoring path already
uses a dict pattern) — do not restart the search for a target from scratch, use this session's ruled-out
candidates (GraspGen: no boundary; Sim-Grasp's label loop: out of scope by design) to avoid repeating them.

For each: find the function(s) that assemble the feature vector/observation used for policy training or
candidate scoring, place a marker per `commit_marker.py`'s convention (this requires one human judgment call
per codebase — which line is "execution has now started" — document that judgment explicitly, it is itself
a claim someone could disagree with), run the tagger, and manually verify any flagged
`EXECUTION_DERIVED` field by reading the surrounding code — do not trust the automated tag alone for a
claim this consequential.

### If a real violation is found

1. Privately disclose to the maintainers before any publication (standard, ethically correct practice for
   this kind of finding — do not publish a "gotcha" about someone else's code without giving them a chance
   to respond or fix it first).
2. Test the impact: does removing/fixing the leaked feature and re-evaluating change reported numbers? This
   is the strongest possible evidence, mirroring this project's own retrospective demonstration on its own
   pipeline.
3. Write up as the external-validation section of the counterfactual-critic/audit paper (`TRO_PAPER_OUTLINE.md`
   §4, or `paper_risk_gated_vla_draft.md`'s eventual merge target) — NOT as a separate paper, per the
   external review's explicit recommendation, unless the finding turns out to generalize across multiple
   independent systems with consequential, conclusion-changing impact (an upside case to notice, not the
   plan to bet on).

### If no violation is found

Report it honestly as a negative/clean result — but ONLY alongside Stage 1's detection-rate evidence, so a
reader can distinguish "this codebase is clean" from "the tool didn't look hard enough." A clean scan
without Stage 1's report is not worth including in the paper at all.

## Known limitation to state honestly regardless of outcome

Per the external review: pure static analysis cannot always establish the *actual* runtime path in dynamic
Python (dynamic dispatch, monkey-patching, config-driven branching, etc.). A hybrid static + runtime-taint/
dataflow trace would be a more defensible design than static analysis alone. This project's tool is
static-only. State this as a limitation in the paper regardless of how Stage 1/2 turn out — do not let a
clean Stage 1 report imply the tool is unconditionally sound.
