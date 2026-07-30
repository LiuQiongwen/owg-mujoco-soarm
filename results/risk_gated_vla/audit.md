# Phase 0 Audit — Causal Validity & Evaluation-Design Review of `world_model/`

**Date**: 2026-07-30 (session date; repo clock unreliable for exact time — see `logs/`)
**Repo HEAD at audit time**: `45e4c874e99f65a49e597c3db32d6728e3947a3d` (clean working tree)
**Scope**: `world_model/train_mlp_predictor.py`, `world_model/rerank_grasps.py`,
`scripts/collect_mujoco_transitions.py`, `scripts/eval_wm_reranking_full.py`,
`data/transition_logger.py`, `results/run_full_01/` (the source of `wm_reranking_results.md`'s
headline +15.6pp claim), and cross-checked against `CAUSAL_VALIDITY_METHOD.md`,
`AUTO_TAGGER_ALGORITHM.md`, `RULED_OUT_METHODS.md`, `IDEA_REPORT.md`.
**Method**: read every line of the files above; traced actual code paths (not comments/docs) per
`CAUSAL_VALIDITY_METHOD.md`'s own stated discipline; reproduced the seeding logic standalone;
spot-checked the claim against the real `results/run_full_01/results.csv`.
No code was modified in this phase.

---

## Summary verdict

**Do not treat `wm_reranking_results.md`'s +15.6pp as valid evidence for anything.** The
feature set the world-model critic consumes is PRE_EXECUTION-admissible (no leaked
execution-derived input — this part is clean). But the **evaluation that produced the headline
number compared the two methods on non-overlapping random scenes and non-overlapping candidate
pools**, despite `results/run_full_01/hybrid_report.md` explicitly and incorrectly documenting the
opposite ("sharing the same K=10 grasp candidates (same seed)"). This is exactly the class of bug
`CAUSAL_VALIDITY_METHOD.md` §4.3 warns about (trusting a description of the code instead of
tracing the code), and the same general failure family as `RULED_OUT_METHODS.md` rows 7–13 (the
unseeded-placement-sampler bug) — this project has hit this specific bug shape multiple times
before, on different subsystems, independently. Per constraint #4, this result is **not assumed
valid** and Phase 1 is required to re-measure it properly before any claim is made.

---

## 1. Repo / environment state

- `git status`: clean. HEAD `45e4c87`.
- conda env `tango`: `torch==2.10.0+cu128`, CUDA available.
- GPU: local RTX 3060, 6GB, currently idle (816MiB used by something else — check before large
  training runs). No cloud GPU in use or required so far.
- `world_model/mlp_predictor.pkl` dated 2026-05-15 20:42 — predates this project's
  `causal_validity_audit/` tooling (built mid-July) and the whole `IDEA_REPORT.md` idea-discovery
  effort (started 2026-07-15) entirely. **Zero mentions of `world_model`/`mlp_predictor` anywhere
  in `RULED_OUT_METHODS.md`, `IDEA_REPORT.md`, `paperA_data/README.md`, or
  `TRO_PAPER_OUTLINE.md`** — this direction was never folded into the project's later, more
  rigorous audit discipline. It is stale, not previously vetted, and not previously killed either.

## 2. Data provenance — what actually trained `mlp_predictor.pkl`

`data/transitions/meta.json` (505 episodes, still present locally, gitignored):

| obj_name key | count |
|---|---|
| banana | 130 |
| cylinder | 130 |
| cracker | 80 |
| mustard | 80 |
| drill | 80 |
| `YcbBanana` (raw YCB name, wrong key — collection artifact) | 5 |

All 5 evaluated object classes were present in training (resolves an initial concern that the
model might have been trained banana/cylinder-only per `collect_mujoco_transitions.py`'s CLI
default — it wasn't; a full run with `--objects all` was used). The 5 stray `YcbBanana`-keyed rows
are a minor data-hygiene artifact (wrong key format, likely an earlier partial run before the
`obj_key` convention was finalized) — negligible at n=5/505, noted but not blocking.

`train_mlp_predictor.py`'s train/test split is a flat random 80/20 row shuffle
(`np.random.default_rng(42).permutation`), not grouped by scene or object instance. This is **not**
a leakage bug here specifically: each row is already an independently, freshly-spawned random
scene (object reloaded and resettled every episode in `collect_mujoco_transitions.py`), so no two
rows share a physical scene the way, e.g., multiple candidates from one Piper scene would. A
random row split is defensible for this data-generating process. (This is different from — and
should not be conflated with — the Piper cross-embodiment leave-scene-out requirement documented
in `CAUSAL_VALIDITY_METHOD.md`, which exists because Piper scenes *do* produce multiple candidate
rows per scene.)

## 3. Feature-level PRE_EXECUTION admissibility audit (Definition 3, `CAUSAL_VALIDITY_METHOD.md`)

Traced every field in the 22-dim feature vector (`data/transition_logger.py::build_feature`) to
its actual computation call site:

| Field | Computed at | Provenance | Reasoning |
|---|---|---|---|
| `grasp_pose` (x,y,z,yaw,opening,obj_height) | `sample_grasp()` / `_sample_grasp()`, called before `execute_grasp`/`_execute()` | **PRE_EXECUTION** | Pure function of `obj_pos_before` + RNG; no execution state involved |
| `obj_pos_before`, `obj_quat_before` | `env.get_obj_pos/get_obj_pose`, captured after settle steps, before grasp execution | **PRE_EXECUTION** | Static scene state `G` in Definition 2's sense |
| `pc_stats_before` | `compute_pc_stats(obs_before, oid)`, same pre-grasp capture point | **PRE_EXECUTION** | Derived from the pre-grasp point cloud only |
| `success`, `dz`, `fell_off` (labels `y`) | computed after `execute_grasp`/`_execute()` | **EXECUTION_DERIVED** | Correctly used only as supervision targets, never as model input (Corollary, `CAUSAL_VALIDITY_METHOD.md`) |
| `geo_score` (candidate-selection heuristic) | `geo_score()`, function of `obj_pos`+`pc_stats` (pre-grasp) | **PRE_EXECUTION** | Admissible |
| `wm_score` / `success_prob`/`dz_pred`/`fell_prob` (critic outputs used for selection) | `score_grasps()`, function of the admissible 22-dim feature above | **PRE_EXECUTION** | Admissible — the critic itself never sees post-execution state at inference time |

**Verdict: no execution-derived feature leaks into the critic's inference-time input.** This part
of the pipeline is causally valid as designed, and is a materially different situation from the
Piper cross-embodiment leak documented in `CAUSAL_VALIDITY_METHOD.md` (`quality_score`/
`correction_proxy`, which genuinely were execution-derived). `causal_validity_audit/provenance.py`
does not currently have a registry entry for this pipeline's fields; recommend adding one (see
Open Items) so this conclusion is machine-checked going forward, not just asserted here.

## 4. Evaluation-design audit — the actual problem

`scripts/eval_wm_reranking_full.py::trial_seed()`:

```python
def trial_seed(base_seed, obj_key, method, idx):
    return (base_seed * 10_000_000
            + _OBJ_IDX.get(obj_key, 0) * 100_000
            + _METH_IDX.get(method, 0) * 1_000     # <-- depends on method
            + idx) % (2 ** 32)
```

`_METH_IDX = {"geometry": 0, "world_model": 1}`. The per-trial RNG seed **depends on which method
is being evaluated**, not just on `(object, trial_idx)`. That RNG is then used, in order, to draw:
object spawn offset (`cx`, `cy`) *and* all `k_grasps=10` candidate poses. Because `geometry` and
`world_model` get seeds 1000 apart, **every "paired" trial in this eval actually samples a
different object placement and a completely disjoint candidate pool between the two methods.**

Confirmed two ways, not just by reading the code:

1. **Analytically**, reproducing `trial_seed()` standalone: `geo_seed(cracker, idx=0) = 420200000`
   vs `wm_seed(cracker, idx=0) = 420201000` — different by construction, for every object/idx.
2. **Empirically**, against the actual `results/run_full_01/results.csv` (the file
   `wm_reranking_results.md` was generated from): of the 250 `(object, trial_idx)` pairs with both
   a `geometry` and a `world_model` row, **0/250 have the same chosen candidate position**
   (`grasp_x`/`grasp_y` match to within 1e-6 in zero cases). If the candidate pools were shared,
   near-ties would occasionally produce identical top-1 picks; the fact that literally none match
   is consistent with the pools being drawn from entirely independent RNG streams.

**This directly contradicts `results/run_full_01/hybrid_report.md`**, which states as its
premise: *"For each `(object, trial_idx)` pair we have two outcomes sharing the same K=10 grasp
candidates (same seed)."* That statement is false against the actual code and actual data. The
entire downstream "Adaptive Gating Analysis" in that file — the oracle upper bound (SR=0.856), the
GeoOnly/WMOnly counts, the 300-value threshold grid search — is built on a premise that does not
hold, and none of its numbers should be cited.

**Consequence for `wm_reranking_results.md`:** the two-proportion z-test used there (`ztest_2prop`)
is *statistically* the correct test for two independent (unpaired) samples, and that is in fact
what this data actually is — so the reported z-statistics and p-values are not miscomputed given
what was actually collected. But an *unpaired* comparison on independently-random scenes cannot
support the paper's implicit claim ("the world-model critic makes a better decision on the same
situation than geometry would") — it only shows that two independent batches of random scenes had
different average outcomes, which is confounded by scene-difficulty variance across the two
batches, not isolated to the selection method. Object-level effects that looked large (mustard
+42pp, cracker +30pp) could partly or wholly reflect which random scenes happened to be easier in
one batch vs the other, not the critic's selection quality on shared decisions.

## 5. Cross-check against this project's other findings — an unresolved internal tension

`train_geo_ebm_grasp.py` / `tango_robot/ui.py`'s GeoEBM work (2026-07-12/13, documented in
`paperA_data/README.md`) explicitly motivates switching away from raw-pose scoring because
*"raw pose carries near-chance success signal for several objects (scene-grouped CV, not a leakage
artifact)"*, and moves to a 6-dim object-relative geometric/affordance feature set instead. The
`world_model/` MLP audited here scores candidates using **raw world-frame pose + minimal scene
context** (`grasp_pose` in absolute coordinates, `obj_pos`/`obj_quat`, coarse point-cloud stats) —
much closer to the GeoEBM finding's "raw pose" category than to its "geometric features" category
— yet reports a large, clean-looking effect. Given §4's finding that the eval design confounds
selection method with scene difficulty, this tension is not necessarily a contradiction (the
`world_model` report's apparent signal may simply be scene-variance noise that a properly paired
design will wash out) — but it should not be assumed resolved either way. **Phase 1's harness will
directly settle this**, using this project's own established feature philosophy question as an
explicit, falsifiable prediction rather than an assumption in either direction.

## 6. Relationship to already-excluded routes (constraint #1)

None of `RULED_OUT_METHODS.md`'s already-closed routes are being re-proposed:
- Not raw-pose *candidate generation* (OT-CFM route) — this is discriminative scoring of
  externally-sampled candidates, a different mechanism.
- Not MPC-style online correction — no in-the-loop action modification.
- Not affordance-auxiliary SmolVLA — no VLA training-signal auxiliary loss involved.
- Not execution-derived candidate-selection features — confirmed clean in §3.

The proposed direction (a pre-execution RGB-D/state critic gating a policy vs. a safe baseline) is
a materially new combination not previously tested or killed in this repo's history.

## 7. Open items before Phase 1 can start

1. **Fix the seeding bug for Phase 1's own harness** — do not reuse
   `eval_wm_reranking_full.py::trial_seed()` as-is; Phase 1's harness must derive one seed per
   `(object, scene_seed)` shared identically across every compared method (random / geometry /
   world-critic / oracle), consistent with what `EXPERIMENT_PLAN.md` already established as this
   project's correct convention for paired comparisons.
2. **Register `world_model/`'s fields in `causal_validity_audit/provenance.py`** so §3's
   conclusion is enforced by the existing gate, not just asserted in prose (tracked as a Phase 1
   prerequisite, small patch).
3. Existing `world_model/mlp_predictor.pkl` may be reused as-is for Phase 1's smoke test (its
   *training* data provenance is clean per §2–3) but its held-out evaluation numbers
   (`wm_reranking_results.md`, `hybrid_report.md`) must be **fully re-measured**, not cited.
4. No real hardware, no paid cloud GPU, and no data overwrite/deletion needed for Phase 1 — none
   of the four pause conditions are triggered. Proceeding to Phase 1.

## Addendum (2026-07-30, mid-Phase-1): two further bugs found while building the paired harness — one project-wide

While implementing Phase 1's harness (`scripts/risk_gated_vla_phase1_eval.py`) and sanity-checking
it against cracker/mustard/drill, two additional problems surfaced beyond the seeding bug in
Section 4 above. Both were found and fixed collaboratively (one by this session, one by the user)
during live debugging — recorded here for the same reason Section 4 was: this pipeline's numbers
must not be trusted until every stage is independently verified.

**A. `scripts/eval_wm_reranking_full.py::_execute()` (and `collect_mujoco_transitions.py`'s
equivalent) never required a verified lift.** Its return value is `contact or grasped or lifted`,
where `contact` is checked immediately after gripper closing, before any lift attempt — i.e. mere
transient touching of the object counts as success on its own, independent of `grasped`/`lifted`.
Empirically this saturates: 150/150 sampled candidates (15 scenes x 10 candidates, 2 objects)
"succeeded" under this criterion regardless of candidate pose. **This means every number in
`wm_reranking_results.md`/`results/run_full_01/` — not just the pairing issue in Section 4 — used
a success criterion far more lenient than this project's own standard.** Phase 1's harness now
calls the production `env.grasp()` primitive instead of reimplementing execution logic.

**B. The production primitive itself had a second, project-wide bug.** `env.grasp()` dispatches to
`_execute_grasp_physics_topdown()`, the method backing `GRASP_MODE_PHYSICS_WELD` — the mode
CLAUDE.md names "recommended for all evaluation," used throughout `paperA_data/`'s pilots and
(per CLAUDE.md's own instruction) the standard for the RA-L submission's reported numbers. Its
weld-attach gate was `weld_obj = contact_ids[0] if contact_ids else None` — **any single-jaw
contact triggered kinematic weld-attachment**, not the bilateral (both-jaw) contact the mode's own
name and documented protocol require. Fixed (uncommitted, `tango_robot/env_soarm.py`) to gate on
`metrics_post.get("bilateral_contacts", 0)` before allowing weld. **This is a pre-existing bug in
shared, project-wide code, not something introduced by this study** — it predates this session and
was only surfaced because this study happened to build a harness that stress-tested it directly.
Flagging clearly rather than fixing further: this may affect prior `physics_weld_after_bilateral`
results project-wide (including RA-L's), which is out of scope to re-verify here — a separate,
deliberate task, not something to fold into this study's conclusions.

**C. Workspace mismatch from the July mount-rotation breaking change.** `eval_wm_reranking_full.py`
and `collect_mujoco_transitions.py` hardcode `_CENTRE_Y = -0.40`, valid under the *pre-rotation*
mount geometry (see `ab05889`, this session, and `paperA_data/README.md`'s "BREAKING CHANGE" entry
predating it). Under the current rotated mount, that placement produces ~4.6cm of descend-IK
error — the arm does not reliably reach the intended candidate at all. Phase 1's harness uses
`EVAL_CENTRE_Y = -0.30` (this project's current validated target-zone value) with
`IK_TOPDOWN_BIAS=0.1` (the null-space secondary task from `ab05889`'s round-6 fix) set via
environment variable, reducing IK error to ~0.04cm and restoring genuine bilateral contact.

**Combined effect, verified empirically**: legacy criterion + wrong workspace -> compounds in
non-obvious directions (the lenient contact criterion can register spurious "success" even on a
badly-missed reach, masking the workspace bug). Corrected pipeline (env.grasp(), bilateral gate,
EVAL_CENTRE_Y, IK_TOPDOWN_BIAS=0.1), n=10 scenes/object, 3 objects, pilot-scale only:

| Method | SR |
|---|---|
| random | 26.7% |
| geometry | 53.3% |
| world_critic (old `mlp_predictor.pkl`) | 33.3% |
| oracle (any of 10 candidates succeeds) | 80.0% |

**This is a pilot (n=10/object), not a result** — non-saturated and directionally sane (oracle >
geometry > random, world_critic between random and geometry) but not yet independently re-verified
by this session at the time of writing, and far below Phase 1's pre-registered n>=50/object gate.
Proceeding to re-verify via a fresh smoke-check + scale to the formal run.

## Files referenced (unmodified)

`world_model/train_mlp_predictor.py`, `world_model/rerank_grasps.py`,
`data/transition_logger.py`, `scripts/collect_mujoco_transitions.py`,
`scripts/eval_wm_reranking_full.py`, `data/transitions/meta.json`,
`results/run_full_01/results.csv`, `results/run_full_01/hybrid_report.md`,
`CAUSAL_VALIDITY_METHOD.md`, `RULED_OUT_METHODS.md`, `IDEA_REPORT.md`.
