# Stage 2 evidence traceability

Scoped re-verification (2026-08-05), not a re-exploration: re-checks only the specific claims that
will be written into `paper_tro.tex`'s external-boundary-check section, against the actual code at
the pinned commits, in place, right now. Every claim below has a file path + line range a reader
could go check themselves.

---

## Case 1 — GraspGen: candidate-scoring input boundary

- **Repository**: `github.com/NVlabs/GraspGen`
- **Commit**: `2dd8852e1be60f5f9d277fafcc621835cdf59110` (2026-06-21)
- **Audit question**: does the discriminator that ranks not-yet-executed grasp candidates take any
  execution-derived input?
- **Inspected path**: `grasp_gen/models/discriminator.py::GraspGenDiscriminator.forward`
  (lines 194–260) and its call site, `grasp_gen/grasp_server.py::score_grasps_with_discriminator`
  (lines 28–58, called from `grasp_gen/samplers/graspmoe.py:256` for OBB-candidate ranking and from
  `GraspGenSampler.sample` for diffusion-candidate ranking).
- **Automatic tagger output**: not run — GraspGen has no `env.step`-shaped execution boundary at all
  (no simulator handle, no physical-actuation call anywhere in the scoring path), so there is no
  post-marker taint-propagation question the tagger's mechanism is built to answer. This was
  determined by direct reading, not tool output — stated honestly as such, not implied to be an
  automated result.
- **Manual conclusion**: `score_grasps_with_discriminator`'s only inputs are `points_centered` (an
  (N,3) point cloud) and `grasps_centered` ((M,4,4) candidate poses) — confirmed directly at
  `grasp_server.py:28-58`; the `data` dict built for `forward()` contains only `points`, `inputs`
  (point cloud + zero color channels), and `grasps`. No outcome label, no post-execution state, no
  simulator call anywhere in this function or its two callers.
- **Final classification**: within the inspected candidate-scoring path, no flow of post-execution
  outcome information into pre-execution candidate scores was found.
- **Evidence locations**: `grasp_gen/models/discriminator.py:194-260`,
  `grasp_gen/grasp_server.py:28-58`, `grasp_gen/samplers/graspmoe.py:256`.
- **Limitations**: this checks the scoring/ranking path only, not the full repository (training
  code, data generation, or other entry points were not inspected). Does not establish that no
  other GraspGen code path could leak information — only that the specific candidate-ranking path
  cited in the paper does not, as verified by direct reading, not exhaustive tool coverage.

## Case 2 — Sim-Grasp: the pattern that motivated Category 12

- **Repository**: `github.com/junchengli1/Sim-Grasp`
- **Commit**: `dd0957d0ca06986272775937f8281a3fe1a9518b` (2024-05-05)
- **Audit question**: does this codebase contain the subscript-assignment field-writing pattern
  Category 12 was added to recognize, and did it originate here?
- **Inspected path**: `grasp_sampling/grasp_simulation.py::main_simulation_loop`, lines 143 and 149.
- **Automatic tagger output**: confirmed by direct re-run against the currently-integrated tool
  (`causal_validity_audit/auto_tagger.py`, commit `dc817e7`):
  `tag_file(".../grasp_simulation.py", "main_simulation_loop")` →
  `field_provenance={'simulation_quality': 'PRE_EXECUTION'}, marker_found=False`. Reproduces the
  originally documented output exactly.
- **Manual conclusion**: line 143/149 write
  `new_candidates[object_number]["grasp_samples"][ori_ind]["simulation_quality"] = 1` / `= 0` — a
  literal, real instance of `container[...][...]["field"] = value`, the exact pattern
  `causal_validity_audit/test_fixtures/fixture_12_subscript_assignment.py` was written to isolate
  and that `_subscript_field_name()` was added to recognize. Confirmed this is the real origin of
  Category 12, not a retrospectively-constructed example.
- **Final classification**: inspection of Sim-Grasp exposed a real, unsupported field-writing
  pattern; this motivated the Category 12 tool extension. Not a claim that Sim-Grasp itself has a
  causal-validity violation (see Case 3).
- **Evidence locations**: `grasp_sampling/grasp_simulation.py:143,149`;
  `causal_validity_audit/auto_tagger.py::_subscript_field_name`;
  `causal_validity_audit/test_fixtures/fixture_12_subscript_assignment.py`.
- **Limitations**: none specific to this case — directly reproducible, low ambiguity.

## Case 3 — Sim-Grasp: the near-miss (label vs. pre-execution feature)

- **Repository / commit**: same as Case 2.
- **Audit question**: is `simulation_quality` (Case 2's field) a pre-execution candidate-selection
  feature (would be a violation) or a post-execution label (exempt by this project's own
  labels-are-exempt corollary)?
- **Inspected path**: `main_simulation_loop` (lines 104–152) and its call site (line 371, inside
  `sim_suction_simulation()`).
- **Automatic tagger output**: `marker_found=False` (see Case 2) — the tool defaults every field to
  `PRE_EXECUTION` when no `CAUSAL_VALIDITY_COMMIT_POINT()` marker is present (this project's own
  convention, which no external codebase has), so **the tagger's raw output alone does not answer
  this question** — placing a marker requires a human judgment call about where "execution has
  begun" actually is, exactly as `commit_marker.py`'s design requires. This is the case
  `AUDIT_TOOL_VALIDATION_PLAN.md` documents as "not yet meaningful evidence... a demonstration the
  extraction mechanism works, nothing more" before the manual trace below was done.
- **Manual conclusion**: `main_simulation_loop` runs a post-placement *stress test* — it lowers a
  ground-plane prim under the object in a loop (`xform_prim.set_world_pose(...)`,
  `world.step(...)`, lines 128-133) and then checks whether the object's position dropped
  (`pose - pose_ori >= -10`, lines 135-149) before writing `simulation_quality`. This requires
  physical simulation to have already run inside this very function — genuinely execution-derived.
  Tracing where the candidate pose is actually committed: `RobotSpawner.spawn()`
  (`grasp_simulation.py:219-236`) places the gripper directly at `good_suctions_translation[env_i]`
  / `good_suctions_rotation[env_i]` (the candidate pose) via `XFormPrim(...)`, called at line 359 —
  **before** `main_simulation_loop` is called at line 371. So the causal order is: candidate
  committed (spawn) → stress-test physics runs → `simulation_quality` label written. Labels are
  exempt from the PRE_EXECUTION requirement by this project's own criterion (a label is a training
  target, not a live-selection input) — `simulation_quality` being execution-derived is correct and
  expected, not a violation. Treating this as a "found bug" would have been unsupportable.
- **Correction found during this re-verification, not in the original write-up**:
  `AUDIT_TOOL_VALIDATION_PLAN.md`'s narrative cited `handle_action_for_env`
  (`grasp_simulation.py:77-101`, containing `articulation_controller.apply_action(actions)` at line
  100) as the function that commits the candidate action before `main_simulation_loop` runs. This
  is **not verifiable** — `grep -rn "handle_action_for_env(" --include="*.py" .` across the entire
  Sim-Grasp repository finds only the `def` line itself; the function is defined but never called
  anywhere in this codebase. The actual commit mechanism is `RobotSpawner.spawn()` (direct
  kinematic placement via `XFormPrim`, not an IK-then-`apply_action` control loop). The
  labels-are-exempt conclusion is unaffected (the causal order — commit before stress-test — still
  holds via `spawn()`), but the specific function named as "the action-commanding code" in the
  earlier plan document was wrong and should not be repeated in the paper. Cite `RobotSpawner.spawn`
  (lines 219-236) and its call site (line 359), not `handle_action_for_env`.
- **Final classification**: correctly excluded, not counted as a violation. `simulation_quality` is
  a post-execution label, consistent with this project's own labels-are-exempt corollary.
- **Evidence locations**: `grasp_sampling/grasp_simulation.py:104-152` (the loop),
  `grasp_sampling/grasp_simulation.py:219-236,359` (the actual commit mechanism, corrected),
  `grasp_sampling/grasp_simulation.py:371` (call site, after spawn).
- **Limitations**: `handle_action_for_env`'s original purpose is unknown (dead code, possibly
  vestigial from an earlier version of this pipeline, possibly for a code path not exercised by
  `sim_suction_simulation()`) — not investigated further, since it is unreachable from the traced
  entry point and therefore irrelevant to this specific claim.

---

## Summary table for paper use

| Case | Method | Result | Strength |
|---|---|---|---|
| GraspGen scoring boundary | Manual code reading (no tagger applicable — no execution boundary exists) | No post-execution flow found in inspected path | Qualitative, path-scoped |
| Sim-Grasp Category 12 origin | Automatic tagger (confirmed by re-run) + manual pattern match | Real pattern, confirmed origin of a tool extension | Tool output, directly reproducible |
| Sim-Grasp near-miss | Automatic tagger (necessary but insufficient — marker_found=False) + manual call-graph trace | Correctly excluded (label, not feature); one factual correction made to the original narrative | Qualitative, hybrid tool+manual |

None of these three should be described in the paper with quantitative-validation language
("accuracy," "verified absence of leakage throughout the codebase"). They are boundary/specificity
checks with directly-citable evidence, not a benchmark. Only Case 2 has a tool-output component
that's independently re-runnable; Cases 1 and 3 are manual reading, honestly labeled as such.
