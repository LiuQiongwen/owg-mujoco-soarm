# Phase 3 — Real-Hardware Validation Protocol (design only, not executed)

**Status: design document. No real-hardware trial has been run under this protocol and none
will be, without an explicit, separate safety confirmation at the time of execution** — matching
this study's own rule ("不要在没有安全确认的情况下自动执行真实机械臂") and this session's
established practice of getting explicit human presence/supervision confirmation before any real
`env.grasp()`-equivalent command reaches the physical arm.

## 0. What this validates, and what it doesn't

Phases 1-2 established (`final_report.md`): an object-relative counterfactual critic
(`object_counterfactual`, `results/risk_gated_vla/counterfactual_models_20260730/`) beats
geometry top-1 selection in simulation, on two disjoint held-out scene batches (dev-test +14-16pp,
p<0.01; frozen confirmation +14.0pp, p=3.24e-4, live-executed). Phase 3 asks whether that
*simulation* result transfers to the *physical* SO-ARM101 — a sim-to-real gap check, not a
re-run of Phases 1-2's statistical claims. A physical pilot that fails to replicate the sim
effect is not a bug in Phases 1-2; it is the expected, well-documented category of finding this
project has hit before (`paperA_data/README.md`'s Phase 3 SO-ARM101 pilots, rounds 1-6: "mechanism
works, open-loop precision doesn't transfer precisely").

## 1. Object choice — flagging a mismatch with the original ask

The task briefing named **Pear and MustardBottle**. The trained critic
(`world_model/train_counterfactual_critic.py::OBJECTS`) only has a one-hot encoding for
**{cracker, mustard, drill}** — Pear was never part of any simulated data collection in this
study (Phases 0-2 used exactly CrackerBox/MustardBottle/PowerDrill throughout). Feeding Pear
through the trained critic means its object one-hot is all-zero — an out-of-distribution input
the critic was never trained to handle, not a like-for-like generalization test.

**Recommendation**: run the physical pilot on **MustardBottle and CrackerBox** (both in-
distribution for the trained critic, and both real YCB objects this project already owns and has
handled on real SO-ARM101 hardware in earlier Phase 3 rounds — see `paperA_data/README.md`).
Pear can be added as an explicit, separately-labeled OOD generalization check afterward, but
should not be presented as validating the same claim as the in-distribution objects. This
recommendation is stated here, before any data is collected, precisely so it isn't a post-hoc
justification if Pear performs unexpectedly (rule: no post-hoc object/seed/threshold selection).

## 2. Platform and reused infrastructure (nothing new to build for connectivity)

- **Arm**: SO-ARM101, already connected and verified reachable this session
  (`paperA_data/scripts/real_hw_connect.py`, port `/dev/ttyACM0`,
  `MAX_RELATIVE_TARGET_DEG=30.0`, calibration at
  `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_follower.json`).
- **Backend**: `robots/soarm_real_backend.py::SOARMRealBackend` — relative-delta-clamped motion,
  the same safety pattern already used for this project's Phase 3 SO-ARM101 pilots.
- **Camera**: RealSense D435i (already purchased per `IDEA_REPORT.md`'s asset list). Pose
  estimation pipeline: `cameras/noise_characterization.py::estimate_pose_from_rgbd`
  (RANSAC plane removal -> ROI crop -> largest-component segmentation -> centroid + PCA yaw),
  built for `LINE_B_EXPERIMENT_PLAN.md` and currently blocked there on camera repositioning —
  **that blocker is shared with this protocol and must be resolved first** (Stage 0 below).
- **Geometry**: current (post-2026-07-10) rotated mount, `EVAL_CENTRE_Y=-0.30`,
  `IK_TOPDOWN_BIAS=0.1` — the same constants this study's Phase 1/2 simulation used, already
  verified to produce low descend-IK error (~0.04cm) and genuine bilateral contact in sim.
- **Grasp execution**: `env.grasp()` / `_execute_grasp_physics_topdown`'s real-hardware analogue
  does not yet exist as a single call — Phase 3 needs a thin real-hardware equivalent of
  `execute_candidate()` (this study's `scripts/risk_gated_vla_phase1_eval.py`), built on
  `SOARMRealBackend` + `robots/trajectory.py`'s replay pattern instead of direct MuJoCo motor
  commands. This is new code to write (Stage 1 below), not new infrastructure to design from
  scratch.

## 3. Stage 0 — Prerequisites (blocked, same blocker as `LINE_B_EXPERIMENT_PLAN.md`)

- [ ] D435i repositioned overhead the real tabletop, ~0.5-0.8m working distance.
- [ ] One calibration capture of the empty table (plane-fitting reference,
      `cameras/noise_characterization.py` already expects this).
- [ ] Camera-to-robot-base extrinsic calibration (`scripts/solve_handeye.py`,
      `scripts/capture_handeye_sample.py` — already built this session, not yet run against a
      repositioned camera).
- [ ] Fixed, marked physical placement zone on the table for the object (matching the sim's
      spawn region convention: object centered, small controlled offset range, not free-placed
      by hand each trial — inconsistent placement would break the paired design in Stage 2).

**Nothing past this point should be attempted until Stage 0 is complete and independently
re-confirmed at execution time** (camera repositioning is a physical setup change this session
cannot verify from software alone).

## 4. Stage 1 — Real-hardware candidate pipeline (buildable now, no motion required)

New, minimal code needed (mirrors `scripts/risk_gated_vla_phase1_eval.py`'s structure so the
statistical/audit pipeline built in Phases 0-2 can be reused unchanged):

1. `real_build_pool()`: capture RGB-D, run `estimate_pose_from_rgbd` for object position,
   segment the point cloud for `pc_stats` (same 9-dim layout as `compute_pc_stats` — reuse the
   function, feed it real segmented points instead of sim `obs["points"]`/`obs["seg"]`), sample
   K candidates the same way `_sample_grasp` does in sim (same spread/yaw/opening ranges,
   grounded in the *real* detected object position, not a sim ground-truth position).
2. `real_score_pool()`: identical call to `score_candidates()` with the trained
   `object_counterfactual` ensemble — no changes needed, the critic's input contract
   (`obj_pos_before`, `pc_stats_before`, `candidate_pose`, object one-hot) is platform-agnostic
   by construction (this was verified as PRE_EXECUTION-admissible in `audit.md` Section 3,
   independent of whether the pose/pc_stats come from sim or a real camera).
3. `real_execute_candidate()`: drive `SOARMRealBackend` through the candidate pose via IK (reuse
   `tango_robot/env_soarm.py`'s IK solve offline to get joint targets, matching the existing
   sim-to-real trajectory pattern already used in `paperA_data/scripts/real_hw_replay_feedback_close.py`),
   with the `max_relative_target` safety clamp always active. Success criterion: real, physically
   verified — object retained in gripper after lift and a settle period (not a single instantaneous
   contact check; this study's own Phase 0/1 audit found exactly this class of leniency bug
   invalidated an earlier evaluation, and the same discipline applies here).

This stage produces code and can be fully built/tested (dry-run, no arm motion, using logged/mock
camera frames) before Stage 0's camera repositioning is even complete.

## 5. Stage 2 — Paired trial design (matches this project's established convention)

Per-object, per-trial: **fixed placement, matched-block design** — the same object placement is
used for BOTH compared methods within a block (geometry top-1 vs. critic top-1), analogous to
Phase 1's shared-candidate-pool design but adapted for real hardware where re-placing the object
identically twice is not possible. Two valid real-hardware pairing strategies, pick one explicitly
before running (not after seeing results):

- **Option A (preferred if placement repeatability is good)**: place object once, capture pool,
  execute geometry's pick, manually restore object to the same marked zone (not necessarily
  identical millimeter position — record the actual measured placement each time), execute
  critic's pick on a fresh capture. Blocked/paired by placement-attempt index, not exact position.
- **Option B (if restoration isn't reliable)**: alternate which method goes first, block on
  trial index within each object, and treat the comparison as a randomized (not exactly paired)
  block design — use a two-proportion test instead of McNemar's if pairing can't be trusted, and
  say so explicitly in the writeup rather than reporting a paired test on unpaired data (this
  project's `RULED_OUT_METHODS.md` rows 7-13 are a direct cautionary example of trusting an
  assumed-valid pairing that wasn't).

## 6. Stage 3 — Staged scale-up with a pre-registered decision gate

- **Pilot**: 3-5 trials/condition/object (matching the task briefing's own instruction), 2
  objects (cracker, mustard) x 2 methods (geometry, critic) = 4 conditions x 3-5 trials = 12-20
  total physical grasp attempts.
- **Decision gate** (pre-registered here, before any physical trial runs): if the pilot's
  direction is consistent with sim (critic >= geometry on both objects, even if not
  statistically significant at this n) AND no safety incident occurred, proceed to scale-up. If
  the pilot's direction reverses on either object, or a safety incident occurs, **stop and report
  the sim-to-real gap honestly** — do not scale up chasing a positive result, and do not discard
  the pilot as "just bad luck" without an explicit, pre-stated reason.
- **Scale-up** (only if the pilot gate passes): 20-30 trials/object/condition, matched-block
  design from Stage 2, McNemar's exact test (or two-proportion if Option B pairing was used),
  matching Phase 1/2's statistical convention exactly.

## 7. Safety checklist (must be re-confirmed live, not assumed from this document)

1. **Physical supervision**: a human present and able to physically intervene/power-off for
   every single motion command, for the entire session — non-negotiable, established this
   session's own precedent for real-hardware commands on this project.
2. **`max_relative_target` clamp active** on every `SOARMRealBackend` motion call (30 deg/command
   default, matching `real_hw_connect.py`'s established value — do not raise without a specific
   reason).
3. **Joint range verification**: confirm IK solutions for all sampled candidate poses stay within
   `tango_robot/env_soarm.py`'s `jnt_range` limits before any command is sent (dry-run/log every
   solved pose first; refuse to execute any solution needing clamping/saturation rather than
   silently clamping it and moving anyway).
4. **Workspace clearance check**: fixed placement zone (Stage 0) must be verified clear of
   anything except the target object and the tray before each trial block starts.
5. **E-stop / immediate power path**: confirm exactly how to kill motor power before the first
   motion of the session (this is a re-confirmation step, not a one-time setup item).
6. **Speed**: start at reduced playback speed (matching `scripts/replay_trajectory.py`'s
   `--speed 0.5` convention already established for first real-hardware tests of a new
   trajectory/policy in this project) for at least the pilot stage; only increase to full speed
   after the pilot gate passes and with a separate, explicit go-ahead.
7. **Collision check**: verify the object's fixed placement zone and the arm's approach path
   don't bring the arm within collision range of the camera mount, tray, or table edge — this
   project has hit exactly this class of issue before (round-3 Phase 3 pilot: fast approach
   replay physically knocked the object away before closing).

## 8. What Stage 3's result would and would not settle

**Would settle**: whether the sim-measured critic advantage (+14-16pp) survives the sim-to-real
gap at all, on the specific objects/placements tested, at pilot or scale-up sample size.

**Would not settle**: generalization to Pear or PowerDrill (not tested here per Section 1's
recommendation), whether the risk gate (already shown to add no value in sim, `final_report.md`)
would behave differently on real sensor noise (a real, open question — real camera noise could in
principle make the uncertainty estimate more informative than it was against clean sim
observations, or could make it worse; this protocol doesn't test the gate at all, only top-1
critic selection, to keep the physical pilot's scope minimal and interpretable).
