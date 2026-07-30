# Paper A raw data archive

## 🔍 Depth-noise vs shared-reference-cloud diagnostic (2026-07-13): sensor noise is a minor factor; the real culprit is GeoEBM's cached single-reference point cloud

Follow-up to the GeoEBM Pear/TomatoSoupCan pilot above. Before investing in a
"perception algorithm" research direction (real RealSense D435i integration
+ depth-denoising, motivated by literature like R2SGrasp's "Real-to-Sim
Feature Enhancer" and DiffuDepGrasp), ran a cheap offline diagnostic
(`scripts/depth_noise_feature_degradation_check.py`, no camera/hardware
needed) injecting D435i-realistic synthetic depth noise (Intel's published
RMS-error curve, ~1.06mm at the assumed 0.55m working distance; edge/void
dropout; small-object sparsification) onto the existing clean cached point
clouds, and measuring how much the 3 point-cloud-derived geometric features
(`local_point_density`, `normal_consistency`, `contact_width_ratio`)
degrade in GroupKFold AUC.

| Severity | Pear AUC | TomatoSoupCan AUC |
|---|---|---|
| Clean (0 injected noise, but still the SHARED cached reference cloud) | 0.610 | 0.515 (≈chance) |
| Mild | 0.593 | 0.513 |
| Medium | 0.569 | 0.535 |
| Severe (2x D435i RMS error) | 0.537 | 0.500 |

**Key finding**: injected sensor noise alone only costs Pear ~7pp of AUC
(0.610→0.537) and TomatoSoupCan is already at chance even with ZERO
injected noise. The dominant factor is not depth-sensor noise -- it's that
GeoEBM's live-inference design uses ONE FIXED cached reference point cloud
per object (`scripts/cache_object_pointclouds.py`, one replayed scene) for
every trial, instead of a genuine per-trial point cloud. This single design
choice alone accounts for the drop from the true per-scene AUC (Pear 0.901,
TomatoSoupCan 0.846, both computed on the original dataset's real per-scene
point clouds) down to 0.610/0.515 -- an order of magnitude larger effect
than realistic sensor noise on top of it.

**Can support**: a real depth camera (D435i, already purchased), once
properly integrated, captures a FRESH point cloud every trial -- this
directly fixes the dominant problem (shared-reference-cloud approximation),
not a secondary one. **This reframes the "perception algorithm" research
direction**: the priority is verifying real per-trial point cloud capture
integration, not designing a depth-denoising algorithm (which this
diagnostic shows would target a comparatively minor effect). Depth
denoising / noise-robustness may still be worth a lighter follow-up check
once real per-trial capture is working, but is not the primary blocker.

**Cannot support**: whether a REAL per-trial point cloud (once the D435i is
integrated) actually recovers AUC close to the original 0.901/0.846 --
this diagnostic only shows the shared-cache approximation is the dominant
known cost, not that fixing it recovers full performance. Real camera noise,
calibration error, and real (non-simulated) object placement variability
could still cost additional AUC in ways this synthetic-noise-only check
can't capture. Needs a real hardware validation once the camera is
integrated, not just this offline check.

## ⚠️ GeoEBM physical pilot (2026-07-13): works on Pear (72%, parity with baseline), catastrophically fails on TomatoSoupCan (8% vs baseline ~92%) -- does not generalize

Follow-up to the killed affordance-auxiliary-VLA proposal (see below): pivoted
to a "Geometry-Conditioned EBM" -- same InfoNCE + self-mined-hard-negative
training recipe as the existing EBM v2 (`train_ebm_grasp.py`, RA-L paper),
but scoring the 6-dim live-computable geometric feature subset instead of
raw pose (`train_geo_ebm_grasp.py`, new file). Two real bugs found and fixed
before getting a usable model, both confirmed via `GEO_EBM_CEM_DEBUG=1`
diagnostic logging in `tango_robot/ui.py`'s `_geo_ebm_sample_candidates`:

1. **Point cloud / live object position mismatch**: the cached per-object
   reference point cloud (`scripts/cache_object_pointclouds.py`, one fixed
   seed=1 replay per object) has different absolute world coordinates than
   the object's actual live position at inference time (different spawn RNG
   convention between the data-collection scripts and the real evaluation
   harness -- a recurrence of the same class of bug documented in the
   "🔧 BREAKING CHANGE" entry below). Fixed by recentring the cached cloud
   onto the live CoM at inference time (`pcd = pcd_ref + (gx - ref_x, gy -
   ref_y, 0)`), using a `ref_xy` saved alongside the cache.
2. **Point cloud downsampling diluted local density to near-zero**: uniform
   random downsampling of the whole scene (table + object + background) to
   3000 points left only ~9 points within 3cm of the object itself -- most
   CEM-mining candidates landed in bbox regions with too few points, hitting
   `normal_consistency`/`contact_width_ratio`'s `min_pts` degenerate
   fallback (returns 0.0), producing a saturated/identical score across the
   entire population (confirmed: all 5 final CEM candidates scored EXACTLY
   the same logit, 7.123, regardless of position). Fixed by cropping to a
   0.35m radius around the object BEFORE downsampling (preserves local
   density: 46-81 points within 3cm, vs ~9 before), instead of uniform
   sampling over the whole scene.

After both fixes, single-trial CEM diagnostics showed real score
differentiation (12.1/11.1/11.1/10.6/10.5, not identical) and candidates
converging to 1.8-5.6cm from the true object (down from 10-16cm) -- a
genuine, verified fix, not just a training re-run.

**Physical pilot results** (`demo.py --stage 4`, same harness as all prior
Paper A comparisons, n=25/object):

| Object | GeoEBM | Baseline (historical) |
|---|---|---|
| Pear | 72% (18/25) | 72% |
| TomatoSoupCan | **8% (2/25)** | **~92%** |

**Can support**: the two bugs above are real and the fixes are verified
(CEM no longer saturates, candidates converge near the true object) -- this
diagnostic methodology (point-cloud recentring, crop-before-downsample,
`GEO_EBM_CEM_DEBUG=1` per-iteration convergence logging) is reusable for any
future point-cloud-conditioned candidate scorer in this codebase. Pear
reaching exact parity with baseline (72% both) after the fixes is a real,
non-trivial result -- notably better than OT-CFM's 52% on Pear.

**Cannot support**: that GeoEBM is a viable general method -- TomatoSoupCan's
8% is not noise (n=25, consistent near-zero success across most seeds, not
a few unlucky trials) and is far worse than every other method tried on
this object this project (OT-CFM 78%, DDPM/Remove-OT ~100%). Root cause not
yet diagnosed -- plausibly the 0.35m crop radius or CEM search-range
assumptions were implicitly tuned against Pear's size/geometry and don't
transfer to a differently-shaped/sized object (TomatoSoupCan is a cylinder,
notably different aspect ratio from Pear). **Do not report this as a working
method or invest further training time before diagnosing the TomatoSoupCan
failure specifically** -- a single-object positive result has misled this
project before (see the whole "new-method search" thread above).

## 🔍 New-method search for T-RO/IJRR (2026-07-12): five routes killed offline (zero real-hardware time spent), one route drafted as a proposal

## 🔍 New-method search for T-RO/IJRR (2026-07-12): five routes killed offline (zero real-hardware time spent), one route drafted as a proposal

After the RA-L submission (2026-07-11), separately from the real-hardware
imitation-learning thread (ACT policy on 41 Pear teleop demos), ran a
dedicated new-method search targeting a T-RO/IJRR-level *learned* core
method to replace the dead OT-CFM. Five candidate routes were tested and
killed using **existing sim data only** (`grasp_6dof/dataset/lggsn_candidates_v9.jsonl`,
1000 candidates/object x 7 objects), before any physical trial was burned:

1. **Retrieval-Anchored Residual Flow Matching (RARFM)** -- retrieve nearest
   real/sim anchor pose, apply residual correction. Single-object anchor
   bank: -18.0pp vs naive mean-shrinkage control. Oracle same-object
   single-NN: still -17.4pp (proves it's single-NN estimator noise, not a
   data-scale problem -- yaw std is huge, ~0.87-0.92 rad, near-symmetric/
   multimodal, so any single retrieved anchor is a noisy point estimate).
   k-NN(10)-averaged oracle: ties control (0.0pp) -- fixes the noise but
   adds no value. Object-centroid-normalized coordinates: still -14.0pp
   (rules out "wrong coordinate frame" as the explanation).
2. **Geometric-feature-space generation + nearest-neighbour back-projection**
   -- naive nearest-centroid-to-typical-success-descriptor selection: only
   +4.6pp mean lift over random pick, weaker than what the already-trained
   LGGSN pairwise reranker plausibly achieves on the same features. No
   clear incremental value shown over existing infra.

**Key diagnostic finding** (the one genuinely new, decisive, reusable
result from this search): logistic-regression AUC for predicting true
physical grasp success is **~chance (0.48-0.54) from raw world-frame pose**
for Pear/Mustard/Can -- the exact objects hit hardest by every past
generative failure -- but **0.85-0.92 from the existing 12-dim LGGSN-style
object-relative geometric features**, pooled pose AUC=0.581 vs geom
AUC=0.725. This is a clean, one-number explanation for why every method
that generates/corrects raw pose (OT-CFM, C²OT, RARFM) has struggled while
LGGSN reranking and consensus selection (which operate on/benefit from
relative geometry) keep working. Root cause of *why* raw pose carries so
little signal: all 7 objects are spawned within ~0.4-4cm of the same table
position across scenes (confirmed by comparing each object's own success-
region centroid to the global grand mean), so world-frame (x,y) carries
almost no object-specific information to begin with.

**Proposal drafted from this finding** (not yet novelty-checked externally
-- Codex/GPT-5.4 returned 401 all session, as in every other attempt this
project has made): `paperA_data/new_method_affordance_auxiliary_proposal.md`
-- affordance-auxiliary multi-task fine-tuning of a small VLA (SmolVLA +
LoRA), with a second auxiliary head reusing the killed Phase-1 MPC
project's cheap `_settle_at_pose` sub-process data as an *offline*
representation-learning signal (not an online correction loop, which is
what killed Phase 1). Toy-scale validation (2-layer MLP proxy, not real
SmolVLA): auxiliary geometric-feature supervision raises a 32-dim
bottleneck's success-probe AUC from 0.775±0.004 to 0.791±0.012 (4/5 seeds
improved) -- small but real and non-artifactual, the first candidate this
search did *not* kill. Self-reviewed novelty verdict (honest, no external
reviewer available): NOVEL-BUT-INCREMENTAL -- affordance-auxiliary VLA
training as a mechanism class already exists (AffordVLA arXiv:2605.17517,
SG-VLA arXiv:2603.22760); what's new here is the quantified diagnostic
motivation and the specific small-data/single-GPU instantiation, not the
mechanism itself.

**Can support**: a decisive, cheap (zero real-hardware time), reusable
methodology for killing candidate directions before physical investment --
now demonstrated on 5 more routes on top of the routes killed earlier in
Paper A. The pose-vs-geometric-feature AUC gap is a solid, quotable,
reusable diagnostic finding regardless of which method direction is chosen
next.

**Cannot support**: whether the affordance-auxiliary proposal survives a
real (non-toy) SmolVLA-scale check (Stage 1 in the proposal doc, not yet
run) -- the toy MLP result is suggestive, not decisive. Do not re-propose
routes 1-2 above (or the MPC/C²OT/OT-CFM routes killed earlier) in future
sessions without new evidence.

## ❌ Stage 1 update (same day, 2026-07-12): affordance-auxiliary VLA route also killed -- toy-MLP signal was noise

Ran Stage 1 of `paperA_data/new_method_affordance_auxiliary_proposal.md` for
real: regenerated 1400 real scene images (deterministic object-spawn replay,
`scripts/regen_lggsn_scene_images.py` -- a reusable new asset, first real
images for this benchmark), downloaded SmolVLM2-500M, and reran the
single-task-vs-multi-task probe-AUC comparison using the actual SmolVLA
action-expert (direct fine-tune, not LoRA -- lerobot's PEFT wrapper refuses
a from-scratch action-expert) instead of the earlier toy 2-layer MLP.
3 seeds x 300 steps x 1500 rows: advantage = +0.017 / -0.032 / +0.001,
mean -0.005 -- sign flips across seeds, noise-level. The toy-MLP's
+1.6pp (4/5 seeds positive) did **not** survive the jump to real
architecture scale -- exactly the failure mode the proposal doc's own risk
section anticipated ("the toy MLP's +1.6pp was itself close to noise").
**This route is now killed, joining routes 1-5** (RARFM variants,
geometric-feature-space generation). Did not proceed to the planned
cross-object few-shot-transfer follow-up (no in-distribution baseline
effect to test transfer of). Do not re-propose affordance-auxiliary VLA
training for this benchmark without new evidence -- the 1400 real images
are a reusable asset for whatever comes next, the mechanism itself is not.

## ❌➡️✅ Phase 3 real-robot pilot, round 6 (2026-07-11): `topdown_bias`'s first version targeted the wrong quantity -- fixed to target actual EEF orientation via the rotational Jacobian

Direct follow-up to round 5. Real-hardware test of the mount-rotation-fixed
trajectory (round 5) still failed: user reported "抓夹前伸的抓取姿势，容易把
物品推出去而不是夹住" (the gripper's forward-extending posture pushes the
object away instead of gripping it), consistent with a non-top-down
approach. Diagnosed the *first* `topdown_bias` fix (added same day, before
this entry) as wrong: it targeted
`shoulder_lift+elbow_flex+wrist_flex -> 0` on the assumption this matched
`HOME_QPOS`'s own top-down orientation. **Directly checked via forward
kinematics and found this assumption false**: `HOME_QPOS`'s own EEF Z-axis
(the jaw approach/closing direction) is `[-0.05, 0, 0.999]` -- HOME points
nearly straight **up**, not down. The original (unbiased) trajectory's
Z-axis was `[-0.045, -0.606, 0.794]` (tilted, mostly up); the first
`topdown_bias` fix made this *worse*, not better (`[-0.047, 0.107, 0.993]`,
even more upward) -- consistent with, and now fully explaining, the user's
report.

**Fix**: rewrote `_ik_step`'s null-space secondary task
(`tango_robot/env_soarm.py`) to target the actual EEF Z-axis alignment with
world $-Z$ via the rotational Jacobian (`mj_jacSite`'s `jacr` output) and a
standard rotation-vector error (`cross(z_current, z_desired)`), instead of
the joint-angle-sum proxy. Verified: at `topdown_bias=0.1`, Z-axis becomes
`[-0.342, 0.004, -0.94]` -- now genuinely pointing down, with position error
still small (0.047cm, well within the 0.5cm tolerance). Quick pilot check
(n=5/condition, Pear): 2/5 (bias=0) vs 3/5 (bias=0.1) -- no regression.
Recorded a corrected trajectory
(`trajs/pear_consensus_orient3_gen1_v4_topdown_fixed.json`): pre-close
pan=-1.1° (safe), Z-axis=`[-0.341, -0.067, -0.938]`.

**Can support**: the null-space secondary-task *technique* is sound (as
theory predicted, it doesn't sacrifice position accuracy); the bug was in
which target quantity the first implementation optimized for, not the
technique itself. Always verify a claimed reference orientation (like
"HOME = top-down") via direct FK rather than assuming it from a
naming convention or a proxy quantity.

**Cannot support**: whether this real EEF-orientation fix actually resolves
the "pushed away instead of gripped" symptom on real hardware -- not yet
tested with the corrected trajectory. Next step before further diagnosis:
real-hardware trial of `pear_consensus_orient3_gen1_v4_topdown_fixed.json`.

## 🔧 BREAKING CHANGE (2026-07-10): rotated the arm's mount in simulation (`ROBOT_BASE_EULER`) -- invalidates ALL prior paper-reported numbers, must be re-verified before citing anything against them

Direct follow-up to round 4. User provided a photo of the physical setup and
correctly identified the real issue: the arm's mount, in the simulated
scene, sits exactly at `table_top`'s edge (`table_top` spans y in
[-0.90, 0.00], `robot_base_mount` at y=0) with `HOME_QPOS`'s natural reach
direction (pan=0 -> world +X, confirmed via FK) running PARALLEL to that
edge rather than into the table (world -Y). This is why every real-hardware
attempt so far needed ~90-100 deg of shoulder_pan just to reach the object.
User's explicit decision, after being told this would invalidate existing
comparison data: **"替换成仿真默认设置"** (make it the new simulation
default), not a real-hardware-only opt-in.

**Change**: added `ROBOT_BASE_EULER = "0 0 -1.5707963267948966"` (-90 deg
about Z) to `robot_base_mount` in `tango_robot/env_soarm.py`. Verified via
direct FK: HOME eef moved from `(0.337, ~0, 0.96)` to `(~0, -0.337, 0.96)` --
now pointing into the table. Verified via `_solve_ik_jaw_pos_only`: the
default single-object spawn `(0, -0.30)` now needs shoulder_pan **-0.2 deg**
(was ~97 deg) to reach.

**Immediate breakage found and fixed**: the old `TARGET_ZONE_POS = (0.20,
0.25)` (tray) became fully unreachable under the rotated mount (`ok=False`,
pe=21.8cm, would need pan~-110 deg) -- confirmed the pick zone (~97 deg from
old HOME) and the old tray (~-57 deg from old HOME) are ~154 deg apart, which
cannot both fit inside any single ±60 deg-from-HOME window regardless of
mount rotation (a hard geometric conflict, not a parameter-tuning problem).
Repositioned `TARGET_ZONE_POS = (0.25, -0.30, TABLE_TOP_Z)`, verified cleanly
reachable (pe<1mm, pan~-44 deg) and, unlike the old position, actually sits
on `table_top`'s own footprint (the old tray never did -- y=0.25 is outside
table_top's y range; the old mount's geometry happened to make it reachable
anyway).

**End-to-end verification**: `demo.py --stage 4 --prompt Pear --seed 1
--once --no-semantic` completes the full pick -> lift -> transport -> place
sequence and prints `Done pick` under the new mount + repositioned tray (5th
grasp-candidate retry succeeded; the first 4 failing is normal multi-attempt
behaviour, unrelated to this change).

**Quick directional check (n=9/strategy, not a formal re-verification)**:
re-ran a reduced ikmargin-vs-consensus sweep for Pear (orient_seed in
{5,6,7} x gen-seed base in {1,11,21}, same CFM checkpoint) under the new
geometry: **ikmargin 0/9, consensus 6/9 (67%)** -- same direction as the
original finding (6% vs 68%), if anything cleaner in this small sample. The
consensus-selection mechanism itself appears orientation-independent, as
expected (it's about candidate-pose robustness, not arm mounting) -- good
early signal, but this is n=9, not the n=50 the paper cites.

**Can support**: the rotation is mechanically sound (FK/IK verified,
end-to-end episode completes, tray reachable) and the core consensus finding
survives it directionally in a quick check.

**Cannot support**: any exact number from before this change
(`PAPER_A_CLEAN_SUMMARY.md`'s baseline 79.1% / OT-CFM 69.1% / EBM v2 74.0%,
the Pear ikmargin 6.0%/consensus 68.0% finding, or anything else) without
re-running under the new geometry -- all of it was computed under the old,
unrotated mount and old tray position. **Before citing any Paper A number
going forward, check whether it predates this change** (this entry's
timestamp: 2026-07-10, same day as the round 1-5 real-robot pilots). A full
re-verification (n=50/condition, matching the original protocol) has NOT
been run yet -- only the n=9 directional check above.

**Also unblocking**: real hardware can now, in principle, replicate the
sim's pick geometry directly (no more `--spawn-xy` workaround needed) --
next step is physically re-mounting the real SO-ARM101's base (rotate ~90
deg so its own pan=0 also points into the table instead of along its edge),
then re-running the shoulder_pan=0 calibration check against the NEW
physical orientation before any further real-hardware trial.

## ❌ Phase 3 real-robot pilot, round 4 (2026-07-10): found the actual root cause -- the physical marker was placed ~10cm away from where the trajectory's grasp target actually is

Direct follow-up to round 3. That live attempt (slowed-down approach, feedback
close) still found nothing at the closing point. User's report pinpointed why:
"转到左边才闭合抓夹，根本不可能抓到正前方的物品" (it turned to the left before
closing the gripper -- there's no way it could grasp something directly in
front). This doesn't match either round 2's orientation-deviation explanation
or round 3's approach-speed explanation -- it points to something more basic:
the arm closing far from where the object physically is.

Now that the torch/CUDA environment is fixed (see below), re-ran
`record_consensus_grasp.py` for this exact episode (orient_seed=3,
gen_seed_base=1, consensus) with full stdout captured. The actual printed
grasp target: `approach → xy=(-0.001, -0.323) z=0.787 yaw=0.089`, with
`pre-close metrics: jaw_obj_xy_gap=0.02` confirming the IK successfully
landed within 2cm of the object's real simulated position -- i.e. **the
object's true position for this episode is y≈-0.32, not y≈-0.17** as
recorded in this session's own earlier notes (the value used to justify
placing the physical marker at x=0, y=-0.20).

Independently reproduced by directly loading a single Pear the same way
`ui.py` does for a 1-object pool (`env.load_isolated_obj(..., pos=[0.0,
-0.30, OBJECT_INIT_HEIGHT])`, 100 settle steps -- this is the actual,
deterministic spawn convention for n=1 objects; earlier notes assumed the
`pos=None` random-uniform convention, which does not apply here) and reading
back the settled CoM directly: **(-0.005, -0.297, 0.766)** for orient_seed=3,
**(-0.004, -0.295, 0.766)** for orient_seed=5 -- fully deterministic
(reproduced exactly on a repeat run), and essentially seed-independent
because the deterministic spawn xy dominates over the small
orientation-seeded rolling variance.

**Root cause of rounds 1-3's failures, corrected**: the object's real
position for this single-Pear setup is consistently **x≈0, y≈-0.30** (30cm
in front of the arm base along the marked axis), not y≈-0.20 where the
physical marker was actually placed. That's a ~10cm placement error -- large
relative to a pear's size and the gripper's tolerance -- present in every
real-hardware trial so far (rounds 1-3 all used the same marker position).
This alone is sufficient to explain "grasped from the bottom," "gripper
tilted upward," "turned left with nothing in front of it," and the empty
`Present_Load` readings in rounds 3-4's feedback-close attempts: the arm was
never closing anywhere near the real object to begin with, independent of
the (still real, still worth fixing eventually) orientation-IK and
weld-signal issues documented in the round-2 entry below.

**Can support**: a clear, concrete, correctable next step -- move the
physical marker to (x=0, y=-0.30) measured from the arm base's mounting
point, not (x=0, y=-0.20). This should be checked before attributing any
further failure to orientation/weld/speed mechanisms.

**Cannot support**: any conclusion about whether the round-2/round-3
mechanisms (unconstrained IK orientation, sim-only weld closing signal,
approach-speed collision risk) are still practically relevant once object
placement is corrected -- they were never cleanly isolated from this larger
placement error. Re-test with the corrected marker position before deciding
whether those are still live issues or were secondary to this one.

### Aside: torch/CUDA environment was fully broken this session, now fixed

Unrelated to the sim-to-real work above but blocked it directly (couldn't
re-run `record_consensus_grasp.py` for hours): `torch` (2.10.0) could not be
imported at all (`undefined symbol: cuptiActivityEnableDriverApi`, then
`undefined symbol: ncclCommShrink`) because ~10 `nvidia-cuda-*-cu12` packages
and `triton` were pinned to old versions (leftover from a prior incomplete
torch upgrade), not matching what torch 2.10.0 actually requires. Fixed by
`pip install`-ing the exact versions torch's own package metadata declares
(`nvidia-cublas/cufft/curand/cusolver/cusparse/nccl/nvjitlink/nvtx/cuda-nvrtc/
cuda-runtime/cudnn-cu12` + `triton`, versions read from `importlib.metadata`).
Hit `OSError: [Errno 28] No space left on device` mid-install (root
filesystem was at 99%, 2.4GB free) -- freed ~12GB via pip/npm cache purges,
disabled old snap revisions, and a 7-day journal vacuum (all safe, reversible
caches; deliberately did NOT run `apt autoremove`, which a dry-run showed
would remove the entire wine-staging install backing this machine's MT4/MT5
trading setup, and did NOT purge old kernels, which triggered an unwanted
NVIDIA driver/kernel upgrade as a side effect in a dry-run). `torch` and the
full `tango.policy` import chain confirmed working after the fix.

## ⚠️ Phase 3 real-robot pilot, round 3 (2026-07-10): feedback-driven closing fix built and validated, but exposed a new (safety-relevant) failure mode -- fast approach replay physically knocked the object away before closing ever started

Direct follow-up to round 2's fix. Built
`paperA_data/scripts/real_hw_replay_feedback_close.py`: replays joint_pos +
gripper faithfully only through an auto-detected pre-close settled pose (the
end of the longest near-constant run in the gripper channel before its
global minimum -- validated to land on the same index, 1401/2530, across
three independent trajectory files, which cross-checks against
`_settle_at_pose`'s fixed 400-step structure), then closes the real gripper
in small increments driven by real Feetech `Present_Load` feedback (read via
`backend.bus`, not exposed by `SOARMRealBackend`'s public API) plus a
position-stall backup, stopping on genuine contact resistance instead of
blindly replaying the sim-only "closed to ~0" signal documented in the round
2 entry below.

**Calibration (`real_hw_calibrate_load.py`)**: closing the real gripper fully
with nothing between the jaws gave a clean, low, consistent baseline
(`Present_Load` magnitude 32-48 through most of the travel, rising to 68 only
at full mechanical closure) with tight position tracking throughout (no
stall). Set `--load-threshold 120` (~2x the empty-close ceiling) for the live
attempt.

**Live attempt** (same trajectory/marker setup as round 2, `--speed 0.5`):
replay executed without error, but the feedback-close phase saw a load
profile *indistinguishable from the empty-jaws calibration* (32-52, rising to
68 at full closure, no stop triggered) -- i.e., nothing was between the jaws
by the time closing started. User's direct report explains why: "运动过程太快，
东西刚碰到被甩出去了，碰到东西没有抓起来只甩臂" (the motion was too fast --
the arm made contact with the object and flung it away before ever grasping
it). No damage / no one hurt (confirmed with user).

**Interpretation**: the object wasn't missing from the start (round 2's
orientation-deviation finding is still real and likely contributed to the
approach being off-target), but the dominant proximate cause of *this*
specific failure is that the pre-close **approach** segment (idx 0-1401,
replayed at 0.5x recorded sim speed) swept the arm through/near the object's
actual real-world position fast enough to physically knock it away, rather
than settling near it -- by the time the new load-feedback closing logic
started monitoring, there was nothing left to detect. This is a new,
safety-relevant failure mode distinct from both round 1 (calibration/
placement) and round 2 (sim-only closing/orientation shortcuts): **naive-
speed replay of a trajectory computed for a slightly different real object
position is not just imprecise, it can be an unsafe collision risk at the
approach phase**, independent of whatever happens at the gripper.

**Can support**: the feedback-driven closing mechanism itself works as
designed -- it correctly reported "no resistance detected" rather than
falsely claiming a successful grasp, which is exactly the failure mode round
2 flagged trajectory replay as prone to. The auto-detected pre-close index
heuristic is validated and reusable.

**Cannot support**: a working end-to-end real-hardware grasp yet. Before the
next attempt: replay the pre-close (approach) segment substantially slower
than 0.5x -- possibly much slower, e.g. 0.1-0.15x -- so any incidental
proximity to a not-quite-correctly-placed real object is gentle enough to
observe and abort rather than a fast sweep-through. This is now a prerequisite
for safety, not just precision.

## ❌ Phase 3 real-robot pilot, round 2 (2026-07-10): fixture controlled for object placement, grasp still failed -- root cause is NOT calibration/placement, it's two sim-only shortcuts baked into the recorded trajectory itself

Follow-up to the round-1 pilot below, after ruling out shoulder_pan calibration
and setting up a physical marker/fixture (user-confirmed placement, x=0,
y=-0.20 in simulation coordinates) to control for the object-placement
confound. Replayed a second consensus trajectory chosen specifically because
its simulated Pear settled-CoM position was the closest match to the marker
(orient_seed=3, gen_seed_base=1, `trajs/pear_consensus_orient3_gen1.json`,
`success=True` in sim). Replay executed technically without error (2530
points, clean disconnect). **Physical result**: user's direct observation --
"没有，他不是从上往下抓的，是从底部抓的，而且我没有放置托盘" (did not grasp;
approach was not top-down, it grasped from the bottom) and, on follow-up,
"机械抓夹是向上翘的" (the gripper was tilted upward).

Controlling for object placement did not fix the failure, which rules out
"imprecise manual object placement" as the dominant explanation this time
(round 1's leading hypothesis) and points to something wrong with the
recorded trajectory itself. Diagnosed by reading the trajectory JSON directly
(`trajs/pear_consensus_orient3_gen1.json`) against `tango_robot/env_soarm.py`
source -- no new simulation run needed. Found two distinct, real, and
previously undocumented mechanisms, both sim-only shortcuts with no
real-hardware analog:

**1. The final descent IK does not constrain gripper orientation to
top-down.** At the trajectory's actual grasp point (idx≈1138-1265, the
lowest-z hold, confirmed distinct from the later tray-placement point at
idx≈2450-2529 which matches `TARGET_ZONE_POS=(0.20,0.25)` exactly),
`shoulder_lift+elbow_flex+wrist_flex = +25.3°`, versus `0°` for `HOME_QPOS` /
a strict top-down pose. `_settle_at_pose`'s descent phase uses
`_solve_ik_jaw_pos_only` (position-only IK), and the code's own comment
explains why: "jaw_topdown was tested and reverted twice ... the orientation
constraint corrupts jaw centering at SO-ARM101 reach limits, causing
symmetric objects (**Pear**) to regress catastrophically" (`env_soarm.py`
~line 1737-1741). This is a deliberate, documented design choice for
simulation success-rate reasons -- but it means the actual deployed grasp
orientation for Pear is **not** guaranteed to match the "roll=π, pitch=0,
fixed top-down" assumption stated in `paper_final.tex`'s Limitations section.
This trajectory's IK solution happened to converge to a +25° pitch deviation,
which replayed on real hardware as the visually-reported "gripper tilted
upward."

**2. `GRASP_MODE_PHYSICS_WELD`'s post-close behavior has no real-hardware
analog.** Per `env_soarm.py`'s own design comment (~line 97-109): bilateral
contact at the moment of closing gates a kinematic weld that then teleports
the object to follow the end-effector during lift/transport, because "6mm
sphere colliders cannot generate sufficient friction force to lift ~0.15kg
objects against gravity." The bilateral-contact gate at close time is a real
physical check, but once the weld engages, the object's simulated position is
decoupled from true jaw width -- and the recorded gripper-opening channel
keeps commanding full closure (0.001, near `GRIP_CLOSED`) through the entire
transport phase (idx≈1600-2150 in this trajectory) regardless. Replaying that
"closed to 0.001" command sequence on the real gripper, which has no
kinematic weld to fall back on and must hold the object by genuine friction
alone, is not the same physical event as what produced a `success=True` label
in simulation.

**Can support**: the mechanism (record in sim, replay on hardware) still
executes technically without error -- this is an execution-fidelity finding,
not an infrastructure failure. Both root causes are readable directly from
already-recorded data and existing source comments, no new hardware trials
needed to confirm them.

**Cannot support**: naive joint-space trajectory replay of a
`physics_weld_after_bilateral` grasp as a valid sim-to-real transfer method
for this project's real-hardware validation plan, as currently designed. The
round-1 pilot's calibration/placement explanation is superseded for this
specific failure mode -- shoulder_pan calibration and object placement were
both controlled for in round 2, and the grasp still failed for reasons
intrinsic to the recorded trajectory. Phase 3's real-robot plan
(`/home/lina/.claude/plans/floating-crunching-yeti.md`) needs to be revisited
before further hardware trials: either avoid relying on the weld-assisted
close/transport signal (e.g., replay only through the pre-close settled pose
and let the real gripper's own contact/current feedback drive closure), or
scope the real-hardware claim down to what trajectory replay can actually
validate (arm reaches the intended pre-grasp pose) rather than full
pick-success reproduction.

## ⚠️ Phase 3 real-robot pilot, round 1 (2026-07-10): mechanism works, open-loop replay does not transfer precisely -- confirms the anticipated sim-to-real gap

First physical-hardware test of the year-long roadmap
(`/home/lina/.claude/plans/floating-crunching-yeti.md`, Phase 3): validated
consensus candidate selection (Pear, orient_seed=5, gen_seed_base=1, a
confirmed-successful trial from the same-day main-branch re-verification
above) by recording its full joint-space trajectory in simulation
(`paperA_data/scripts/record_consensus_grasp.py`, driving the real
`RobotEnvUI`/`demo.py` pipeline directly rather than re-deriving its logic --
see that script's docstring for why a standalone re-derivation attempt failed)
and replaying it on the physical SO-ARM101
(`paperA_data/scripts/real_hw_replay.py`).

**Infrastructure verified working end-to-end**: serial connection, motor
handshake, calibration loading (found at lerobot's default user-cache
location, from a prior calibration run in a separate project), joint-position
read/write, a minimal single-joint motion test (±10° wrist_roll, other 4
joints held exactly at 0.00° delta), and full 2530-point trajectory replay
(reset → move_joints → set_gripper sequence) -- all completed without errors,
safe disconnect (torque disabled) after each step. Two real, non-hardware
environment bugs found and fixed along the way (recorded in
`paperA_data/scripts/real_hw_connect.py`'s docstring): this project's own
`datasets/episode.py` collides with the HuggingFace `datasets` package
`lerobot` needs internally (worked around via import ordering, no source
files modified); `scservo_sdk` (`feetech-servo-sdk`) was not installed in the
`tango` env (installed via the exact extras pin from
`/lena/projects/lerobot/pyproject.toml`).

**Physical execution result**: the arm correctly executed the full recorded
motion sequence (approach → descend → close → lift → move to place), but the
grasp itself did not land on the real object -- user's direct observation:
"角度太偏左了，抓取位置也不同步" (the angle is off to the left, and the grasp
position is not synchronized [with the real object]), with the overall motion
shape (grasp on the left side, then move right for placement) matching the
trajectory's intended structure.

**Can support**: the mechanism -- record a consensus-selected trajectory in
sim, replay it on hardware -- works correctly as an *execution* pipeline (the
arm faithfully reproduces the recorded joint sequence). It does **not**,
on this first attempt, reproduce simulation's grasp *precision*, because
`SOARMRealBackend.execute_grasp()` is intentionally unimplemented (per its
own docstring: "Online IK on real hardware is outside this module's scope")
-- replay is pure open-loop joint-sequence reproduction with zero real-time
perception of where the physical object actually sits. The recorded
trajectory's target coordinates were computed relative to simulation's
randomly-spawned object position; reproducing that exact position by manually
placing a real object is effectively impossible to do precisely by eye. This
is exactly the sim-to-real gap `paper_final.tex`'s Limitations section
already names ("Sim-to-real transfer requires alignment of the physical
SO-ARM101's calibration, contact dynamics, and depth-camera intrinsics with
the simulation geometry") -- now confirmed empirically rather than
anticipated.

**Follow-up diagnostic (same session): base-rotation calibration checked out.**
To isolate "calibration/coordinate-frame mismatch" from "object not placed
where simulation assumed" as competing explanations for the reported angle
offset, ran a targeted, minimal test
(`paperA_data/scripts/real_hw_check_shoulder_pan_zero.py`): commanded ONLY
`shoulder_pan` (the base-rotation joint, most directly responsible for a
left/right bias) to 0 degrees -- simulation's `HOME_QPOS` zero for this joint,
which `robots/soarm_real_backend.py`'s `HOME_DEG` is explicitly derived from
-- leaving all other joints untouched. User's direct visual confirmation:
the arm pointed straight ahead ("向前"), matching the expected "0 = facing
forward" convention. **This rules out a base-rotation calibration/coordinate-
frame bug** as the primary explanation, despite the calibration file being
imported from a different project (openvla) rather than calibrated
in-project -- and despite the user having recalibrated multiple times in
that other context without resolving the symptom, which is itself consistent
with the root cause being elsewhere (recalibration wouldn't fix an
open-loop object-placement gap). The reported "angle offset" during the
full trajectory replay is therefore best explained by the already-identified
dominant cause -- open-loop replay reproducing a trajectory computed for
simulation's object position, not the real object's actual (manually,
imprecisely placed) position -- not by a separate calibration defect.

**Cannot support**: a success-rate claim for consensus selection on real
hardware from this single n=1 pilot in either direction -- this entry
documents infrastructure validation and a qualitative failure mode, not a
statistically meaningful physical replication of the simulation finding. Also
cannot rule out smaller, joint-specific calibration inaccuracies beyond
shoulder_pan (only the base-rotation joint was isolated and checked) -- if a
future attempt controls for object placement (e.g., a fixture that
guarantees precise, repeatable positioning) and the grasp still misses
systematically, revisit per-joint calibration for the remaining 4 joints.

## ✅ RESOLVED (2026-07-10): Pear ikmargin-vs-consensus finding re-verified on confirmed main-branch code -- exact match

Before using the Pear consensus finding (ikmargin 6.0% vs consensus 68.0%, Fisher's
exact p=5.8e-11 -- the flagship result behind the year-long roadmap's Phase 3 real-robot
plan) as the basis for physical hardware validation, checked whether it might be
affected by the seeding-bug timeline: `--consensus-n`/`--ikmargin-n` (and the
`sample_poses(seed=...)` parameter they structurally require) were only ever added to
`tango_robot/ui.py`/`demo.py` in commit `10814cd` (2026-07-09 23:44, the seeding-bug fix
commit), but the file mtimes of the original Pear data
(`phase1_v2/ikmargin_Pear.jsonl`, `phase1_matched_n10/consensus_n10_Pear.jsonl`) show
they were generated 2026-07-08 09:37-10:46 -- over a full day *before* that commit. The
data-collection scripts ran from `.claude/worktrees/fix-eval-seeding`, whose exact
uncommitted code state at that specific timestamp can't be reconstructed from git
history alone (worktrees accumulate uncommitted changes; the commit date only marks
when work was finally formalized, not when it started working).

**Re-ran the identical protocol** (`paperA_data/scripts/reverify_pear_ikmargin_vs_consensus.sh`:
orient_seed 5-9 × gen-seed base 1,11,...,91, n=10 ensemble, `cfm_allobj_ot.pt`) on
current main-branch HEAD (confirmed to include the seeding fix). Result:

| | ikmargin | consensus | Fisher's exact p |
|---|---|---|---|
| Original (2026-07-08, worktree) | 6.0% (3/50) | 68.0% (34/50) | 5.8e-11 |
| Re-verification (2026-07-10, main branch) | 6.0% (3/50) | 68.0% (34/50) | 5.81e-11 |

**Can support**: the worktree's code at data-collection time already had the complete
seeding fix -- the match is exact (same counts, same p-value to 3 significant figures),
not just directionally consistent. This is also a nice independent confirmation that
the fix itself produces fully deterministic, reproducible results given the same seeds.
The Pear consensus finding is trustworthy and ready to serve as the basis for Phase 3
real-robot validation (`/home/lina/.claude/plans/floating-crunching-yeti.md`).

**Cannot support**: that every other pre-fix-commit data file in this repository is
automatically safe by the same reasoning -- this check was specific to Pear's
ikmargin/consensus data. If a similar timeline question comes up for other data
generated from the same worktree before 2026-07-09 23:44, re-verify rather than
assume the same conclusion transfers.

## ❌ CONCLUDED (2026-07-10): Phase 1 (MPC-style real-time correction world model) does not work at this data scale -- three consecutive physical pilots net negative

**Scope note**: this section documents work on the *next* project (a year-long
follow-on roadmap, `/home/lina/.claude/plans/floating-crunching-yeti.md`, pursued
after `paper_final.tex` was finalized for RA-L submission), not `paper_final.tex`
itself. Recorded here because it reuses this session's infrastructure, conventions,
and objects, and because the roadmap's Phase 0 explicitly locks `paper_final.tex`'s
numbers as the reference baseline for everything below.

**Motivation and mechanism.** The roadmap's Phase 1 was originally "retrieval-augmented
EBM," then briefly "world-model rollout scorer," then corrected (after checking current
world-model literature) to **MPC-style real-time correction**: a real, previously
unused decision point exists in `tango_robot/env_soarm.py`'s `physics_weld_after_bilateral`
grasp mode -- right before closing the gripper, the code already computes
`get_grasp_debug_metrics()` (including `jaw_obj_xy_gap`, the same metric that
root-caused EBM v1's catastrophic failure earlier this session) but discards it and
closes unconditionally. The idea: train a small model to predict the outcome of a
local correction (Δx, Δy, Δyaw) applied just before closing, search a handful of
candidate corrections, and apply the best one -- a genuine real-time decision, not a
static candidate-scorer wearing a "world model" label.

**Two real bugs found and fixed during this work (both worth remembering independent
of the final outcome):**

1. **Directional information loss.** The first training pass used the scalar
   `base_jaw_gap` as the only context feature -- this can say *how far off* the
   current settle is, but not *which way* to correct. Top-1 delta-selection accuracy
   was 16.7%, barely above the ~11% chance level for 9 options/group. Fixed by adding
   the directional offset vector (`base_off_x`/`base_off_y`, jaw midpoint minus object
   centre) as a feature -- accuracy jumped to 37.5%.
2. **Spawn-range train/deployment mismatch.** `paperA_data/scripts/collect_mpc_correction_data.py`
   copied its object-spawn constants from `scripts/record_trajectory.py`
   (`_CENTRE_Y=-0.40`, spread `0.06`, giving object y in `[-0.46,-0.34]`) without checking
   they matched the actual evaluation harness. `tango_robot/ui.py`'s real spawn path
   (`env.load_isolated_obj`, the one every physical pilot and the whole paper's Baseline/
   OT-CFM/EBM numbers go through) uses `r_x=uniform(-0.15,0.15)`, `r_y=uniform(-0.35,-0.10)`
   -- these two ranges barely overlap. The correction model was trained on object
   positions from a different part of the workspace than the one it was asked to correct
   at physical-pilot time. Found only because the user explicitly asked to sanity-check
   simulation object/arm positions (a good instinct -- worth remembering: **when an ML
   fix doesn't transfer from offline metrics to physical results, check for a data-
   generation/deployment distribution mismatch before concluding the method itself is
   wrong**). Verified the rest of the scene geometry (`ROBOT_BASE_POS="0 0 0.785"` at the
   table's front edge, `ee_position_limit` centred on the base, `TARGET_ZONE_POS` on the
   opposite side with its own floor geom) is internally self-consistent -- no further
   geometry bugs found.

**Three physical pilots (3-object, n=25, Pear/TomatoSoupCan/CrackerBox, paired same-seed
vs. the locked `pilot_baseline_*.jsonl`), all net negative:**

| Pilot | Fix applied before this run | Pear | TomatoSoupCan | CrackerBox | Pooled |
|---|---|---|---|---|---|
| Round 1 (gap-regression target, wrong spawn range) | directional feature | 64.0%→36.0% (p=0.092) | 88.0%→96.0% | 40.0%→32.0% | **−9.3pp** |
| Round 2 (bilateral-classification target, wrong spawn range) | retarget to predict success directly | 64.0%→28.0% (p=0.023, sig.) | 88.0%→80.0% | 40.0%→28.0% | **−18.7pp** |
| Round 3 (bilateral target, fixed spawn range) | spawn-range fix | 64.0%→36.0% (p=0.065) | 88.0%→84.0% | 40.0%→32.0% | **−13.3pp** |

(Raw per-seed data: `paperA_data/worldmodel_trajs/pilot_mpc_correction_*.jsonl` (round 1),
`pilot_mpc_correction_bilateral_*.jsonl` (round 2), `pilot_mpc_correction_v2_*.jsonl` (round 3),
all compared against the same locked `pilot_baseline_*.jsonl`.)

Each fix was well-motivated and each offline validation metric improved substantially
(37.5%→72.7%→50.0% top-1 delta-selection accuracy across the three model variants,
all far above the ~11% chance level) -- **but none of it translated into a better, or
even a neutral, physical result.** Fixing the second bug (spawn-range mismatch)
improved the pooled number somewhat (−13.3pp vs. −18.7pp) but did not flip the sign.

**Can support:** the correction mechanism, as designed and implemented (small MLP,
1080-row/~120-candidate-group dataset, local ±0.03m/±0.2rad search range), does not
reliably improve physical grasp success at this data scale -- it is net harmful, and
the harm concentrates overwhelmingly on Pear (the object with the fewest training
examples throughout this entire session). Pear's fragility is now corroborated by
**five independent diagnostics** across this session and the follow-on work: OT-CFM's
largest per-object failure, Stratified-OT's one completely-unmoved object, EBM v2's
hyperparameter sensitivity (72%→48% under a stronger mining schedule), and both
correction pilots (rounds 2 and 3) here. **Offline validation metrics (MAE, top-1
delta-selection accuracy) did not predict physical outcome in any of the three
rounds** -- this is itself a transferable methodological lesson: a proxy-metric
offline gate, however much it improves, is not a substitute for a physical pilot,
and should not be treated as sufficient evidence before scaling to a full evaluation.

**Cannot support:** that a larger dataset, a different architecture, or further
hyperparameter tuning would fix this -- none of those were tried, and the pattern
(three different, individually-reasonable fixes, three negative results) is
suggestive but not proof that the mechanism is fundamentally unworkable at any scale.
Also cannot support a specific mechanistic explanation for *why* Pear is uniquely hard
across so many unrelated methods -- geometry (round/symmetric, easy to overshoot a
correction into a worse configuration) and data scarcity (fewest positive examples)
are both plausible, untested hypotheses.

**Decision:** stop iterating on this specific mechanism (per the roadmap's own
pre-registered risk log, which anticipated this exact outcome and its honest framing).
The infrastructure built here -- `_settle_at_pose()` (the reusable settle-and-measure
primitive in `env_soarm.py`), the data collection and training scripts, and the
now-corrected spawn-range convention -- remains available for Phase 3 (real-robot
validation) or a redesigned Phase 1, regardless of this outcome. The roadmap's Phase 2
(6-DoF extension) was scoped as "extend Phase 1's model," which no longer has a
working Phase 1 to extend -- needs revisiting before proceeding, separately from this
write-up.

## ⚠️ EBM v2 hyperparameter robustness check: Pear specifically is sensitive, TomatoSoupCan/CrackerBox are not (2026-07-10)

Follow-up to the paper's own flagged Limitation ("single hyperparameter setting each for Stratified-OT
and EBM v2, neither was swept") — before submitting, checked whether EBM v2's reported parity result
(74.0% vs. baseline's 77.7%, 6 objects, p=0.294) is specific to the one hard-negative-mining schedule
reported, or holds under a meaningfully different choice.

**What changed**: retrained a variant, `ebm_allobj_v2b.pt` (`train_ebm_grasp.py`, same InfoNCE +
static/uniform/hard-negative recipe as v2), with a stronger adversarial-mining schedule —
`EBM_K_STATIC=3 EBM_K_UNIFORM=3 EBM_K_HARD=6 EBM_HARD_POP=64 EBM_HARD_ITERS=6` (v2 was
`K_STATIC=4 K_UNIFORM=4 K_HARD=4 HARD_POP=32 HARD_ITERS=3`) — more self-mined adversarial pressure,
less static/uniform coverage. Converged cleanly (positive-ranked-top1 acc 0.819, vs. v2's 0.777).
Smoke-tested (2 trials, Pear seeds 1-2, `--verbose 1`, confirmed candidates and rankings vary
seed-to-seed, not a degenerate fallback) before running the full batch
(`scripts/run_ebm_v2b_check_fixed_code.sh`, same 3 objects/25 seeds as every other diagnostic this
session: Pear/TomatoSoupCan/CrackerBox).

**Result** (raw: `phase0_diag_extended/ebm_v2b_check_*.jsonl`; existing v2 numbers from
`ebm_v2_check_*.jsonl` for direct, same-seed comparison):

| object | EBM v2 (existing) | EBM v2b (stronger mining) | paired McNemar exact p |
|---|---|---|---|
| Pear | 72.0% (18/25) | **48.0% (12/25)** | 0.146 (ns at n=25, but 9 v2-only vs 3 v2b-only discordant pairs) |
| TomatoSoupCan | 100.0% (25/25) | 100.0% (25/25) | 1.0 (byte-for-byte identical, same seeds) |
| CrackerBox | 28.0% (7/25) | 28.0% (7/25) | 1.0 (byte-for-byte identical, same seeds) |

- **Can support**: TomatoSoupCan and CrackerBox's EBM v2 numbers are robust to this hyperparameter
  change — literally identical outcomes per seed, not just similar aggregate rates. **Pear is not**:
  a −24pp point-estimate drop under stronger mining, not statistically significant at this sample size
  but too large to dismiss as noise. Pear is also the object with the fewest training positives (248,
  lowest of all 7 — see `train_ebm_grasp.py`'s dataset printout) and the one Stratified-OT could not
  move at all (see the OT-coupling section above) — three independent diagnostics now converge on
  "Pear specifically is fragile," which is a more precise and more defensible claim than the paper's
  previous blanket "neither was swept."
- **Cannot support**: that EBM v2's reported parity result is fully hyperparameter-robust in general —
  it demonstrably is not, for Pear. Also cannot support a causal mechanism for *why* stronger mining
  hurts Pear specifically (small-data instability in the mining CEM itself vs. something else) — not
  tested here.
- **Paper impact**: `paper_final.tex`'s Limitations section rewritten (was: "we report one
  hyperparameter setting each... neither was swept") to report this finding directly, staying within
  the 8-page hard limit (recompiled and reverified after the edit: `latexmk -pdf && pdfinfo`).
  Also fixed a real RA-L compliance gap while reviewing formatting requirements: `IEEEkeywords` had
  9 entries, exceeding RA-L's stated 2–5 keyword limit — trimmed to 5. Double-blind compliance
  (no author/affiliation leaks, no identifying links, self-citation `owg2024` already anonymized as
  "[Author(s) omitted for blind review]") was checked and found already compliant. Note for the
  authors: RA-L is 6 free pages + up to 2 extra pages with page charges (8 max) — this paper is
  exactly at 8, so it will incur the maximum extra-page charge; trimming to 6 was not attempted given
  the amount of content, but is a cost/quality tradeoff the authors should be aware of before
  submission.

## ✅ ROOT CAUSE FOUND AND FIXED (2026-07-09/10): the seeding bug behind every instability finding below, full clean re-evaluation, and the paper's final (honest, mixed) narrative

**This section supersedes the "broader implication, not resolved here" note at the bottom of the
TomatoSoupCan section directly below.** That note asked whether a documented pattern of run-to-run
non-determinism (Scissors' 3 different measurements, TomatoSoupCan's reversal at n=50) pointed to
something systemic. It did: `sample_poses`/`sample_poses_ddpm` (the CFM/DDPM inference-time
samplers) never received a per-trial `seed` parameter in `tango_robot/ui.py`'s main code path.
Combined with a module-level `torch.manual_seed(42)` in `train_cfm_grasp.py`/`train_diffusion_grasp.py`
(executed once on import), every trial in a freshly-started process drew generator noise from
whatever state that fixed seed left behind at import time -- **independent of the trial's nominal
`--seed`**. Concretely: 25 nominally-distinct OT-CFM trials per object were, in practice, close to
25 repeats of one candidate. Baseline (random-CoM sampling) was unaffected -- it draws from
`numpy`'s properly-seeded generator, not the torch-seeded generator network -- which is exactly why
only the learned-generator conditions showed suspiciously narrow, high success rates in every
earlier draft of `paper_final.tex`, and why Scissors' instability and TomatoSoupCan's reversal
(both documented below, both found *before* this root cause was known) only ever showed up as
individually-explained anomalies rather than a single mechanism.

**Fix**: developed on a separate worktree (`worktree-fix-eval-seeding`, commit `10814cd`) and
merged into `main` (commit `1434533`, 2026-07-09) after the user chose, when presented with the
alternative of a narrower patch, to fully merge and re-run rather than leave main and the
worktree diverged. `demo.py`/`tango_robot/ui.py`/`train_cfm_grasp.py`/`train_diffusion_grasp.py`
now pass a real per-trial `seed=` into both the spawn-orientation draw (`np.random.seed`) and the
CFM/DDPM sampling call itself.

**Before deciding how to write the paper, checked for other bugs of the same shape** (per explicit
user instruction) rather than assuming this was the only one: re-verified LGGSN reranking, the
spawn-position fix, and DDPM's own sampling path. None had a second instance of the same class of
bug. Concluded the fix was complete and proceeded to a full clean re-evaluation.

**Full clean re-evaluation** (all 7 objects, `n=50`/object/condition, 350 trials/condition pooled,
fixed code, `tango` env — scripts `run_clean_seed1_25_6obj.sh` + `run_clean_scissors_seed26_50.sh`
+ prior Scissors recheck data; aggregated by `scripts/run_final_paper_stats.py` ->
`formal_results/final_paper_all_methods.csv`):

| Comparison | Baseline | OT-CFM / EBM v2 | Δ (pp) | z | p |
|---|---|---|---|---|---|
| Pooled, 7 objects, OT-CFM | 79.1% (277/350) | 69.1% (242/350) | **−10.0** | −3.02 | **0.0025** |
| Pooled, 6 objects excl. Scissors, OT-CFM | 77.7% (233/300) | 66.0% (198/300) | **−11.7** | −3.18 | **0.0015** |
| Pooled, 6 objects excl. Scissors, EBM v2 | 77.7% (233/300) | 74.0% (222/300) | −3.7 | −1.05 | 0.294 (ns) |

**This reverses the paper's original headline claim** (previously reported as a ~+12pp OT-CFM win
built on the unseeded 175-trial evaluation) **to a significant loss.** This is the single most
consequential finding of the whole session and is now the paper's actual thesis.

**Diagnosis, not just reporting the reversal**: rather than discard OT-CFM, used the fixed pipeline
as a diagnostic instrument. Three-object (Pear/TomatoSoupCan/CrackerBox) controlled comparisons at
`n=25` (`run_removeOT_check_fixed_code.sh`, `run_ddpm_check_fixed_code.sh`,
`run_stratifiedOT_check_fixed_code.sh`) isolate the failure to minibatch OT coupling specifically:

- **Remove-OT** (identical architecture/data, no OT coupling): Pear 72% (= baseline exactly),
  TomatoSoupCan 100% (baseline 92%), CrackerBox 28% (baseline 42%, the one object no method
  recovers on).
- **DDPM** (same data, diffusion instead of flow matching, still no OT coupling): same pattern —
  tracks baseline far more closely than OT-CFM.
- **Stratified-OT** (per-object OT coupling instead of mixed-minibatch coupling — the discrete-class
  limit of C²OT's condition-aware fix, Cheng & Schwing, ICCV 2025, arXiv:2503.10636; implemented via
  `CFM_STRATIFY_OT=1` in `train_cfm_grasp.py`): partially recovers TomatoSoupCan (78%→88%, still
  below baseline) and CrackerBox (28%→44%, now above baseline, but CrackerBox is noisy for every
  method), but **leaves Pear completely unmoved** (52%, identical to unfixed OT-CFM) — the object
  where OT-CFM's failure is largest. Read as: the fix is not wrong in principle, but Pear has the
  fewest training examples of the three objects and a per-object OT sub-batch of that size may be
  too small for any coupling scheme to help.

This matches Cheng & Schwing's mechanism (condition-agnostic minibatch OT coupling creates a
training-time prior skewed by which noise samples the solver assigned to which object, that
inference-time sampling — drawing from the full unconditional prior — was never trained to match)
almost exactly, and localizes the problem to the coupling step, not to conditional flow matching or
learned candidate generation in general.

**EBM v1 → v2**: separately implemented an energy-based scoring alternative
(`train_ebm_grasp.py`, `tango_robot/ui.py`'s `_load_ebm_model`/`_ebm_sample_candidates`, CEM search
over (x,y,yaw)), motivated by Implicit Behavioral Cloning (Florence et al., CoRL 2021) as a
small-data-friendlier alternative to an ODE/SDE generator.

- **v1 (naive, static contrastive labels, `ebm_allobj.pt`)**: catastrophic failure, 0–16% success.
  Root-caused via a standalone diagnostic script replicating the CEM loop: search converged to
  `mean_xy≈[1.67,−0.26]` normalized (~8.5cm real-world offset from CoM, matching the observed
  ~9.8cm gap in failed physical trials) where the model gave artificially high confidence
  (logit=4.55) despite that region being entirely unconstrained by training data — the textbook IBC
  "unconstrained energy exploitation" failure: inference-time search finds and exploits any region
  the training loss never penalized.
- **v2 (adversarial hard-negative mining, `ebm_allobj_v2.pt`)**: rewrote training to mine hard
  negatives from the model's own current beliefs during training (short CEM search each step,
  `K_HARD=4` per positive alongside `K_STATIC=4` logged failures and `K_UNIFORM=4` uniform-coverage
  negatives, InfoNCE/cross-entropy loss) — exactly the mechanism IBC prescribes and v1 omitted.
  Verified via the same standalone diagnostic: CEM now converges near the true CoM (<2mm offset)
  with a modest, well-calibrated confidence (logit≈−0.05), not a confidently-wrong one.
  Full 6-object (excl. Scissors) `n=50` evaluation (`run_ebm_v2_full7obj_n50.sh`): **74.0% vs.
  baseline's 77.7% (p=0.294, not significant)** — statistical parity, not a clean win, and
  significantly *beats* baseline on exactly one of six objects. Reported as a mixed, honest result.

**Final paper narrative** (fully rewritten in `paper_final.tex`, new title "When Generative
Candidates Do Not Beat Random Sampling: A Seeding-Bug Audit and Diagnosis for 6-DoF Grasp
Generation"): no generative candidate method tested (OT-CFM, Remove-OT, DDPM, Stratified-OT, EBM
v1, EBM v2) reliably and significantly beats a well-reranked random baseline in this small-data
(~400 examples/object), single-GPU regime. The one significant effect in either direction is
OT-CFM's specific, avoidable harm from condition-agnostic minibatch OT coupling. The one method
that does robustly help is training-free: consensus candidate selection from an ensemble of
independent draws (documented in the ikmargin-vs-consensus section below). This is reported as the
paper's actual contribution — a confirmed evaluation bug and its consequence, a physically-grounded
transfer of a generative-modeling diagnosis (to our knowledge the first on physically executed
robot task success rather than a generation-quality proxy), and a documented EBM failure-and-fix
pair — rather than spun as a win. Compiles cleanly to exactly 8 pages (RA-L's hard limit, references
included, zero headroom left) as of 2026-07-10, `latexmk -pdf paper_final.tex && pdfinfo`.

## ⚠️ TomatoSoupCan's Table II direction reverses at n=50 -- not applied to the table, disclosed as a limitation instead (2026-07-09)

Follow-up to the Scissors instability finding below: while investigating whether the same kind
of cross-run drift affects other objects (motivated by finding `logs/eval_baseline_nosem_v2.log`
gives PowerDrill baseline 17/25 vs. the adopted run's 19/25 -- not chased further, out of scope),
extended TomatoSoupCan and PowerDrill from 25 to 50 seeds (25 new seeds each, both Baseline and
OT-CFM+LGGSN conditions, exact harness reverse-engineered from the original source logs' own
printed headers: `logs/eval_baseline_nosem.log` and `logs/eval_cfm_ot_nosem_current.log`, both
confirmed byte-for-byte matching Table II's published counts before extending). Smoke-tested
first (seed=26, both objects, both conditions, all 4 ran and completed cleanly) before committing
to the full 100-trial batch (`scripts/run_seed26_50_tomatosoupcan_powerdrill.sh`, 0 timeouts).

**TomatoSoupCan's OT-CFM rate on the 25 *new* seeds is 76% (19/25)** -- sharply lower than the
original 25 seeds' 100% (25/25). Pooled to n=50: **Baseline 94.0% (47/50) vs. OT-CFM 88.0%
(44/50), a -6.0pp reversal** of the original +8.0pp direction (still not significant either way:
z-test p=0.29, Fisher p=0.49). The original 25-seed result was a favorable draw, not an
underpowered-but-real effect. PowerDrill's direction holds at n=50 (Baseline 72.0%[36/50] vs.
OT-CFM 84.0%[42/50], +12.0pp, still p=0.15/0.23, not significant).

**Decision (2026-07-09, user call)**: do not overwrite Table II's TomatoSoupCan/PowerDrill rows
with the n=50 figures -- doing so for only 2 of 7 objects would make the table's own stated
methodology ("25 seeds each") inconsistent across rows. Instead, `paper_final.tex`'s Per-Object
Analysis text now discloses the reversal directly and retracts the "suggesting real but
seed-limited headroom" claim that had been written into that same paragraph during the Tier 1
contact-feature work earlier in this session -- that claim is now known to be wrong, not merely
unproven. The contact-feature diagnostic's TomatoSoupCan finding (`local_point_density` p=0.00123)
is reframed as describing that diagnostic's own sample only, not evidence that OT-CFM finds
better landing positions than random sampling.

Formalized via `scripts/run_seed26_50_stats.py` ->
`formal_results/seed26_50_tomatosoupcan_powerdrill.csv`; raw per-seed data in
`phase0_diag_extended/seed26_50_{TomatoSoupCan,PowerDrill}_{baseline,otcfm}.jsonl`.

**Broader implication, not resolved here**: between this finding and the Scissors instability and
the PowerDrill-baseline-v2 discrepancy noted above, there is now a documented pattern of
run-to-run non-determinism affecting multiple objects and conditions in the 175-trial main
evaluation, not just Scissors. A full re-audit of all 7 objects at higher seed counts would be
needed to know how much of Table I/II/III is affected -- out of scope for this session, flagged
here for whoever picks this up next.

## ⚠️ Separate issue, same object: `paper_final.tex`'s own Scissors number was also unreplicated (found + fixed 2026-07-09)

This is unrelated to the CFM name-matching fallback bug below -- it affects `paper_final.tex`'s
*own* 175-trial main evaluation (Table I/II/III), which is a completely different dataset from
everything else in this directory (175 trials = 7 objects x 25 seeds, no `--gen-seed` variation,
run via `scripts/quick_eval.sh` in the `tango` conda env, which was `owg-mujoco` until a
`conda rename -n owg-mujoco tango` -- confirmed via `~/miniforge3/envs/tango/conda-meta/history`).

**What happened**: `paper_final.tex`'s Baseline (82.3%) and OT-CFM+LGGSN (94.3%) numbers were
built by adding a separately-measured 25-trial Scissors block to an existing 150-trial (6-object)
total, after `a04a62c`'s VHACD-tunnelling physics fix (see `logs/eval_scissors_fix_summary.log`).
That Scissors block was measured **25/25 = 100%** on 2026-06-26, with a real per-seed log
(`logs/eval_scissors_baseline.log`, `logs/eval_scissors_cfm.log` -- genuine `[✓]` per seed, not an
estimate). A later commit (`cf58a7d`, 2026-06-28) claimed a re-run found **23/25 = 92%** and
labeled the 06-26 figure "imputed" -- inaccurate framing (it wasn't imputed), but the underlying
discrepancy is real: two actual measurements of the same nominal condition gave different counts,
and `cf58a7d`'s corrected number was never propagated back into `paper_final.tex` (it doesn't
touch that file at all, and no later commit does either -- confirmed by `git log -- paper_final.tex`).

**Resolution (2026-07-09)**: rather than trust either historical number, re-ran the identical
condition from scratch (`scripts/run_scissors_recheck_2026-07-09.sh`, 25 seeds each,
baseline + OT-CFM, `tango` env, smoke-tested first). Result: a **third** different number,
**22/25 = 88%** for both conditions (same 3 failing seeds -- 1, 7, 10 -- in both, consistent
with Scissors falling back to the same random-CoM sampler in both conditions, per the CFM
name-matching issue below). The smoke-test itself already disagreed with the 06-26 log at the
exact same seed (seed=1: success there, failure here), which is why neither historical number was
trusted blindly. No configuration drift was found -- `configs/objects/ycb_mujoco_manifest.yaml`'s
Scissors entry is byte-identical to what `eval_scissors_fix_summary.log` describes. The most
likely explanation is that the 4cm box-proxy fix is only marginally above the gripper's 4cm
minimum opening, making this specific object's outcome sensitive to small run-to-run numerical
differences (contact-solver iteration order, floating-point accumulation) -- a property of this
object's geometry, not evidence that the pipeline itself is broken.

**Adopted the freshest measurement (22/25 = 88%) per explicit user decision**, and propagated it
through every dependent number in `paper_final.tex`: Table I (Baseline 82.3%→80.6%, OT-CFM
94.3%→92.6%, z 3.49→3.29), Table II (Scissors row 100%/100%→88%/88%, All row updated to match),
Table III (every "vs. Baseline" and "vs. Full" delta and p-value recomputed -- see below for which
ones changed qualitatively), abstract, intro contributions list, Related Work, Discussion ("Why OT
Coupling Matters" rewritten -- see below), and Conclusion. Formalized in
`scripts/run_scissors_recheck_stats.py` -> `formal_results/scissors_recheck_corrected_totals.csv`;
raw per-seed data in `phase0_diag_extended/scissors_recheck_{baseline,otcfm}.jsonl`.

**Every ablation-table significance classification (SIG/ns) is unchanged** -- only the absolute
percentages and precise p-values shifted, because the 6-object (non-Scissors) totals were held
fixed and Scissors' identical-across-conditions count (22/25 both) shifts every row's absolute
value by the same constant, preserving relative comparisons almost exactly. **One qualitative
claim did have to be softened**: the original text argued "standard CFM without OT coupling drops
*below* baseline (78.9% < 82.3%)" as supporting evidence that OT coupling matters. With the
corrected baseline (80.6%), that specific pairwise comparison is no longer statistically
significant (78.9% vs 80.6%, $p=0.69$) -- rewritten to rest the argument on the comparison that
remains robust throughout (Remove-OT vs. the full pipeline: $-13.7$pp, $p<0.001$), rather than the
now-weaker vs.-Baseline framing. Similarly, DDPM's relationship to baseline flips sign (was
$-0.6$pp, now $+1.1$pp) but stays non-significant either way, so no claim needed to change there
beyond the number.

**Not in scope for this pass**: the other 6 objects' Baseline/OT-CFM/GRC-6DoF/DDPM counts were
*not* independently re-verified (only Scissors was, per the specific discrepancy found) -- if a
similar cross-date drift exists for any other object, it has not been checked. GRC-6DoF's own
82.9% and Remove-OT/DDPM's own 78.9%/81.7% are carried forward unchanged from the existing record;
only their *deltas relative to Baseline/Full* were recomputed, since those two reference points
moved.

## ✅ RESOLVED (2026-07-09): Scissors excluded from Paper A, clean 5/6-object results published

The Scissors fallback bug described in the CRITICAL section below was confirmed by
re-executing the exact matching code and re-deriving the 40/50-tie and 0-discordant-trial
evidence independently on 2026-07-09. Decision: **exclude Scissors entirely from Paper A**
rather than attempt a same-day rerun. The other 6 objects (Banana/Pear/MustardBottle/
CrackerBox/PowerDrill/TomatoSoupCan) are unaffected — confirmed clean by running the same
matching code against all 7 object names.

What changed on disk:
- **New authoritative summary**: `formal_results/PAPER_A_CLEAN_SUMMARY.md` — the single file
  to cite from the paper. Consolidates the corrected contact-feature Bonferroni/BH table
  (15 tests, 5 objects) and the corrected exp1_variance method comparison (`ALL_excl_scissors`
  scope, 6 objects).
- **New file**: `formal_results/contact_features_bonferroni_bh_5obj_clean.csv`
  (`scripts/run_contact_features_stats_5obj_clean.py`) — Scissors dropped entirely from both
  the p-values and the correction family (15 tests, not 18). No significance conclusion
  changed vs. the old 18-test file (checked row-by-row).
- **Existing files annotated, not deleted**: `contact_features_bonferroni_bh_6obj.csv` and
  `exp1_variance_significance.csv` now carry an `excluded_reason` column — non-empty only for
  Scissors rows, pointing back to this README and to the clean replacement file. Both are kept
  on disk for provenance/audit but are no longer the files to cite.
- **New pooled scope**: `exp1_variance_significance.csv` gained an `ALL_excl_scissors` row
  (n=300/method, the 6 valid objects) alongside the original `ALL` (n=350/method, still
  includes the invalid Scissors rows, kept for comparison only). One conclusion changes here:
  OT-CFM vs. DDPM's unpaired tests move from non-significant (p≈0.051) to significant
  (p≈0.037) once Scissors' diluting 80%/80%/80% tie is removed. The paired McNemar result
  (the statistically preferred test) was already significant and is numerically unchanged,
  since Scissors contributed exactly 0 discordant pairs.

See `formal_results/PAPER_A_CLEAN_SUMMARY.md` for full numbers and the file map of what to
cite vs. what's kept for provenance only.


Copied verbatim (md5 verified) from the Claude Code job scratchpad
`/home/lina/.claude/jobs/b899ad73/tmp/` on 2026-07-08, because that directory
is not part of any git repo and is subject to cleanup. This is a straight
copy — no files were regenerated or edited.

## ⚠️ CRITICAL: every "Scissors" data point in this repo is not OT-CFM/CFM-noOT/DDPM data (found 2026-07-08)

`tango_robot/ui.py`'s `_cfm_sample_candidates()` matches an object name to a checkpoint's trained
conditioning key via `key = obj_name.lower()...`; `for k in vis_map: if k in key or key.startswith(k)`.
All three checkpoints (`cfm_allobj_ot`, `cfm_allobj`, `ddpm_allobj`) were trained on the identical 7-key
set `{banana, pear, mustard, cracker, drill, can, cylinder}`. `"scissors"` does not match any of these
keys (no substring/prefix relationship) — **confirmed by direct execution of the exact matching code**,
not inferred. When the match fails, `_cfm_sample_candidates` returns `None`, and `ui.py` silently falls
back to **uniform-random CoM-based candidate sampling** — the pre-CFM baseline, not any of the three
generative methods. Elsewhere in this same codebase (`scripts/collect_lggsn_data.py:443`) there IS an
explicit `"scissors": "cylinder"` alias for a different pipeline (LGGSN training-data collection) — the
runtime path used for every experiment in this repo (`demo.py` → `ui.py`) never implements that alias,
so the intended fallback (map scissors to the "cylinder"/`YcbMediumClamp` conditioning class) never
happens here.

**Evidence this actually occurred, not just a theoretical risk**: in `exp1_variance/`, Scissors' success
rate is *exactly* 40/50 (80.0%) for OT-CFM, CFM-noOT, **and** DDPM — identical to the decimal, whereas
every other object's rate varies across methods. The paired McNemar test in
`formal_results/exp1_variance_significance.csv` finds **zero discordant trials** between any pair of
methods on Scissors (`mcnemar_p=nan`, 0/0 in the discordant-count columns) — i.e., all three methods
produced the identical trial-by-trial outcome across all 50 trials, which is only possible if they were
all secretly running the same non-CFM fallback mechanism (indifferent to which checkpoint was loaded),
not three different generative models.

**Practical consequence**: exclude Scissors from any claim comparing OT-CFM/CFM-noOT/DDPM — its rows in
`exp1_variance_significance.csv` are internally correct arithmetic but describe "the random-sampling
fallback vs. itself under three labels," not a method comparison. The `phase0_diag_extended/` Scissors
contact-feature diagnostic data (added 2026-07-08, see below) is likewise **not characterizing OT-CFM's
behavior** — it characterizes the random-CoM fallback's behavior. This is a *different and more
fundamental* problem than the "thin/flat object, structural floor effect" note already on that data
(that note is still true as an independent observation, but secondary to this one). No other object
in this repo is affected — Banana/TomatoSoupCan/Pear/MustardBottle/CrackerBox/PowerDrill all match
their checkpoint's trained keys correctly (verified by running the exact matching logic against all 7).

## Methods reference — precise, code-grounded definitions (compiled 2026-07-08)

Answers to specific methods-section questions, each traced to the exact source line, not paraphrased
from memory. Worktree paths below are relative to
`/lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding` unless stated otherwise.

**1. "Generation-isolated" setup — GT identity or GT segmentation?**
Neither exactly, but closer to GT identity, and stronger than either: every trial spawns **exactly one
object** in the scene (`n_objects: 1` in the loaded config, confirmed via `[DEBUG] loaded objects:
[(1, 'Pear')]`-style prints in every run). All generation scripts also pass `--no-semantic`, which sets
`OWG_NO_SEMANTIC=1`; `tango/policy.py`'s no-semantic fast-path matches the text prompt directly against
the simulator's own `id_to_name` registry (`tango/policy.py:218-240`) — no vision model, no segmentation
mask, no GPT-4o grounding call. So there is no segmentation step to bypass: with one object in the scene
and its identity known from spawn time, "which object is the target" is never actually a vision problem
in this dataset.

**2. Exact success criterion**
The recorded `"success"` field = **`success_grasp`**, not `success_target` (whether it landed correctly
in the tray) — these are different and the data uses the former. Chain: `env_soarm.py`'s physics_weld
grasp routine (~line 1778) computes `success = bool(grasped_ids) and lifted`, where `grasped_ids` is
non-empty only if bilateral jaw contact was detected (`check_grasped_id()`) **and** a kinematic weld was
triggered (MuJoCo sphere colliders alone can't generate enough friction to lift against gravity, so the
sim rigidly attaches the object to the gripper once bilateral contact confirms a valid grasp), and
`lifted = obj_z > Z_TABLE_TOP + 0.07` (object center must rise >7cm after the lift move). **`table_contact`
is computed and printed but does NOT gate success** — confirmed by an actual trial with
`table_contact=False, success=True`. This `success_grasp` bool propagates up through
`put_obj_in_tray()` → `ui.py:step()`, which prints `"Done {action} {input}"` (i.e. `"Done pick 1"`) only
when `success_grasp` is true (`ui.py:756-762`) — this exact string is what every shell script's
`grep -q "Done pick"` checks. `success_target` (whether the object also ended up correctly in the tray)
is computed and logged alongside but is **not** what any script in this repo checks.

**3. Object selection rationale**
- `exp1_variance/`'s 7 objects = the full trained-object roster of the CFM/DDPM checkpoints (see #9's
  answer — `{banana, pear, mustard, cracker, drill, can, cylinder}`, mapped to
  Banana/Pear/MustardBottle/CrackerBox/PowerDrill/TomatoSoupCan/**"Scissors" incorrectly** — see the
  CRITICAL note above; there is no evidence this was a deliberate representative sample, it's simply
  "every object the checkpoints were conditioned on" (modulo the Scissors bug).
- The original 3-object diagnostic set (Pear/MustardBottle/CrackerBox) has **no documented rationale**
  found anywhere in the copied scripts or job history — it predates this archive and appears to be an
  earlier, unrecorded choice.
- The 3 objects added 2026-07-08 (TomatoSoupCan/PowerDrill/Scissors) were chosen explicitly because
  they're the only remaining `exp1_variance` objects with a real success/fail split under OT-CFM (Banana
  is 100% success, degenerate for a success-vs-fail comparison) — this rationale **is** documented
  (`phase0_diag_extended/` section below) and was correct at the time, but Scissors is now known to be
  invalid for an unrelated reason (the CFM name-matching bug), leaving 2 valid additions, not 3.

**4. OT-CFM sampling configuration**
`train_cfm_grasp.py:41`: `ODE_STEPS = int(os.environ.get("CFM_ODE_STEPS", "20"))` — **20 steps**, default
value, never overridden by any script in this repo (no script sets `CFM_ODE_STEPS`). Integrator: explicit
forward Euler (`train_cfm_grasp.py`'s `sample_poses`: `dt = 1.0/steps`; `x = x + model(t, x, cond) * dt`
per step) — not RK4 or an adaptive-step method.

**5. DDIM eta / DDPM_STOCHASTIC**
`train_diffusion_grasp.py:176`: `eta = 1.0 if os.environ.get("DDPM_STOCHASTIC") == "1" else 0.0`. No
script in this repo sets `DDPM_STOCHASTIC`, so **eta=0.0 — fully deterministic DDIM reverse sampling**
was used throughout, despite the checkpoint being called "ddpm_allobj.pt" and the data files being named
"DDPM". Step count: `DDIM_STEPS` defaults to 100 (`train_diffusion_grasp.py:42`) but
`experiment1_other_methods.sh` explicitly passes `DDIM_STEPS=50` — **50 steps were used**, not the
default 100. (The initial noise `x_T` is still drawn from a `--gen-seed`-seeded generator, so different
seeds still produce different candidates even though the reverse process itself is deterministic given
that noise.)

**6. Tests reported in `exp1_variance_significance.csv`**
Three tests per (scope, method-pair) row: unpaired two-sided **Mann-Whitney U** (`scipy.stats.mannwhitneyu`),
unpaired two-sided **Welch's t-test** (`scipy.stats.ttest_ind(equal_var=False)`), and paired two-sided
**McNemar's exact test** (`scipy.stats.binomtest` on the discordant-pair count, matched by identical
`(object, orient_seed, gen_seed)` triples across methods). Reported at 8 scopes (pooled "ALL", n=350/method,
plus each of the 7 objects, n=50/method) × 3 method pairs = **24 rows total**.

**7. IK-margin / reachability metric — exact definition**
`tango_robot/headless_ik.py`'s `solve_ik_jaw_pos_only(target_jaw_mid, iters=800, pos_tol=5e-3, n_outer=8)`:
runs numerical IK (800 total iterations split across 8 outer re-anchoring passes) to move the gripper
jaw-midpoint geometry to `target_jaw_mid` (world-frame xyz, **position only, no orientation term**).
`pe = ||achieved_jaw_midpoint - target_jaw_mid||` (Euclidean, metres; reported as `ik_pe_mm` = `pe*1000`
elsewhere). **`ik_ok` = `pe < pos_tol` = `pe < 5mm`.** The "IK-margin" candidate-selection strategy picks
the candidate with the **smallest `pe`** among its ensemble, regardless of whether that `pe` clears the
5mm threshold.

**8. Complete contact-feature list**
Exactly 3, all defined in `grasp_6dof/grasp_sampler.py`, all operating on points transformed into the
gripper frame and filtered to a `gripper_width × gripper_width × 4cm` box centered at the candidate pose:
- `local_point_density`: fraction of the episode point cloud inside that box.
- `normal_consistency`: std-dev of the z-component of Open3D-estimated surface normals of the points
  inside that box (`radius=0.02, max_nn=30`, consistently oriented; returns 0.0 if <5 points or the
  normal-orientation step is degenerate, e.g. a flat/coplanar patch).
- `contact_width_ratio`: inter-decile span (p90−p10) of the in-box points along the gripper's x-axis,
  divided by `gripper_width`; returns 0.0 if <10 points or `gripper_width<=0`. `<0.3` is documented in
  the source as indicating "thin object with poor jaw engagement."

`local_point_density`/`normal_consistency`/`contact_width_ratio` are exactly the 3 features that go
  into `formal_results/contact_features_bonferroni_bh*.csv`. `ik_pe_mm` (from #7) is used as a *4th*,
  separate variable in `scripts/phase1_step2_causal.py`'s causal analysis, but is **not** one of the
  3 "contact features" and is not part of the 9/18-test Bonferroni family.

**9. Number of tests corrected for**
**9** in `contact_features_bonferroni_bh.csv` (3 features × 3 objects: Pear/MustardBottle/CrackerBox).
**18** in `contact_features_bonferroni_bh_6obj.csv` (3 features × 6 objects) — but per the CRITICAL note
above, Scissors' 3 rows there are not valid data, so **15 is the effective, citable test count** for
that file (3 features × 5 valid objects), even though the on-disk Bonferroni α (0.05/18=0.00278) was
computed treating it as 18. If citing the 6-object file, recompute α at n=15 (0.00333) rather than
reusing the stored 0.00278, or just cite the 3-object file's 9-test family plus TomatoSoupCan/PowerDrill
as a separate, smaller family.

**10. Is the "consensus" comparison pool-aligned (n=10)?**
Yes, **in the authoritative file only**: `formal_results/ikmargin_vs_consensus_matched_n10.csv` uses
`phase1_matched_n10/consensus_n10_*.jsonl` (`--consensus-n 10`) against `phase1_v2/ikmargin_*.jsonl`
(`--ikmargin-n 10`) — both pool size 10. The **original** `formal_results/ikmargin_vs_consensus.csv`
(pool 10 vs pool 5) is still on disk but explicitly marked superseded/do-not-cite in this README. If
citing "the consensus vs ikmargin result," always mean the `_matched_n10` file.

## exp1_variance/ — sampler variance experiment (7 objects x 5 orient_seed x 10 gen_seed = 350 trials/file)

- `raw_results.jsonl` = **OT-CFM** (ckpt `cfm_allobj_ot.pt`), produced by `scripts/experiment1_otcfm_variance.sh`
- `raw_results_CFM-noOT.jsonl` = **CFM-noOT** (ckpt `cfm_allobj.pt`)
- `raw_results_DDPM.jsonl` = **DDPM checkpoint (`ddpm_allobj.pt`) sampled with `DDIM_STEPS=50`** — this is NOT
  an independently-trained "DDIM model"; it's the DDPM model sampled via 50-step DDIM. Don't relabel it as a
  third method on equal footing with the two CFM variants without noting this.

Both shell scripts are in `scripts/`.

## phase1_v2/ + phase1_pilot/ — consensus vs IK-margin candidate-selection strategies

- `phase1_v2/ikmargin_{Pear,MustardBottle,CrackerBox}.jsonl` — IK-margin strategy, **ensemble size 10**
  (`--ikmargin-n 10`, pick the candidate with lowest IK error out of 10), all 3 objects, 50 trials each
  (5 orient_seeds x 10 ensemble repetitions).
- `phase1_v2/consensus_MustardBottle.jsonl` — consensus strategy, **ensemble size 5** (`--consensus-n 5`,
  pick the candidate closest to the median of 5), MustardBottle only, 50 trials (5 orient_seeds x 10 reps).
- `phase1_pilot/consensus_trials_n10.jsonl` — consensus strategy, **ensemble size 5**, Pear+CrackerBox,
  100 trials total (50 each).
- `phase1_pilot/consensus_trials.jsonl` — consensus strategy, ensemble size 5, Pear+CrackerBox, only
  **5 repetitions** instead of 10 (earlier/smaller pilot, 25 each). Superseded by `consensus_trials_n10.jsonl`.

  **⚠️ Correction (2026-07-08, caught while writing the formal significance-test scripts below): an
  earlier version of this README said the consensus files were "ensemble_n=10, matched to ikmargin".
  That was wrong** — checked `scripts/phase1_v2_full.sh` and `scripts/phase1_consensus_n10.sh` directly:
  consensus was always run with `--consensus-n 5`. The "n10" in `consensus_trials_n10.jsonl`'s filename
  means **10 repetitions of a 5-candidate ensemble**, not "ensemble size 10". So **every consensus-vs-
  ikmargin comparison in this repo compares a 5-candidate pool against a 10-candidate pool** — ensemble
  size is confounded with strategy choice. This is now stated explicitly in
  `formal_results/ikmargin_vs_consensus.csv` and must be carried into any paper text: do not claim this isolates
  "selection rule" as the only variable.
- `phase1_v2/pear_ensemble_reconstruction.json` — diagnostic-only, Pear only: per-ensemble candidate
  `pe_ik` values + which candidate ikmargin picked, used to explain why ikmargin fails badly on Pear (6%
  success vs consensus's 52%).

Generating script: `scripts/phase1_v2_full.sh` (+ `phase1_consensus_pilot.sh`, `phase1_consensus_n10.sh`
for the pilot-stage files).

- `phase1_v2/ikmargin_TomatoSoupCan.jsonl` — added 2026-07-09, same design (ensemble size 10, 5
  orient_seeds x 10 reps) via `scripts/run_ikmargin_n10_tomatosoupcan.sh`, standalone (does not touch
  or re-run the original 3 objects). See `formal_results/ikmargin_vs_consensus_matched_n10.csv`'s
  TomatoSoupCan row for why.

## phase1_matched_n10/ — ensemble-size-controlled consensus re-run (2026-07-08)

- `consensus_n10_{Pear,MustardBottle,CrackerBox}.jsonl` — consensus strategy re-run at **`--consensus-n 10`**
  (matching ikmargin's pool size), same 5 orient_seeds x 10 ensemble_bases grid as `phase1_v2/ikmargin_*.jsonl`
  (identical `ensemble_base` values: 1,11,21,...,91), 50 trials/object, 150 total. Generated by
  `scripts/run_consensus_n10_matched.sh`. This directly closes the ensemble-size confound described above —
  see `formal_results/ikmargin_vs_consensus_matched_n10.csv` below for the controlled comparison.
- `consensus_n10_TomatoSoupCan.jsonl` — added 2026-07-09 via `scripts/run_consensus_n10_tomatosoupcan.sh`,
  identical grid, standalone run (does not re-touch the original 3 objects). Confirmed with a live
  smoke-test first (`--verbose 1`, checked candidate poses actually vary across gen_seed and the
  outcome differs) before committing to the full 50-trial batch, given the Scissors fallback-bug
  precedent.

**Environment note for reproducing this run**: the `tango` conda env's own `torch==2.10.0` and
`nvidia-cuda-cupti-cu12==12.8.90` (installed together 2026-06-29, unchanged since — the same versions
that produced the original 2026-07-03 data) got shadowed by a stray user-level
`~/.local/lib/python3.10/site-packages/nvidia-cuda-cupti-cu12==12.1.105` (installed 2026-07-07 by an
unrelated task on this shared machine; Python's default user-site precedence lets it leak into any
conda env). A matching stray `libnccl` caused the same problem one layer deeper. Neither `torch`/`cupti`
in `tango` nor the stray user-level package were touched — the fix is purely at invocation time:
`LD_PRELOAD` tango's own `libcupti.so.12` and `libnccl.so.2` ahead of anything else (see the exported
`LD_PRELOAD` line at the top of `run_consensus_n10_matched.sh`). This keeps user-site enabled, so
`mujoco`/`open3d`/etc. (never installed into `tango` itself, always resolved via user-site, exactly as
during the 2026-07-03 run) continue to work unchanged. **Net effect: this run used the identical
torch/cupti/nccl stack as the original exp1_variance/phase1_v2 data — old and new results are on the
same footing.** (Minor harmless leftover: while diagnosing this, `pygments`/`regex`/`safetensors`/
`tokenizers`/`typer`/`pydantic`/`requests`/`mujoco` were also pip-installed directly into `tango`'s own
site-packages during an abandoned alternate fix attempt (`PYTHONNOUSERSITE=1`); they don't conflict
with anything and weren't reverted, but the `LD_PRELOAD` approach above is what actually made this run
work, not those installs.)

## phase0_diag/ — base diagnostic dataset (150 trials: Pear/MustardBottle/CrackerBox x 50 each)

- `trials.jsonl` — raw success/fail records, produced by `scripts/phase0_full_diagnostic.py`
  (see also `scripts/phase0_diagnostic_rerun.sh`)
- `data_with_ik.json` — `trials.jsonl` + per-trial IK reachability (`ik_ok`, `ik_pe_mm`) and pose (x/y/z/yaw)
- `ui_grasp_exec_snapshot.jsonl` — 300 records, used to backfill `width` (gripper opening) onto `data_with_ik.json`
- `data_with_contact_feats.json` — `data_with_ik.json` + contact geometry features
  (`local_point_density`, `normal_consistency`, `contact_width_ratio`), produced by
  `scripts/phase2_contact_features.py`. **This is the source of the Bonferroni-corrected
  Mann-Whitney table** (MustardBottle's normal_consistency/local_point_density survive
  Bonferroni with large effect sizes; Pear's normal_consistency survives; CrackerBox does not).

## phase0_diag_extended/ — diagnostic pipeline extended to 3 more objects (2026-07-08)

Extends phase0_diag/'s 3-object diagnostic pipeline to **TomatoSoupCan, PowerDrill, Scissors**
(150 more trials), turning the "object-dependent reliability" pattern from a 3-point observation
into a 6-point one. Object choice: these are the only 3 remaining exp1_variance objects with a real
success/fail split under OT-CFM (Banana is 100% success — degenerate for a success-vs-fail
comparison, so it was excluded).

- `trials_new3.jsonl` — same design as the original (`phase0_diagnostic_rerun.sh`): single-draw,
  OT-CFM checkpoint, 5 orient_seeds x 10 gen_seeds = 50 trials/object. Success rates
  (TomatoSoupCan 80%, PowerDrill 82%, Scissors 80%) match `exp1_variance/raw_results.jsonl`'s
  OT-CFM rates for these objects exactly, as expected (identical design, freshly re-executed).
- `ui_grasp_exec_snapshot_new3.jsonl` — 300 raw records (150 `mode=="tray"` pose records, paired
  1:1 with `trials_new3.jsonl`), same pairing convention as the original.
- `data_with_contact_feats_new3.json` — IK reachability + the same 3 contact geometry features as
  `phase0_diag/data_with_contact_feats.json`, produced by `scripts/run_phase0_2_extended_analysis.py`
  (mirrors `phase0_full_diagnostic.py` + `phase2_contact_features.py`'s logic exactly, applied to the
  3 new objects — kept as a separate file rather than merged into the original, since the two were
  generated in separate runs and merging raw pose logs across runs would risk an ordering mismatch).

Generated by `scripts/run_phase0_extended.sh` (trials) + `scripts/run_phase0_2_extended_analysis.py`
(IK + contact features). Uses the same `LD_PRELOAD` environment workaround documented above.

**⚠️ Scissors here is not OT-CFM data at all — see the CRITICAL note at the top of this file.** All 50
Scissors trials silently used the random-CoM-sampling fallback (name-matching bug in
`_cfm_sample_candidates`), not the OT-CFM checkpoint used for the other 5 objects in this dataset.

**Separately, and secondary to the above**: Scissors' contact features are exactly 0.0 for all 50
trials, for all 3 features, which would be true regardless of which sampler produced the candidates —
`grasp_6dof/grasp_sampler.py`'s own docstring on `normal_consistency` says *"Low → flat surface
(scissors failure)"*: the metric is points-inside-a-narrow-gripper-bbox, close to zero by construction
for thin/flat objects. Treat Scissors' all-zero rows in `formal_results/contact_features_bonferroni_bh_6obj.csv`
as doubly compromised: (1) not OT-CFM candidates, and (2) this feature set can't discriminate for this
object's geometry regardless. Not usable as evidence about OT-CFM's contact-feature signal for any
thin/flat object — would need a rerun with the `scissors`→`cylinder` alias fixed to mean anything.

**PowerDrill `lateral_score` post-hoc follow-up (2026-07-09)**: PowerDrill is the only object with a
clean null across all 3 contact features in `contact_features_bonferroni_bh_5obj_clean.csv` (all
p_raw≥0.17), left unexplained in the paper. `scripts/run_powerdrill_lateral_score.py` tests a 4th,
previously-unused feature (`grasp_6dof/grasp_sampler.py`'s `lateral_score`, gripper-axis vs. object
principal-axis alignment — valid here because it varies per-trial via yaw, unlike `elongation_ratio`
which is ~constant per object and was deliberately *not* tested this way) against the same 50 already-
logged trials (no new grasp trials, only 5 lightweight object-spawns to get point clouds for PCA).
Result: `formal_results/powerdrill_lateral_score_posthoc.csv` — Mann-Whitney p=0.062 (marginal, ns at
α=0.05), rank-biserial=0.40 (moderate effect), fail-trials skew toward gripper-perpendicular-to-axis,
success toward parallel. **Deliberately reported as a single, uncorrected, post-hoc exploratory test —
do not fold into the existing 15-test Bonferroni/BH family** (that would require re-deriving every
already-cited p-value/significance flag in the 5-object-clean file). Per-trial data saved to
`phase0_diag_extended/data_with_lateral_score_powerdrill.json`. Note PowerDrill's mesh is only weakly
elongated (`elongation_ratio≈1.075`), so this finding should be read as inconclusive, not explanatory —
PCA's "principal axis" isn't a strongly stable direction for a near-isotropic point cloud.

## scripts/

Analysis/generation scripts copied alongside their data for provenance:
`phase1_step2_causal.py`, `phase2_contact_features.py`, `phase1_causal_check.py`,
`phase0_full_diagnostic.py`, `experiment1_otcfm_variance.sh`, `experiment1_other_methods.sh`,
`phase1_v2_full.sh`, `phase1_consensus_pilot.sh`, `phase1_consensus_n10.sh`,
`phase0_diagnostic_rerun.sh`, `FIX_DESIGN.md` (seeding-bug fix design notes, not applied to
production code as of this copy), plus (2026-07-08) `run_consensus_n10_matched.sh`,
`run_ikmargin_vs_consensus_matched.py`, `run_phase0_extended.sh`,
`run_phase0_2_extended_analysis.py`, `run_contact_features_stats_extended.py`, plus (2026-07-09)
`run_contact_features_stats_5obj_clean.py`, `run_ikmargin_n10_tomatosoupcan.sh`,
`run_consensus_n10_tomatosoupcan.sh`, `run_powerdrill_lateral_score.py`.

**Caveat on `phase1_step2_causal.py`**: this script only prints to stdout, it does not write a
results file. Its hardcoded `OBJECTS = ["Pear", "CrackerBox"]` means it does not cover
MustardBottle. Any previously-seen printed p-values from this script exist only as free text in
an old session transcript, not as a saved, reproducible result — re-run it against the data in
this directory to get citable numbers.

## formal_results/ — formal, code-generated statistical outputs (2026-07-08)

These three files replace every number that previously existed only as free text in an old
session transcript. Each was produced by running the matching script in `scripts/` against the
raw data in this directory (not against the original job scratchpad) — anyone can regenerate them
with `python3 scripts/<name>.py` and get byte-identical numbers. **Any number quoted in the paper
from these three analyses must cite the specific row/file below, not the old transcript.**

### `formal_results/exp1_variance_significance.csv` (from `scripts/run_exp1_significance.py`)

Pairwise comparison of OT-CFM / CFM-noOT / DDPM(50-step DDIM) on the exp1_variance data. Reports,
per scope (pooled "ALL" and per-object) and per method pair: success rates, unpaired Mann-Whitney U,
unpaired Welch's t-test, and **paired McNemar's exact test** (the statistically appropriate one here,
since all three methods share the identical (object, orient_seed, gen_seed) trial grid).

- **Pooled (ALL, n=350/method)**: unpaired tests are not significant at α=0.05 (Mann-Whitney/t-test
  p≈0.11 for OT-CFM vs CFM-noOT, p≈0.05 for OT-CFM vs DDPM, p≈0.71 for CFM-noOT vs DDPM). The paired
  McNemar test, which has more power because it uses the matched trial design, **is** significant:
  OT-CFM vs CFM-noOT p=0.036, OT-CFM vs DDPM p=0.008; CFM-noOT vs DDPM remains non-significant (p=0.63).
- **Per-object breakdown** shows this pooled effect is driven almost entirely by **Pear** (OT-CFM 56%
  vs CFM-noOT/DDPM 76%, McNemar p=0.041 / p=0.021) and **TomatoSoupCan** (OT-CFM 80% vs CFM-noOT 98% /
  DDPM 100%, McNemar p=0.012 / p=0.002). CrackerBox, MustardBottle, PowerDrill show no significant
  pairwise difference; Banana is 100% for all three methods (test undefined, see `note` column).
  **Scissors' `mcnemar_p=nan` (0 discordant trials across every pair) is not "no difference between
  methods" — see the CRITICAL note at the top of this file: all three "methods" silently ran the same
  non-CFM random-sampling fallback for this object, so there was nothing to differ between. Exclude
  Scissors when citing this table for a method comparison.**
- **Can support**: "OT-CFM is significantly less reliable than CFM-noOT and DDPM(DDIM-50) specifically
  on Pear and TomatoSoupCan (paired test); CFM-noOT and DDPM(DDIM-50) are statistically indistinguishable
  from each other on every object tested."
- **Cannot support**: any ODE-vs-SDE claim, any AUC number, or a blanket "method X is better than method
  Y overall" — the pooled McNemar result is driven by 2 of 7 objects, not a uniform effect, and pooling
  across objects with very different baseline rates (44%–100%) needs the per-object table alongside it,
  never just the pooled row.

### `formal_results/contact_features_bonferroni_bh.csv` (from `scripts/run_contact_features_stats.py`)

9 tests (3 contact features x 3 objects) on `phase0_diag/data_with_contact_feats.json`: Mann-Whitney U,
raw p, rank-biserial effect size, Bonferroni-corrected significance (α=0.05/9=0.00556), and proper
Benjamini-Hochberg adjusted p-values (via `scipy.stats.false_discovery_control`, not just a threshold).
Numbers match the independently hand-verified table from the previous inventory pass exactly.

- **Can support**: MustardBottle's `normal_consistency` (p=0.00029, rank-biserial=−0.60) and
  `local_point_density` (p=0.00030, rank-biserial=−0.62) survive Bonferroni with large effect sizes;
  Pear's `normal_consistency` (p=0.00068, rank-biserial=−0.47) survives Bonferroni with a medium-large
  effect. These three are robust findings under the strictest correction.
- **Cannot support**: "CrackerBox has no pre-execution contact signal" as an unqualified claim —
  CrackerBox's `local_point_density` (p=0.0119) and `contact_width_ratio` (p=0.0322) fail Bonferroni but
  **pass** Benjamini-Hochberg (p_BH=0.021, 0.048). Which correction method you pick changes the CrackerBox
  conclusion; state the method explicitly whenever citing CrackerBox.

### `formal_results/contact_features_bonferroni_bh_6obj.csv` — **extended, 6 objects** (from `scripts/run_contact_features_stats_extended.py`, 2026-07-08)

Same procedure as the 3-object table above, extended to 18 tests (3 features x 6 objects) using
`phase0_diag_extended/data_with_contact_feats_new3.json` for the 3 new objects. The 3-object table
above is not wrong and can still be cited on its own, but this is the fuller picture — includes a
`degenerate_all_zero` column flagging Scissors' structural floor-effect rows (see the data-quality
note above; these show `p_raw=1.0` by construction, not because nothing distinguishes success/fail).
**Scissors' rows are additionally not OT-CFM data at all — see the CRITICAL note at the top of this
file — so exclude them from the object count entirely rather than reading them as a 4th category;
this is effectively a 5-object result (Pear/MustardBottle/CrackerBox/TomatoSoupCan/PowerDrill) plus
one object (Scissors) that needs a rerun before it says anything.**

Reading the (valid) 5 objects by outcome, not just individually:
- **Bonferroni-surviving signal (large effect)**: MustardBottle (`normal_consistency` p=0.00029,
  `local_point_density` p=0.00030), Pear (`normal_consistency` p=0.00068), and now **TomatoSoupCan**
  (`local_point_density` p=0.00123, rank-biserial=−0.625) — a new object joining this group.
- **Signal only under the more lenient Benjamini-Hochberg correction**: CrackerBox (2 features) and
  TomatoSoupCan's `normal_consistency` (p_BH=0.039).
- **No signal under either correction**: **PowerDrill** — all 3 features non-significant even under
  BH (p≥0.17). This is the first object with a clean, unqualified null result on this feature set.
- **Excluded — not a real data point**: Scissors (see above).

- **Can support**: the "object-dependent" pattern is not a binary (signal / no-signal) split — across
  5 valid objects it's a spectrum: robust signal (Pear, MustardBottle, TomatoSoupCan) → correction-dependent
  signal (CrackerBox) → clean null (PowerDrill). Any paper claim about object-dependence should describe
  this spectrum, not collapse it to "some objects work, some don't", and should count this as n=5, not n=6.
- **Cannot support**: a specific geometric/physical property (e.g. "compliance", "size", "symmetry")
  as *the* explanatory variable for where signal exists — no such property was measured or tested here;
  the grouping above is purely empirical (which p-values came out significant), not mechanistically
  explained. That would need a follow-up analysis relating a measured object property to the outcome
  group, which was not done.

### `formal_results/ikmargin_vs_consensus_matched_n10.csv` — **authoritative, ensemble-size-controlled** (from `scripts/run_ikmargin_vs_consensus_matched.py`, 2026-07-08; TomatoSoupCan row added 2026-07-09)

Fisher's exact test, both strategies now at **matched ensemble size 10** (ikmargin: existing
`phase1_v2/ikmargin_*.jsonl`; consensus: new `phase1_matched_n10/consensus_n10_*.jsonl`). This closes
the confound in the original comparison below and is the number to cite going forward.

| object | ikmargin (n=10 pool) | consensus (n=10 pool) | Fisher's exact p |
|---|---|---|---|
| Pear | 6.0% (3/50) | 68.0% (34/50) | **p=5.8e-11** |
| MustardBottle | 50.0% (25/50) | 68.0% (34/50) | p=0.103 (ns) |
| CrackerBox | 44.0% (22/50) | 44.0% (22/50) | p=1.0 (exact tie) |
| TomatoSoupCan | 34.0% (17/50) | 64.0% (32/50) | **p=0.0048** |

- **Can support**: the Pear finding not only survives the ensemble-size fix, it gets *stronger*
  (p=5.8e-11 vs the original confounded p=4.1e-7) — ruling out ensemble size as an alternative
  explanation. "On Pear, the ikmargin selection rule performs dramatically and significantly worse than
  consensus, at matched candidate-pool size" is now a clean, controlled finding.
  `phase1_v2/pear_ensemble_reconstruction.json` still gives the per-candidate diagnostic detail for why.
  CrackerBox is an exact tie (22/50 both strategies) at matched pool size — reinforces the running theme
  that CrackerBox shows no selection-strategy signal under any method tested in this repo.
- **New, not previously visible**: giving consensus a fair 10-candidate pool raised its MustardBottle
  rate from 56% (at pool 5) to 68% — closer to significance (p=0.103, down from p=0.69) but still not
  significant at α=0.05. Consensus numerically leads ikmargin on MustardBottle now, but call this
  "suggestive, not established" until more trials are run.
- **TomatoSoupCan added 2026-07-09** (`scripts/run_ikmargin_n10_tomatosoupcan.sh` +
  `scripts/run_consensus_n10_tomatosoupcan.sh`, same 5 orient_seed × 10 ensemble_base grid, same OT-CFM
  checkpoint): motivated by `paper_final.tex`'s "Reliability across generation seeds" paragraph naming
  Pear **and** TomatoSoupCan as objects where single-draw OT-CFM is significantly less reliable across
  seeds (`exp1_variance`'s paired McNemar test) — this closes the gap of only having ikmargin-vs-consensus
  mitigation data for one of the two named objects. Result: consensus significantly beats ikmargin here
  too (64% vs 34%, p=0.0048) — the mitigation generalizes to both flagged objects, not just Pear.
- **Cannot support**: "consensus is a better selection rule than ikmargin" as a uniform, all-objects
  claim — it's a controlled, significant win on Pear and TomatoSoupCan; a tie on CrackerBox; and a
  non-significant lead on MustardBottle.

### `formal_results/ikmargin_vs_consensus.csv` — superseded, kept for record (confounded, ensemble 10 vs 5)

Original comparison: ikmargin (ensemble 10) vs consensus (ensemble 5) — the ensemble-size confound
described above. Kept on disk for transparency (shows what was believed before the matched re-run), but
**do not cite this file's numbers in the paper** — use `ikmargin_vs_consensus_matched_n10.csv` instead.

| object | ikmargin (n=10 pool) | consensus (n=5 pool) | Fisher's exact p |
|---|---|---|---|
| Pear | 6.0% (3/50) | 52.0% (26/50) | p=4.1e-7 |
| MustardBottle | 50.0% (25/50) | 56.0% (28/50) | p=0.69 (ns) |
| CrackerBox | 44.0% (22/50) | 42.0% (21/50) | p=1.0 (ns) |

## Explicitly NOT included (because it does not exist as a file anywhere)

Paper B's real-robot servo current/load data (tissue package / towel / glasses case) has no
backing data file on disk anywhere — it exists only as a hand-summarized table inside a memory
note (`~/.claude/projects/-lena-projects-lerobot/memory/project_paper_b_execution_prep.md`), and
the scripts that produced it (`live_probe_gripper.py`, `signal_watch.py`, etc.) were only ever in
`/tmp` and are gone. Do not treat this archive as covering Paper B.
