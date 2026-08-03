# Joint-Limit-Aware Grasp Orientation Selection: A Training-Free Method That Matches or Exceeds a Learned Correction Model

**[Draft — created 2026-08-01, reframed 2026-08-01 to lead with the positive contribution rather than the
negative-result framing of the first pass (see Drafting notes). Status: Abstract, Contributions, §1-5, §7-8
are full prose with numbers verified against source documents and citations verified against live arXiv
metadata; §6 (Real-Hardware Validation) is an explicit placeholder awaiting Stage 0+ execution on physical
Piper. This is this project's designated real-hardware submission line.]**

**Framing decision (read before editing further):** this paper's headline claim is the wrist-fix method
itself — a real, deployable, statistically confirmed, mechanistically explained contribution. The CR-CFM
comparison is retained (it is real, honest data, and this project's own house style does not hide honest
findings) but demoted from "headline finding" to a **sufficiency ablation**: evidence that the simple method
already captures the available benefit, so a more complex learned correction is not obviously worth its
training/inference cost. That is a positive, efficiency-oriented claim ("simple beats/matches complex"), not
a negative one ("the complex thing doesn't work") — keep every section's prose consistent with this framing.
Do not let word count or table ordering imply the ablation is the main result.

## Abstract

*(Status: sim-only claims are final prose below; the closing real-hardware sentence is a placeholder and
must not be filled in before Section 6 actually executes. Word count currently ~220 — trim toward a target
venue's abstract limit, e.g. RA-L, during final formatting, not before results are locked.)*

Grasp-pose generation pipelines typically check inverse-kinematic reachability as a binary filter, but a
reachable pose can still leave a redundant-DOF arm's joints with no margin to spare — in particular, a
wrist-roll joint solved right at its hardware limit. We show, on a 6-DoF AgileX Piper arm, that this omitted
margin is a strong predictor of grasp failure (Fisher's exact p=1.8e-5, odds ratio 90) and introduce a
training-free method that exploits it: given a grasp target, solve inverse kinematics for both of two
IK-equivalent approach orientations and keep whichever leaves the wrist-roll joint further from its limit.
The method significantly improves grasp success at adequate statistical power (n=152, McNemar's exact
p=0.027, 73.0% vs. a 65.8% baseline) at zero added training or inference cost beyond one extra IK solve. Its
benefit is object-dependent — and, as a second contribution, this dependence is itself cheaply predictable
in advance via an IK-only proxy requiring no physical execution, letting us correctly anticipate and skip an
expensive execution test on an object the method does not help. We further ask whether a more complex
learned correction is still worth adding on top of this simple method: a controlled sufficiency ablation
against a 49K-parameter receding-horizon flow-matching correction model, at matched statistical power
(n=152), shows it adds no measurable benefit (McNemar's exact p=0.33), with an architectural explanation for
why — the learned model's closed loop never re-perceives the object, only the arm's own state. We complete a
five-gate sim-side real-hardware-readiness assessment, including a real safety-coverage gap found and fixed
along the way, and report the method's status under real-hardware execution. [PLACEHOLDER — insert one
sentence stating the real-hardware outcome once Section 6 produces it; do not draft this before that data
exists.]

## Contributions

1. **A joint-limit-aware grasp orientation selection method** (`pick_wrist_friendly_orientation`) —
   training-free, model-independent, deployable with zero additional inference cost — that significantly
   improves grasp success on a real hardware constraint most grasp-pose pipelines do not model. Confirmed at
   n=152 (McNemar p=0.027, 73.0% vs. 65.8% baseline), with the underlying mechanism independently confirmed
   at Fisher's exact p=1.8e-5 (odds ratio 90).
2. **A validated cheap predictor of where the method helps**: an IK-only proxy (no physics execution needed)
   that reproduces a known object's reference failure-pinning rate trial-for-trial once corrected, then
   correctly predicts a *different* object's outcome before its expensive execution test is run — turning
   "does this method help on a new object" into an almost-zero-cost question instead of a per-object physical
   experiment.
3. **A sufficiency ablation against a learned alternative**: isolating the orientation method's contribution
   from a receding-horizon flow-matching correction model (CR-CFM) sharing the same pipeline, at two sample
   sizes (n=32, n=152), shows the simple method alone already accounts for the measured gain — the learned
   model adds nothing further at adequate power, with a mechanistic (not just statistical) explanation for
   why its closed loop does not close around the parts of the environment that would need to change for it
   to add value.
4. **A found-and-fixed real safety-coverage gap**: a scoped action-clamp missed one full pipeline phase; a
   universal clip at the action-output boundary closes it completely (275/1690 → 0/1690 violations),
   documented as a mandatory component of any real-hardware backend for this pipeline.
5. **A completed sim-side Real-Hardware Readiness assessment** (5 gates: baseline comparison, multi-object
   generalization, safety-layer hardening, failure-mode taxonomy, staged rollout procedure).
6. **[PLACEHOLDER, pending hardware access] Real-hardware validation** of the method under the arm's actual
   (possibly narrower) joint-limit configuration — see Section 6.

## 1. Introduction

Grasp-pose generation pipelines routinely check inverse-kinematic reachability and collision-freeness before
proposing a candidate, but a pose that clears both checks can still leave a redundant-DOF arm's joints with
no margin to spare — in particular, a wrist-roll joint solved right up against its hardware range limit.
Reachability is treated as a binary feasibility gate, not as a graded ranking signal, so two IK-equivalent
grasp orientations (e.g. approaching an object from either of two antipodal sides with the same grip
geometry) are typically treated as interchangeable even when one leaves substantially more room at every
joint than the other. We show, on a 6-DoF Piper arm, that this omitted margin is not a minor detail: whether
the wrist-roll joint is pinned at its limit is one of the strongest single predictors of grasp failure we
have found in this pipeline (Fisher's exact p=1.8e-5, odds ratio 90), and it can be cheaply computed and
optimized before any physical execution — a training-free orientation choice that costs one extra IK solve
and nothing else.

Kinematic redundancy has long been exploited for other purposes in grasping and manipulation — e.g. using
spare degrees of freedom to grasp multiple objects at once [Chen & Xu 2023, arXiv:2303.01868] or to track a
manipulability objective through a motion [Jaquier et al., arXiv:1811.11050] — but these treat redundancy as
a resource to spend on a *separate* objective (a second object, a smoothness criterion), not as a ranking
signal for *which of several equally-valid solutions to the same grasp* to prefer. Our framing is narrower
and, we argue, underexplored: among IK solutions that already solve the *same* grasp, choose the one with
the largest hardware safety margin.

This project also asks the natural follow-up question a hardware-constrained fix like this invites: given
that a training-free orientation choice already exists, is a more complex, learned correction mechanism
still worth adding on top of it? We built and evaluated exactly such a mechanism — a receding-horizon,
flow-matching-based trajectory correction model (CR-CFM) — and answer this empirically with a controlled
ablation at adequate statistical power (n=152): no, not on this pipeline. We report this as a sufficiency
finding about our own system, not a general claim about learned correction models, and give a mechanistic
account of why (§5.4) grounded in what the model's closed loop actually closes around.

Contribution arc: mechanism (the joint-limit-pinning predictor, §4.1) → method (the orientation choice,
§4.2) → confirmation (§5.1) → predictable scope (§5.2) → sufficiency check against a learned alternative
(§5.3-5.4) → real-hardware status (§6).

## 2. Related Work

**Kinematic redundancy in grasping and manipulation.** Prior work exploits a manipulator's or hand's spare
degrees of freedom for objectives beyond the single grasp being executed — using redundant DOFs to grasp
multiple objects simultaneously [arXiv:2303.01868], or to shape a manipulability ellipsoid along a whole
trajectory via learned tracking and transfer of manipulability profiles [arXiv:1811.11050]. Whole-body and
real-time redundancy-resolution work in the same family typically optimizes obstacle avoidance or a
secondary task while a primary end-effector objective is held fixed [arXiv:2012.14578; arXiv:1810.03945].
These establish that treating redundant DOFs as an optimization resource (not just an IK-feasibility
formality) is an established idea — our contribution narrows this to a single, previously uncharacterized use
of that resource: ranking among *multiple IK solutions to the identical grasp* by hardware joint-limit
margin, rather than spending redundancy on a second, independent objective.

**Learned correction / receding-horizon control for manipulation, and test-time compute scaling.** CR-CFM
(§4.4/§5.3) belongs to the family of generative, flow- or diffusion-based control policies that allocate
test-time compute to refine or correct an action before or during execution. Recent work in this family
explicitly studies *how much* test-time compute such policies should spend, adaptively — ELASTIC
[arXiv:2606.31132] scales sequential/parallel test-time compute per-instance for generative control
policies; DASIP [arXiv:2511.20906] similarly adapts inference budget per control step for stochastic
interpolant policies; and a training-free execution-horizon adaptation result [arXiv:2602.21445] shows a
flow-based VLA's own internal signal can set how many predicted actions to actually execute, without any
additional training. Our sufficiency ablation is best read against this literature as a *lower bound*
case: before spending effort on adaptive test-time compute for a correction model, our result shows it is
worth first checking whether a training-free geometric prior already captures the available benefit on the
specific pipeline in question — a question this project's own architecture-level analysis (§5.4) suggests
generalizes to any RHC design whose closed loop does not re-perceive the object.

**Evaluating manipulation policies under simulation.** Our paired, McNemar-tested evaluation protocol
(§4.5) sits within a broader concern about whether simulation-based manipulation-policy comparisons
transfer meaningfully — both a benchmarking-oriented treatment of sim-to-real policy evaluation
[arXiv:2508.11117] and a study of evaluating real-world manipulation policies inside simulation
[arXiv:2405.05941] raise related methodological concerns about evaluation validity that this project's
paired-trial, adequately-powered design is intended to address directly (though neither directly overlaps
with the joint-limit framing above); cite for methodological grounding, not as a competing method.

**Reporting negative/null findings.** We follow this project's established practice, and the position
argued in [arXiv:2406.03980], of reporting a null finding (§5.3's sufficiency ablation) with the same rigor
as a positive one, rather than omitting or reframing it — while keeping the paper's own headline claim
centered on the positive, validated method (§4.2, §5.1), consistent with that position paper's broader
point that predictive-performance framing alone is an incomplete way to judge a contribution's worth.

*(Literature pass status: searched via arXiv API for kinematic-redundancy/manipulability-aware grasp
planning, learned-correction/test-time-compute-scaling, and sim-based policy evaluation methodology — all
citations above verified against live arXiv metadata, not fabricated. Not yet searched: prior art on
*joint-limit margin specifically as a discrete grasp-orientation ranking criterion* — the closest matches
found treat redundancy as a resource for a separate objective, not this framing; if a closer match exists
it was not surfaced by the queries run so far, and this gap should be called out explicitly in the final
prose as supporting the novelty claim, not silently assumed.)*

## 3. System

**Platform.** All results are collected on a RoboSuite/MuJoCo simulation of the AgileX Piper, a fixed-base
6-DoF arm (joints 1-6) fitted with Piper's stock 2-finger parallel gripper (two prismatic finger joints
coupled by an equality constraint, 0.05 m of travel per finger — matching AgileX's official Isaac Lab asset
specification, widened from an earlier, incorrect community-ported value). The arm and gripper are modelled
as a RoboSuite `ManipulatorModel`/`GripperModel` pair (`piper_robot.py`, `piper_gripper.py`) built from a
MuJoCo XML asset split out of a combined community model, driven by an absolute joint-position PD controller
(`PIPER_JOINT_POSITION_CONFIG`).

**Joint limits (the constraint this paper's method exploits).** All six arm joints are hardware-range-limited
in simulation: joint1 ±2.618 rad, joint2 [0, 3.14] rad, joint3 [−1.637, 1.33] rad, joint4 ±1.832 rad, joint5
±1.22 rad, and joint6 (wrist-roll — the joint this paper's method optimizes against) ±3.14 rad, sourced from
the arm's MuJoCo asset at simulation-build time. Section 6.2 revisits whether the physically-configured
joint6 limit matches this simulated value before any real-hardware claim is made.

**Objects and scene.** Grasp targets are real YCB meshes wrapped as RoboSuite `MujocoXMLObject`s (not
RoboSuite's built-in synthetic YCB-alike primitives): Pear, Tomato Soup Can, Banana, Mustard Bottle, Cracker
Box, Power Drill, and Medium Clamp (7 objects total in the registry; four fail at baseline execution for
reasons unrelated to this paper's method and are out of scope here — see Limitations). This paper's method
is evaluated primarily on Cracker Box (the object shown to benefit) and Pear (the object shown, mechanistically,
not to).

**Pipeline.** The production pick-and-place pipeline (`piper_pick_and_place.py`) executes eight named
phases per trial — approach, descend, close, lift, transit, lower-into-tray, open, retract — each phase
solving joint-space IK for a target end-effector pose (full position **and** orientation; an earlier
position-only version left three DOF free to settle arbitrarily and produced asymmetric, tilted,
single-finger contact instead of a level two-finger grip) against a scratch copy of `qpos` so the IK solve
itself never perturbs the live physics state, then commanding the result via the joint-position controller
for enough physics steps to actually converge under contact dynamics rather than teleporting. Both this
paper's method (§4.2) and the compared CR-CFM correction model (§4.4) act within this same shared pipeline
and phase structure, differing only in how the descend phase's target/trajectory is produced — the
orthogonality that makes the §5.3 sufficiency ablation a clean, isolated comparison rather than a comparison
across otherwise-different systems.

## 4. Method

### 4.1 Joint-limit-pinning as a failure predictor

Diagnostic scan across established trials: when the IK solution for a grasp target lands the wrist-roll
joint (joint6) pinned at its hardware limit, the trial fails 82% of the time (9/11) vs. 4.8% (1/21) when it
does not (Fisher's exact p=1.8e-5, odds ratio 90) — state this as the paper's foundational empirical
observation, motivating the method below.

### 4.2 Joint-limit-aware orientation selection (the method)

`pick_wrist_friendly_orientation`: for a given grasp target, solve IK for both the nominal approach
orientation and its 180°-flipped equivalent (same grip geometry, opposite approach side); keep whichever
leaves joint6 further from its limit. Training-free, model-independent, zero added inference cost beyond one
extra IK solve; applies identically regardless of which descend-execution method is used downstream — this
orthogonality is what makes the §5.3 sufficiency ablation possible.

### 4.3 IK-only applicability predictor

Describe the proxy metric (captures the *seed* joint configuration entering the descend phase, not the
solved IK target — note this as a real, fixed bug worth stating plainly: the initial version measured the
wrong quantity and could not reproduce the reference pinning rate; state the fix and the verification
criterion, namely trial-for-trial reproduction of a known reference rate) and the pre-registered decision
rule it supports (skip expensive execution testing for objects the proxy predicts will show a low pinning
rate).

### 4.4 Sufficiency ablation design (CR-CFM comparison)

Briefly describe CR-CFM (49K-parameter flow-matching correction network, receding-horizon control loop) only
as much as needed to define the ablation: since the orientation method is orthogonal to the descend-execution
mechanism, it can be paired with either plain interpolation or CR-CFM, isolating each component's own
contribution.

### 4.5 Evaluation protocol

Paired trial design (same trial_id/seed under every compared condition), McNemar's exact test as the
methodologically correct paired test, 3-repeat majority vote per trial, power analysis governing the
n=32→n=152 scale-up decision.

## 5. Results

### 5.1 The method: confirmed improvement

| Finding | Statistic |
|---|---|
| joint6-pinning predicts trial failure (motivating diagnostic) | Fisher exact p=1.8e-5, odds ratio 90 |
| Orientation method vs. baseline, Cracker, n=152 | 111/152 (73.0%) vs. 100/152 (65.8%); McNemar exact p=0.027 |
| Real-baseline comparison (Gate 1): plain interpolation vs. method+CR-CFM | 19/32 (59.4%) vs. 26/32 (81.2%); McNemar exact p=0.0156 |

Lead with this table. Note for drafting: Gate 1's number pairs the method *with* CR-CFM (historical framing,
run before the sufficiency ablation existed) — state explicitly in-text that §5.3 attributes this win fully
to the orientation method, so Gate 1 should be read as "the combined pipeline beats plain baseline," with
§5.3 clarifying which component deserves credit.

### 5.2 Applicability is object-dependent, and predictable in advance

| Object | IK-proxy pinning rate | Execution-verified pinning | Method effective? |
|---|---:|---|---|
| Cracker | 30% (6/20; reference 11/32=34.4%) | Yes (reference rate) | **Yes** (p=1.8e-5 mechanism, p=0.027 confirmatory) |
| Pear | 5% (1/20) | 0/16 (0%), fully concordant 6/8 vs. 6/8, zero discordant | No — correctly predicted null, not a failure of the method |
| Mustard | 5% (1/20) | Not run — skipped per pre-registered decision rule | Predicted no; untested |

Frame as a second positive contribution: the method's own scope of applicability is cheaply knowable in
advance (§4.3), avoiding a brute-force per-object execution sweep (only 7 objects exist in the Piper
registry; 4 already fail at baseline for unrelated reasons, making naive scaling infeasible regardless). Pear
is a scope-characterization result, not a shortcoming — a method whose applicability boundary is unknown
would be the weaker paper.

### 5.3 Sufficiency ablation: is a learned correction model worth adding on top?

| Comparison | n | Result | Test |
|---|---:|---|---|
| baseline+method vs. CR-CFM+method | 32 | 26/32 both conditions, 0 discordant pairs | tie |
| baseline+method vs. CR-CFM+method (scaled) | 152 | 115/152 (75.7%) vs. 110/152 (72.4%); 17 discordant, 11 favor baseline, 6 favor CR-CFM | McNemar exact p=0.33 |
| Perturbation test (one-time joint disturbance, descend phase) | 8 (both conditions) | 3/8 (37.5%) both conditions, 0 discordant | tie |

Frame the takeaway positively: the orientation method alone already captures the measurable benefit at
adequate power (5× the initial sample does not reveal a hidden CR-CFM advantage; the small discordance that
does appear tilts toward the simpler method) — a practical efficiency result (no training data, no model, no
added inference latency needed) rather than a claim that learned correction models cannot work in general.

### 5.4 Why the learned model adds nothing here (mechanistic explanation)

`target_qpos` is IK-solved once before CR-CFM's receding-horizon loop begins; the loop re-reads only arm
state. The perturbation test's null result follows directly: MuJoCo's PD tracking recovers from a one-time
joint disturbance regardless of open-loop or RHC execution, since neither architecture is asked to react to a
*changed goal* — only a real object-position perturbation would test that, and CR-CFM as implemented has no
mechanism to re-solve against fresh perception mid-loop either. Frame as a scoping statement about *this*
architecture, not a general claim about learned correction models.

### 5.5 Real-Hardware Readiness (sim-side assessment)

| Gate | Result |
|---|---|
| 1. Real-baseline comparison | PASSED, McNemar p=0.0156 |
| 2. Multi-object generalization pilot | Object-dependent as characterized in §5.2 (Pear) |
| 3. Safety-layer hardening | PASSED — found and fixed a real coverage gap (scoped clamp missed the `lower_into_tray` phase; universal `clip_action_to_real_limits` verified 275/1690→0/1690 violations) |
| 4. Failure-mode taxonomy | PASSED — all known failures benign/recoverable (low terminal velocity, no divergent trajectories) |
| 5. Staged real-rollout procedure | Documented, not yet executable without hardware (see §6) |

## 6. Real-Hardware Validation

**[PLACEHOLDER SECTION — fill in as each stage below actually executes. Do not draft prose claiming a stage
complete before it has run.]**

### 6.1 Software readiness (complete, sim-only work)

- `piper_sdk` v0.6.1 installed; every `piper_real_backend.py` method body verified line-by-line against
  official `piper_sdk/demo/V2/*.py` examples (correct interface class, unit-conversion factors, arm-enable
  polling pattern, control-mode preconditions).
- Gripper unit conversion resolved (`finger_qpos_to_span_m()`); `PiperTrajectoryReplayer.replay()` can now
  drive the real gripper instead of raising `NotImplementedError`.

### 6.1b Perception readiness (shared project infrastructure, reused not rebuilt)

All results in this paper through §5 use ground-truth simulated object pose (a deliberate simplification
enabled by MuJoCo, orthogonal to this paper's own method — see §4.2). Real-hardware execution needs the
object's actual pose, which this project already has embodiment-agnostic tooling for, built for the
parallel SO-ARM101 real-hardware line and directly reusable here without modification:
- `cameras/base.py::CameraBase` / `cameras/realsense_stub.py::RealSenseCamera` — camera abstraction and a
  real Intel RealSense D435i driver, robot-agnostic by construction.
- `cameras/noise_characterization.py::estimate_pose_from_rgbd` — a complete, already-implemented classical
  pose-estimation pipeline (depth→point cloud, table-plane removal via a pre-calibrated plane model, DBSCAN
  clustering, centroid position + PCA-based yaw), validated so far only against synthetic data with known
  parameters (a closed-loop self-check), not yet run on a real capture.
- `scripts/solve_handeye.py` — eye-to-hand camera-to-robot-base extrinsic calibration solver
  (`cv2.calibrateHandEye`), robot-agnostic; consumes samples from `scripts/capture_handeye_sample.py`.

**What is genuinely new for Piper, and what is not**: none of the above software needs to be rewritten for
this arm — the abstraction was already built to be shared. What Piper-specific work remains is entirely
physical/data, not code: (1) mount/position a camera overhead of wherever the physical Piper sits, (2)
capture a table-plane calibration reference, (3) run the hand-eye capture+solve procedure against Piper's
own base frame (a different transform than whatever SO-ARM101's setup will produce, even with the same
camera hardware), (4) run `estimate_pose_from_rgbd` on the first real captures and confirm it behaves
sanely before trusting it for a grasp target. Building this once, here, is intentionally shared investment —
the same real-pose-estimation capability is a prerequisite for this project's separate future language-
grounded-grasping work (`tango/policy.py`'s VLM pipeline, currently SO-ARM101-only), not just for this
paper's real-hardware section.

### 6.2 Known open blocker

`piper_sdk`'s documented *default* wrist-roll (joint6) limit is ±2.09439 rad (±120°) — 60° narrower than the
±3.14 rad this entire investigation's mechanism and method threshold assume (sourced from the simulation
XML). This is a reconfigurable soft limit (`MotorAngleLimitMaxSpdSet`, official demos widen it toward ±170°,
still short of ±180°). **This must be resolved before any method trajectory is replayed on real hardware** —
the method's threshold logic (`joint6_limit=3.14` in `pick_wrist_friendly_orientation`) may need
recomputation against whatever limit the physical arm actually reports.

### 6.3 Staged rollout plan (not yet started)

1. [ ] Query the physically-configured joint6 limit (`SearchMotorMaxAngleSpdAccLimit` /
   `GetAllMotorAngleLimitMaxSpd`); compare against 3.14 rad; recompute the method's threshold if different.
   **Report the queried value and the delta here once run.**
2. [ ] Gripper-agnostic joint-only trajectory replay (no grasp) — verify timing and confirm the safety clip
   layer behaves correctly on real hardware before any contact is attempted.
3. [ ] Mount/position the camera over Piper's table; capture the table-plane calibration reference; run
   `scripts/capture_handeye_sample.py` + `solve_handeye.py` against Piper's own base frame. **Record the
   solved cam→base transform and sample count here once run** (§6.1b — reused infra, Piper-specific
   physical step).
4. [ ] Run `estimate_pose_from_rgbd` on real captures of a single known-placed object; sanity-check the
   returned position/yaw against a hand measurement before trusting it as a grasp target for step 5.
5. [ ] Single conservative real grasp attempt, small `max_relative_target`, replaying an orientation-method
   trajectory (not CR-CFM — §5.3 already shows it is unnecessary) on Cracker (the only object with a
   confirmed sim-side positive effect), using the real perceived pose from step 4 as the grasp target.
   Record outcome.
6. [ ] Scale sample size per Gate 5's staged procedure (n=5→10→20), compared against the Gate 1 sim figure
   (81.2%) as the stop/go signal.

**Sequencing note**: steps 3-4 (perception) are not, strictly, required to validate this paper's own claim
— a manually-measured, fixed object placement would let step 5 run without them, faster. They are included
here because the user has decided to build real-perception capability now as shared infrastructure (reused
by this paper's real-hardware trial and by future language-grounded-grasping work alike), not because the
wrist-fix method itself needs perception. If timeline pressure ever makes the perception steps the
bottleneck, running steps 5-6 first on a manually-measured placement (with a note in this section that
perception was added afterward) remains a valid fallback that does not weaken the method's own claim.

## 7. Limitations

1. Real-hardware validation is, as of this draft, entirely unexecuted (Section 6) — every claim above
   Section 6 is simulation-only.
2. The method is confirmed on one object (Cracker) and confirmed inapplicable (not "failing," per §5.2's
   framing) on a second (Pear); general coverage across the full object distribution is untested beyond the
   two directly executed cases plus one IK-proxy-only prediction (Mustard).
3. The joint6-limit discrepancy (§6.2) means the sim-validated threshold is not guaranteed to be correct
   as-is on physical hardware without the Section 6.3 step 1 verification.
4. The sufficiency ablation (§5.3) is scoped to this pipeline's specific CR-CFM design (IK solved once,
   re-reads only arm state); it does not claim learned correction models cannot add value in manipulation
   generally, only that this particular architecture's closed loop does not close around the parts of the
   environment that would need to change for it to add value here.

## 8. Conclusion

We presented a training-free grasp-orientation selection method that exploits a real hardware constraint —
wrist-roll joint-limit margin — largely ignored by grasp-pose pipelines that treat reachability as a binary
filter rather than a ranking signal, and confirmed it statistically (n=152, McNemar p=0.027) with an
independently-verified mechanistic explanation (Fisher p=1.8e-5). We showed the method's scope of
applicability is itself cheaply predictable in advance via an IK-only proxy, avoiding brute-force per-object
physical testing. A controlled sufficiency ablation against a learned receding-horizon correction model
found the simple method already captures the available benefit at adequate statistical power, with an
architectural account of why the learned model's closed loop does not close around the parts of the
environment that would need to change for it to add value — reported as an efficiency finding about this
system, not a general claim against learned correction models. [PLACEHOLDER: restate real-hardware outcome
here once Section 6 executes; do not draft this sentence before that data exists.]

---

## Drafting notes (remove before submission)

- **Reframing history**: the first pass of this skeleton (2026-08-01, same day) led with "falsifying CR-CFM"
  as the headline, mirroring this project's other honest-negative-results papers. User feedback: worried
  this reads as "too much negative data" relative to real contribution, especially given
  `paper_final.tex`/RA-L was rejected partly for being negative-result-heavy — see [[project_ral_rejection]].
  Reframed same day to lead with the orientation method as the positive contribution, with the CR-CFM
  comparison retained as a sufficiency ablation, not deleted (all underlying data unchanged, only framing/
  ordering). If further revision is needed, keep this principle: never delete or soften honest negative
  data, only ensure the *positive, deployable, real* contribution is what a reader encounters first in the
  title, abstract, and contributions list.
- **Literature pass (2026-08-01)**: ran arXiv-API searches (kinematic redundancy/manipulability-aware grasp
  planning, learned-correction/test-time-compute scaling, sim-based policy evaluation methodology) via the
  `research-lit` skill; every citation now in §1/§2 was verified against live arXiv metadata (title+abstract
  fetched directly, not taken from a search snippet) before use — none fabricated. No local paper library or
  Zotero/Obsidian MCP was configured for this project, so the search was arXiv-API + web only. Explicitly
  NOT yet found: a close prior-art match for "joint-limit margin as a discrete grasp-orientation ranking
  signal among IK-equivalent solutions" specifically — §2's closing paragraph states this as a real gap, not
  an oversight; re-run a targeted search pass before final submission in case something was missed, and
  do not remove that caveat sentence without doing so.
- Source of truth for every number above: `tango_robot/piper_robosuite/PIPER_FINDINGS_SUMMARY.md` (concise
  synthesis) and `tango_robot/piper_robosuite/cr_cfm/IMPROVEMENT_PLAN.md` (full chronological log, 200+
  entries — the numbers above were cross-checked against `README.md`'s dated entries, but re-verify against
  `IMPROVEMENT_PLAN.md` directly before finalizing any table, since that is the primary log).
- Venue/template not yet chosen. This project's existing LaTeX infrastructure (`interact` class, used for
  `paper_tro.tex`/`paper_advanced_robotics.tex`) is available if a T-RO/Advanced-Robotics-style venue is
  chosen; confirm before converting this markdown draft to LaTeX. Given this paper's distinct core claim, it
  is its own submission target, not another §4.x subsection of `TRO_PAPER_OUTLINE.md`. Current lean: RA-L
  (letter format fits a single, focused, rigorously-confirmed method claim; also directly addresses the
  prior rejection reason once Section 6 has real data), with `Advanced Robotics` as a lower-bar fallback.
- Do not backfill Section 6 with sim numbers or projected/expected results — every checkbox in §6.3 must be
  a real, executed, dated result before this section can be written as anything other than a placeholder.
- When real-hardware numbers land, update the Abstract and Contributions item 6 in the same pass — those are
  the two places most likely to silently drift out of sync with Section 6.
