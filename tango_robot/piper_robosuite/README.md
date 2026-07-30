# Piper simulation data archive

Companion to `paperA_data/README.md`'s documentation convention (newest-first,
each entry states what the finding **can** and **cannot** support). This
covers the Piper + RoboSuite simulation built to replace SO-ARM101 as the
T-RO paper's real-hardware validation target (Piper hardware is available;
SO-ARM101 real-hardware work has been superseded, not merged with this).

## 📊 2026-07-22: Paper-readiness Priority 1 -- wrist-fix ALONE (no CR-CFM) shows ZERO measurable effect on Pear, unlike its confirmed p=0.0156 significance on Cracker

Part of the post-decisive-finding paper-readiness plan (4 items, run in strict order). Tested plain baseline
vs. baseline+wristfix directly on Pear (trials 3000-3007, paired, 3-repeat majority vote, no CR-CFM in
either arm) -- isolating wrist-fix's own effect from the CR-CFM confound, mirroring the same isolation logic
used in the critical Cracker ablation. **Result: plain_baseline 6/8 (75%) vs. baseline_wristfix 6/8 (75%) --
fully concordant, ZERO discordant pairs** (3000-3004 and 3006 succeed under both; 3005 and 3007 fail under
both). Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Step 1" entry under the new "Paper-readiness
supplementary experiments" section.

**Honest interpretation**: this does not contradict the Cracker finding -- wrist-fix specifically prevents
`robot0_joint6` (wrist-roll) from pinning against its hardware limit, and if Pear's grasp geometry never
drives joint6 anywhere near that limit in the first place, the fix has nothing to correct. Proceeding to
Priority 2 (check whether Pear's failing trials 3005/3007 show the same joint6-pinning signature Cracker's
failures did) to confirm this mechanistic explanation directly rather than leaving it as a plausible guess.

## 🎯 2026-07-22: Paper-readiness Priority 4 CLOSED -- built and validated a cheap IK-only predictor of wrist-fix benefit; Mustard predicts the same null result as Pear, so the expensive execution test was skipped by design

Follow-up to the "do we need 50 objects?" question: rejected brute-force object-count scaling (infeasible --
only 7 objects exist in the Piper registry, 4 already fail at baseline for unrelated reasons; also
infeasible given observed compute cost) in favor of a falsifiable, cheap PREDICTOR -- does an object's
naive (un-fixed) grasp orientation drive joint6 toward its hardware limit, measured via IK solves alone
(no physics execution beyond transit_high+approach)?

**Found and fixed a real bug first**: the initial version of this proxy measured the wrong quantity (the
SOLVED IK target's joint6, not the arm's CURRENT/seed qpos entering descend -- these differ because
`cr_cfm/inference.py`'s `build_template_x0` defines its first row as the seed, not the target). The buggy
version couldn't even reproduce Cracker's own known 34.4% pinning rate (got 1/32). Fixed by capturing the
right quantity (`primary_seed`, matching Priority 2's already-correct convention) -- the corrected proxy
then reproduced Cracker's reference rate EXACTLY: **11/32 pinned, trial-for-trial identical** to the
original diagnostic.

**Corrected cross-object comparison** (20 trials/object, trial_id 4000-4019): **Cracker 6/20 (30%, matches
reference) vs. Pear 1/20 (5%) vs. Mustard 1/20 (5%)**. Mustard predicts the SAME null result as Pear, not
Cracker's positive signal. Per the pre-registered decision rule, a low predicted pinning rate does not
warrant the expensive execution-based confirmatory test -- that's the entire point of building a validated
cheap proxy. **Skipped the Mustard execution test by design**, not by omission.

**Closing synthesis for Priority 4**: wrist-fix is a real, statistically confirmed, but OBJECT-DEPENDENT
improvement -- it helps precisely (and only) when an object's grasp geometry drives joint6 toward its
hardware limit, a property now shown to be predictable in advance via cheap IK geometry alone, rather than
requiring expensive execution to discover per-object. Full numbers and methodology in
`cr_cfm/IMPROVEMENT_PLAN.md`'s "Priority 4" section. **This closes all 4 items of the paper-readiness plan.**

## 📈 2026-07-22: Paper-readiness Priority 3 -- Cracker's critical ablation scaled from n=32 to n=152 (matching Stage 12's own power level): still not significant, and the small amount of discordance that DOES appear tilts toward baseline, not CR-CFM

The n=32 critical ablation (baseline+wristfix vs. CR-CFM+wristfix) found a perfect tie -- 26/32 both, zero
discordant pairs. Striking, but also exactly what an underpowered n=32 sample would produce even with a
small real effect present (Stage 12's own power analysis says ~150 paired trials are needed for 80% power
at this effect size). Ran the same 15 additional trial ranges Stage 12 used for its own n=152 confirmatory
result (120 more paired trials, `scratchpad/ablation_crcfm_isolation.py`, 15 independent short-lived
processes to avoid the known `mega_confirm.py` memory-leak pattern).

**Pooled n=152: baseline+wristfix 115/152 (75.7%) vs. CR-CFM+wristfix 110/152 (72.4%).** 17 discordant
pairs -- **11 favor baseline, only 6 favor CR-CFM** -- McNemar's exact test **p=0.33**, not significant.
5x the sample size does not reveal a hidden CR-CFM advantage; if anything the direction reverses toward
baseline. This is a stronger result than the n=32 tie: it rules out "not enough data to see it" as an
explanation and directly confirms, at adequate statistical power, that CR-CFM adds no value beyond the
wrist-fix on Cracker. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Priority 3" subsection under
"Critical ablation." Proceeding to Priority 4 (cross-object generalization, reframed around a cheap
IK-based joint6-pinning predictor rather than brute-force object-count scaling -- see next entry).

## 🔬 2026-07-22: Paper-readiness Priority 2 -- confirms Priority 1 mechanistically: Pear NEVER approaches the joint6 limit, on any of 16 runs, success or failure

Direct follow-up to Priority 1's null result. Instrumented `ArmIK.solve_multi_seed` to capture the seed qpos
entering the descend phase on the plain-baseline path (the non-CR-CFM equivalent of the `x0[0][5]` capture
used in Cracker's original joint6-pinning diagnostic), across all 8 Pear trials x both wristfix conditions
(16 runs total). **Zero of 16 show `pinned` (|abs(joint6) - 3.14| < 0.05)** -- the largest magnitude seen
anywhere is 2.632 rad, still >0.5 rad from the limit. The two trials that fail regardless of wristfix (3005:
joint6=0.754, 3007: joint6=-0.429) sit near the CENTER of the joint's range, about as far from the limit as
possible -- and on both, `pick_wrist_friendly_orientation` independently chose the *same* orientation with
or without the fix enabled, meaning the fix's own selection logic agrees there was nothing to flip.

**This confirms Priority 1's null result mechanistically, not just statistically**: Pear's grasp geometry
(small, round object, approach angles already concentrated in a narrow arc) never drives the wrist into the
region Cracker's flat, wide box geometry does. The wrist-fix is real, mechanism-grounded, and object-scoped
by design -- exactly the profile expected of a fix targeting one specific, previously-identified failure
mode, not a claim that it should generalize universally. Full write-up in `cr_cfm/IMPROVEMENT_PLAN.md`'s
"Step 2" entry. Proceeding to Priority 3: scaling the Cracker critical ablation from n=32 to n=152.

## 🏁 2026-07-21: FINAL DECISIVE FINDING -- plain baseline+wristfix outperforms CR-CFM on Pear too; this closes the entire investigation with a clear, cross-object conclusion

Closed the last gap: ran the ONE comparison the entire Pear investigation (v1->v2->v3, 25%->62.5%->62.5%)
never tested -- baseline+wristfix ALONE, no CR-CFM, on the same trials. Result: **baseline+wristfix 6/8
(75%) vs. CR-CFM v3 (143 trajectories)+wristfix 5/8 (62.5%)** -- the one discordant trial favors BASELINE.
Plain interpolation with zero training investment outperforms CR-CFM's best, most-data-scaled checkpoint on
Pear. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "FINAL DECISIVE FINDING" entry.

**The complete, three-part, cross-object picture**: (1) Cracker -- CR-CFM ties baseline+wristfix exactly, 0
discordant pairs across 32 trials; (2) Pear -- CR-CFM actually UNDERPERFORMS baseline+wristfix; (3)
mechanism -- CR-CFM's RHC loop cannot react to anything in the environment, only to its own drift toward an
already-fixed target, explaining why extensive investment (Stages 1-15, Pear's 3.5x data scale-up) never
closed a gap that baseline+wristfix never had in the first place.

**This is the authoritative conclusion of the entire investigation.** The one validated, real contribution
is the wrist-orientation-selection fix -- simple, training-free, model-independent, statistically confirmed
at the mechanism level (p=1.8e-5), and now shown to match or beat CR-CFM on every tested object while
needing no training data, no model, no real-time inference, and none of CR-CFM's own unresolved failure
modes. **Any real deployment or paper built on this work should use plain baseline + wrist-fix as the
actual system, not CR-CFM.** The CR-CFM investigation remains valuable as the honest, rigorous diagnostic
process that led to finding this fix -- not as a validated control algorithm.

## 🔬 2026-07-21: Perturbation test -- CR-CFM still shows zero advantage under disturbance, and the reason is architectural: its closed loop only closes around the ARM's own state, never the object's

Direct follow-up to the critical ablation's null result: hypothesized the unperturbed test conditions never
gave CR-CFM's closed-loop RHC a chance to show value over open-loop baseline, since nothing ever required
reactive correction. Injected a one-time joint-space disturbance (+-0.1 rad, uniform) partway through the
descend phase (physics step 50) via the existing `step_hook` mechanism, for both conditions. Result:
**baseline 3/8 (37.5%), CR-CFM 3/8 (37.5%) -- still fully concordant, zero discordant pairs.** Both dropped
substantially from their unperturbed 6/8 (the perturbation is real and consequential), but by exactly the
same amount on exactly the same trials.

**Root cause, now understood precisely**: `target_qpos` is IK-solved ONCE before descend starts, for BOTH
conditions -- CR-CFM's RHC loop only re-reads the ARM's own qpos each iteration, never the object's
position or a re-solved target. Perturbing the arm's joint state tests something neither architecture needs
special adaptivity for: both are driven by absolute position commands toward the same fixed target, and
MuJoCo's PD tracking recovers from a one-time disturbance regardless of open-loop vs. RHC planning. **CR-CFM's
actual closed loop closes around its own execution, not around the environment** -- it has no mechanism to
react to a changed goal, only to its own drift relative to an already-fixed one. Full analysis in
`cr_cfm/IMPROVEMENT_PLAN.md`'s perturbation-test entry.

**This is not "the test needs to be harder" -- it's an architectural finding.** The correct next test would
perturb the OBJECT's position (invalidating the computed target), but CR-CFM as currently implemented has
no mechanism to re-solve IK against fresh perception mid-RHC-loop either -- demonstrating real closed-loop
value would require an actual architecture change (re-solving the object-relative target each iteration,
not just re-reading arm state), not another evaluation of the existing system.

## 🚨 2026-07-21: DECISIVE FINDING -- CR-CFM adds NO measurable value beyond the wrist-fix on Cracker; the entire confirmed win is attributable to the wrist-orientation fix ALONE, not flow-matching

Critical ablation, identified as the single most important missing experiment for the project's core
technical claim: Gate 1 compared plain baseline (NO wrist-fix) against CR-CFM+wrist-fix -- but the wrist-fix
is a grasp-orientation-selection mechanism, completely orthogonal to and independent of whichever
descend-execution method is used. It could equally apply to the plain baseline. This ablation tested
exactly that: baseline+wrist-fix vs. CR-CFM+wrist-fix, same 32-trial paired protocol as Gate 1.

**Result: 26/32 (81.2%) for BOTH conditions. Zero discordant pairs across all 32 trials -- every single
trial produced an identical majority-vote outcome.** CR-CFM (the 49K-parameter flow-matching model, RHC,
and every CR-CFM-specific mechanism from Stages 1-15) contributes nothing measurable beyond the
wrist-orientation fix on Cracker. The entire Gate 1 win (81.2% vs. 59.4%, p=0.0156) is fully explained by
the wrist-fix alone. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s critical-ablation entry.

**What this does NOT invalidate**: Stage 12's result (wrist-fix helps CR-CFM itself, p=0.027) remains
correct -- it just turns out that benefit isn't specific to CR-CFM at all. **What this means for the
project's framing**: the extensive Stage 1-15 diagnostic investigation remains a rigorous, honest,
valuable systems-diagnosis narrative (the "Embracing Negative Results" framing, arXiv:2406.03980) -- but
the claim "our learned flow-matching policy beats the baseline" is NOT supported once the wrist-fix
confound is controlled for. **The real, validated, useful contribution is the wrist-orientation-selection
fix itself** -- simple, training-free, model-independent, and statistically confirmed at the mechanism
level (p=1.8e-5 on the joint6-pinning correlation) -- not a claim about CR-CFM or learned control. Any
paper or report built on this work should be reframed accordingly: this is the story of a rigorous
diagnostic process that found a real, simple fix, not the story of a validated new learned-control
algorithm. This finding supersedes any earlier framing in this document implying CR-CFM itself was
confirmed to add value.

## ⛔ 2026-07-21: Pear approach-angle audit REJECTS the leading hypothesis -- Cracker's Stage 1 fix cannot transfer because Pear never had that problem

Direct test, using Cracker's exact Stage 1 audit methodology, of whether Pear's residual 62.5%-vs-75% gap
is an angle-diversity/sparse-tail problem like Cracker's was. Computed Pear's approach-angle distribution
across all 143 collected trajectories: **96.5% already fall within the same (-30, 60) range** Stage 1 found
for Cracker, with only 5 outliers total -- Pear's data was never in the sparse-tail failure regime that
fix addresses. Applying the same angle-range filter would remove a negligible 5/143 trajectories. Full
numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Pear approach-angle audit" entry. **This specific hypothesis is
REJECTED** -- not because the underlying "Geometric Entropy" principle is wrong, but because it doesn't
apply here. **Updated conclusion**: the residual gap (1 discordant trial out of 8) may not even be
statistically distinguishable from parity at this sample size, and if it is real, it likely needs the same
depth of dedicated, multi-stage diagnostic investment Cracker received (Stages 6-15), not a quick
transferable fix. This closes the Pear angle-diagnostic thread with a clean, informative negative result.

## 📊 2026-07-21: Pear data scaling PLATEAUS at 143 trajectories -- the "more data" trend was real but not linear, closing the Pear investigation with an honest, non-obvious finding

Direct follow-up to the promising v1->v2 result below: collected 62 more Pear trajectories (81 -> 143
total, now exceeding Cracker's own 127; more environment-level timeout interruptions, each resumed rather
than restarted). Trained `cr_cfm_pear_v3.pt` (858 descend segments), re-ran the identical Gate 2 protocol:
**`v3` scores 5/8 (62.5%) -- IDENTICAL to `v2`'s win rate**, not further improved, despite nearly doubling
the data again. The specific failing trial changed (3004 now succeeds, 3006 newly fails) but the aggregate
outcome plateaued exactly where `v2` left it. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "v3 RESULT"
entry. **Final honest conclusion**: data volume closed most of the original gap (25% -> 62.5%) in one step,
then plateaued -- the remaining ~12.5pp to baseline parity needs the same kind of dedicated mechanistic
diagnosis Cracker received (Stages 6-13), not simply more of the same data collection. This closes the
Pear multi-object investigation as a well-characterized, honestly-reported result: substantially but not
fully explained by data insufficiency, with a clear, specific direction for any future continuation
(a Stage-1-style diagnostic pass on Pear specifically) rather than an open-ended "collect more data" loop.

## 📈 2026-07-21: Pear data-scale follow-up CONFIRMS the "insufficient data" diagnosis -- doubling training data took CR-CFM+wristfix from 25% to 62.5%, near-parity with baseline's 75%

Direct test of the diagnostic conclusion below: collected 41 more Pear trajectories (81 total, up from the
original 40; two collection runs were interrupted by an environment-level timeout unrelated to memory, each
resumed from the highest saved trial_id rather than restarting). Retrained `cr_cfm_pear_v2.pt` (486 descend
segments vs. `v1`'s 240, same hyperparameters otherwise, healthy loss curve). Re-ran Gate 2's exact
evaluation protocol (same 8 trials, paired): **`v2` scores 5/8 (62.5%), up from `v1`'s 2/8 (25%)** -- vs.
baseline's unchanged 6/8 (75%). Only ONE discordant trial remains (3004), down from `v1`'s four. Full
numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "v2 RESULT" entry. **This is a genuinely positive update to
Gate 2's FAILED verdict**: the original failure was real and correctly reported, but it is a fixable
data-scale gap, not a fundamental cross-object generalization limitation -- the same architecture and
mechanisms that failed badly at 40 trajectories perform respectably at 81, with no other change. Suggests
continued data collection toward Cracker's 127-trajectory scale is a more promising path than a
Stage-1-style angle diagnostic (the other candidate hypothesis), though not pursued further here.

## 🔬 2026-07-21: Gate 2 follow-up -- Pear's failure has a DIFFERENT root cause than Cracker's, ruling out reusing Stage 8-15's fixes

Applied Cracker's own diagnostic tools (Stage 7's velocity-divergence check, Stage 12's joint6-pinning
check) to Pear's 4 CR-CFM-specific regressions (trials 3000, 3001, 3004, 3006) vs. 2 successes (3002, 3003).
Result: **zero of 6 trials show joint6 pinning; zero show any velocity divergence** -- every trajectory,
failure or success, is smooth and numerically well-behaved (~0.07-0.11 velocity norms throughout, matching
the "healthy" Cracker signature). Neither established Cracker root cause applies to Pear. Failures show
moderate `dist_to_tray` (0.21-0.38m, real partial progress, not a clean miss) vs. successes landing almost
perfectly (0.005-0.016m) -- consistent with plain model imprecision from insufficient training data (Pear:
40 trajectories vs. Cracker's 127) rather than a mechanical/numerical issue. Full table in
`cr_cfm/IMPROVEMENT_PLAN.md`'s "Gate 2 follow-up diagnostic" entry. **Conclusion**: Cracker's specific fixes
(Stage 8-15) do not transfer to Pear because Pear's problem isn't the same problem -- a future attempt
would need substantially more Pear-specific training data and possibly its own angle-range diagnostic pass
(Stage 1-style), not a reapplication of Cracker's mechanisms.

## 📋 2026-07-21: Real-Hardware Readiness Gate 5 -- staged hardware rollout procedure documented (not executable without physical access)

Unlike Gates 1-4, Gate 5 is not a sim-testable claim -- it is the actual procedure that should govern the
first real Piper trials, following the same "Sim-to-Lab-to-Real"/ISAACS staged-trust philosophy cited for
Gate 3. Four stages documented in `cr_cfm/IMPROVEMENT_PLAN.md`'s Gate 5 entry: (A) tethered no-load dry
run, (B) low-speed supervised dry run with a real object including analogs of Gate 4's known failure
shapes, (C) full-speed supervised trials building sample size gradually (n=5->10->20) and compared against
Gate 1's 81.2% sim figure as a stop/go signal, (D) unattended operation only after a clean Stage C pass.
Pre-flight checklist requires Gate 3's `clip_action_to_real_limits` wired into the real backend as a
non-negotiable last line of defense, PLUS hardware-level motor limits as defense in depth, and scopes
everything to the Gate-1-confirmed config on Cracker specifically. **This completes all 5 gates of the
Real-Hardware Readiness Plan** -- 1/3/4 tested and passed in sim, 2 tested and honestly failed (Pear), 5
specified as the governing procedure for when physical hardware access begins.

## 🏁 2026-07-21: Real-Hardware Readiness Gate 4 PASSED -- all known Cracker failures classify as benign/recoverable, none hazardous; PLAN COMPLETE (Cracker-scoped)

Retrieved full diagnostics (`terminal_velocity`, `final_eef_residual`, `dist_to_tray`) for the 6 known
CR-CFM+wristfix failures from Gate 1's n=32 Cracker comparison (trials 1000, 1007, 1104, 1106, 1202, 1301).
Striking uniformity: terminal_velocity consistently low (0.006-0.010) across all 6, none showing the
high-velocity/divergent signature from the pre-Stage-12 instability; final_eef_residual's Z-component
consistently ~0.031-0.041m regardless of outcome -- every failure is a controlled, gentle near-miss, not a
wild excursion. `dist_to_tray` splits them into two sub-patterns (clean miss, object never engaged vs.
gentle partial progress that fell short) -- both benign, neither hazardous. Full table in
`cr_cfm/IMPROVEMENT_PLAN.md`'s Gate 4 entry.

**Real-Hardware Readiness Plan: COMPLETE.**

| Gate | Result |
|---|---|
| 1. Real-baseline comparison | PASSED (McNemar p=0.0156) |
| 2. Multi-object pilot | **FAILED** (Pear: 25% vs. baseline 75%) |
| 3. Safety-layer hardening | PASSED (found + fixed a real coverage gap) |
| 4. Failure-mode taxonomy | PASSED (zero hazardous failures) |
| 5. Staged hardware rollout | Not yet started -- the natural next step, Cracker-scoped only |

**Overall verdict**: real-hardware readiness is confirmed for Cracker specifically, not the method in
general. Gates 1/3/4 give real, checked confidence for a staged (tethered/low-speed/supervised) Cracker
deployment. Gate 2's failure was reported honestly rather than hidden -- any claim of general multi-object
readiness would be premature until other objects get their own dedicated diagnostic investment.

## ✅ 2026-07-21: Real-Hardware Readiness Gate 3 PASSED -- found and fixed a real, previously-invisible safety gap: Stage 8's clamp only protected 1 of the pipeline's several phases

Verified the concern behind Gate 3 directly rather than assuming it: intercepted the actual action sent to
`env.step` for trial 1007 (the known ~370rad-divergence case). Without any protection, the commanded joint
value reached **151.7 rad -- 48x the real ±3.14 rad hardware range**. Stage 8's existing
`clamp_waypoints_to_limits` reduced this correctly, but **125/1690 actions were still out of range** --
traced to `lower_into_tray`, a phase using a completely different code path
(`move_to_interpolated`/`solve_and_move`) the scoped clamp never covered. Implemented
`clip_action_to_real_limits` (new, `piper_pick_and_place.py`): a UNIVERSAL clip at the lowest common point
before any action reaches `env.step`, independent of which phase produced it. Verified with Stage 8's clamp
disabled: caught **all 275 violations across the entire pipeline (275/1690 -> 0/1690)**. Full details in
`cr_cfm/IMPROVEMENT_PLAN.md`'s Gate 3 entry. **Decision**: Gate 3 PASSED -- `clip_action_to_real_limits` is
documented as MANDATORY for any real-hardware backend, independent of win rate. This is the single most
safety-relevant finding of the real-hardware readiness plan: a scoped, mechanism-specific safety fix
(Stage 8) silently left most of the pipeline unprotected, and only a universal, unavoidable clip at the
action-output boundary closes the gap completely.

## ⛔ 2026-07-21: Real-Hardware Readiness Gate 2 FAILED -- CR-CFM+wristfix generalized to Pear is markedly WORSE than baseline, exact opposite of Gate 1's Cracker result

Collected 40 successful Pear demonstrations (36/60 full run + 4/5 smoke test, matching arXiv:2511.01770's
~30-50 trajectory precedent), fine-tuned a Pear-specific checkpoint using Cracker's `v6_narrowed` default
hyperparameters (no angle-range narrowing -- that was itself a Cracker-specific Stage 1 finding, not
reapplied here), training loss looked healthy (0.0045 -> 0.0005, clean plateau). Paired evaluation (same
protocol as Gate 1, 8 trials x 3 repeats): **baseline 6/8 (75%) vs. CR-CFM+wristfix 2/8 (25%)** -- ALL 4
discordant pairs favor the baseline, zero favor CR-CFM, the exact opposite pattern from Gate 1's Cracker
win. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s Gate 2 entry. **Honest read**: the training pipeline
itself works end-to-end for a new object (data collection, fine-tuning, evaluation all ran cleanly) -- the
failure is in result quality, not infrastructure. Cracker's `v6_narrowed` needed real, hard-won,
diagnosis-driven tuning (the whole Stage 1-12 investigation) to reach its confirmed performance; reusing
those exact defaults for a geometrically very different object (Pear: round/ellipsoidal vs. Cracker:
rectangular box) via a quick 40-trajectory fine-tune, without repeating any of that diagnostic depth, did
not transfer. **Decision**: Gate 2 FAILS. This reinforces, rather than resolves, the multi-object
generalization concern flagged before this gate was attempted -- real-hardware deployment should stay
scoped to Cracker (where Gate 1 is confirmed) until other objects get their own dedicated diagnostic
investment; Gates 3-5 remain valid work but should not be read as extending past Cracker specifically.

## ✅ 2026-07-21: Real-Hardware Readiness Gate 1 PASSED -- CR-CFM+wristfix significantly beats the plain interpolation baseline, McNemar p=0.0156, the cleanest result of the entire project

First head-to-head, properly paired comparison this session between the current best CR-CFM configuration
(`v6_narrowed` + `wrist_friendly_orientation=True`) and the plain, non-learned interpolation baseline this
whole project set out to replace -- run on the same trial_ids/seeds across all 4 established ranges (32
trials, 3 repeats/condition/trial). Result: **baseline 19/32 (59.4%) vs. CR-CFM+wristfix 26/32 (81.2%)**.
**McNemar's exact test: p=0.0156 -- significant**, and every one of the 7 discordant pairs favors CR-CFM;
zero favor the baseline. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Real-Hardware Readiness Plan"
section, Gate 1 entry. **Caveat flagged honestly**: baseline's 59.4% here is notably below the historically-
cited 75% reference figure -- most likely because this specific 32-trial set (which became this project's
standard evaluation set early on, and includes several trials that turned into this session's best-known
hard cases) happens to be harder for the plain baseline specifically, not a methodological inconsistency;
the paired comparison on this set is the one that matters and it is unambiguous. **Gate 1: PASSED** --
proceeding to Gate 2 (multi-object pilot) of the real-hardware readiness plan.

## ⚠️ 2026-07-21: Stage 15 (convergence-stall-based adaptive budget) REJECTED -- identical pooled outcome to both wristfix-alone and Stage 13's blunt constant; the intended per-trial selectivity did not materialize

Second item from the improvement-strategy literature search: a `stall_aware` mechanism designed to be
structurally different from Stage 13's blunt global `max_iterations=20` -- tracks per-iteration joint-space
residual and only extends budget for trials that are ACTUALLY stalling (recent-window improvement rate
below a threshold at a checkpoint iteration), leaving already-converging trials untouched. Smoke test found
a real bug in the first design (comparing against iteration 1 masks a fast-then-plateau pattern) and fixed
it with a recent-window comparison instead -- which then correctly triggered on trial 1007 and fixed it,
but ALSO over-triggered on known-healthy trials (1001, 1002), flagging a real risk before scaling up. Full
4-range test confirmed the risk was real: **pooled win rate 26/32 (81.2%) and disagreement 3/32 (9.4%) --
both EXACTLY IDENTICAL to wristfix-alone**, with the identical specific trade Stage 13 made (trial 1007
fixed, trial 1206 broken). The "recent-window improvement rate" signal is confounded by ordinary RHC
deceleration near convergence, which looks statistically similar to genuine stalling -- a structural
limitation of the signal, not a calibration slip. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s Stage 15
entry. **Decision**: REJECTED -- `wrist_friendly_orientation=True` alone remains the confirmed default;
73.0% (111/152) remains the number to cite. This closes both items from the improvement-strategy literature
search (Stage 14's angle-tolerance extension, Stage 15's stall-aware budget) without further gains beyond
Stage 12's confirmed win.

## ⚠️ 2026-07-21: Stage 14 (small-angle orientation tolerance sweep) REJECTED -- the target trial got worse, not better, as tolerance increased

Extension of Stage 12's confirmed exact-binary orientation check (0/180 degrees only) to a small-angle
tolerance sweep, motivated by grasp-planning literature's general practice of sampling finer orientation
increments. Implemented `_rotate_grasp_about_approach` + `angle_tolerance_deg` param on
`pick_wrist_friendly_orientation` (opt-in, default 0.0 = exact prior behavior). Calibration on trials
1000-1005 (0/10/20 degrees): known-reliable trials 1001-1005 never regressed at any tolerance -- but trial
1000 (the intended target) got WORSE, not better (1/3 success at tolerance=0 -> 0/3 at tolerance=10 and 20).
Correction along the way: trial 1000 turned out to be a genuinely unstable trial even at the exact binary
check, not the clean deterministic failure Stage 12's isolated smoke test suggested. Full numbers in
`cr_cfm/IMPROVEMENT_PLAN.md`'s Stage 14 entry. **Decision**: REJECTED, no full tuning-range test run --
calibration alone showed a clear negative trend. Reverting to `angle_tolerance_deg=0.0`
(Stage 12's exact binary check) as the default; 73.0% (111/152) remains the number to cite. The literature's
generic "sample finer orientation increments" precedent does not transfer to this symmetric-gripper,
narrow-axis-grasp setup -- only the exact 0/180 pair are true grasp-preserving equivalents here.

## 🔬 2026-07-21: Stage 13 -- trial 1007's residual failure fully explained (wrist-fix also resolves the Stage 7 divergence, and the approach angle sits in the training data's dense region), but the follow-up fix (`max_iterations=20`) is net-neutral, NOT adopted

With `wrist_friendly_orientation=True`, trial 1007's iteration-1 velocity norms drop to 0.08-0.10 (matching
known-stable trial 1001 exactly, down from 6.04->1387 without the fix) and all repeats become bit-identical
-- confirming the wrist-fix ALSO resolves Stage 7/10's numerical-divergence finding as a side effect, not a
separate mechanism. Approach angle (32.5 degrees) sits squarely in the training data's dense region, ruling
out the Stage 1 data-sparsity hypothesis too. What remains: a fully stable, deterministic trajectory that
simply doesn't converge close enough within `max_iterations=12` (~3.2cm residual, not shrinking with more
iterations). Sweeping `max_iterations` on trial 1007 alone found a real but NON-MONOTONIC effect (12/16
fail, **20 succeeds**, 30 fails again) -- residuals barely change across the sweep, so success hinges on
fine-grained timing, not distance. Full 4-range test of `wrist_friendly_orientation` + `max_iterations=20`:
**exactly the same pooled win rate (26/32=81.2%) as wristfix alone** -- it fixed trial 1007 in tuning but
broke trial 1206 in fresh, a redistribution with no net gain and slightly worse disagreement (12.5% vs
9.4%). Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s Stage 13 entry. **Decision**: `max_iterations=20` is
NOT adopted -- keep the confirmed Stage 12 defaults (`wrist_friendly_orientation=True`, `max_iterations=
12`, 73.0%/111-152 as the number to cite). Trial-1007-class compound failures remain open; a future fix
should look for a genuinely adaptive, per-trial budget signal rather than a single global constant.

## ✅ 2026-07-21: Stage 12 CONFIRMED -- wrist-friendly grasp orientation is a statistically significant improvement (n=152, McNemar's exact p=0.027), first newly-confirmed win since Stage 1

Committed to the full confirmatory run the power analysis called for (~150 total paired trials): 13 new
ranges (1600-2807, 104 more paired trials), run via a memory-safe per-range-chunked script after the first
attempt (one giant Python process) leaked memory unboundedly and had to be killed with zero results.
Combined with the original 48 for **n=152 total paired trials**: 16 discordant pairs favor
`wrist_friendly_orientation`, only 5 favor baseline. **McNemar's exact test: p=0.027 -- significant.**
Marginal totals: wrist-fix 111/152 (73.0%) vs. `v6_narrowed` baseline 100/152 (65.8%) -- a confirmed ~7.2pp
real gain, and a materially larger, more robust baseline estimate than Stage 3's original 68.8%/n=32 (the
two are close and consistent, just now backed by 4.75x more data). Full numbers in
`cr_cfm/IMPROVEMENT_PLAN.md`'s Stage 12 "FINAL RESULT" entry. **Decision**: `wrist_friendly_orientation=
True` is now CONFIRMED, not merely promising -- adopted as the new default (`v6_narrowed` +
`wrist_friendly_orientation=True`). **73.0% (111/152) is now the number to cite**, not 68.8%. Trial 1007
remains a clean concordant failure under both conditions -- this fix helps broadly (the underlying
joint6-pinning mechanism, p=1.8e-5) but does not resolve the hardest, likely-compound-cause outliers,
which remain open for a future, different mechanism.

## 📐 2026-07-20: Stage 12 power analysis -- McNemar's test (the methodologically correct test for this paired design) gives p=0.18, and ~150 total trials (100 more) would be needed for 80% power

Correction: every trial was run paired (same trial_id/seed under both conditions), so Fisher's exact test
on marginal totals (used for the n=32/n=48 comparisons above) wasn't the right test -- McNemar's exact test
on the full 48-trial paired table (matching this project's own established convention) gives **p=0.18**: 7
discordant pairs favor `wrist_friendly_orientation`, only 2 favor baseline. Closer to significance than the
Fisher framing suggested, but still not there. Simulated power analysis (assuming the current 18.75%
discordance rate and 7:2 split hold as the true effect): reaching 80% power would need **~150 total paired
trials, about 100 more than the 48 already run (~13 more range-pairs, est. 4-6 hours more compute)**. Full
numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s power-analysis entry. **Status**: analysis complete, the large
confirmatory run has NOT been launched -- decision on whether to commit that compute, versus pivoting to
finding a complementary mechanism for the remaining compound-failure cases (trial 1007-class), is pending.

## 📉 2026-07-20: Stage 12 confirmatory extension (n=48/arm) -- still not significant, and the two brand-new ranges alone show a near-tie -- tempers, not confirms, the earlier 4-range signal

Ran 2 more genuinely new ranges (1400-1407, 1500-1507), BOTH conditions on each, to properly power the
significance test the 4-range result (p=0.39) couldn't provide. New ranges alone: wristfix 14/16 (87.5%)
vs. baseline 13/16 (81.2%) -- Fisher **p=1.00**, essentially a coin flip apart. Combined with the original
4 ranges: **wristfix 40/48 (83.3%) vs. baseline 35/48 (72.9%), Fisher p=0.32** -- barely moved from n=32's
p=0.39 despite 50% more data. The honest read: the original 4-range result's apparent strength was
disproportionately driven by ONE range (validation's 50%->87.5% jump); a larger, previously-untouched
sample dilutes rather than confirms it -- the same kind of single-range-looks-better-than-it-is pattern
this project has now caught multiple times (Stage 3, Stage 4). Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`.
**Decision**: `wrist_friendly_orientation=True` remains the working default (never regressed on any of 6
ranges tested, and the underlying joint6-pinning diagnostic, p=1.8e-5, is a separate and still-solid
finding) but is NOT a proven win -- 68.8% (`v6_narrowed` alone) remains the number to cite; further n=8
smoke-test batches have clearly diminishing informativeness (p barely moved despite +16 trials) and are not
worth continuing without either a properly power-analyzed sample size or a complementary fix for the
compound-failure cases (trial 1007-class) this fix alone doesn't resolve.

## 📈 2026-07-20: Stage 12 full 4-range result -- most promising numbers since Stage 1 (81.2% pooled, every range tied-or-improved), but NOT statistically significant (p=0.39) -- report as unproven, not confirmed

Follow-up to the tuning-range tie: since it was a TIE (not a clean loss like Stages 8-11) and the
underlying diagnostic is unusually strong, ran the full 4-range protocol anyway.

| Range | wrist-fix | `v6_narrowed` baseline |
|---|---|---|
| Tuning | 6/8 (75%) | 6/8 (75%) -- tie |
| Held-out | 6/8 (75%) | 5/8 (62.5%) -- improved |
| Fresh | 7/8 (87.5%) | 7/8 (87.5%) -- tie |
| Validation | 7/8 (87.5%) | 4/8 (50%) -- improved substantially |
| **Pooled** | **26/32 (81.2%)** | **22/32 (68.8%)** |

Every range tied or improved, none regressed -- a genuinely different shape than any prior stage. But
Fisher's exact test on the pooled counts: **p=0.39, not significant**. Disagreement rate exactly unchanged
(9.4% both) -- any real effect converts some failures to successes rather than reducing instability. Full
numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s Stage 12 "Full 4-range RESULT" entry. **Decision**: adopting
`wrist_friendly_orientation=True` as the new tentative default for continued evaluation (zero ranges
regressed, mechanism is well-diagnosed), but NOT citing 81.2% as a proven number anywhere -- 68.8%
(`v6_narrowed` alone) remains the last statistically validated figure; 81.2% is flagged explicitly as
encouraging-but-unconfirmed, needing a larger n before being trusted as a real gain. This is consistent
with, not a departure from, the same discipline that caught Stage 3's validation dip and Stage 4's
tuning-range-only PACE signal as noise -- a good number alone was never sufficient here, and isn't now.

## 🔬 2026-07-20: Stage 12 -- wrist-friendly grasp orientation (root-cause fix, EARLIER pipeline layer than Stages 8-11): strongest diagnostic of the whole project (p=1.8e-5), but tuning-range result is an exact tie, not a clear win

Direct follow-up to Stage 11's closing recommendation ("look earlier in the pipeline, at how `target_qpos`
is computed"). Found `robot0_joint6` (wrist roll) has a genuine HARDWARE limit (`robot_arm.xml`:
`limited="true" range="-3.14 3.14"`, matching AgileX Piper's real spec). Diagnostic scan across all 32
established trials (tuning+held-out+fresh+validation): when the IK solution for the grasp target lands
joint6 pinned exactly at that limit, the trial fails 82% of the time (9/11) vs. 4.8% (1/21) when it doesn't
-- **Fisher's exact p=1.8e-5, odds ratio 90**, by far the strongest single predictor found this entire
project. Implemented `pick_wrist_friendly_orientation`: solves IK for both the computed grasp orientation
and its 180-degree-flipped equivalent (same grip, approaches from the object's other side), keeps whichever
leaves joint6 further from its hard limit. Smoke test: fixed trial 1007's pinning exactly as designed
(joint6 3.140 -> 0.282) but task success did not follow (compound difficulty remains); trial 1000 was
unaffected (both orientations still hit the limit for that yaw). Full tuning-range test: **6/8 (75%),
disagreement 2/8 -- an exact tie with `v6_narrowed`'s baseline**, not a clear win, so the full 4-range pool
was not run per this project's established discipline. Real but small/mixed effect: trial 1000 (a clean,
deterministic 3/3 failure in every prior stage this session) gained partial success (1/3); trial 1006 (more
stable under baseline) gained new disagreement. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 12"
entry. **Honest read**: the underlying diagnostic remains a real, strong finding (p=1.8e-5) -- but fixing
one grasp-orientation choice alone is not a sufficient standalone fix, particularly for the hardest,
likely-compound-cause trials. Worth revisiting combined with `v6_narrowed`'s existing mechanisms or at
larger n before any stronger claim, rather than treated as resolved.

## ⛔ 2026-07-20: Stage 11 REJECTED at gate Step 1 -- refutes "stabilize + extend budget," closes out the entire Stage 8-11 family of RHC-descend-internal fixes

Implemented and smoke-tested the difficulty-aware design proposed above: `sample_corrected_trajectory`
gained `return_diagnostics`, `move_to_cr_cfm_descend` gained `difficulty_aware`/
`difficulty_extended_max_iterations`, using Stage 10's calibrated velocity-norm trigger to arm subdivision
and extend the iteration budget (12->18 or 12->24) ONLY for detected-hard trials. Trigger fired correctly
on trial 1007 in both budget conditions -- but task success still failed all 6 runs, AND the final residual
was WORSE than baseline in both (0.093/0.091 vs 0.030), not merely unimproved. Extending 18->24 barely
moved the outcome (0.0930->0.0907) -- ruling out "just needed more time": the stabilized trajectory
converges to an actual fixed point well before either ceiling, and that fixed point is simply wrong. Full
numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 11... RESULT" entry. **This closes out the whole Stage
8-11 family** (value clamp, displacement clamp, integration subdivision, subdivision+budget) -- all four
rejected, with the last three all converging on the same finding: stabilizing trial 1007's execution
removes the divergence but does not produce success, and often makes the deterministic outcome worse than
the baseline's noisy one. **Updated leading hypothesis**: the fixed, pre-computed `target_qpos` itself
(from `ik.solve_multi_seed`, solved once before the RHC loop starts) may be a poor target for this trial's
geometry -- baseline's occasional better-looking chaotic outcomes were likely lucky noise landing nearer a
workable state, not evidence a clean reachable trajectory exists for a stabilized policy to find. **Next
lead, if pursued**: look earlier in the pipeline, at how `target_qpos` is computed, rather than at anything
inside the descend-execution mechanism -- four independent, well-motivated attempts at that layer have now
been rejected.

## 📋 2026-07-20: Stage 11 PROPOSED (not yet implemented) -- difficulty-aware adaptive iteration budget, combining Stage 7's detector with Stages 9/10's confirmed-but-incomplete fixes

Fourth literature search this project, grounded directly in Stages 9/10's shared finding (two independent
mechanisms both eliminate trial 1007's divergence but neither reaches success within the fixed
`max_iterations=12` budget). Found a real, active research family matching the exact needed combination:
ELASTIC (arXiv:2606.31132, state-dependent test-time compute for generative control policies, 34% latency
cut at matched quality), DASIP (arXiv:2511.20906, per-instance difficulty classifier picks integration step
budget, 2.6-4.4x compute reduction at matched success), and AutoHorizon/"VLA Knows Its Limits"
(arXiv:2602.21445, confirmed training-free, uses an internal model signal to adapt EXECUTION horizon,
matches oracle performance on LIBERO/RoboTwin). All three establish "detect difficulty from a self-generated
signal, spend extra test-time compute only on hard cases" as a validated pattern, not a risky invention --
and this project's own signal is already fully specified: Stage 10's calibration found the model's own
iteration-1 velocity magnitude cleanly separates known-easy (max ~0.13) from known-hard (6.04) trials, no
new classifier needed. Proposed design: when that signal fires, enable Stage 10's `adaptive_subdivide`
(already implemented) AND extend `max_iterations` for that trial only (candidates 18/24, to be calibrated)
-- easy trials keep the cheap default of 12, untouched. Full design, pre-registered gate (trial-1007 smoke
test -> easy-trial budget-neutrality check -> tuning-range gate -> 4-range pool), and literature citations
in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 11" entry. **Status: proposed, not yet implemented** -- awaiting
confirmation before writing the trigger/branch code and running the first smoke test.

## ⚠️🔬 2026-07-20: Stage 10 (adaptive Euler subdivision) REJECTED as tested -- second independent mechanism to eliminate divergence without fixing task success, pointing squarely at Stage 9's iteration-budget diagnosis

Literature-grounded follow-up to Stage 9 (AdaFlow, arXiv:2402.04292; "From Euler to Dormand-Prince,"
arXiv:2605.00836): calibration found a dramatic, EARLY-detectable signature -- known-stable trial 1001's
per-substep velocity magnitude stays bounded ~0.08-0.13 across all 6 Euler substeps; known-unstable trial
1007's is already 6.04 (~50x higher) at the VERY FIRST substep, then compounds exponentially to 1387 by the
last substep. Implemented `cr_cfm_adaptive_subdivide`: when a substep's velocity exceeds a threshold
(0.5, calibrated between the two trials' ranges), subdivide it into finer sub-steps instead of one large
jump -- refining integration resolution rather than capping the output (Stage 8) or displacement (Stage 9).
Smoke test on trial 1007: task success unchanged (still fails all 6 runs), final residual actually WORSE
(0.031->0.095) -- but repeat-to-repeat divergence again vanishes COMPLETELY (bit-identical across all 3
repeats), the second independent mechanism (after Stage 9's rate clamp) to fully stabilize the trial
without producing success. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 10... RESULT" entry.
**Updated conclusion**: the numerical divergence is real and causally implicated in run-to-run VARIANCE
(twice confirmed), but is not what stands between trial 1007 and task success -- the more likely blocker is
the fixed `max_iterations=12` budget itself, independent of numerical stability. **Next test**: raise
`max_iterations` alone (no clamp, no subdivision) to check whether budget is the real remaining constraint.

## ⚠️🔬 2026-07-20: Stage 9 (rate/displacement clamp per waypoint) REJECTED as tested -- but CONFIRMS the causal mechanism via bit-identical repeats

Follow-up to Stage 8: capping how far qpos may move per waypoint relative to the arm's actual current
position (`cr_cfm_max_step_per_waypoint`, chain-clamped, preserves direction) is mechanistically distinct
from Stage 8's redundant value clamp -- MuJoCo limits the final qpos value, never the approach rate.
Smoke test on trial 1007 (3 repeats, values 0.2 and 0.1 rad, calibrated against stable trial 1001's normal
~0.001-0.15 rad range): BOTH clamp values made terminal_velocity and final_eef_residual substantially worse
(0.0101->0.24/0.68, residual 0.031->0.035/0.107) and the trial still failed in all 9 runs -- capping
displacement forces more RHC iterations to cover the same distance, but `max_iterations=12` is fixed, so
the clamped runs simply ran out of budget mid-approach. **Genuine side-finding**: repeat-to-repeat
divergence vanished COMPLETELY under both clamp values (bit-identical across all 3 repeats) -- direct
causal confirmation that slowing the aggressive first motion eliminates the chaos-triggering event Stage 7
localized; the mechanism is now confirmed, the fix just isn't usable as implemented because it interacts
badly with the fixed iteration budget. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 9... RESULT"
entry. **Not yet tried**: raising `max_iterations` alongside the rate clamp, to give the gentler approach
room to actually converge -- a different, untested combination, not a repeat of this rejection.

## ⚠️ 2026-07-20: Stage 8 (clamp raw model output to joint limits) REJECTED at the trial-level smoke test -- redundant with a constraint MuJoCo already enforces

Follow-up to Stage 7: found the model's raw Euler-integrated output diverges numerically to ~370 radians
for trial 1007's first RHC iteration (vs. sane 0.001-0.15 rad deltas for stable trial 1001, same
checkpoint) -- invisible to every prior task-space diagnostic, which only sees the already-clipped physical
outcome. Implemented `cr_cfm_clamp_waypoints_to_limits` (opt-in) to clip the model's raw waypoints to each
joint's real range before ever commanding them. Smoke test on trial 1007 (3 repeats, clamp on vs. off):
outcomes nearly identical (terminal_velocity ~0.0101 both ways, final Z-residual ~0.031 vs ~0.030, all 6
runs still fail) -- the clamp changes almost nothing. Root cause: MuJoCo joints with `limited="true"`
enforce hard range constraints directly in the physics solver, so qpos can never exceed `jnt_range`
regardless of how extreme the commanded target is -- the explicit clamp was redundant with a constraint
already being enforced downstream. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 8... RESULT"
entry. **The underlying diagnostic finding still stands** (the model's output genuinely diverges
numerically for at least this trial -- a real, previously-unknown behavior) -- just not one this particular
fix addresses. **Reframed candidate for a future stage**: a RATE/displacement clamp on how far qpos can
move per waypoint relative to the arm's ACTUAL current position (forcing a gradual multi-iteration approach
instead of one maximally fast attempt), mechanistically distinct from this stage's absolute-value clamp and
not yet tested.

## 🔬 2026-07-20: Stage 7 diagnostic -- divergence precisely localized to a ~10-physics-substep window right after a fast, large initial Z-drop, consistent with a high-velocity first-contact event

Follow-up to Stage 6's correction (`ik.solve_multi_seed` ruled out -- it's called once, before the RHC loop,
not per iteration; the only remaining candidate inside the loop capable of introducing real divergence is
`env.step` itself). Added a custom `step_hook` (zero source changes -- `run_pick_and_place` already accepts
one) logging eef Z at EVERY physics step instead of only once per RHC iteration, for 3 repeats of trial
1007. Result: substeps 0-30 are bit-identical across repeats; a fast, large Z-drop at substeps 25-29
(~13cm in 4 substeps) is ALSO bit-identical; then divergence first appears at **substep 31**, growing to a
peak of 0.086cm by substep 38, before partially resetting at the next RHC replan. This pins the divergence
to a specific, narrow window immediately following a fast initial descent -- the signature of a genuine
floating-point-path-dependent contact event in MuJoCo's own solver, not diffuse/system-wide noise. Full
numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 7... RESULT" entry. **New candidate surfaced**: the trigger
is tied to how FAST/LARGE the first Z-drop is -- a lever none of Stages 2/4/5 addressed (they targeted
output smoothness, chunk-cut timing, and replanning cadence respectively, not first-contact velocity).
Proposed Stage 8: cap the first RHC iteration's max per-step Z-velocity for a gentler initial contact --
a self-derived hypothesis this time, not literature-sourced, pending confirmation before implementing.

## 🔬 2026-07-20: Stage 6 diagnostic -- direct measurement of run-to-run divergence CONFIRMS it's real, but REFINES the mechanism from "sustained chaos" to a bounded, transient, trial-specific branch-sensitivity spike

Pure measurement, no intervention: 4 repeats each of trials 1006 (known disagreement), 1007 (this session's
worst chaos outlier), 1001 (stable control), tracking eef-Z divergence between repeats at every RHC
iteration (already recorded, zero new code needed). Result: 1001's repeats are **bit-identical** (zero
divergence throughout); 1006 is bit-identical in 5 of 6 repeat-pairs, sub-millimeter and non-growing in the
6th; **1007 spikes to 2.32cm divergence at exactly iteration 3, then decays back to ~0.001cm by iteration
12** -- all 4 repeats still fail despite the traces numerically reconverging. This is the opposite shape
from classic sustained/exponential chaotic growth (which the Lyapunov-exponent literature check was
looking for) -- it's a bounded, transient, self-correcting spike concentrated at ONE specific iteration,
and it's trial-specific, not system-wide (most trials/repeats show zero divergence at all). Full numbers
and mechanism discussion in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 6... RESULT" entry. **Reframed leading
hypothesis**: a discrete branch-sensitivity event (most likely `ik.solve_multi_seed`'s seed/argmin selection
for the descend target, re-solved fresh every RHC iteration -- a discrete choice that can flip
discontinuously on sub-millimeter differences, unlike the flow-matching model's own forward pass, separately
verified bit-deterministic earlier this session) rather than continuous chaotic amplification of the kind
Stages 2, 4, and 5 implicitly targeted. **Next concrete step** (not yet run): log which IK seed/branch gets
selected at each RHC iteration across repeats of trial 1007 and check whether it flips exactly at iteration
3 -- a direct, checkable causal test rather than another indirect policy/control-layer intervention.

## ⚠️ 2026-07-20: Stage 5 (RHC replanning frequency sweep, `execute_steps` ∈ {1,3,4}) REJECTED at the smoke-test gate -- none beat `v6_narrowed`'s own tuning-range number

Motivated by Stage 4's own conclusion (two independent mechanisms, one training-time one inference-time,
both left the disagreement rate untouched) -- tested whether the RHC loop's replanning frequency itself
(fixed at `execute_steps=2` since early this session, never swept) is a controllable knob. No new code, no
retraining, `v6_narrowed` unchanged. Pre-registered gate: cheap tuning-range-only smoke test first, only a
value that BEATS the current default's tuning-range number (6/8, 75%) proceeds to the expensive full
4-range pool. Result: `execute_steps=1` (more frequent replanning) underperforms at 5/8 (62.5%) -- opposite
of the "correct drift before it compounds" hypothesis; `execute_steps=3` and `4` (less frequent replanning)
both exactly TIE the default at 6/8 (75%), not beat it, and fail on the identical two trials the default
does (1000, 1007) -- notably, the current default is the ONLY one of the four values tested that solves
trial 1007, a small piece of evidence against a smooth "replanning frequency" trend existing at all. None
of the three swept values cleared the gate, so no full 4-range pool was run -- exactly the discipline
meant to avoid spending the larger compute budget chasing a smoke-test tie. Full numbers in
`cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 5... RESULT" entry. **Decision**: `v6_narrowed` with `execute_steps=2`
remains current best/default; 68.8% remains the number to cite. Three independent mechanisms across three
different layers (training-time smoothing, inference-time valley detection, control-layer replanning-
frequency sweep) have now all failed to move the ~9-12% instability rate -- the leading hypothesis going
forward is genuine MuJoCo physical chaos under contact, not a fixable property of the policy or control
loop; a future attempt in this direction should test that hypothesis directly rather than trying a fourth
variant of "smooth/gate/repace the plan."

## ⚠️ 2026-07-20: Stage 4 (PACE-style adaptive execution length) REJECTED -- tuning-range gain (75%→87.5%) does not survive the 4-range pool; pooled result is a statistical wash, slightly below `v6_narrowed`

Second literature search (grounded in Stage 2's null result, redirected to inference/control-layer methods
instead of more training-loss terms) surfaced PACE (arXiv:2606.00537): training-free, analyze the generated
chunk's own speed profile and commit only up to the first low-speed valley, instead of RHC's fixed
`execute_steps=2`. No retraining -- built on `v6_narrowed` unchanged. Tuning range alone looked strong
(6/8→7/8, 75%→87.5%, with trial 1007 -- this session's worst chaos-outlier case -- flipping to a clean 3/3
success), but per this project's standing discipline this was checked against the full 4-range pool before
being trusted: held-out 5/8→4/8, fresh 7/8→5/8, validation 4/8→4/8. **Pooled: 22/32 (68.8%) → 20/32
(62.5%)**, Fisher's exact p=0.79 (not significant -- statistically indistinguishable from `v6_narrowed`, not
a confirmed regression). Disagreement rate unchanged (9.4%→9.4%) -- adaptive execution length did not touch
run-to-run instability at all, same "zero measured effect on the target mechanism" pattern as Stage 2's
Lipschitz regularization. Fisher's exact test on tuning-range-alone vs. the other three ranges pooled:
p=0.20 -- the tuning-range signal itself was not statistically real, the same first-range-looks-better
pattern already caught once this session (Stage 3's validation check). Full numbers and methodology in
`cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 4... RESULT" entry. **Decision**: reverted to `v6_narrowed` with fixed
`execute_steps=2` as current best/default; 68.8% remains the number to cite. Two independent mechanisms
(training-time Lipschitz smoothing, inference-time PACE valley-detection) have now both failed to move the
disagreement rate -- worth remembering if instability is revisited: look elsewhere (RHC replanning
frequency itself, or the physical-chaos hypothesis directly) rather than another variant of either.

## ✅ 2026-07-19: Stage 1 of the improvement plan CONFIRMED -- narrowed training data (127/155 trajectories, dominant approach-angle arc only) beats the previous best on both win rate AND stability

New default checkpoint: `cr_cfm_cracker_v6_narrowed.pt`. Combined majority-vote win rate across all three
probe ranges: **67% -> 75%** (now exactly tying baseline), with disagreement rate roughly halved (~25% ->
12.5%) as a bonus, not a trade-off -- two of three ranges improved independently, none regressed. Full
numbers and methodology in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 1... RESULT" entry. This confirms
Geometric Entropy's (arXiv:2606.20871) counter-intuitive finding -- for this small model/small dataset,
narrower training diversity outperformed broader coverage, contrary to the "collect more sparse-angle
data" instinct.

## ⚠️ 2026-07-19: Stage 2 (Lipschitz regularization) REJECTED -- no stability gain, real win-rate cost

Tested on top of `v6_narrowed` (`cr_cfm_cracker_v7_lipschitz.pt`, `lambda_lip=0.01`): combined win rate
dropped 75%->66.7%, while the disagreement rate stayed EXACTLY unchanged at 12.5% -- zero measured benefit,
real cost. Rejected per the plan's own pre-registered gate; reverted to `v6_narrowed` as current best/
default. Full numbers in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 2... RESULT" entry -- matches this session's
own established pattern (a well-motivated, literature-grounded hypothesis a direct measurement rejects
rather than confirms, same as the earlier TCR-zeroing result).

## 📌 2026-07-19: Stage 3 blind validation -- Stage 1's improvement holds up, but the honest pooled figure is ~69%, not 75%

Ran `v6_narrowed` on trial_id 1300-1307, a range genuinely never touched by any decision in this plan:
4/8 (50%), below the 75-87.5% seen on the other three ranges. Checked directly (Fisher's exact,
p=0.22) and confirmed this is ordinary n=8 sampling noise, not a real regression -- consistent with the
earlier range-variance finding. **Pooling all four independently-evaluated ranges (32 trials, the correct
statistical treatment): 22/32 = 68.8%** -- this is the number that should be cited going forward, not the
more favorable 75% from the first three ranges alone. Still a real, substantial improvement over the
pre-Stage-1 estimate (67%) and the original ~50% single-run headline, just not literally tying baseline.
Full detail in `cr_cfm/IMPROVEMENT_PLAN.md`'s "Stage 3... RESULT" entry.

## 📋 2026-07-19: staged improvement plan, see `cr_cfm/IMPROVEMENT_PLAN.md`

Literature-grounded (`Geometric Entropy`, arXiv:2606.20871; `Robust Behavior Cloning via Global Lipschitz
Regularization`, arXiv:2506.19250), two-stage plan targeting the two confirmed open problems below:
approach-angle data-narrowing (test first, no new code) and Lipschitz regularization for the confirmed
~25% instability (test second, needs new loss code). Pre-registered decision gates in that file -- read it
before starting either stage.

## 🔬 CR-CFM (Contact-Robust Conditional Flow Matching) — Stage A in progress, no paired result yet (2026-07-18): `tango_robot/piper_robosuite/cr_cfm/`

New algorithm direction, structurally different from execution-control rows 1-13 (RULED_OUT_METHODS.md):
those are all hand-designed rules on a single fixed descend pass; this is a learned, temporal-
consistency-regularized flow-matching trajectory generator (49K params, 1D-conv + FiLM, `cr_cfm/model.py`),
trained on the team's own recorded Piper descend trajectories rather than DROID/BridgeData V2 (resource
reality check: compute is a single RTX 3060 6GB-class GPU historically, dual A100/A800 80GB available for
a short 1-2 week window later — every one of the 5 frontier world-model/VLA directions surveyed except a
small-scale flow-matching decoder needs infrastructure this team doesn't have; see conversation history).
Novelty check on the specific combination (asymmetric TCR + flow matching for contact-robust descent) was
found open (no exact match; PreAfford and Exp-Force are the nearest, narrower precedents) but not yet
formally logged as a citable check.

**Can support**: full pipeline runs end-to-end (data collection → drift-augmented flow pairs → training →
Euler-sampled inference → closed-loop execution via a new `cr_cfm_descend=True` mode in
`run_pick_and_place`) — validated on n=15 phase-tagged cracker trajectories. Two real bugs found and fixed
during this build: (1) `PiperTrajectoryRecorder`'s phase tagging silently no-op'd because
`hasattr(step_hook, "set_phase")` checked a bound method object instead of its owner (fixed via
`__self__` resolution, see `_set_phase()` in `piper_pick_and_place.py`); (2) `build_naive_x0`'s
straight-line joint-space interpolation produced an out-of-distribution inference-time input relative to
training's real PD-converged trajectory shapes — confirmed by n=6 smoke test (broke 2 previously-easy
baseline successes, one failing at near-zero drift, ruling out "just miscentered" as the explanation) —
fixed via `build_template_x0` (affine re-targeting of the dataset's own mean trajectory shape), which
recovered both broken cases in a follow-up n=6 check.

**Cannot support**: any claim that CR-CFM beats baseline yet. Post-fix n=6 check was a coin flip (2
discordant pairs, 1 each direction) against baseline on the same trials — real improvement over the
broken version, not yet evidence of being better than doing nothing extra.

## ⚠️ Trial 1007's anomalous "success" (drift=28.67cm) audited (2026-07-18): genuine run-to-run physics nondeterminism under cr_cfm_descend, not a one-off collision fluke — a real, unresolved methodology gap

**Can support**: re-ran trial 1007 (identical seed, identical checkpoint, identical code) 4 times under
`cr_cfm_descend=True`: outcomes were True/28.67cm, False/69.16cm, True/32.82cm, False/67.70cm —
qualitatively different every time, including the object being launched clear off the table (z drops from
0.908 to ~0.058) on 2 of 4 runs. Isolated the cause: the flow model's own forward pass is bit-deterministic
given identical input (verified directly: 5 repeated calls, max abs diff 0.0000000000), so the divergence
enters upstream, in the physics trajectory. **Baseline (no cr_cfm) on the same trial_id ALSO shows small
run-to-run nondeterminism** (final XY differs by a few mm across 2 runs) — this noise floor apparently
always exists in the simulation, but is normally invisible because it doesn't flip binary outcomes for
well-behaved trajectories. Under `cr_cfm_descend`, at least some generated trajectories (1007's included)
sit near a genuine physical bifurcation point (object teetering at the table edge), where that same
mm-scale noise gets chaotically amplified into "stays on table" vs. "launched onto the floor."

**Cannot support**: any interpretation of 1007 as a real grasp -- it was one lucky draw from what is
actually a wide, unit-cost-uncontrolled outcome distribution for that trial under the current model, not a
reproducible result. Also cannot yet support that this is confined to trial 1007 specifically, or that it
will improve with more training data (untested) -- it may reflect the current 15-trajectory model
generating physically aggressive/undamped waypoint sequences baseline never produces, which is itself a
negative finding about the model, separate from the x0-distribution fix.

**Methodology implication, not yet resolved**: a single run per trial_id may not be a valid signal for
`cr_cfm_descend` specifically if some fraction of generated trajectories are this chaos-prone --
McNemar's paired design assumes a fixed, reproducible outcome per (trial_id, condition) pair, which does
not hold here. Before trusting Step 3's paired pilot, should either (a) check how many trials in the
scaled-up eval show high run-to-run variance (e.g. 3 repeats per trial_id, flag any with disagreement) and
report that rate honestly, or (b) determine whether more training data reduces this instability before
declaring it a fundamental property of the approach.

## 📈 Four sequential, honestly-diagnosed fixes — win rate 0/8 → 1/8 → 0/8 → 2/8 → 4/8 on the same 8-trial probe set (2026-07-18)

Each step below was a distinct, verified root-cause fix, not hyperparameter search — every one changed a
specific, previously-measured mechanism, and every number here is from the identical trial_id 1000-1007
probe set for direct comparability.

1. **Data scaling alone (n=15→41), open-loop**: win rate flat (1/8→1/8), but run-to-run disagreement across
   3 repeats dropped from real chaos (4 different outcomes across 4 reruns of trial 1007) to 0/8 trials
   disagreeing — confirms more data fixes *reproducibility*, not accuracy, when the architecture is still
   open-loop.
2. **TCR zeroed in the final 30% of the horizon** (hypothesis: over-regularization was suppressing real
   deceleration near contact): mean terminal velocity got WORSE (8.44→11.11; trial 1007's alone went
   66.12→87.79), win rate unchanged (1/8→1/8) — **hypothesis rejected** by direct measurement, not
   assumption. Loss weighting was never the mechanism.
3. **Receding-horizon control** (execute only 2 of 16 generated waypoints per iteration, re-plan from the
   arm's real current state each time, instead of committing to the full chunk open-loop): terminal velocity
   collapsed to a tight, outlier-free range (mean 0.086, max 0.12, vs. the open-loop version's 11-88 range)
   — the runaway-velocity/table-launch failure mode is gone. Win rate alone didn't improve yet (0/8 at
   n=41) but this fixed a real, distinct, previously-measured problem (dynamic instability), setting up data
   scaling to actually help for the first time.
4. **Data scaling with RHC (n=41→155)**: win rate 0/8→2/8 — the first real accuracy gain, only possible
   once RHC had removed the instability data scaling alone couldn't fix.
5. **Root-caused the remaining ~10cm Z-axis shortfall via a per-iteration Z-trace**: NOT "needs more
   iterations" (ruled out directly — some trials' Z-trace is literally frozen, Δz~0.0001, by iteration 6;
   others show a genuine period-2 limit cycle, bouncing between two fixed heights forever, not slow
   convergence). Root cause: the conditioning vector (`x0_start - dataset_mean_start`) has no signal for
   *remaining distance to the true target* — once the arm's height resembles a familiar mid-trajectory
   state, cond≈0 regardless of whether the real target is still 10cm away, so the model had no way to tell
   "haven't started" from "still short." Fixed by concatenating remaining-distance-to-target
   (`target_qpos - x0_start`) into cond (6→12 dims) — exact at inference (target_qpos is already IK-solved
   before RHC starts), unlike the drift feature's dataset-mean proxy. **Win rate 2/8→4/8** (25%→50%) on
   retrain, with genuine descending Z-traces replacing frozen plateaus on the flipped trials.

**Persistent outlier**: trial 1007 has been the worst-behaved case under every single configuration tried
(open-loop, RHC, both conditioning schemes) — under the final conditioning fix its Z-trace is now the most
chaotic yet measured (0.613→1.606→1.481, residual=57cm). Worth investigating as a possibly genuinely
pathological object placement for this model, independent of whatever else gets fixed, rather than assuming
the next fix will resolve it too.

## 🔬 Fifth fix (remaining-distance-only conditioning, drift term dropped) + held-out validation (2026-07-18)

Ablated the v2 conditioning (12-dim: drift-from-dataset-mean concatenated with remaining-distance-to-target)
against a 6-dim remaining-distance-ONLY variant, same probe set, same everything else (n=155, RHC
execute_steps=2, max_iterations=12): **remaining-only won, 5/8 vs 4/8**, and tamed trial 1007's Z-residual
from a 57cm outlier down to a normal ~8.6cm, consistent with every other trial. Absolute-position drift was
adding noise for this 49K model, not useful signal -- promoted to the actual pipeline default (`train.py`,
`move_to_cr_cfm_descend`), not left as a one-off ablation checkpoint.

**Held-out validation, addressing the overfitting risk flagged above**: re-ran the exact same
retrained-through-the-real-pipeline checkpoint on trial_id 1100-1107 -- a range never touched by any fix or
tuning decision this session. **Result: 4/8 (50%)**, close to the tuning set's 5/8 (62.5%) and dramatically
above every earlier stage (0/8 → 1/8 → 0/8 → 2/8 → 5/8 across the five sequential fixes) -- the improvement
generalizes, not just tuned-set noise. Z-residual on the held-out set is equally tight (8.39-8.94cm) to the
tuning set (8.51-8.94cm) -- the ~8.5-9cm shortfall is a genuine, stable, reproducible property of the current
model, not overfitting artifact.

**Cannot support**: that this ~8.5-9cm Z-shortfall is resolved -- it is smaller and far more consistent than
the original ~10cm finding, but still present in literally every trial tested (16 total across both probe
sets), meaning `descend_refresh`'s plain interpolation is still doing real work closing the final gap in
every successful trial, not eliminated. Also cannot support that n=8 held-out is itself a fully trustworthy
number (still small) -- it corroborates the direction and rough magnitude of the tuning-set result, not a
final claim.

**Next steps, in order**: (1) investigate trial 1007 specifically now that it behaves normally under
remaining-only conditioning (or accept it as no longer a special case); (2) run the actual n=20 paired pilot
against baseline (fresh-env-per-trial, per rows 12-13's lesson) -- this is now overdue and the natural next
step given a held-out-validated improvement exists; (3) investigate the persistent 8.5-9cm shortfall itself
if further improvement is wanted, now that the confound (drift-term noise) has been removed and the
remaining gap is presumably a genuine model-capacity or `max_iterations`/`converge_tol` limitation.

**Candidate causes for the 8.5-9cm shortfall, checked against the actual code before recording (2026-07-18)**
-- two of three initially-proposed hypotheses do not apply to this implementation and are recorded here as
ruled out, not as open leads, so the next session doesn't re-investigate them from scratch:
- **Data-tail dissipation / deadband (real per-step joint deltas do shrink near the true end of recorded
  segments, confirmed earlier this session)**: plausible in kind, but the MAGNITUDE doesn't fit -- a deadband/
  damping effect predicts a residual of a few mm (the genuinely negligible final motion), not 8.5-9cm, which
  is roughly 60-90% of the total ~10-14cm descent. This reads as "most of the vertical drop is never
  attempted," not "the last few mm are slow." Closest lead, but likely incomplete on its own -- worth
  checking whether `max_iterations=12` is simply insufficient for this model's actual per-iteration progress
  rate (not yet measured directly: how many iterations would it take to close a fixed 9cm gap at the
  per-iteration rate observed in the Z-traces already collected?), before inventing a new mechanism.
- **Action-space normalization deadlock**: RULED OUT -- there is no normalization step anywhere in this
  pipeline (`x0`/`x1`/`target_qpos` are raw joint-space radians throughout, no min-max scaling, no z-score).
  Nothing to underflow. Do not re-propose without first checking `data.py`/`model.py` for a normalization
  step that doesn't exist.
- **Camera parallax / near-contact geometric distortion**: RULED OUT -- this pipeline uses NO camera or
  perception pipeline. `target_qpos` comes from exact IK against the object's true simulated position
  (`env.get_object_positions()`, a privileged sim read), not a vision estimate. No parallax is possible in
  the current architecture; this would only become relevant if/when Line B's real perception pipeline is
  ever integrated with this line (not currently planned).

## 🔬 Sixth fix (sub-segment training augmentation) — CONFIRMED root cause, mechanism fixed, win rate unchanged (2026-07-19)

Direct audit found the real cause of the 8.5-9cm shortfall (sharper than any of the three hypotheses above):
every training example had `x0` built from a FULL trajectory's own start, so the "remaining-distance"
conditioning value was NEVER small during training (measured: joint-space L2 norm 0.277-1.29 across all 155
real examples) -- but RHC's later iterations present exactly this small-remaining-distance regime as the arm
approaches the target, a value the model never saw and had to extrapolate to. Fixed via `DescendDataset.
load(augment_subsegments=True)`: additionally resamples 6 intermediate sub-segments per trajectory (starting
30%-through to near the end, always ending at that trajectory's own true final waypoint), giving the model
dense training coverage of small remaining-distance values (930 total segments from the same 155 recordings,
median remaining-distance dropped from ~0.47 to ~0.008 in joint-space L2 norm -- confirmed directly before
spending training time on it).

**Can support**: the targeted mechanism fixed exactly as diagnosed -- mean Z-residual on successful trials
dropped from a consistent ~8.5-8.9cm to a consistent ~3.2-3.4cm (roughly 60% reduction), confirming the
training-distribution-gap hypothesis was correct, not just plausible.

**Cannot support**: that this fixes overall grasp success. **Win rate on BOTH probe sets is exactly
unchanged** (tuning 5/8→5/8, held-out 4/8→4/8) -- individual trials flipped in both directions (1004
success→failure, 1002 failure→success) but the totals didn't move. The residual distribution is now visibly
BIMODAL: some failing trials show the new small residual (~2.8cm, e.g. 1102) while others remain stuck at the
old ~10cm plateau (1000, 1007, 1100, 1104, 1106) -- meaning this fix helped a real subset of trials close the
Z-gap almost fully, but whatever is actually failing in the others (most likely XY alignment, not yet audited
with the same rigor as Z) was untouched by it. Trial 1007 specifically is unaffected or mildly regressed
(8.63cm→10.04cm) -- still the standing outlier across every fix tried.

**Honest reframing**: the 8.5-9cm Z-shortfall was real and is now substantially reduced for the cases it
applied to, but it was NOT the dominant bottleneck for overall grasp success -- fixing it was necessary
groundwork, not the breakthrough it looked like it might be. The actual binding constraint on win rate
remains unidentified. Do not report this fix as having improved win rate; it improved a specific, correctly-
diagnosed mechanism that turned out to be necessary but not sufficient.

**Next steps, in order**: (1) audit XY-plane residual with the same rigor already applied to Z (the
`final_eef_residual` diagnostic already captures X/Y, just hasn't been analyzed as carefully) -- this is the
most likely actual bottleneck now that Z is substantially fixed for the trials it helps; (2) investigate why
trial 1007 is untouched by every fix tried so far, now that it's isolated as a genuine standing outlier
rather than one of several failure modes; (3) the n=20 paired pilot against baseline, still not yet run,
should wait until win rate shows real movement -- running it now against an unchanged number would not be a
good use of the pilot.

## 🔬 XY-plane residual audit (2026-07-19): high variance, not systematic bias -- the actual bottleneck is unreliable horizontal placement on harder trials, not a fixable calibration offset

Pooled `final_eef_residual`'s X/Y components across all 16 trials from both probe sets (tuning + held-out),
split by outcome:

| | X mean / std (cm) | Y mean / std (cm) |
|---|---|---|
| Success (n=10) | 1.13 / 0.22 | 0.39 / 0.14 |
| Failure (n=6)  | 1.95 / 1.09 | 1.74 / 1.96 |

**Can support**: successful trials land at a small, remarkably CONSISTENT offset (tight std both axes) --
the arm reliably lands ~1.1cm off in X, ~0.4cm in Y, apparently within tolerance. Failed trials show 5x the
X variance and 14x the Y variance, with individual XY-norm errors spanning 0.56-5.22cm -- genuinely
scattered, not a fixed directional offset. This rules out a hand-eye-calibration-style systematic bias (which
would show a consistent large offset, not high variance) -- the actual problem is the model's horizontal
placement being unreliable specifically on harder cases, not miscalibrated on all of them.

**Cannot support**: that XY residual alone explains every failure -- trial 1102 fails despite SMALL residuals
on both Z (2.75cm) and XY (0.56cm), meaning something outside what `final_eef_residual` captures (most likely
contact dynamics during the final close, which `pre_close_drift_cm` already tracks separately but hasn't been
cross-referenced against this analysis) is responsible for at least that one case. Not every failure reduces
to a single diagnosed mechanism.

**Implication for next steps**: since this is high-variance model unreliability rather than fixable bias, the
likely fix direction is different from the Z-shortfall fix (which was a training-distribution gap with a
clean data-augmentation solution) -- reducing variance on hard cases probably needs either more training data
density specifically for difficult trials, or a genuine architecture/capacity increase, not another targeted
data augmentation. Worth treating as a separate, harder problem rather than assuming the same fix pattern
will work twice.

Files:
`cr_cfm/{data,model,losses,train,inference,eval_pilot,collect_seed_trajs,audit_1007,stability_check}.py`,
checkpoints at `/tmp/.../scratchpad/cr_cfm_cracker*.pt` (scratchpad — not yet copied to a durable path;
`cr_cfm_cracker_n155_v4_remainingonly.pt` is the current best/default).

## 📌 2026-07-19: training-set approach-angle audit revises the XY hypothesis from "multimodal averaging" to "long-tail data sparsity" — verified on the training set, NOT yet confirmed against probe-trial failures

**Can support**: histogrammed the horizontal (XY) approach direction (start→end displacement angle of the
descend phase) across all 155 training trajectories. Not a clean bimodal split (the "left-approach vs.
right-approach" hypothesis proposed before checking) -- instead, a **dominant cluster** of 127/155 (~82%)
concentrated in a ~90° arc from -30° to 60°, plus a **sparse, scattered minority** of 28/155 (~18%) spread
thinly across 60°-180° and a small sub-cluster near -165°. This is a real, direct measurement, not inferred.

**Revised mechanism**: the earlier "Category B: multimodal vector-field blurring" framing implied two
well-populated competing modes cancelling each other out. The actual data shows one well-populated mode and
a thin, scattered tail -- better described as **data support deficiency**: the model likely has a
well-constrained, confident vector field within the dominant arc (consistent with successful trials'
tight residual std of 0.14-0.22cm, all apparently landing within this arc) and an under-constrained,
higher-variance field wherever training density is thin. This is a data-density problem, not a
representational-capacity problem -- the fix direction differs accordingly (targeted data collection in the
sparse region, not mode-clustering or a bigger model).

**Cannot support yet**: that this actually explains the probe-set failures. The probe trials' (1000-1007,
1100-1107) own approach angles were never recorded (`PiperTrajectoryRecorder` wasn't attached during those
eval runs) -- the correlation between "failed trial's approach angle" and "training data's sparse 18%
region" is the load-bearing claim of this whole diagnosis and has NOT been directly checked. Do not treat the
dominant/sparse split alone as confirmation that this is what's failing the probe trials -- it is a strong,
well-evidenced lead, not yet a closed case.

**Next-session backlog, in this order**:
1. Instrument `move_to_cr_cfm_descend` (or wrap the probe eval loop) to record each trial's own approach
   angle, using the same start→end XY displacement definition as the training-set audit, for direct
   comparability.
2. Re-run both probe sets (1000-1007, 1100-1107) with this instrumentation and check directly: do failed
   trials' approach angles fall disproportionately in the sparse 60°-180°/-165° region? This is the actual
   test of the hypothesis above, not optional follow-up.
3. Only if step 2 confirms the correlation: decide between targeted data collection (recording more
   trajectories specifically at sparse approach angles) vs. other fixes -- do not pre-commit to a fix before
   the correlation is verified, per this session's own established discipline (every fix tonight that worked
   was verified against direct evidence first, and the one that didn't -- TCR zeroing -- was the one instance
   of acting on a plausible-sounding mechanism before checking it directly).
4. A third, still-untouched probe range should be used for final validation once a fix is attempted --
   1000-1007 and 1100-1107 have now been used across six sequential tuning decisions each, and the ongoing
   overfitting risk flagged earlier this session still applies.

## 🧪 2026-07-19: approach-angle correlation confirmed as real but partial -- necessary, not sufficient, and a direct counter-example rules out a purely angle-deterministic story

Instrumented `move_to_cr_cfm_descend` to record each trial's own approach angle (same start->end XY
displacement definition as the training-set audit) and re-ran both probe sets (16 trials total,
1000-1007 + 1100-1107) with `cr_cfm_cracker_n155_v5_subseg.pt`.

**Can support**: approach angle is a real, necessary factor. **9/9 successes fall inside the dense -30 deg
to 60 deg training arc; zero successes in the sparse region.** Successes cluster narrowly within that arc
(5.4 deg to 50.5 deg, an interior core, not spread across the full -30/60 window). Dense-arc failures that
do occur (1000: -1.6 deg, 1100: -19.1 deg, 1106: 59.0 deg) sit near the EDGES of the window, consistent with
a soft density gradient rather than a hard cutoff.

**Cannot support**: that approach angle is sufficient to predict success. 4 of 7 failures also fall inside
the dense arc (1000, 1006, 1100, 1106) -- being in the well-populated region does not guarantee success.

## ⚠️ 2026-07-19 CORRECTION: the "1005-vs-1006 paradox" is NOT a real, stable phenomenon -- it was a run-to-run nondeterminism artifact, caught before it was written into a paper

The original "counter-example" above (trial 1005 success, trial 1006 failure, near-identical approach
angles) was built entirely on ONE single-run report of trial 1006 as a failure. Before writing a Discussion
section and a matplotlib comparison figure around it (per plan), re-ran trial 1006 four independent times
(with and without a trajectory recorder attached, to rule out instrumentation artifacts): **all 4 repeats
showed success (dist=0.013-0.014), consistent with the ORIGINAL sub-segment-checkpoint evaluation's own
report of 1006 as a success** -- only the later "approach-angle" evaluation run reported it as a failure,
and that appears to have been the anomalous one, not the other three.

**This directly matches the SAME run-to-run chaos already discovered and explained for trial 1007 hours
earlier this session** -- the mechanism was never fixed, only reduced in frequency by the RHC and
conditioning fixes. Building a paper narrative and a dedicated figure around a "stable, reproducible
counter-example" that turned out to be a single nondeterministic sample would have been presenting noise as
a finding -- caught before any time was spent on the figure or the writing, not after.

**Broader process gap this revealed**: the stability check (repeated runs confirming no outcome
disagreement) was performed ONCE, early in this session, for an intermediate checkpoint -- then three more
checkpoint changes were made (remaining-only conditioning, sub-segment augmentation) without ever re-running
it. Re-ran it properly against the CURRENT best checkpoint (`cr_cfm_cracker_n155_v5_subseg.pt`), 3 repeats
per trial, across both full probe sets:

- **Tuning set (1000-1007): 2/8 trials (25%) show genuine disagreement** -- trials 1002 and 1004. Trial 1006
  is CONFIRMED stable (3/3 success), closing the paradox investigation as a non-issue for that specific pair.
- **Held-out set (1100-1107): 0/8 disagreement** -- fully reproducible.
- **Combined: 2/16 (12.5%) genuine instability**, concentrated entirely in the tuning set.

**Interesting wrinkle, not necessarily bad news**: under majority vote across the 3 repeats (2/3 agreeing),
trial 1004 -- originally reported as a single-run FAILURE in the headline 5/8 tuning-set number -- is
actually 2-success/1-failure, i.e. majority SUCCESS. The true, repeat-robust tuning-set win rate may be
**6/8, not 5/8** -- the instability doesn't necessarily mean the reported numbers overstate performance; it
means single-run point estimates carry real uncertainty in either direction that this session's evaluation
protocol (one run per trial_id) was never designed to resolve.

**Implication for any future evaluation, including the proposed N=30 scale-up**: single-run-per-trial_id is
not a fully reliable protocol at ~12.5% instability -- any serious follow-up evaluation (paper-bound or not)
needs multiple repeats per trial_id with majority-vote (or a formal mixed-outcome treatment) as the unit of
comparison, not a single sample. Do not report a win-rate number from a single-run N=30 sweep as if it were
noise-free; either budget for repeats, or explicitly caveat the single-run protocol's known ~12.5%
per-trial uncertainty in any write-up.

## 📈 2026-07-19: majority-vote re-evaluation on a THIRD, fresh trial range confirms the true win rate is materially higher than the single-run headline number -- 65-75%, not ~50%

Ran the same 3-repeat stability-check protocol on trial_id 1200-1207 -- a range never touched by any
tuning decision, distinct from both the tuning set (1000-1007) and the held-out set (1100-1107) used so
far. Majority-vote results across all three ranges now available:

| Range | Majority-vote win rate | Disagreement rate |
|---|---|---|
| Tuning (1000-1007) | 6/8 = 75% | 2/8 (25%) |
| Held-out (1100-1107) | 4/8 = 50% | 0/8 (0%) |
| Fresh (1200-1207) | 6/8 = 75% | 2/8 (25%) |
| **Combined (n=24)** | **16/24 ≈ 66.7%** | 4/24 (16.7%) |

**Can support**: the true, repeat-robust win rate is materially higher than the ~50% single-run estimate
that has been the headline number reported all session -- two of three independently-evaluated ranges tie
baseline (75%) exactly, only one shows a real deficit. The ~25% instability rate replicates independently
on the fresh range (matching the tuning set's rate, not a fluke specific to one range) -- this appears to be
a genuine, recurring property of the current checkpoint, not sampling noise from a single evaluation.

**Cannot support**: a clean "CR-CFM matches baseline" claim -- there is real, unexplained range-to-range
variance (75%/50%/75%) that itself needs an explanation before any overall number is reported with
confidence. The leading candidate, consistent with the earlier approach-angle-density finding, is that
different trial_id ranges happen to sample different proportions of dense-vs-sparse approach angles by
chance -- this is a testable hypothesis (check each range's own approach-angle distribution against the
training-set density map), not yet checked.

**Revised next steps**: (1) check whether the held-out range's lower win rate correlates with a higher
proportion of sparse-region approach angles compared to the other two ranges -- this would explain the
variance with the SAME mechanism already found, rather than requiring a new one; (2) if confirmed, report
win rate as a function of approach-angle density rather than a single pooled number, which would be both
more honest and a more precise, publishable claim than either the pessimistic ~50% or the optimistic ~75%
read; (3) the N=30 scale-up and ablation arms should now be run with the SAME majority-vote (3 repeats)
protocol from the start, not single-run.

## ⚠️ 2026-07-19: the approach-angle-density explanation for range-to-range variance is REJECTED, not confirmed -- checked directly, and it doesn't hold

Extracted approach angles for the fresh range (1200-1207) and compared sparse-region fractions across all
three evaluated ranges:

| Range | Sparse-angle fraction | Majority-vote win rate |
|---|---|---|
| Tuning (1000-1007) | 1/8 (12%) | 75% |
| Held-out (1100-1107) | 2/8 (25%) | 50% |
| Fresh (1200-1207) | 2/8 (25%) | 75% |

**Cannot support** the hypothesis from the entry above: held-out and fresh have IDENTICAL sparse-angle
fractions (25% each) but completely different win rates (50% vs. 75%). If angle density explained the
range-to-range gap, these two should perform similarly -- they don't. This specific explanation is now
cleanly ruled out by direct check, not just left unconfirmed. **The individual-trial-level finding still
holds** (successes cluster in the dense arc, within any given range) -- what's ruled out is using that same
mechanism to explain WHY one range's aggregate rate differs from another's. The actual cause of held-out's
underperformance is genuinely unknown.

**Additional finding while checking this**: this single extra run flipped trial 1201 from its earlier
3-repeat-confirmed "stable success" classification to a failure -- a 4th independent sample contradicting
what looked like a clean case. The true instability rate is likely higher than the measured ~12.5-25%; 3
repeats is not always sufficient to catch it.

**Honest status at the close of this investigation**: win rate under proper majority-vote evaluation is
materially better than the ~50% single-run headline (combined 16/24 ≈ 67% across three ranges), real
instability (~25%, possibly higher) is a confirmed, recurring property of the current checkpoint, and the
range-to-range variance driving the gap between the best and worst observed ranges has a ruled-out
explanation but no confirmed one yet. This is an honest, open question for the next session, not a solved
one forced to a premature close -- do not report a single pooled win-rate number in any paper material
without this caveat attached.

## ✅ 2026-07-19 FINAL CORRECTION: the range-to-range variance itself is not statistically distinguishable from n=8 sampling noise -- there was likely nothing to explain

Before investigating further, checked whether the 75%/50%/75% pattern across ranges is even a real
difference. **Fisher's exact test, held-out (4/8) vs. tuning+fresh pooled (12/16): p=0.36 -- not
significant.** Held-out's rate sits only 1.0 standard deviation below the combined 66.7% mean, which is
ordinary sampling variation at n=8 (binomial std ≈0.167 at this n and rate). The angle-density investigation
(entry above) and everything that would have followed it were very likely chasing an explanation for a
difference that was never established as real in the first place.

**Consolidated, final honest state of this whole investigation**: the single trustworthy number is the
**combined win rate across all three ranges, 16/24 ≈ 67%**, evaluated with proper repeat + majority-vote
methodology -- not 50%, not 75%, and no story about which trial ranges are "better." Real, confirmed
findings that survive: (1) win rate improved from a broken 0% to a validated ~67% via six honestly-diagnosed
fixes; (2) ~25%-or-higher single-run instability is a genuine, replicated property of the current checkpoint,
mandating repeat-based evaluation for anything built on top of this; (3) two plausible-sounding
explanations (the 1005-vs-1006 "paradox," the angle-density range-variance story) were each caught and ruled
out by direct verification before reaching a paper draft. **Next session's correct first move is a
properly-powered N (with repeats) to get a real confidence interval around 67%** -- not further
investigation of the range-to-range gap, which this check closes as very likely noise.

## ⚠️ SUPERSEDED NUMBERS BELOW (2026-07-17): `PiperMultiObjectScene`'s placement sampler was found to be unseeded — every `np.random.seed(trial_id)` call in every entry below had ZERO effect on object placement. Fixed in `piper_multi_object_scene.py`. Re-verification status: execution-control mechanisms (binary hard-stop, sustained-contact stop, force-admittance, pre-narrow) — 3 of 4 original "worse" findings do NOT replicate, see `RULED_OUT_METHODS.md` rows 7-10. Orientation-aware grasping ablation — re-verified, original "no significant difference" framing does NOT hold cleanly (object-dependent, Pear notably favors fixed), see that entry below. Best-vs-consensus candidate selection — re-verified, null CONFIRMED on all 3 objects (not an RNG-bug artifact), absolute success rates much higher than originally measured, see that entry below. The gripper-controller double-scaling fix and its success-rate point estimates are unaffected (single-condition estimates, not paired comparisons). Still NOT re-verified: centroid-offset re-checks. Treat those p-values with appropriate caution.

## 🔧 Trajectory recorder/replayer built and verified in sim, NOT hardware-tested (2026-07-17): `piper_trajectory.py`, the "natural next piece" `piper_real_backend.py` flagged as missing

`PiperTrajectoryRecorder`/`PiperTrajectoryReplayer`, analogous to `robots/trajectory.py`'s SO-ARM101
pair. Required a small, consistent change to `piper_pick_and_place.py`: threaded an optional
`step_hook` parameter through all four `move_to*` functions and `run_pick_and_place` itself, called
after every `env.step()` from within the recording boundary — deliberately the same boundary as the
causal-validity commit marker (the initial spawn-settle loop is not recorded, since it has no
real-hardware equivalent).

**Can support**: verified end-to-end in simulation. Recorded a real trajectory (trial 903, cracker,
1210 points, 24.92s) via `run_pick_and_place(..., step_hook=recorder.snap)`; save/load round-trips
exactly (joint positions and metadata byte-identical after reload). Replayer tested against a mock
backend matching `PiperRealBackend`'s actual interface: correctly calls `reset()` then `move_joints()`
in order, and — by design — raises `NotImplementedError` on the first gripper command rather than
guessing a unit conversion. `replay(..., gripper_agnostic=True)` correctly drives all 1210 joint
targets with no gripper calls, useful for validating arm motion/timing independent of that blocker.

**Cannot support**: any real-hardware replay yet. The recorded gripper value is the sim's raw
action-space ctrl signal (`env.sim.data.ctrl[-1]`), not `PiperRealBackend.set_gripper`'s metres
convention — no utility in this codebase reads actual gripper finger width from sim state, and
guessing a conversion would repeat the exact unguarded-unit-assumption pattern that caused the
double-scaling bug (see the root-cause entry elsewhere in this file). Resolving this (or building a
real metres-reading utility) is required before any real gripper replay; deliberately left as an
explicit, loud `NotImplementedError` rather than a silent guess.

**Can support**: a `PiperRealBackend` class (`piper_real_backend.py`) written against the documented `piper_sdk` (agilexrobotics/piper_sdk) API, following the same safety-conscious pattern as `robots/soarm_real_backend.py` -- relative-delta motion clamping (`max_relative_target`), an availability guard so the module imports cleanly without `piper_sdk` installed (confirmed: not installed in the `tango` env as of this entry), and `execute_grasp()` deliberately left unimplemented for the same reason SO-ARM101's backend leaves it unimplemented: real hardware should replay a joint-position sequence already solved and validated in simulation, not re-solve IK live against physical state.

**Cannot support**: that any of this actually works against real hardware. The physical Piper arm was not connected to the development machine while writing this (confirmed only `/dev/ttyACM0`/`/dev/ttyACM1` present, no CAN interface, no `piper_sdk` installed) -- every method that would touch hardware (`get_joint_positions`, `get_gripper_opening`, `move_joints`, `set_gripper`) raises `NotImplementedError` with an explicit "VERIFY" marker instead of guessing unit/scale conventions (radians vs. degrees vs. raw encoder counts for `JointCtrl`; the full multi-parameter signature of `GripperCtrl`; the exact return schema of `GetArmJointMsgs`/`GetArmGripperMsgs`) that a web search could not confirm from documentation alone. Getting these wrong silently, rather than failing loudly, is exactly the kind of mistake this project's gripper-scaling bug (this file's earlier entries) should make everyone here paranoid about repeating -- especially now with a 6-DoF arm capable of real physical damage, not a MuJoCo sim.

**Before the first real connection**: (1) `pip install piper_sdk`, (2) read `piper_sdk/demo/V2/`'s actual example scripts in the installed package and replace every `NotImplementedError`/"VERIFY" marker in `piper_real_backend.py` with a confirmed call, (3) start `max_relative_target` very small, (4) do not copy the README example's `judge_flag=False`/`start_sdk_joint_limit=False`/`start_sdk_gripper_limit=False` without deliberately deciding those are the right settings (they disable some of the SDK's own built-in safety checks).

**Not yet built**: a Piper-specific trajectory recorder/replayer (analogous to `robots/trajectory.py`'s `TrajectoryReplayer`) -- needed before `execute_grasp`-equivalent real-hardware execution is possible at all, matching how SO-ARM101's real backend requires replaying a pre-recorded `Trajectory`, not live IK.

**Files**: `piper_real_backend.py`.

## ⚠️ Pre-narrow descend (v4, 2026-07-16): also negative -- 4th independent mechanism tried in this axis, all negative or null. Closing the "candidate selection / execution control" axis for real this time.

**Can support**: a working implementation of a structurally different idea from v1-v3 (which all tried to manage contact AFTER it happens) -- narrow the gripper by ~2cm (calibrated: 1 `GRIPPER_CLOSE` step from full open takes the span from ~12.0cm to ~10.1cm, `scratchpad/calibrate_gripper_width*.py`) BEFORE descending past the object's own height range, reducing the collision cross-section during the risky vertical pass in the first place, rather than reacting once contact occurs. `run_pick_and_place(..., pre_narrow_descend=True)`; `descend_refresh` also holds the narrowed width (fixed a first-draft bug where it would have snapped back to full-open, defeating the point).

**Result**: paired n=20 (same trial_ids as v1-v3, for direct comparability): pre_narrow 20% vs. baseline 50% (8 discordant favoring baseline vs. 2 favoring pre_narrow, McNemar p=0.11) -- **worse, not better**, the same direction as v1 and v3.

**Cannot support**: that this specific narrowing amount helps. Plausible explanation, not yet verified: the achievable narrowing via the existing action interface is coarse (only ~12.0cm or ~10.1cm are cleanly reachable in one step; finer widths would need bypassing the normal action path, not attempted) -- a 15% reduction may simply not be enough to change the underlying dynamics, or a narrower-but-still-much-wider-than-the-object gripper doesn't meaningfully reduce edge-catch risk the way hoped, since there's still slack on both sides.

**This is the 4th independent, differently-motivated attempt within the "candidate selection / execution control" axis, and the 3rd specifically targeting execution dynamics during descend (v1 binary hard-stop, v3 force-aware admittance, v4 pre-narrowing) -- all negative or null.** Combined with Direction 2's closure in `IDEA_REPORT.md` (candidate reranking, any form, has no valid pre-execution signal) and the original candidate-selection heuristic's cross-embodiment failure, this axis has now been searched thoroughly enough that continuing to generate and test new variants within it has a low expected return. Closing it for now rather than continuing to iterate.

**Files**: `piper_pick_and_place.py` (`pre_narrow_descend` parameter), `test_pre_narrow.py` (scratchpad), data at `pre_narrow_cracker_1000-1020.json`.

## ⚠️ Force-aware admittance descend (v3, 2026-07-16): negative, with a mechanistic explanation -- the "candidate selection / execution control" research axis is being closed for now

**Can support**: a working implementation reading MuJoCo's own exact contact forces (`mj_contactForce`, `move_to_force_compliant_descend`) and using them to continuously damp the arm's advancement toward the descend target -- a genuine admittance-style response (not the binary hard-stop of v1/v2), directly informed by CoorGrasp's mechanism. Diagnostic data confirmed the force signal itself is clean and informative: across 20 trials, contact force cleanly separated into a "normal" band (~1.7-2.4, mostly successful trials) and a "problem" band (~17-41, 0/7 successful) -- detection is not the issue.

**v1 (threshold=0.3, gain=0.5)**: paired n=20, force_compliant 55% vs. baseline 75% (6 discordant favoring baseline vs. 2 favoring compliant). Root cause: threshold far too low relative to the observed "normal" force band, so damping was active on nearly every trial regardless of whether anything was actually going wrong, capping normal descends at ~83% completion for no benefit.

**v2 (threshold=8.0, gain=2.0, recalibrated against the observed force distribution)**: paired n=20 (same trial_ids), force_compliant 50% vs. baseline 75% (8 discordant favoring baseline vs. 3 favoring compliant) -- **worse, not better**, and introduced a NEW catastrophic-drift case (36.4cm) on a trial where baseline succeeded cleanly (drift 0.07cm). Recalibration fixed the "always damping" problem but did not produce an improvement.

**Cannot support**: that force-magnitude-aware admittance damping, at least in this direct form, helps. **Mechanistic explanation for why, not just an empirical shrug**: damping the arm's advance toward the target means the open (not-yet-closed) gripper spends MORE time lingering near/in contact with the object under load, not less -- the opposite of what's needed. CoorGrasp's real mechanism works because contact happens with FINGERS ALREADY FORMING A GRIP (post wrench-balance-criterion phase transition), where coordinated arm motion can genuinely compensate a held object's position error. Here, contact during `descend` happens with the gripper still open and not yet committed to a grip -- slowing down doesn't stabilize anything, it just extends the exposure window during which an open, still-approaching finger can keep nudging the object further off-target.

**Verdict: the "candidate selection / execution control" research axis is being closed for now, after a reasonably thorough search.** Tried across this axis: training-free geometric heuristics (fail to transfer cross-embodiment), learned rerankers (pointwise/pairwise, single/cross-embodiment, conditioned/unconditioned -- no valid pre-execution signal found), binary contact-triggered hard-stop compliant descend (v1/v2, null), and now force-magnitude-aware admittance damping (v3, negative with a real mechanistic explanation). Per this project's own stated go/no-go framing going into this pilot: this is the signal to step outside the "which candidate to pick, or how gently to arrive at it" framework entirely, not to keep re-tuning parameters within it. A more structurally different approach (e.g. a genuinely two-stage strategy -- a deliberate, low-force nudge to align the object within the gripper's span BEFORE any commitment to a final descend/close, rather than trying to combine "get close" and "get centered" into a single interpolated pass) was not tried and remains a real, structurally distinct candidate if this line is revisited -- distinguished from everything tried here by not assuming the SAME single approach trajectory has to both close the distance and fix any misalignment.

**Files**: `piper_pick_and_place.py` (`move_to_force_compliant_descend`, `_object_contact_force_magnitude`), `test_force_compliant.py` (scratchpad), data at `force_compliant_cracker_1000-1020.json` (both v1 and v2 threshold settings, same trial_ids for direct comparison).

## ⚠️ Compliant/contact-aware descend pilot (2026-07-16): the geometry-only version of CoorGrasp's mechanism does not show a real improvement -- honest null result after two independent batches

**Can support**: a working, tested implementation of the literature-informed next step proposed in the entry directly below (`move_to_compliant_descend` in `piper_pick_and_place.py`, opt-in via `run_pick_and_place(..., compliant_descend=True)`) -- stop advancing through the descend waypoints once contact with the target object is detected, instead of blindly driving to a precomputed target regardless of what's physically happening. Uses MuJoCo's own contact list (`env.sim.data.contact`), no new hardware.

**v1 (single-instant trigger) — tried and reverted**: stopping the instant ANY contact was detected made things measurably worse (paired n=20: baseline 70% vs. compliant 40%, 10 discordant pairs favoring baseline vs. 4 favoring compliant). Diagnosed why by checking WHEN each trial triggered: every trial that stopped early (step 28-58 of a 150-step budget) failed 10/10; every trial that "stopped" near step 105-106 (i.e. essentially at the natural end of the interpolation anyway) succeeded about as often as baseline. A single brief/incidental touch early in the trajectory was being treated the same as genuine sustained grazing contact.

**v2 (require `sustain_steps=5` consecutive contact-positive steps before triggering)**: fixed the premature-stop failure mode. First paired batch (n=20, trial_id 1000-1019) looked promising: compliant 75% vs. baseline 55%, 8 discordant pairs favoring compliant vs. 4 favoring baseline (McNemar p=0.39, not significant but a 2:1 favorable ratio). **Did not replicate on a second batch** (n=20, trial_id 1020-1039): baseline 65% vs. compliant 50%, this time 7 discordant pairs favoring baseline vs. 4 favoring compliant -- the opposite direction. **Combined across both batches (n=40): baseline_only=11, compliant_only=12, McNemar p=1.0000 -- a coin flip.** Pooled success rates nearly identical (60% baseline vs. 62.5% compliant). This is the same "promising small batch, null on replication" pattern that has recurred repeatedly across this session's other pilots (centroid-offset re-measurement, conditioning mechanism) -- treated with the same discipline: report the null, don't keep the flattering first number.

**Cannot support**: that geometry-only contact detection (no force/torque magnitude, just binary "touching or not") is sufficient to implement CoorGrasp's mechanism usefully. The source paper's own version requires real tactile force sensing (Tac3D) specifically to know not just THAT contact happened but how much force and in what direction -- information this binary-contact implementation structurally does not have. A hard stop-on-sustained-contact is a much cruder signal than a force-magnitude-aware admittance response, and this pilot's null result is consistent with that gap mattering, not with the underlying mechanism being wrong.

**Honest read**: this closes the immediately-triable, hardware-free version of this literature lead. The remaining, more faithful version of CoorGrasp's mechanism (using actual contact force magnitude/direction, which MuJoCo also computes internally and could in principle be read via `mj_contactForce` without new physical hardware, unlike a real deployment) has not been tried yet and is a more promising next step than this binary-contact version, if this line is picked back up.

**Files**: `piper_pick_and_place.py` (`move_to_compliant_descend`, `_has_object_contact`, `_object_contact_geoms`), `test_compliant_descend.py` (scratchpad), data at `compliant_descend_cracker_{1000-1020,1020-1040}.json`.

## 🔍 Literature check on the crystallized root cause (2026-07-16): two independent investigations (this file's Cracker debugging + IDEA_REPORT.md's cross-embodiment reranker closure) both point at execution-time approach/descend contact dynamics, not candidate selection -- literature agrees this is a real, distinct problem, and a hardware-free next step exists

**Can support**: literature grounding for pivoting effort from "which candidate to pick" to "how the arm executes the approach to whichever candidate is picked" -- directly actionable given this project's existing (sim-only, no new hardware needed) infrastructure.

Searched contact-aware/compliant grasp approach literature after `IDEA_REPORT.md`'s Direction 2 closed on the finding that no pre-execution feature (kinematic or otherwise) predicts Cracker's success -- the real determinant is what happens physically during `descend`. Found real, active-2025-2026 corroboration:

- **"Language-Guided Grasping under Partial Observation"** (arXiv:2603.07866): baseline methods found to fail "predominantly due to approach collisions" -- grasp candidates "locally plausible but not executable once approach clearance is considered." Same failure category as this project's own finding, independently observed elsewhere.
- **AutoDex** (arXiv:2606.23689): explicitly frames the generator as "responsible for proposing kinematically and geometrically feasible candidates but NOT certifying stability under real contact dynamics" -- candidates are filtered/selected only after testing through real execution, not scored a priori. Matches this project's own conclusion that pre-execution scoring structurally cannot see what matters.
- **CoorGrasp** (arXiv:2607.03557, "Coordinated Contact Control for Adaptive Dexterous Grasping Under Uncertainty"): the closest direct mechanism match -- separates approach from grasping via a wrench-balance criterion, and critically, uses **coordinated arm motion (not just finger closing) to compensate for object position errors once gentle contact is made**, instead of a fixed target the arm blindly drives to. Reported +10pp under shape uncertainty, +0.9pp under 2cm position uncertainty vs. a no-arm-coordination baseline (UR5 + LEAP hand, 15,000 sim grasps + 8 real objects). **Requires tactile force sensors (Tac3D) -- hardware this project doesn't currently own**, and no public code found.
- Closed-loop visual servoing during final approach (multiple 2025 papers, e.g. arXiv:2001.05650 and MDPI act14010025): continuously re-servos the grasp target as the gripper approaches, instead of committing to one fixed pose computed before descent starts -- conceptually a continuous version of this project's own "pre-close refresh" (a single-shot version of the same idea, found earlier to only partially help).

**Key implication, not present in any of the papers found**: CoorGrasp's mechanism needs real tactile hardware, but **this project's Piper environment is a MuJoCo simulation, which already computes exact contact forces/geom distances internally for free** (`env.sim.data.contact`, `mj_contactForce`, etc.) -- no sensorless-estimation research or new hardware purchase is needed to prototype an admittance-style "detect contact force during descend, back off or slow down" controller here, unlike a real robot deployment. This is a lower-cost, immediately-triable next step this project's own infrastructure uniquely enables, that the literature's real-hardware-focused solutions don't have to (and mostly don't) offer for free.

**Cannot support**: a validated fix -- this is a literature scan, not a pilot. The natural next step (not yet done) would be a sim-only prototype: during `descend`, monitor `env.sim.data.ncon`/contact force magnitude on the target object in real time, and pause/back-off/reduce the commanded delta whenever contact is detected before the intended grasp height, instead of blindly continuing to the precomputed target -- directly informed by CoorGrasp's mechanism but implemented with privileged sim contact data instead of real tactile sensors.

## ⚠️ Resolved why the two centroid-offset measurements disagreed (MuJoCo's own `body_ipos` is authoritative) -- but using it doesn't improve Cracker/Mustard either (2026-07-15)

**Can support**: a clean explanation for the earlier "bbox-center vs original z-slice offset" discrepancy -- a collision hull's raw `(min+max)/2` bbox center is NOT its true volumetric/mass centroid unless the hull is a perfectly symmetric box, and MuJoCo already computes the real answer for every body at compile time: `model.body_ipos[body_id]`, the center of mass in body-local frame, used directly by the physics engine. Checked cracker: `body_ipos = [0.0275, -0.0108]`, almost exactly matching the ORIGINAL `[0.0306, -0.0102]` (within ~3mm) -- confirming the original 2026-07-14 z-slice measurement was basically right all along, and the bbox-center recomputation (tried and reverted twice, previous entry) was measuring the wrong quantity, not a more-correct one.

**Cannot support**: that this resolution actually helps. Re-measured all four tracked objects from `body_ipos` and swapped them into `OBJECT_CENTROID_OFFSET_LOCAL` (pear and mustard changed the most, Y components shifting meaningfully; can/cracker barely moved). Tested on matched n=20 batches: **Cracker 45%->35% (7/20, mild regression)**, **Mustard unchanged at 65% (13/20)** despite mustard's offset changing more than cracker's. No validated improvement anywhere -- reverted back to the original values. Centroid-offset precision, at least the version measurable this way, does not appear to be the dominant remaining lever for either object's failure rate at this point.

**Side note on methodology**: a same-code, same-trial_id (400-404) re-run of the Cracker n=5 sanity check produced 2/5 this time versus 3/5 earlier in the session -- genuine run-to-run non-determinism in the simulation itself (plausible given the contact-rich, sometimes near-chaotic dynamics this whole investigation has been tracing: small floating-point differences in a borderline scrape-or-clear-the-corner case can cascade to a different final outcome). This is a real caveat for every n=5 "quick check" in this file's history, not just this entry -- n=20 pilots are the right scale to trust directionally; single small-n re-checks can legitimately disagree with each other even with zero code change.

## ⚠️ Two follow-up fix attempts for Cracker's remaining drift-linked failures, both tried and reverted (2026-07-15): the mechanism is sustained grazing contact from a still-imperfectly-centered target, not motion speed or the collision-hull-based offset

**Can support**: a clearer picture of WHY Cracker's post-gripper-fix failures still correlate with high `pre_close_drift_cm` (previous entry) -- a full step-by-step trace of a fresh high-drift trial found continuous single-finger contact starting partway through the interpolated `descend` phase and persisting/escalating through `descend_refresh`, ending with the object being pushed/knocked well off-target (in one case landing ~5cm lower than table height) well BEFORE the close command was ever issued. Also: two independently-motivated fix attempts for this, both tested on matched n=20 Cracker batches (trial_id 400-419) and both reverted because they measurably made success WORSE, not better -- ruling out two plausible-looking explanations rather than leaving them untested.

**Cannot support**: a fix for the remaining ~55% Cracker failure rate -- both obvious next moves from the available diagnosis were tried and failed; the actual fix is still open.

**Attempt 1: interpolate `descend_refresh` (like `descend` itself was fixed earlier).** Reasoning: the trace showed a violent excursion right around the `descend_refresh` window, matching the exact mechanism that abrupt, non-interpolated moves caused for `descend` before that was fixed. Tested: success dropped from 45% to 30% (n=20). Likely explanation: the two situations aren't the same. `descend`'s problem was a single large, high-velocity jump colliding hard with an object it was never near before. `descend_refresh`'s problem (per the trace) is **sustained low-level grazing contact** against a target that's still slightly off -- interpolating that motion doesn't reduce contact, it just means the arm spends MORE total time in light contact with the object, giving a slow push more time to accumulate a larger net displacement than a fast pass-through would. Reverted.

**Attempt 2: re-apply the directly-measured (collision-hull bbox center) `OBJECT_CENTROID_OFFSET_LOCAL` for Cracker** (see the older "centroid-offset re-measurement attempted and reverted" entry below -- that A/B ran BEFORE the gripper-scaling fix, so it was confounded by ~0.1mm of real clearance making any target basically fail regardless of which offset was more accurate; re-testing after the gripper fix seemed like the obvious way to get a clean answer). Tested on the same fixed gripper, clean n=20 batch: **2/20 (10%)**, far worse than the original value's 45%. This independently confirms (not just replicates) that the 8-vertex collision-hull bbox center is the wrong quantity here, under two different gripper-clearance regimes now. Reverted back to the original `[0.0306, -0.0102]`.

**Honest read**: the remaining Cracker failures are real, still-open, and now well-characterized (sustained grazing contact against an imperfectly-centered target, not a motion-speed problem and not fixable by the specific alternate offset measurement tried) -- but neither of the two most obvious fixes worked. The next investigation should probably question why the collision-hull-derived offset disagrees with the original z-slice-based one (rather than treating "read it from the live geom" as automatically more correct), or measure the actual finger-to-object clearance on both sides at the moment contact begins (not just at the end state), to see whether the object is off-center along the closing axis specifically or something else (yaw error, an asymmetry in how the two fingers individually track their shared commanded position) is at play.

## 🎉 ROOT CAUSE FOUND AND FIXED, n=20 pilot (2026-07-15): the gripper controller was silently double-scaling the open/close command -- real travel was ~0.1mm, not 7.6-10cm. Cracker 0%→45%, Mustard 0%→65%, Pear 65% (not a regression from its prior 70-90%)

**Can support**: the actual, dominant root cause of this entire project's cross-object grasp generalization failure, found by following the finger-clearance investigation (previous entries below) one level deeper than "where does the arm end up" into "what does the gripper actually do when told to open." Fixed with a one-line controller config change (`tango_robot/piper_robosuite/piper_controller_config.py`). Validated with an n=20/object pilot (trial_id 400-419, two n=10 batches each): **Cracker 9/20 (45%)**, **Mustard 13/20 (65%)**, **Pear 13/20 (65%, not a regression from its prior 70-90% -- consistent within pilot-scale noise)**. All three moved from a hard 0% (Cracker, Mustard) or an already-working baseline (Pear) to real, substantially-above-zero rates with no other change than this one config fix.

**Cannot support**: that this is a finished/tuned system -- Cracker's n=20 result was NOT uniform (1/10 on trial_id 400-409, 8/10 on 410-419), and every single Cracker failure in the full n=20 batch had a large `pre_close_drift_cm` (2.6-14.2cm), while every success had near-zero drift (0.04-0.19cm, with one exception at 3.06cm that still succeeded) -- meaning drift-inducing contact during `descend` (documented in the entries below) still happens for a meaningful fraction of Cracker spawn poses even with the gripper's real clearance restored, and is very likely the next real bottleneck to chase, not assumed away. Mustard's one big-drift trial (417, 11.0cm) also failed, same pattern.



The previous entry's ground-truth finger-clearance measurement found the gripper's real open span was inconsistent and much smaller than expected (3.6-8.3cm instead of the ~10cm the 2026-07-14 width fix was supposed to guarantee). Chasing that discrepancy (not just accepting it and moving to "option (b)") led to the actual bug: `env.robots[0].gripper` is driven through robosuite's composite-controller stack, and `PiperGripper.format_action` (piper_gripper.py) does something unusual compared to other robosuite grippers -- instead of returning a normalized `[-1, 1]` action, it tracks and returns an **absolute real-units joint position** (-0.05 to -0.004 metres, incrementally, already clipped to the gripper's true ctrlrange). robosuite's `SimpleGripController` (the `"GRIP"` controller type) does not know this and applies its own separate real-units mapping on top (`vels = bias + weight * desired_qvel`, derived from the actuator's own min/max) -- silently double-scaling an already-real-units value as though it still needed converting from a normalized range.

Traced empirically: at a verified, fully-converged `current_action = -0.05` ("fully open" per the gripper's own internal state), the actual `data.ctrl` written to the finger actuators was **-0.02815** -- nowhere near either range endpoint. At "fully closed" (`current_action = -0.004`), it computed to -0.0271. **The real difference between commanded fully-open and fully-closed was ~0.0001m** -- a tenth of a millimetre. Every finger-width investigation this whole project (the original 7.6cm-vs-10cm width fix, today's centroid-offset re-measurement, the descend-phase interpolation, the pre-close pose refresh, the approach-precision retry loop, the mink collision-avoidance integration) was carried out on top of a gripper that could barely move its fingers at all, without anyone knowing it -- because `PiperGripper.current_action` (the Python-level tracked state) looked completely correct in isolation; only checking the ACTUAL `data.ctrl` value written to the physics engine exposed the gap.

First fix attempt (didn't work): add `input_min`/`input_max` matching `format_action`'s real range to the gripper sub-config. Traced further into `robosuite/robots/robot.py` (~lines 958-969) and found it rebuilds the gripper's part-controller config from scratch when wiring up the composite controller, copying only `"type"` and `"use_action_scaling"` from the config dict -- `input_min`/`input_max` are silently dropped, so this had no effect (confirmed: ctrl stayed at -0.02815).

**Actual fix**: `"use_action_scaling": False` in the gripper sub-config (`piper_controller_config.py`). `SimpleGripController.set_goal()`/`.run_controller()` both gate their entire rescaling logic behind `if self.use_action_scaling:`; with it off, the action passes straight through to `data.ctrl` unchanged -- exactly right, since `format_action` had already done the real unit conversion itself. Verified: `data.ctrl` now exactly matches `current_action` (-0.05000 at full open), and the true open span (measured finger-collision-geom to finger-collision-geom) is **12.01cm**, comfortably wider than every tested object's narrow-axis width.

**Honest read**: this was hiding in plain sight the entire session (and likely since the gripper was first wired up, well before this session started) -- every other fix today (interpolated descend, pre-close refresh, centroid-offset re-measurement, the abandoned approach-retry loop, mink) was chasing symptoms of a gripper that had almost no real clearance to work with, no matter how precisely the arm was positioned. With real clearance restored, Cracker and Mustard immediately jumped from a hard 0% to a provisional 60% on the very first n=5 check, with no other changes. This doesn't mean the investigation was wasted -- the interpolated-descend and pre-close-refresh fixes are still real, still validated on their own terms, and probably still contribute now that they have real margin to work with instead of none. But this gripper-scaling bug was almost certainly the dominant lever the whole time.

## ⚠️ Ground-truth finger-clearance measurement (2026-07-15): rules out an axis-mismatch bug, confirms the object is under ONGOING contact disturbance through the whole descend, not a one-time knock

**Can support**: direct empirical proof that the gripper's real closing axis (measured from live `finger7`/`finger8` world positions) exactly matches what `compute_grasp_orientation`/`grasp_mat` assumes it to be (cos similarity = 1.0000 in an isolated test) -- this specific "wrong axis" hypothesis is ruled out. Also: evidence (via the same measurement, applied to live failing trials) that the object is under continuous, ongoing contact disturbance from the open gripper through the entire `descend` phase, not a single isolated knock -- meaning a "read the pose once right before closing" strategy is fundamentally insufficient, because the pose is still actively changing at the moment it's read.

**Cannot support**: a fix. This narrows the search space but does not yet resolve Cracker's 0% rate.

Directly measured -- independent of any offset formula, straight from live geom positions -- the real closing axis (vector between `gripper0_right_finger7_collision` and `gripper0_right_finger8_collision` world positions) while the eef was driven to the `DOWN_ORIENTATION` reference pose. Result: closing axis vs. world X, cos=0.9994; closing axis vs. eef's own local-x (world-frame), cos=1.0000. This confirms the code's "align the object's narrow axis with the gripper's closing direction" logic is targeting the mechanically correct axis -- ruling out a whole class of hypothesis (e.g. a 90 deg or otherwise mismatched axis convention between `grasp_mat` and the real gripper hardware) that would have explained the persistent one-finger-contact pattern just as well.

Applying the same live measurement to actual (failing) Cracker trials, right before the close command, produced physically inconsistent-looking numbers at first glance -- measured gripper span varying 3.6-8.3cm across trials instead of a fixed ~7.5cm (the value the isolated DOWN_ORIENTATION test gave for the same nominal "open" control target), and the object's own measured extent along the closing axis varying 9-23cm despite Cracker's true narrow-axis width being ~6-7cm. Once the axis-mismatch hypothesis was ruled out, the correct read of these numbers became clear: **the fingers are not actually reaching their commanded open span, because they are already being mechanically obstructed by contact with the misplaced object** -- consistent with the `descend`-phase contact trace from the previous entry, which showed contact starting partway through descent and persisting continuously (not a single snag) all the way to the bottom, with the object visibly drifting and likely rotating under that ongoing contact. This also explains why the "pre-close refresh" fix (re-reading the object's pose once, right before closing) only partially helped: it reads a pose that is itself still being actively disturbed by contact at the moment of reading, so the "fresh" target can already be stale again by the time closing starts.

**Implication for what to try next**: the fix needs to prevent the open gripper from touching the object at all before the intentional close begins, not just react faster to the disturbance after the fact. Two candidate directions, not yet implemented: (a) tighten the XY convergence requirement before `descend` is allowed to start (currently `approach` typically leaves ~0.4-0.9cm residual error, which combined with Cracker's height is enough to graze it on the way down), or (b) for tall objects specifically, avoid ever passing the open gripper alongside the object's own height range at all -- e.g. an approach that reaches the final XY position while still well above the object's top, THEN descends on a path with a tighter real-time XY correction loop (closer to what the mink Stage 1/2 experiments attempted, though those had their own convergence problems in this same tight-clearance scene).

## ⚠️ Descend-phase table launch found and fixed; centroid-offset re-measurement attempted and reverted (2026-07-15): Cracker still 0%, but the failure mode changed twice

**Can support**: two real, verified fixes to the DLS (non-mink) pipeline -- (1) interpolating the `descend` phase instead of a single abrupt PD move, and (2) re-reading the object's true pose immediately before the close command instead of trusting a value computed several phases earlier. Both are now in `piper_pick_and_place.py`. Also: a live-model-based method for directly measuring an object's true collision-geom centroid offset (`model.mesh_vert` on the compiled model), and a clear demonstration that this specific measurement, despite being mathematically sound, made Cracker's outcome worse in a direct A/B and was reverted.

**Cannot support**: any claim that Cracker (or Mustard) now succeeds more often -- still 0/5 in every configuration tested today. The root cause of the remaining miscentering is still open.

Picking up from the mink Stage 2 entry above: switched back to the original (non-mink) `ArmIK`/DLS pipeline to isolate variables, since mink's own convergence issues were confounding the picture. Traced a full Cracker trial step-by-step through `transit_high -> approach -> descend` with object-position and contact logging at every step, expecting to find a subtle few-mm centering error. Found something much bigger instead:

**Finding 1 -- `descend` was launching Cracker off the table.** With `descend` as a single non-interpolated `move_to` (the original code), a trace showed the object's Z position collapsing from 0.908 (table height) to 0.068 (the floor) within 1.5 seconds of sim time -- not a nudge, a full ejection. Mechanism: `descend` travels ~13cm straight down starting just 3cm above the object's own top surface (by design, via `approach_height_for`), so for a 21cm-tall box the last ~10cm of that descent runs alongside the object's own height range. `approach`'s IK typically leaves ~0.4-0.9cm of residual XY error (it rarely converges to exact 0), and a single fast PD move covering that whole descent has enough momentum to catch a corner hard and launch the box. **Fix**: interpolate `descend` the same way the post-grasp phases already are (10 waypoints instead of 1). Verified via the same trace: the catastrophic floor-launch became a much gentler few-cm nudge that stays on the table.

**Finding 2 -- even after that fix, the object could still drift several cm during approach/descend before closing.** Added a re-read of the object's true position/orientation immediately before the close command (`pre-close refresh` in the logs/phase dict), instead of trusting the value computed at the top of the trial. Combined with Finding 1's fix, average pre-close drift dropped from 5.99cm (mean, pre-fix baseline) to 2.94cm across a matched n=5 batch -- a real, measured improvement in how well the arm's final position tracks the object's actual location. **Still 0/5 success**, including one trial (403) where the refreshed position converged almost exactly (drift 0.33cm, refresh IK error 0.01cm) and the grasp still failed.

**Finding 3 -- traced that near-perfect trial through the close phase and found the true remaining problem: only ONE finger (`gripper0_right_finger8`) ever touched the cracker, for the entire 250-step close, and the object visibly slid sideways under it instead of being pinned symmetrically.** This means the "descend/drift" story from Findings 1-2, while real and worth fixing, was not the (or not the only) actual cause of failure -- the grasp target itself, even when reached exactly, is not centered between the two fingers.

**Finding 4 -- traced this to `OBJECT_CENTROID_OFFSET_LOCAL`, and found the 2026-07-14 measurement of it was itself off.** Read the *actual* collision geom's raw vertices directly from the live compiled MuJoCo model (`model.mesh_vert`, transformed through `geom_pos`/`geom_quat` into body frame -- guaranteed to reflect exactly what physics uses for contact, not an offline approximation) and computed the true 2D bbox center for Pear/Mustard/Can/Cracker. Results disagreed with the coded values substantially: Cracker's true offset was [0.0549, -0.0217] vs the coded [0.0306, -0.0102] (roughly double); Pear's was qualitatively different, including a sign-flipped Y component ([0.0197, -0.0407] true vs [-0.0014, +0.0155] coded). Pear apparently tolerated this because it's small/round with generous clearance; Cracker/Mustard have much less margin.

**Finding 5 -- swapping in the "correct" measured value made things worse, not better, and was reverted.** A direct A/B on the same n=5 Cracker batch: mean pre-close drift went from 2.94cm to 7.43cm (up to 12.44cm on one trial), and a repeat of the descend trace showed continuous single-finger scraping contact from waypoint 2 through waypoint 10 of the descent (versus brief, intermittent contact before). The math behind the new value is verifiably correct for the exact geom MuJoCo uses -- but empirically it did not produce a better-centered grasp. This means the true source of the miscentering is not fully captured by "a single constant local-frame XY offset," at least not the way it's being measured and applied here. Possible unexplored explanations: the 8-vertex collision proxy (`cracker_g1`) may not behave the same as its own bounding box under contact (a coarse convex hull's *contact* geometry near a corner isn't the same as its *centroid*), the interaction between this offset and the yaw-dependent `grasp_orientation_from_quat` axis choice may not compose the way assumed, or there may be a genuine sim artifact in how contact forces near a corner get resolved. **Reverted to the original 2026-07-14 values** rather than ship an empirically-worse change; kept Findings 1-2's fixes since those were independently verified to help (reduced drift, eliminated the catastrophic table-launch) even though neither flipped the final outcome.

**Honest read**: this session made genuine, verified progress on two real bugs (table-launch, stale pre-close target) but the actual determining bottleneck for Cracker turned out to be one level deeper than either of those -- systematic miscentering between the gripper's two fingers at the moment of contact, which does NOT reduce to the single local-offset model this project has been using since 2026-07-14. The right next step is probably to stop trying to patch the constant-offset model further and instead directly measure, for a given failing trial, the actual finger-to-nearest-face distances on both sides at first contact (not inferred from a precomputed offset) to determine whether the offset model is fundamentally the wrong shape for this problem, or whether there's a separate bug in how the yaw/axis and offset combine.

## ⚠️ mink collision-aware IK, Stage 2 full-pipeline test (2026-07-15): does not fix Cracker -- two new, distinct failure modes found

**Can support**: a working mink-based collision-aware reach (`piper_mink_ik.py`), and two honest, reproducible findings about why it doesn't rescue Cracker: (1) the reactive collision-avoidance QP itself frequently gets stuck in a local minimum well short of the target for this scene's tight arm/mount/table/object geometry, and (2) even in the trials where it converges cleanly, the downstream grasp still fails -- meaning the pre-grasp collision-sweep problem was not Cracker's only (or even its dominant) obstacle to a successful grasp.

**Cannot support**: any claim that mink integration improves Cracker's (or any other object's) end-to-end success rate -- it does not, in this pilot.

Stage 1 (previous entry in this file's history, superseded by this one) prototyped `mink`'s `CollisionAvoidanceLimit` on a single hand-picked Cracker trial (trial_id=400) and got a clean result: IK converged to 0.0004m position error with zero unwanted arm/mount contact with the object. That looked like strong evidence the "arm sweeps through tall objects" bug (documented below, "Root-cause investigation...") was genuinely fixable. Stage 2 integrated it into the full pipeline (`piper_pick_and_place_mink.py`: mink drives the pre-grasp reach, the existing `ArmIK`/`move_to_interpolated` machinery handles everything after the gripper closes, since collision-avoiding against an object you're now holding would fight the grasp) and ran it end-to-end on Cracker.

**Result: 0/10 success across two n=5 batches** (trial_id 400-404, run twice under two different `minimum_distance_from_collisions` settings -- see below), identical to the pre-mink baseline.

Two distinct problems, found by tracing individual trials in detail (same methodology as the earlier root-cause investigation):

1. **The collision-avoidance QP itself gets stuck.** With the Stage-1 prototype's `minimum_distance_from_collisions=0.01` (1cm), a direct trace of a failing trial showed velocity collapsing to ~0.0000 by step ~15 while position error was still stuck at 0.11-0.16m -- the QP had decided "no further motion" was the locally-optimal (feasible) choice, given the hard 1cm clearance constraint, rather than finding a path around the obstacle. Confirmed this was margin-dependent, not object-dependent, with a same-trial A/B: reducing the margin to 0.003m (2mm) let the identical trial converge to 0.0006m. But rerunning the full n=5 batch with the tighter 2mm margin still got stuck (pos_err 0.12-0.14m) on 4 of 5 trials -- the 1cm value from Stage 1's cherry-picked trial happened to be uniquely favorable, not a generally safe choice for this scene's geometry (the object spawns close to the mount pedestal and table in a fairly tight envelope). This is a known category of limitation for reactive/local velocity-level collision avoidance: it has no global path-planning guarantee and can get stuck in local minima, just via a different mechanism than the un-collision-aware DLS solver it replaced (which instead swept straight through the object).

2. **Even a near-perfect reach doesn't guarantee a successful grasp.** In the one trial (of 10) where mink converged almost exactly to the target (pos_err=0.0010m or 0.0005m depending on the run), the full task still failed (`dist_to_tray` 0.34-0.48m, nowhere close to the tray). A step-by-step contact trace on one such trial showed only a single finger (`gripper0_right_finger7/8_collision`) ever contacting the cracker, both before and after the close phase -- the classic asymmetric one-finger-contact pattern this file's own `piper_pick_and_place.py` docstring already documents as the reason position+orientation IK was adopted in the first place. During "lift", the object stayed at table height and lost even that contact, meaning it was never actually grasped, just nudged by the passing single finger. This points to an unresolved grip-precision issue independent of the pre-grasp approach path -- consistent with the still-open finding in "All 5 bugs fixed, full pilot rerun" below (fixing the sweep-through bug specifically did not fix Cracker's underlying 0% rate, because it was never the sole bottleneck).

**Files**: `piper_mink_ik.py` (`mink_reach()`, `ARM_BODY_NAMES`, `object_geom_ids()`), `piper_pick_and_place_mink.py` (`run_pick_and_place_mink()`, full pipeline: mink for the pre-grasp reach, existing `ArmIK`/`move_to_interpolated` for lift/transit/lower/retract).

**Honest read**: mink's collision awareness genuinely does what it's supposed to (Stage 1's single-trial validation of "zero unwanted contact" was real, not a fluke), but neither the original approach-collision hypothesis nor an off-the-shelf reactive-IK fix for it turns out to be sufficient to move Cracker off 0%. The remaining bottleneck looks like it's in grip precision/finger-contact symmetry at the moment of closing, not in how the arm gets there -- worth investigating directly (e.g. tracing exact finger-vs-object contact geometry at the close moment) before trying any other collision-avoidance library or IK solver, since swapping solvers again would very plausibly hit the same wall.

## ⚠️ All 5 bugs fixed, full pilot rerun (2026-07-14): Pear stable, Cracker still 0%, Mustard REGRESSED

Direct follow-up to the entry below (four bugs, one open) -- fixed the fifth
(per-object `approach_height_for`, widening clearance above tall objects'
own top surface, since the fixed 10cm default was tuned around Pear and
barely cleared Cracker's top at all), then reran a clean n=10/cell pilot on
fresh trial ids with ALL FIVE fixes applied together:

| Object | Before any of these 5 fixes | After all 5 |
|---|---|---|
| Pear | 75-90% (varied across earlier pilots) | 80% (8/10) |
| Mustard | 20% (2/10, earlier baseline) | **0% (0/10)** |
| Cracker | 0% | 0% (0/10) |

**Can support**: fixing individually-verified, real bugs does not
guarantee holistic improvement. Every one of the five fixes was validated
independently at the time it was made (exact centroid match, confirmed
settle timing, confirmed mount contact eliminated, confirmed approach-phase
clearance restored) -- yet Cracker is unchanged and Mustard got WORSE.
Pear, which never exhibited any of these five problems in the first place,
stayed roughly stable, which at least confirms the fixes aren't globally
harmful. The most likely explanation for Mustard's regression: its
grip was already marginal/fragile (established earlier as a grip-force
problem, not a positioning one -- fits the gripper's opening but is heavy
relative to available closing force). Any change to the exact grasp point,
even a mathematically small, "more correct" one (Mustard's own centroid
offset was measured at only 3% of its width, the smallest of all objects
checked), can be enough to disrupt a hold that was already right at the
edge of working. Fixing *precision* doesn't help an object whose failure
mode was never about precision.

**Cannot support**: that there are no further bugs to find for Cracker --
this investigation fixed everything currently identified and Cracker still
does not work, meaning either an undiscovered sixth issue exists, or
Cracker's failure has a different, non-bug root cause entirely (e.g.
container/box shapes may need a different grasp strategy altogether, not
just a more accurate version of the same top-down parallel-jaw approach
that works for Pear). Also cannot support that Mustard's regression is
specifically caused by the centroid fix rather than some other change in
this batch (didn't isolate which of the 5 fixes caused it, given time
already invested) -- flagged as the natural next debugging step if Mustard
matters more than Cracker going forward.

**Honest assessment for the paper**: this five-bug investigation is a
legitimate, well-documented negative result in its own right (found real,
verified, independent bugs; fixing all of them did not fix the underlying
generalization problem) -- arguably more informative than a clean
success would have been, and consistent with this project's established
practice of reporting negative/diagnostic findings rather than only
positive ones. Whether to keep debugging Cracker specifically, focus on
understanding Mustard's regression, or treat "Pear works, most other
shapes don't, and the reasons are only partially bug-shaped" as the actual
finding is an open decision, not something this investigation resolves on
its own.

## 🔧 Root-cause investigation into poor cross-object generalization (2026-07-14): four distinct real bugs, three fixed, one identified but open

Prompted by asking directly why generalization across objects was so poor
(Cracker/Banana 0% despite fitting the corrected 10cm gripper). Traced
Cracker specifically (clean box geometry, no bottle-taper or asymmetric-
shape confounds) step by step instead of continuing to guess. Found four
compounding, independent bugs -- not one root cause:

1. **Body origin != true geometric centroid** (`OBJECT_CENTROID_OFFSET_LOCAL`,
   `true_centroid_xy` in `piper_pick_and_place.py`). Every grasp target was
   computed around `env.get_object_positions()` (the mesh's body origin),
   silently assumed to be the object's centre. Sliced each object's raw
   mesh vertices near grasp height and measured the ACTUAL centroid vs
   body origin: Pear/Mustard/Banana/Can offsets are small and fall mostly
   along the *long* axis (forgiving -- doesn't hurt a two-finger grip much),
   but Cracker's is +3.06cm along the CLOSING axis on a box only 6.5cm wide
   -- 47% of its own width. Verified the fix is exactly correct by comparing
   against the true world-frame centroid computed from the live, settled
   mesh transform (matched to 4 decimal places). **Fixed and applies to all
   objects now, not just Cracker.**
2. **Pose read before the object finishes settling** (`run_pick_and_place`,
   30-step settle-and-hold added before reading the object's pose). The
   placement sampler deliberately spawns objects ~3cm above the table
   (`z_offset=0.03`, added earlier to avoid initial mesh embedding) and
   `env.reset()` does not itself step physics forward -- every grasp height
   this whole session was computed from that stale, pre-settle Z reading.
   XY/orientation were already confirmed stable at spawn (only Z drops
   during settling), so this specifically affected grasp HEIGHT accuracy.
   Caught a real implementation bug while fixing this: the settle loop
   initially used `env.step(np.zeros(action_dim))`, which with the
   absolute JOINT_POSITION controller commands qpos=0 for every joint (a
   completely different pose from READY_QPOS), not "hold still" -- fixed
   to explicitly hold `READY_QPOS`. Also required raising
   `ignore_done=True` on both env classes: the settle steps pushed total
   trial length over RoboSuite's default 1000-step horizon, which was
   already close to the limit before this fix.
3. **Object spawn region overlaps the robot's own stationary mount**
   (`piper_multi_object_scene.py`, sampler `x_range` lower bound tightened
   -0.25 -> -0.14). The mount pedestal is a fixed collision cylinder
   (radius 0.18m, centred at world x=-0.47, reaching to x=-0.29). Traced a
   cracker trial step-by-step through the "approach" phase and found
   `fixed_mount0_pedestal_col` in contact with the object from step 0 --
   BEFORE the arm had moved at all. Cracker (horizontal_radius 0.1085m)
   and Banana (0.122m) are large enough that their sampled edge could
   physically overlap the stationary mount even at spawn, silently
   corrupting every subsequent grasp attempt regardless of how good the
   grasp-planning logic is.
4. **NOT YET FIXED -- arm sweeps through tall objects en route to the
   target**: after fixes 1-3, re-traced Cracker (n=10, fresh trial ids) and
   still got 0/10, but the failure mode changed: no more mount contact, but
   `gripper0_right_finger7_collision` now shows up DURING the "approach"
   phase (before the arm has even reached its intended hover point above
   the object) -- i.e. the arm's IK-solved path from READY_QPOS to the
   approach target isn't a straight vertical descent, and can sweep
   sideways through a tall object (Cracker is 21cm) while still below its
   top edge, especially when the target requires a large joint1 (base)
   rotation. This is a collision-avoidance gap: `ArmIK`/`solve_multi_seed`
   only cares about reaching the target end-effector pose, with zero
   awareness of what the rest of the arm sweeps through along the way.

**Can support**: cross-object generalization was never really about
"gripper width" or even purely "grasp precision" as previously reported --
those were real findings, but incomplete. A meaningful chunk of the
Cracker/Banana failures traces to scene-level and trajectory-level bugs
that had nothing to do with grasp strategy at all. Fixes 1-3 are validated
independently (exact centroid match, confirmed settle timing, confirmed
mount contact eliminated) and now apply to every object, not just Cracker
-- the earlier orientation-ablation and consensus-selection pilots were all
run WITHOUT these fixes, so their absolute success-rate numbers likely
undersell what this pipeline can actually do (though the *relative*
comparisons within each pilot, oriented-vs-fixed and consensus-vs-best,
should be less affected since both arms of each comparison shared the same
bugs equally).

**Cannot support**: that fixing #4 (trajectory collision-avoidance) would
be enough to make Cracker succeed -- with fixes 1-3 alone, Cracker is still
at 0/10. This needs either a waypoint-based approach (rise to a safe
height first, rotate the base at that height, then descend at the target
XY -- avoiding horizontal sweeps at heights below tall objects) or genuine
arm-body collision checking, neither implemented yet. Also cannot support
that Cracker will reach parity with Pear even once #4 is fixed -- there
could be further, still-undiscovered issues; this investigation stopped
at the first newly-identified bug rather than assuming it's the last one.
The prior pilots (orientation ablation, consensus selection) have NOT been
rerun with these fixes and their absolute numbers should not be trusted
as final until they are.

## 🔍 Consensus vs best-IK, robustness check under a more faithful noise model (2026-07-14): same negative result holds

Follow-up to the entry directly below. That pilot's noise model was
criticized in its own writeup: synthetic Gaussian jitter directly on an
already-computed grasp target isn't a faithful reproduction of SO-ARM101's
actual noise source (a learned model's candidate diversity). Built a second,
more faithful noise model to check whether that critique explains the
negative result, or whether it's more fundamental:
`sample_perception_noisy_candidates` (`piper_candidate_selection.py`)
perturbs the TRUE object pose itself -- position AND full 3D orientation
(random-axis rotation, σ=10°, not just in-plane yaw) -- then recomputes
what grasp target EACH noisy pose estimate would imply via
`grasp_orientation_from_quat`. This can produce qualitatively different
candidates (e.g. a large enough orientation error changes which axis
*looks* narrow), which direct kinematic jitter cannot by construction.

**Pilot results** (n=10/cell, same 3 objects, `noise_model=perception`):

| Object | Consensus | Best (ikmargin) |
|---|---|---|
| Pear | 70% (7/10) | 80% (8/10) |
| Mustard | 0% (0/10) | 0% (0/10) |
| Cracker | 0% (0/10) | 0% (0/10) |

**Can support**: the negative result is not an artifact of the first
pilot's noise model being too clean/synthetic. Only 3 of the 6 (object,
noise_model) cells had any discordant pairs at all (Mustard-kinematic:
2, Pear-kinematic: 3, Pear-perception: 5) -- Cracker and Mustard-perception
never disagreed at all, both strategies just failed uniformly. In every
one of those 3 cells with signal, best won more often than consensus
(never the reverse): kinematic-Mustard best_only=2/consensus_only=0,
kinematic-Pear best_only=2/consensus_only=1, perception-Pear
best_only=3/consensus_only=2. That consistent direction across two
independently-designed noise models is a stronger signal than any
individual p-value at this n (none reached significance). The mechanistic
account from the entry below still stands: this pool's noise, however it's
generated, stays well-behaved enough that IK error and true grasp quality
don't decouple the way they apparently did for SO-ARM101's learned
candidates.

**Cannot support**: that NO noise model would ever show the effect --
still haven't tested candidates from an actual trained generator (the
one thing confirmed capable of producing SO-ARM101's effect). That
remains the real replication test, and remains hardware/data-gated (see
below).

## 🔍 Consensus vs best-IK candidate selection (2026-07-14): does not replicate on Piper -- and there's a mechanistic reason it might not be expected to

SO-ARM101's core validated finding is **consensus candidate selection**
(pick the pool candidate closest to the median, not the one with lowest IK
error): Pear 6%→68%, TomatoSoupCan 34%→64%, both significant (Fisher's
exact). Replicating this on Piper as a second, independently-implemented
embodiment would be a real cross-embodiment generalization result for the
paper -- this is what this experiment tested.

**Method** (`piper_candidate_selection.py`, `piper_consensus_experiment_runner.py`,
`piper_consensus_analysis.py`): since Piper's grasp target
(`compute_grasp_orientation`) is deterministic, injected explicit pose noise
-- sample N=10 candidates as Gaussian XY (σ=0.5cm) + yaw (σ=5°) jitter
around the nominal target, per trial. "best" = lowest cheap-IK-error
candidate (ikmargin analogue). "consensus" = candidate closest to the
pool's own median (circular median for yaw, unwrapped relative to the
pool's circular mean -- not a naive arithmetic median on angles). Paired
design: same `np.random.default_rng(trial_id)` draws the identical pool for
both strategies at a given trial_id.

**Pilot results** (n=10/cell, Pear/Mustard/Cracker -- Can/Drill/Clamp
excluded, their failures are unrelated to candidate pose noise):

| Object | Consensus | Best (ikmargin) | McNemar p |
|---|---|---|---|
| Pear | 80% (8/10) | 90% (9/10) | 1.0 |
| Mustard | 0% (0/10) | 20% (2/10) | 0.5 |
| Cracker | 0% (0/10) | 0% (0/10) | undefined (no discordant pairs) |

**Can support**: consensus selection does not show the SO-ARM101 effect
here -- direction is flat-to-negative across all 3 objects (never positive),
opposite of the hoped-for replication, though none of the differences are
significant at this n. Mechanistic explanation: SO-ARM101's candidates came
from a *learned generative model* (OT-CFM/GeoEBM) where individual
candidates could have genuinely different failure modes decoupled from IK
convergence (model artifacts producing a numerically-plausible-but-wrong
candidate) -- consensus protected against picking one of those outliers.
Piper's candidate pool here is synthetic, symmetric Gaussian jitter around
an already-reasoned nominal target (`compute_grasp_orientation` already does
real geometric work), so there's no equivalent "looks good but is wrong"
outlier for consensus to filter; IK error and true grasp quality stay
tightly correlated in this regime, removing the room for consensus's
noise-averaging benefit to show up. **Conclusion for the paper**: consensus
selection's benefit looks conditional on the candidate generator's noise
characteristics (model-driven multi-modality), not a free-standing
ensembling trick that transfers to any noisy-candidate setting.

**Cannot support**: whether consensus WOULD work on Piper if fed candidates
from a genuinely noisy/multi-modal source (a trained grasp proposal model,
or noise injected at the perception/pose-estimation stage rather than
directly on the grasp target) -- this pilot only tested synthetic kinematic
jitter, not a faithful reproduction of SO-ARM101's actual noise source.
Building a trained candidate generator for Piper would need real physical
trial data (SO-ARM101's took ~400 trials/object collected over months);
not attempted here, flagged as a separate, larger, hardware-gated project.

**Re-verified 2026-07-17 under the fixed placement RNG** (n=10/object, trial_id 7000-7009,
kinematic noise model, matching the original protocol exactly): Pear 80% best / 70% consensus
(p=1.0000), Mustard 80% best / 70% consensus (p=1.0000), Cracker 50% best / 50% consensus (0
discordant pairs). **The null finding is confirmed, not overturned** -- consensus still does not
show a benefit on Piper. Notably, absolute success rates jumped dramatically for Mustard (0%/20%
originally -> 80%/70%) and Cracker (0%/0% -> 50%/50%), consistent with the RNG bug having made the
overall task harder-than-intended by drawing unfavorable object placements more often than a truly
random draw would -- but the RELATIVE comparison this pilot exists to test was unaffected either
way. The original mechanistic explanation (Piper's synthetic Gaussian jitter around an
already-reasoned target lacks SO-ARM101's model-based multi-modal outlier structure) holds
independent of the RNG bug and remains the best available account of why this doesn't transfer.

## 🔍 Grip-precision bottleneck survives the gripper-width fix (2026-07-14): width was never the dominant limiter for most objects

After correcting the gripper's max opening from an erroneous 7.6cm (bug --
see entry below) to the verified 10cm, re-tested Cracker (7.17cm), Banana
(7.89cm), and Can (~8.6cm) -- all now geometrically fit with margin.
**All three still failed at 0% success** (Cracker/Banana: n=5, Can: n=5).
Traced the mechanism directly: Cracker showed only ONE finger actually
contacting the object (asymmetric grip) despite full IK convergence;
Banana's IK converged at every phase (approach/descend/lift all report
success) yet the object never left the table. Same "IK says success, grasp
isn't secure" pattern first found with Mustard.

**Can support**: for flat/box-shaped (Cracker) and elongated (Banana)
objects, grip failure is a *precision* problem (asymmetric contact, no
self-centering effect from a curved surface), not a width problem. Only
Pear (round, light, self-centering, forgiving of a few mm of
misalignment) is reliably graspable with the current open-loop,
no-tactile-feedback grasping approach. Mustard survives on a different,
narrower margin (fits geometrically, grip force marginal given 0.6kg mass).

**Cannot support**: whether a different (non-parallel-jaw) grasp strategy,
or force/tactile feedback during closing, would fix Cracker/Banana --
not attempted, would be a materially different grasping approach.

## 🔧 Gripper opening corrected 7.6cm → 10cm (2026-07-14): community-ported model disagreed with AgileX's own spec

Ported gripper (`piper_assets/piper_gripper.xml`, from `soulde/Piper_mujoco`)
used joint7/joint8 slide range `-0.038 to 0` (max opening ~7.6cm, measured
directly via finger body positions). Found AgileX's own official Isaac Lab
asset (`piper_gripper.urdf` in `agilexrobotics/Agilex-College`) specifies
`gripper_joint1/2` range `0 to 0.05` per finger -- true opening ~10cm.
Same class of discrepancy as the joint2-range conflict found earlier this
project (community port vs AgileX's own numbers). Widened
`piper_gripper.xml`'s joint/ctrlrange and `piper_gripper.py`'s clip bounds
to match; verified new opening = 10.0cm via direct finger-position
measurement; confirmed stable under sustained gripper-close commands
(no regression of the earlier QACC/NaN finger-mass fix).

**Can support**: the 7.6cm figure this project was reporting for "how few
objects Piper can grasp" was itself partly a simulation modeling error, not
purely a hardware fact -- see the entry above for what actually happened
once this was fixed (width stopped being the dominant limiter).

**Cannot support**: that 10cm is itself exactly correct for the REAL
physical gripper -- this is AgileX's own official asset, more authoritative
than the community port, but still not independently verified against
actual hardware measurement (no real Piper connected yet this session).

## 🔧 Multi-seed IK + interpolated lift (2026-07-14): two real reliability fixes, both validated with data

1. **Multi-seed DLS restart** (`ArmIK.solve_multi_seed`, `piper_pick_and_place.py`):
   single-seed DLS IK is a local method with no guarantee the seed's basin
   contains a valid solution. Investigated porting AgileX's own analytical
   IK (`piper_kinematics/piper_analytical_ik.hpp`, closed-form wrist-center
   decomposition) but found their DH parameters do not match this project's
   ported MJCF model even after searching all 64 per-joint sign-flip
   combinations (residual error ~2.3, not near zero) -- rejected as
   too risky given the DH mismatch discovery. Multi-seed restart (try a
   diverse fallback set of joint1/elbow seeds on primary-seed failure) was
   the lower-risk fix. Validated: Pear success rate 55%→75% (oriented),
   45%→80% (fixed) purely from this change.
2. **Interpolated lift** (`move_to_interpolated`): traced a held-Mustard
   trial where both fingers had a legitimate, decent-force grip right after
   closing, but the object's height barely changed through the "lift" phase
   -- `move_to`'s single abrupt joint-space target command was yanking the
   arm too fast for a marginal grip to survive the resulting acceleration.
   Interpolating the post-grasp motion through intermediate waypoints
   roughly doubled Mustard's success rate (10%→20%, both strategies).

**Can support**: both fixes are validated by re-running the SAME trial_ids
before/after and observing the change, not just aggregate rate shifts that
could be sampling noise.

## ⚠️ Orientation-aware grasping ablation (2026-07-14): re-verified 2026-07-17, original "no significant difference" framing does not hold up cleanly

**Original finding (RNG-bug-affected)**: compared aligning the grasp yaw to the object's measured
narrow axis (`compute_grasp_orientation`) vs a fixed world-X approach direction
(`DOWN_ORIENTATION`), paired by trial_id, McNemar's exact test. Consistent finding across
single-seed IK, multi-seed IK, and multi-seed+interpolated-lift solver configurations: p-values
0.75-1.0 throughout, never significant.

**Re-verified 2026-07-17 under the fixed placement RNG** (see the file-top pointer note),
n=20/object, trial_id 6000-6019, current solver config (multi-seed+interpolated-lift), objects
matching the T-RO paper's headline set (Cracker/Mustard/Pear) rather than the original
Pear/Can/Mustard:

| Object | Oriented | Fixed | McNemar p | Direction |
|---|---|---|---|---|
| Cracker | 50% (10/20) | 20% (4/20) | 0.146 | favors oriented (not significant) |
| Mustard | 70% (14/20) | 75% (15/20) | 1.000 | null — replicates original exactly |
| Pear | 55% (11/20) | 90% (18/20) | 0.065 | favors **fixed** (nearly significant) |

**Can support**: Mustard's null replicates cleanly. Cracker and Pear do NOT — none individually
reach p<0.05, but the pattern (Cracker leaning oriented, Pear leaning fixed by a large 35pp margin)
is materially different from "no difference across all conditions, p=0.75-1.0 throughout." The
original blanket "doesn't matter which you pick" justification for defaulting to
`use_oriented_grasp=True` is weaker than previously stated.

**Cannot support**: that oriented grasping is proven better or worse overall — the pattern is
object-dependent and no single comparison reaches conventional significance at this n. Cannot
support a recommendation to switch the default without further, larger-n testing per object
(Pear's 90% vs. 55% gap in particular deserves a bigger sample before acting on it either way).

**Note for any paper material**: `paper_tro.tex`/`paper_tro_draft.md` never make an explicit claim
about oriented-vs-fixed grasping — `use_oriented_grasp=True` is used only as an implementation
default, never discussed or justified in the paper text. This finding does not require changing
any reported paper number, but the informal justification for that default (documented here, not
in the paper) should not be cited as "no significant difference" without this caveat.

**Cannot support** (from the original entry, still true): whether orientation awareness would
matter for objects with a more extreme aspect ratio, or under real (non-simulated) perception
noise where the "fixed" direction might not even be roughly reasonable.
