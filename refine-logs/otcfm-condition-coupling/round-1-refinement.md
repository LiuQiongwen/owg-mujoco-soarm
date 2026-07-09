# Round 1 Refinement

## Problem Anchor (copied verbatim from Round 0)

- **Bottom-line problem**: Our RA-L submission's original headline claim — "OT-CFM (minibatch-optimal-transport-coupled conditional flow matching) generates 6-DoF grasp candidates that significantly outperform random-CoM sampling" — does not hold. After fixing a real per-trial seeding bug in the evaluation harness and re-running cleanly (7 YCB objects, n=50 trials/condition), OT-CFM is significantly worse than the random baseline (pooled: 69.1% vs. 79.1%, Δ=-10.0pp, p=0.0025; all 7 objects individually negative). A targeted follow-up shows the failure is specific to the OT-coupled variant: plain CFM without OT coupling and DDPM both track or beat baseline on 2 of 3 objects tested, while OT-CFM alone is consistently worst. This pattern matches "The Curse of Conditions" (Cheng & Schwing, ICCV 2025), validated only on generation-quality metrics on image benchmarks, never on physical task success in robotics.
- **Must-solve bottleneck**: A method-level fix, not just a bug report — show the condition-agnostic OT coupling is the identifiable cause, and that a condition-aware fix restores baseline-beating performance on a physical-execution metric.
- **Non-goals**: No LGGSN redesign. No new perception/pose-uncertainty estimator. No claim that OT coupling is universally bad — scoped to conditional flow matching over a small, multi-class conditioning set with small per-class data.
- **Constraints**: RA-L, 8-page hard limit (7 pages currently). Frozen infra: MuJoCo/SO-ARM101, LGGSN v2, YCB objects, `tango` env. Single-GPU; training is cheap, evaluation wall-clock is the bottleneck. Deadline urgency stated but **not yet confirmed with an actual date — see Anchor Check below.**
- **Success condition**: clean n≥50/condition evaluation showing condition-aware coupling statistically recovers parity with/exceeds baseline, attributable specifically to condition-awareness. If it fails, the paper still stands as a rigorous negative/methodological result.

## Anchor Check

- **Original bottleneck**: build (or salvage) a working, honestly-evaluated grasp-candidate generator that beats random sampling.
- **Does the revised method still address it?** Yes, but with a named reframing (see Drift Warning below) — the contribution is now "diagnose + transfer-validate + attempt-to-fix a recent generative-modeling failure mode, on a physical robot task," not "propose a new generative method." This is still anchored to the same success condition (does OT-CFM, once fixed, beat baseline on the same physical metric?), so it is not abandoning the anchor, but the *kind* of paper this now, is different from what was originally scoped, and I am flagging it rather than smoothing it over.
- **Reviewer (self-review) suggestions rejected as drift**: none — the self-review's suggestions (correct the mechanism, simplify to the discrete case, confirm the deadline) all sharpen the same anchor, they don't change what problem is being solved.

## Simplicity Check

- **Dominant contribution after revision**: unchanged in spirit, sharpened in mechanism — first physical-task-success validation (and attempted fix) of the Curse-of-Conditions failure mode, now using the *exact, discrete-class-appropriate* instantiation C²OT's own authors used for their closest analogous setting (CIFAR-10, 10 discrete classes), not a generic guessed formula.
- **Components removed or merged**: the generic λ-weighted-penalty cost function and its held-out sweep are **removed entirely** and replaced by per-object-stratified OT coupling (run the existing, already-implemented unconditional OT solver independently within each object's ~400 examples, instead of across the mixed 7-object minibatch). This is simpler (no new cost function, no new hyperparameter, no sweep needed) and is the discrete-class limit of C²OT's own released CIFAR-10 configuration (`condition_weight → very large` forces within-class-only pairing).
- **Reviewer suggestions rejected as unnecessary complexity**: none from the self-review — all suggestions were adopted since they simplify.
- **Why the remaining mechanism is still the smallest adequate route**: it is now a for-loop (group minibatch by object label) around a training call that already exists, rather than a new cost matrix implementation. Nothing simpler would still test the actual mechanism (removing OT entirely is a different condition we already have — "Remove-OT" — and is not this claim).

## Changes Made

### 1. Corrected the OT-coupling fix mechanism (Method Specificity, CRITICAL)
- **Reviewer said**: round 0's cost formula was guessed, not verified against C²OT's actual released code; the real mechanism (fetched from `c2ot/ot.py` in the official repo) uses a continuous condition-distance cost (cosine or L2², not a binary indicator), solved by exact linear-sum-assignment, and for their closest discrete-class analogue (CIFAR-10) uses a very large fixed weight rather than a tuned sweep.
- **Action**: replaced the generic weighted-cost-matrix method with **per-object-stratified OT coupling** — mathematically the discrete-class limit of C²OT's own verified approach, and simpler to implement (no new cost function or hyperparameter at all).
- **Reasoning**: this is both more faithful to the cited mechanism and a genuine simplification; it removes an entire sub-experiment (the λ/weight sweep) from the validation plan.
- **Impact on core method**: Core Mechanism section rewritten below.

### 2. Reworded the novelty framing (Contribution Quality, IMPORTANT)
- **Reviewer said**: the dominant contribution should be explicit that the coupling mechanism itself is borrowed verbatim (from released, working code), and the novelty is the transfer to a physical robot task plus the empirical validation — not a newly-designed algorithm.
- **Action**: Novelty and Elegance Argument section reworded accordingly (below).
- **Reasoning**: honesty about what is and isn't new strengthens the paper's credibility and matches what RA-L reviewers actually reward (rigorous application/validation, not claimed algorithmic novelty that doesn't exist).
- **Impact on core method**: no method change, framing/prose change only.

### 3. Flagged the reframing explicitly instead of smoothing it over (Problem Fidelity, IMPORTANT)
- **Reviewer said**: the paper's contribution has shifted kind (from "new method" to "transfer-validation-and-fix study"), and this should be named, not hidden.
- **Action**: added an explicit note in the Problem Anchor and Drift Warning.
- **Impact**: no method change; sets correct expectations for how this reads to an RA-L reviewer (Venue Readiness section reworded to foreground the task-level framing over the OT-math framing).

### 4. Flagged missing deadline information (Feasibility, CRITICAL — unresolved, requires user input)
- **Reviewer said**: "tight timeline" is stated but no actual date is given, and this session has already spent a full day-plus of compute; committing to another multi-day retrain-and-evaluate campaign without knowing the real deadline is irresponsible.
- **Action**: **not resolved in this document** — flagged for the user directly, see message accompanying this refinement. The Compute & Timeline Estimate section below is revised downward (stratified OT removes the sweep, saving time) but still assumes a multi-day window that has not been confirmed.

## Revised Proposal

*(Sections unchanged from Round 0 are summarized; only changed sections are given in full.)*

**Problem Anchor, Non-goals, Constraints, Success Condition**: unchanged from Round 0 (see above), with the explicit caveat added: **this is now named as a transfer-and-validation contribution about a generative-modeling failure mode in a physical robotics setting, not a new generative algorithm** — see Novelty and Elegance Argument.

**Technical Gap**: unchanged — condition-agnostic minibatch OT coupling creates a training/inference prior mismatch under multi-condition minibatches; Remove-OT/DDPM controls already rule out "CFM is broken" and implicate the coupling specifically.

### Core Mechanism (REVISED)
- Input/output: unchanged (256-d SAM embedding condition → 6-DoF pose).
- Architecture: unchanged VelocityNet.
- **Training signal/loss (corrected)**: unchanged flow-matching regression loss. The *only* change from the current (failing) OT-CFM training is: **before computing the OT assignment for a minibatch, partition the batch by object label, and run the existing unconditional OT solver (`scipy.optimize.linear_sum_assignment` on pairwise pose-space L2² cost, which is what `train_cfm_grasp.py` already implements for the "OT" condition today) independently within each object's own examples, then concatenate the per-object assignments back into the training batch.** No new cost matrix, no new hyperparameter, no sweep. This is the discrete-class limit of C²OT's own released CIFAR-10 configuration (`condition_weight=1e8`, which in practice forbids cross-class pairing whenever a same-class pairing exists) — we are directly instantiating that limit rather than approximating it with a tunable penalty.
- Why this is the main novelty: **the mechanism itself is not novel — it is C²OT's own verified fix, in its discrete-class special case.** The contribution is (a) recognizing that our setting is exactly the discrete-class case their own repo already configures for, (b) being the first to test whether this transfers from generation-quality metrics to physical task success, and (c) reporting the result honestly regardless of outcome.

### Novelty and Elegance Argument (REVISED)
Closest work #1 (C²OT, ICCV 2025): identifies and fixes the identical mechanism; we do not claim a new algorithm — we claim a new *validation domain* (physical robot task success, not FID/likelihood) and we reuse their exact discrete-class configuration rather than reinventing a continuous penalty we would need to sweep. Closest work #2 (Joyce et al., IROS 2025): different uncertainty source (perception vs. generation-sampling), explicitly secondary contribution. Elegance argument, sharpened: the "fix" costs zero new code complexity beyond a groupby — this is the smallest possible test of whether the cited mechanism is the actual cause, which is itself methodologically clean (a confound-free ablation, not a new system).

### Claim 1 validation (REVISED — simpler, no sweep needed)
- Minimal experiment: 3-way physical comparison at the training-recipe level (condition-agnostic OT-CFM [existing, failing] vs. per-object-stratified OT-CFM [new, one retrain] vs. Baseline/Remove-OT/DDPM as already-established reference points), n=50/condition, 7 objects, existing fixed-seeding harness.
- No λ ablation needed (removed per Simplicity Check) — the "OT applied across objects" vs. "OT applied within object" comparison IS the ablation.
- Metric, expected evidence: unchanged from Round 0.

### Compute & Timeline Estimate (REVISED)
- Training: unchanged, minutes/checkpoint, now with a simpler code change (groupby, not a new cost function + sweep) — actually *less* engineering risk than Round 0.
- Removed: the λ/target-r sweep and its held-out-split evaluation (not needed under stratified OT).
- Dominant remaining cost: one new n=50×7-object physical re-evaluation of the stratified-OT checkpoint (≈12.5 GPU-hours worst case, likely much less per this session's observed pace) — smaller than Round 0's 4-way estimate since two of the four conditions (Baseline, Remove-OT/DDPM references) already exist and do not need rerunning.
- **Unresolved and CRITICAL**: actual submission deadline not yet provided by the user. This determines whether even this simplified plan (retrain: <1hr; re-evaluation: likely 3-8 hours based on this session's observed per-trial pace) is attemptable. Must confirm before proceeding.

## Status

**Not yet re-scored by an external reviewer** (Codex/GPT-5.4 access unavailable this session). Self-assessed as meaningfully stronger than Round 0 on Method Specificity (now grounded in verified source code) and Feasibility (simpler mechanism, smaller compute ask), but the Problem Fidelity / Venue Readiness reframing concern and the deadline question are substantive enough that I am presenting this to the user for a decision rather than continuing to iterate alone.
