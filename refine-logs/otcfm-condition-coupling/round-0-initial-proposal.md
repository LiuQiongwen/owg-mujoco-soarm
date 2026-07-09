# Research Proposal: Condition-Aware Optimal Transport Coupling for Physically-Grounded Grasp Pose Flow Matching

## Problem Anchor

- **Bottom-line problem**: Our RA-L submission's original headline claim — "OT-CFM (minibatch-optimal-transport-coupled conditional flow matching) generates 6-DoF grasp candidates that significantly outperform random-CoM sampling" — does not hold. After fixing a real per-trial seeding bug in the evaluation harness (CFM/DDPM inference sampling was drawing from an unseeded, effectively-constant torch RNG state instead of varying per `--seed`) and re-running cleanly (7 YCB objects, n=50 trials/condition, MuJoCo/SO-ARM101, physics-based grasp success), OT-CFM is *significantly worse* than the random baseline (pooled: 69.1% vs. 79.1%, Δ=-10.0pp, p=0.0025; all 7 objects individually negative). A targeted follow-up (3 objects, n=25) shows the failure is specific to the OT-coupled variant: plain CFM without OT coupling (identical architecture/data) and DDPM both track or beat baseline on 2 of 3 objects, while OT-CFM alone is consistently worst. This pattern matches a documented mechanism — "The Curse of Conditions" (Cheng & Schwing, ICCV 2025): minibatch OT coupling that ignores the conditioning variable creates a training/inference prior mismatch in *conditional* flow matching, degrading performance. That paper validates the mechanism only on generation-quality metrics (FID, likelihood) on image benchmarks (CIFAR-10, ImageNet); it has never been shown to matter for physical task success in robotics.
- **Must-solve bottleneck**: We need a method-level fix, not just a bug report. The paper must show (a) that the condition-agnostic OT coupling used in our original training is the identifiable cause (not merely correlated), and (b) that a condition-aware coupling fix restores — ideally exceeds — the random-baseline performance that OT-CFM was supposed to beat, closing the loop from diagnosis to remedy on a real physical-execution metric.
- **Non-goals**: Not redesigning the LGGSN reranker (frozen, keep as-is). Not proposing a new perception/pose-uncertainty estimator (that is Joyce et al. IROS 2025's territory — different source of uncertainty, out of scope). Not claiming OT coupling is universally harmful in flow matching — the claim is scoped to *conditional* flow matching over a small, multi-class conditioning set with small per-class data (~400 examples/object), which is exactly our regime and plausibly a common one in robot learning.
- **Constraints**: RA-L, 8-page hard limit (currently at 7 pages before this rewrite). Existing infra frozen: MuJoCo/SO-ARM101 sim, LGGSN v2 reranker, YCB object set, `tango` conda env. Single-GPU (RTX 3060 6GB class) — CFM retraining is cheap (~minutes, ~400 examples/object, 4-layer MLP) but each physical re-evaluation trial costs up to 90s wall-clock, so evaluation compute (not training compute) is the actual bottleneck. Submission timeline is tight — this diagnosis was made late in the review cycle.
- **Success condition**: A clean, properly-seeded n≥50/condition evaluation on the same 7 objects showing (1) vanilla/condition-agnostic OT-CFM underperforms baseline (already have this), (2) condition-aware OT coupling (adapted from C²OT to our 7-class, low-dimensional pose setting) statistically recovers parity with or exceeds baseline, and (3) the recovery is attributable specifically to condition-awareness (ablation: same data/architecture, only the coupling's cost matrix changes). If (2) fails even after a genuine condition-aware retrain, the paper still stands as a rigorous negative/methodological result — but success means we get our "OT coupling helps" claim back, honestly earned this time.

## Technical Gap

Current conditional flow-matching-for-grasping methods (and our own original design) treat minibatch OT coupling as a drop-in upgrade over random pairing, following the unconditional-generation literature (Tong et al. 2023) where OT coupling straightens ODE paths and speeds/improves sampling. This ignores a mechanism identified only very recently (C²OT, ICCV 2025): when a minibatch mixes multiple conditions (here: 7 object identities via SAM visual embeddings), computing one shared OT plan over the whole batch implicitly optimizes an assignment that is blind to which noise sample "belongs" to which condition's target distribution. During training the model only ever sees the *conditionally-skewed* subset of the prior that the OT solver assigned to each condition; at inference, sampling starts from the *full, unconditional* prior, and the model has never learned to map from there. C²OT's fix — add a condition-dependent penalty term to the OT cost matrix so the transport plan itself respects conditioning — was shown to close this gap on image-generation benchmarks.

Naive fixes are insufficient here: more training epochs or more data does not address a *structural* mismatch between the training-time coupling and the inference-time sampling procedure — it is not a capacity or convergence problem, and our Remove-OT/DDPM controls (no OT coupling, same everything else) already show that removing OT coupling alone recovers baseline-competitive performance, which rules out "the CFM approach is broken" as the explanation and specifically implicates the *coupling mechanism*, not the network or data. Larger reranker ensembles or more LGGSN candidates would only mask, not fix, low-quality candidate generation upstream.

The smallest adequate intervention is to adapt C²OT's conditional cost-matrix weighting to our setting: our conditioning is a small, closed set of 7 discrete object classes (not continuous, as in C²OT's main experiments, though C²OT explicitly reports also testing discrete conditions), and our pose space is 6-D (not image pixels), so the adaptation is a substitution of the cost term, not a new architecture — directly matching the skill's "smallest adequate mechanism" principle. A frontier-native alternative (e.g., replacing minibatch OT with an amortized/learned transport map, or using a VLM to re-condition on richer per-instance features) would introduce new trainable components and training complexity disproportionate to the diagnosed bottleneck, and is not what the evidence calls for: the failure is specifically the *condition-blindness* of the cost matrix, not the representational capacity of the conditioning signal itself.

Core technical claim: *condition-agnostic minibatch OT coupling, applied naively to conditional flow matching with a small per-condition sample budget, produces a generator whose physically-executed grasp success rate is significantly below both an unconditioned-coupling ablation and a random-sampling baseline — and this is fixable by making the OT cost matrix condition-aware, without any other change to data, architecture, or downstream reranking.*

Required evidence: (1) the already-collected clean n=50 comparison across all 7 objects (Baseline vs. Remove-OT vs. OT-CFM vs. DDPM) establishing the failure and localizing it to OT coupling specifically; (2) a retrained condition-aware-coupling model (same architecture, same 7-object training set, only the coupling changed) evaluated under the identical clean harness; (3) a physical-execution comparison, not merely a distributional/likelihood one, since that is precisely the gap C²OT never closed.

## Method Thesis

- **One-sentence thesis**: We show that the recently-identified "Curse of Conditions" failure mode in conditional flow matching (Cheng & Schwing, ICCV 2025) is not just a generation-quality artifact but causes measurable physical task failure in robotic grasping, and that adapting their condition-aware OT coupling fix to a small-class, low-dimensional conditional pose-generation setting restores — and lets us honestly evaluate — the candidate-generation advantage the original (flawed) OT-CFM design claimed but never earned.
- **Why this is the smallest adequate intervention**: The fix is a substitution inside an already-existing training loop (the OT solver's cost matrix), not a new module, network, or training stage. Everything downstream (LGGSN reranker, MuJoCo evaluation harness, candidate count) is untouched.
- **Why this route is timely**: Minibatch-OT-coupled flow matching is being adopted rapidly across robotics/generative-control papers (that literature almost always cites Tong et al. 2023's unconditional-OT result without checking C²OT's conditional caveat, which is only 4-5 months old as of this writing); our finding is a concrete, physically-grounded warning shot for a methodological blind spot the field is actively walking into.

## Contribution Focus

- **Dominant contribution**: First demonstration that condition-agnostic OT coupling in conditional flow matching causes measurable *physical task failure* (not just a generation-quality metric regression) in a real robotic manipulation setting, together with a condition-aware coupling fix (adapted from C²OT) that closes the gap — validated end-to-end from training change to physically-executed grasp success rate.
- **Optional supporting contribution**: Test-time consensus-based candidate selection (execute the candidate nearest the pose-space median of N=10 independently-sampled candidates, rather than the one with lowest predicted IK error) is a complementary, training-free reliability lever: on the two objects where raw OT-CFM sampling is least reliable across seeds, it significantly recovers success rate (Pear 6%→68%, Fisher p=5.8e-11; TomatoSoupCan 34%→64%, p=0.0048) — orthogonal to the training-time coupling fix, costs nothing to combine with it.
- **Explicit non-contributions**: No new reranker design (LGGSN untouched). No new perception/pose-estimation uncertainty model (not competing with Joyce et al.). No claim that OT coupling is bad in general — scoped explicitly to small-class conditional settings with limited per-class data.

## Proposed Method

### Complexity Budget
- **Frozen / reused**: LGGSN v2 reranker (unchanged), MuJoCo/SO-ARM101 sim + physics-weld grasp-success criterion (unchanged), VelocityNet architecture (4-layer MLP, unchanged), SAM visual embedding as conditioning signal (unchanged), training data (same ~400 examples/object physical-execution successes), evaluation harness (now-fixed seeding).
- **New trainable component**: One re-trained flow-matching checkpoint, identical to the original except the minibatch OT solver's cost matrix gets a condition-dependent penalty term (per C²OT). This is a training-time change to an existing component, not a new component.
- **Tempting additions intentionally not used**: No learned/amortized transport map. No continuous/richer conditioning signal (e.g. per-instance point cloud) — that would conflate "condition-awareness of the coupling" with "richness of the conditioning signal," muddying the claim. No new reranker features.

### System Overview
```
[unchanged] SAM visual embedding (per-object, 256-d) ──► condition c
[unchanged] Live object CoM (from MuJoCo point cloud) ──► CoM-shift at inference
[CHANGED]   Training: minibatch {(x0_i, x1_i, c_i)} ──► OT solver with
             cost(i,j) = ||x0_i - x1_j||² + λ · 1[c_i ≠ c_j] · penalty
             (condition-aware cost matrix, C²OT-style) ──► reassigned pairs
             ──► flow-matching loss (unchanged)
[unchanged] Inference: sample_poses(cond=c, seed=gen_seed) ──► 5 candidates
[unchanged] LGGSN reranker ──► top-1 candidate ──► physical execution
```

### Core Mechanism
- **Input/output**: identical to current OT-CFM — conditioning vector c (256-d SAM embedding), output 6-DoF pose (x,y,z,roll,pitch,yaw), normalized.
- **Architecture**: unchanged VelocityNet (4-layer MLP, SiLU, zero-init output).
- **Training signal/loss**: unchanged flow-matching regression loss; only the *source of the training pairs* changes — the OT assignment used to pick which x0 pairs with which x1 is computed with a condition-aware cost matrix instead of pure Euclidean cost. Concretely (adapting C²OT's discrete-condition variant, since our 7 object classes are categorical, not continuous): augment the per-pair transport cost with a penalty λ·1[c_i ≠ c_j] before solving the batch-level OT assignment (Sinkhorn or exact), so the solver is discouraged from moving probability mass across conditions within a minibatch. λ is the one new hyperparameter, swept on a held-out split before the main comparison.
- **Why this is the main novelty**: it is the first time this exact, recently-identified failure mode is (a) shown to produce a *physically executed task failure*, not merely a shift in a generation-quality metric, and (b) fixed with a physically-validated re-training, in a robotics setting.

### Optional Supporting Component
- **Only included because it is already fully evidenced, orthogonal, and nearly free to report**: consensus candidate selection at inference time — draw N=10 independent per-trial samples (already-fixed seeding makes this meaningful now), execute the one nearest the (x,y,yaw)-space median.
- **Input/output**: same conditioning and candidate format; only the selection rule among an ensemble changes.
- **Training signal/loss**: none — purely an inference-time selection rule, no additional training.
- **Why it does not create contribution sprawl**: it requires zero changes to the trained model, is already fully validated with real physical-execution data (2 objects, Fisher-exact significant), and is presented explicitly as a secondary, orthogonal lever — not a second core mechanism competing for the paper's spotlight.

### Modern Primitive Usage
- No LLM/VLM/RL component is introduced beyond what already existed (SAM embedding for conditioning). The "frontier" angle here is methodological/diagnostic (a very recent, 2025 finding applied for the first time to embodied/physical evaluation), not an added foundation-model module — consistent with "the smallest adequate mechanism wins" when the bottleneck is a training-pair-assignment procedure, not a representational one.

### Integration into Base Generator / Downstream Pipeline
The condition-aware coupling only touches the offline training script (`train_cfm_grasp.py`); the trained checkpoint is a drop-in replacement for the existing `cfm_allobj_ot.pt`, loaded by the same unchanged `ui.py` inference path (already fixed for per-trial seeding this session). No changes to LGGSN, no changes to the MuJoCo evaluation harness beyond what's already fixed.

### Training Plan
Re-use the existing ~400-examples-per-object physical-execution success dataset (7 objects). Train three checkpoints under the identical fixed-seeding harness for a controlled 4-way comparison: (i) Baseline [no learned generator], (ii) Remove-OT / plain CFM [already trained, existing checkpoint], (iii) original condition-agnostic OT-CFM [already trained, existing checkpoint — this IS the failing one], (iv) new condition-aware OT-CFM [only new training run required]. Sweep λ (condition-penalty weight) on a small held-out validation split (e.g. leave-one-orientation-seed-out) before committing to the main n=50 physical comparison, to avoid tuning against the physical-execution test set itself.

### Failure Modes and Diagnostics
- **Failure mode**: condition-aware coupling recovers *distributional* quality (e.g., matches training yaw statistics better) but still fails to beat baseline physically. **Detection**: compare physical success directly, not just distributional diagnostics. **Fallback**: report as a rigorous negative result — the mechanism transfers to robotics but the specific fix does not fully close the gap in a low-data regime; still a defensible, honest RA-L contribution given the diagnostic rigor already in hand.
- **Failure mode**: λ sweep overfits to the small held-out split, e.g. picks degenerate λ→∞. **Detection**: check the recovered coupling isn't just approaching "condition-agnostic OT with each condition's own private batch" trivially (i.e., confirm λ is in an interior regime, not saturating). **Fallback**: report the sweep curve, not just the best point.
- **Failure mode**: CrackerBox's independent, apparently condition-unrelated collapse (all three generative methods underperform baseline there) confounds the pooled statistic. **Mitigation**: already flagged as likely a separate geometry-specific effect (thin/flat-object contact-feature floor effect, consistent with the paper's existing contact-feature diagnostic); report CrackerBox's result transparently but do not let it dominate the pooled claim — present per-object breakdown alongside the pooled number, as the paper already does elsewhere.

### Novelty and Elegance Argument
Closest work #1 (Cheng & Schwing, ICCV 2025, C²OT): identifies and fixes the identical mechanism, but only ever measures FID/likelihood on image benchmarks. We are — to our knowledge — first to show this failure mode causes physically-measurable task failure, and first to validate the fix on a downstream success-rate metric rather than a generation-quality proxy. Closest work #2 (Joyce et al., IROS 2025, consensus-driven grasp uncertainty): uses ensemble disagreement across *pose-estimation* outputs (perception uncertainty) to predict grasp failure; our supporting contribution uses ensemble disagreement across *generative-sampling* outputs (candidate-generation uncertainty) as a selection rule, a different uncertainty source entirely, explicitly scoped as secondary. The paper's elegance argument: one training-time mechanism-level fix (dominant) + one already-validated, free, orthogonal inference-time lever (supporting) — not a pile of unrelated modules.

## Claim-Driven Validation Sketch

### Claim 1 (dominant): Condition-agnostic OT coupling causes physical task failure in conditional grasp-pose flow matching, and condition-aware coupling fixes it.
- **Minimal experiment**: 4-way physical comparison (Baseline / Remove-OT / condition-agnostic OT-CFM / condition-aware OT-CFM) on the same 7 objects, n=50/condition, identical fixed-seeding harness already built this session.
- **Baselines/ablations**: Remove-OT and DDPM already serve as the "OT coupling is the specific culprit, not CFM itself" ablation (already collected, n=25 on 3 objects — can extend to n=50/7-objects if time permits). λ=0 (degenerates to original condition-agnostic OT-CFM) is the internal ablation confirming the fix's mechanism, not just "a different training run."
- **Metric**: physical grasp success rate (existing `physics_weld_after_bilateral` criterion), two-proportion z-test / Fisher exact, pooled and per-object.
- **Expected evidence**: condition-aware OT-CFM significantly closer to (or exceeding) baseline than condition-agnostic OT-CFM, at matched training data/architecture.

### Claim 2 (supporting): Consensus-based candidate selection is an orthogonal, training-free reliability improvement.
- **Minimal experiment**: already complete — ikmargin vs. consensus at matched ensemble size n=10, Pear and TomatoSoupCan.
- **Baselines/ablations**: ikmargin (lowest predicted IK error) as the alternative selection rule at matched pool size — already isolates the selection rule from ensemble size.
- **Metric**: Fisher's exact test on success/fail counts.
- **Expected evidence**: already have it — report as-is, framed as a secondary, complementary contribution.

## Experiment Handoff Inputs
- **Must-prove claims**: Claim 1's 4-way comparison is the load-bearing result; everything else in the paper is secondary to it now.
- **Must-run ablations**: λ=0 internal check; CrackerBox handled as a flagged exception, not swept under the rug.
- **Critical datasets/metrics**: existing 7-object YCB physical-execution dataset; physics_weld_after_bilateral success criterion; existing fixed-seeding harness (merged into main this session).
- **Highest-risk assumption**: that a condition-aware coupling retrain, with no other changes, is sufficient to recover baseline-beating performance in a ~400-examples-per-object regime. If it partially closes but doesn't fully recover, the paper's framing shifts from "we fixed it" to "we diagnosed it and partially mitigated it," which is a real risk to the "success condition" as currently defined.

## Compute & Timeline Estimate
- **Estimated GPU-hours**: training itself is minutes per checkpoint (small MLP, ~400 examples/object); the dominant cost is physical re-evaluation: 4 conditions × 7 objects × 50 seeds × up to 90s/trial ≈ up to 35 GPU-hours worst case for the full clean comparison (likely much less in practice, per this session's observed per-trial times).
- **Data/annotation cost**: none — reuses existing physical-execution training data and existing MuJoCo benchmark objects.
- **Timeline**: λ sweep + retrain: <1 day. Full clean 4-way re-evaluation: 1-2 days of (mostly unattended, background) compute given the 8-page RA-L deadline pressure already in play.
