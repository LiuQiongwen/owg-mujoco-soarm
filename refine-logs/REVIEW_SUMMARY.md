# Review Summary — Method Refinement History

**Date**: 2026-05-15

---

## V1 Starting Point (raw direction)

"World model alignment + SO-ARM real robot migration"

**Weaknesses identified:**
- Too broad: "world model alignment" could mean visual domain adaptation, physical parameter
  estimation, neural world model (NeRF), or feature normalization — needs scoping.
- SO-ARM migration is engineering, not a research contribution, unless it validates a
  transferability claim.
- No connection to the existing LGGSN net=0 diagnosis: ignoring this wastes the most
  valuable piece of evidence already in hand.

---

## V2 Direction (focused refinement)

"Object-centric feature normalization fixes LGGSN failure modes + enables geometry-grounded
real-robot transfer"

**Strengths:**
- Uses existing failure analysis (H saturation, yaw ambiguity) as the Problem Anchor.
- ORFN is a deterministic feature transform — no retraining, no data collection needed.
- Geometric features are inherently sim-to-real transferable (no visual domain gap).
- SO-ARM migration becomes the transfer validation experiment.

**Weaknesses:**
- obj_principal_yaw estimation (PCA on N~5 candidates) may be noisy.
- H_rel requires object height estimate — need a robust real-RGBD estimator.
- Risk: if ORFN helps in sim but real GR-ConvNet generates different candidates,
  the real-robot experiment could fail for an independent reason (GR-ConvNet domain gap).

**Decision**: Accept V2 direction with explicit risk table. Do not add visual adaptation
(rejected: breaks geometric transferability and adds visual domain gap problem on top).

---

## V3 Final (accepted)

ORFN as dominant contribution. PCA-based principal axis estimation is O(N), zero-cost.
H_rel computed from within-episode z_min/z_max (already in feature set). Risk of
GR-ConvNet domain gap mitigated by using depth-only input channel.

**Complexity explicitly rejected:**
- Per-object model heads (data cost too high, no generalization)
- Visual embedding (breaks transferability)
- GC-LGGSN (empirically failed, documented in EXPERIMENTS.md §4.5)
- Score-delta gate (non-discriminative across objects, KW p>0.3)

---

## Key Reviewer Concerns to Address in Validation

| Concern | Addressed by |
|---------|-------------|
| Is net=0 improvement just sampling noise? | 25-seed paired eval with McNemar test (already done); ORFN should show ≥+3 net with p<0.10 at N=25 |
| Does ORFN actually fix Scissors specifically? | Per-object breakdown required (not just aggregate net) |
| Is SO-ARM transfer zero-shot or does it use real data? | Must be zero-shot (sim-trained weights, no fine-tuning) |
| Why not just collect real data and retrain? | Address in Discussion: data efficiency argument (ORFN achieves transfer with 0 real trials) |
