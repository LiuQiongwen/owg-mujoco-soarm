# Discussion Notes — OWG Stage-4 LGGSN Experiments

> Notes for the Experiments / Discussion section of the paper.
> All numbers verified against `logs/` via `scripts/paired_stats.py` and
> `scripts/summarize_score_delta_gate.py` (2026-03-26).

---

## Paper-ready summary (English, ~300 words)

**Results.**
We evaluate the LGGSN reranker (Stage 4) against the GR-ConvNet baseline (Stage 3)
in 150 paired trials (25 seeds × 6 YCB objects) under identical PyBullet scenes.
The v2 model with context features (`dist_to_centroid`, `z_rel`) achieves an
aggregate net improvement of **0** (13 improvements, 13 regressions).
Object-level effects are strongly heterogeneous:
CrackerBox gains +4 trials (52%→68% success rate), MustardBottle gains +2 (84%→92%),
while Scissors loses −4 (80%→64%) and PowerDrill/TomatoSoupCan each lose −1.
McNemar's test yields p ≥ 0.12 for all objects at N=25, confirming that
no individual effect clears statistical significance at this sample size.
Bootstrap power analysis shows that N=25 provides 93.9% power for CrackerBox
(the largest observed effect, δ≈+0.33 Cohen's h) but only 13.5% power for
Scissors (δ≈−0.36)—two objects with nearly equal effect magnitudes but
opposite signs require N ≥ 150 seeds each for reliable detection.

**Failure analysis.**
Post-hoc geometric analysis reveals two orthogonal failure modes.
For *Scissors* (`flat_frac = 0.457`, σ_H ≈ 0.001), the H feature saturates
at zero for nearly half of all candidates; the resulting near-identical LGGSN
scores (max–min spread < 0.002 in 84% of episodes) reduce selection to an
implicit `dist_to_centroid` ordering, which biases towards the blade tip—
a geometrically poor grasp region.
For *PowerDrill* (σ_yaw = 0.195 rad, highest among all objects), world-frame
yaw is ambiguous across multiple valid and invalid approach orientations;
without an object-frame encoding, LGGSN cannot resolve this ambiguity.

**Experiment 3: Score-delta gate.**
We investigate whether gating on model-output uncertainty
(score spread = max−min over candidates < δ) can suppress harmful reranking.
Retrospective simulation on the 25-seed log shows that LGGSN scores are
uniformly near-saturated across *all* objects (54% of episodes have
spread < 0.0001; max observed spread = 0.0095).
At the empirically optimal δ = 0.0005, the gate fires on 80% of episodes,
yielding total net = +3—a marginal improvement driven entirely by partially
suppressing the Scissors regression (+1 over 5 saved trials).
However, this comes at the cost of also suppressing 4 valid CrackerBox gains.
The 95% bootstrap CI at δ = 0.0005 spans [−2, +5] for Scissors and
[+0, +5] for MustardBottle, indicating that the gate's benefit is not
distinguishable from sampling noise at N=25.
The fundamental limitation is that score saturation is not object-selective:
the spread distribution is statistically indistinguishable across all six objects
(Kruskal-Wallis p > 0.3), so no threshold simultaneously protects Scissors
without suppressing CrackerBox.

**Ablation.**
At N=10 seeds, ablation condition B (`dist_to_centroid` only, val acc = 0.558)
shows net = +7 vs. D (12-dim baseline, net = −1).
However, this is entirely within bootstrap noise: the 95% CI for condition B's
pooled net spans [+2, +13], while condition D spans [−5, +3], and the
distributions overlap substantially.
The 25-seed primary run (condition A, both features) converges to net = 0,
consistent with regression-to-the-mean from an initial N=10 result of +4.

---

## Key claims for the paper

| Claim | Evidence | Caveat |
|-------|----------|--------|
| LGGSN consistently helps CrackerBox | +4 net across 25 seeds, power 93.9% | Single object, single eval set |
| Global reranking is net-neutral | aggregate net = 0 / 150 trials | Gains cancel losses; no stat sig |
| Score saturation explains Scissors failure | 84% episodes spread < 1e-3; flat_frac=0.457 | Retrospective; no ablation of H feature alone |
| Score-delta gate cannot selectively protect Scissors | Spread distribution not object-discriminative (p>0.3) | N=25 power insufficient for definitive conclusion |
| N=25 insufficient for most objects | Power < 20% for PowerDrill, Scissors, TomatoSoupCan | Requires N≥150 for these objects |

---

## Suggested framing for Discussion

> We find that LGGSN-based reranking improves grasp success for objects with
> stable geometry and non-trivial height variation (CrackerBox, MustardBottle),
> but introduces consistent regressions on flat-featured objects (Scissors)
> and orientation-ambiguous objects (PowerDrill). These two failure modes
> have distinct root causes—feature saturation and frame ambiguity—and
> cannot be jointly addressed by input-geometry or model-output gating strategies
> at the current sample size.
> Our analysis suggests that per-object adaptivity (e.g., learning to abstain
> on object classes identified as score-saturating at training time) is a more
> promising direction than post-hoc gating.

---

## Statistical methodology note

All CIs are **paired bootstrap 95%** with N=10,000 resamples on the net-improvement
statistic (improvements − regressions). Significance tests use **McNemar's exact
binomial** test for N_disc < 25 and McNemar chi-squared with continuity correction
otherwise. No multiplicity correction is applied (exploratory analysis).
Effect sizes are **Cohen's h** on paired success proportions.
