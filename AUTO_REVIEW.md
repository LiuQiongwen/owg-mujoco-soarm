# Auto Review Log — LGGSN Pairwise Reranker

**Paper**: "When Geometry Is Not Enough: Object-Dependent Failure Modes of Learning-to-Rank Grasp Selection in Open-World Manipulation"
**Target venue**: ICRA 2027 (double-blind, Seoul, May 2027)
**Started**: 2026-06-03
**Max rounds**: 4

---

## Round 1 (2026-06-03)

### Assessment Summary
- **Score: 5.5 / 10**
- **Verdict: NOT READY**
- Reviewer model: Claude Sonnet 4.6 (direct, xhigh effort)

### Key Criticisms
1. **[W1 - Severe]** Related work missing 2023-2025 language-conditioned grasping literature (RT-2, OpenVLA, GraspGPT, SpatialVLM, AnyGrasp, etc.) — only 10 citations total
2. **[W2 - Severe]** Cross-platform results contradict (CrackerBox: MuJoCo −2 vs PyBullet +4; PowerDrill: +4 vs −1) with no explanation — undermines diagnostic framework credibility
3. **[W3 - Medium]** Scissors floor effect (0/25 both stages) counted in totals but failure mode analysis attributed to reranking — conflates grasping failure with reranker failure
4. **[W4 - Medium]** Feature ablation at N=10 is underpowered (±5–6 trial CIs); results presented as conclusions without caveat
5. **[W5 - Medium]** GC-LGGSN evaluated on only 2 objects — "gate is insufficient" conclusion is over-generalised
6. **[W6 - Minor]** Abstract buries the diagnostic framework value; last two sentences weak

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score: 5.5/10 — NOT READY for ICRA 2027**

**Strengths:**
The paper makes a valuable contribution as a rigorous negative-result analysis. The paired evaluation design is methodologically sound. The flat_frac predictor (ρ=−0.96) is a genuine finding with practical implications. Statistical power analysis is refreshingly honest and rare in robotics papers.

**Critical Weaknesses:**

W1 — Related work (severity: HIGH): The paper cites only 10 works, with no coverage of 2023-2025 language-conditioned grasping. For ICRA 2027, reviewers will immediately flag missing: GraspGPT (RA-L 2023), AnyGrasp (IEEE T-RO 2023), RT-2 (CoRL 2023), OpenVLA (arXiv 2024), SpatialVLM (CVPR 2024), language-driven grasp detection (CVPR 2024). The paper positions itself as addressing language-conditioned open-world settings but doesn't engage with the contemporary literature in this space.

W2 — Cross-platform divergence (severity: HIGH): CrackerBox is −2 in MuJoCo but +4 in PyBullet. PowerDrill is +4 in MuJoCo but −1 in PyBullet. These sign reversals are unexplained. If the diagnostic framework (flat_frac, σ_H) is truly predictive, why does the same geometry produce opposite outcomes on different platforms? This needs a dedicated discussion section, not just tables side-by-side.

W3 — Scissors floor effect (severity: MEDIUM): 0/25 on both stages means there's no task signal from Scissors in the MuJoCo primary evaluation. The feature saturation analysis is valid (score spread data supports it), but the paper conflates "LGGSN causes regressions on Scissors" (which requires observed task failure attributable to reranking) with "LGGSN scores would degrade grasps if they succeeded" (which is the actual claim supported by the data). The PyBullet −4 result does support the causal claim, but this should be made explicit.

W4 — Ablation power (severity: MEDIUM): N=10 × 6 = 60 trials per ablation condition. Bootstrap 95% CI ≈ ±5. The net difference between Cond A (+4) and Cond D (−1) is 5 trials — exactly at the CI width. The paper should note this explicitly rather than presenting ablation numbers as definitive feature contributions.

W5 — GC-LGGSN scope (severity: MEDIUM): "A 302-parameter context gate is insufficient to resolve the underlying feature-space conflict" — this conclusion from 2 objects is too strong. The 2-object choice is reasonable for probing the hardest case, but the conclusion should be scoped accordingly.

W6 — Abstract (severity: LOW): The final two sentences ("Our analysis establishes when geometry-only reranking helps or harms, and identifies concrete requirements for successor methods") are weak. The concrete requirements should be named in the abstract, and the ρ=−0.96 predictor should be framed as the main positive contribution.

**Verdict:** NOT READY. The paper's core analysis is sound and the negative result is valuable, but W1 and W2 need to be addressed before submission. W1 is a writing fix; W2 requires either additional analysis or a dedicated discussion section explaining the platform divergence.

</details>

### Actions Taken
- **W1**: Extended Related Work with 8 new 2023–2025 citations (AnyGrasp, RT-2, OpenVLA, GraspGPT, SpatialVLM, GNFactor, Language-Driven Grasp Detection, RoboAgent key fix); rewrote §2.4 as "Language-Conditioned and Open-World Grasping"
- **W2**: Added new §5.1 "Cross-Platform Divergence" explaining contact-model, depth-map, and kinematics differences between MuJoCo/SO-ARM101 and PyBullet/Panda
- **W3**: Rewrote Scissors analysis paragraph to explicitly distinguish floor-effect from reranking failure; grounded causal claim in PyBullet data
- **W4**: Added "Statistical caveat" paragraph to §4.4 ablation section with explicit CI width and power requirement
- **W5**: Softened GC-LGGSN conclusion to scope "2-object hardest-case probe" framing
- **W6**: Rewrote Abstract final 2 sentences to surface the 3 design requirements and ρ=−0.96 predictor

### Results
- LaTeX compiles cleanly (exit 0, no undefined references after bibitem key fix)
- 6 targeted fixes applied, no new experiments required
- Word count increased ~250 words (Related Work +180, Discussion +120, Ablation caveat +70, Abstract −20)

### Status
Proceeding to Round 2

---

## Round 2 (2026-06-03)

### Assessment Summary
- **Score: 6.5 / 10**
- **Verdict: ALMOST**
- Reviewer: Claude Sonnet 4.6 (direct)

### Key Criticisms
1. **[W2']** Cross-platform paragraph used "We attribute" (causal claim) without ablation data
2. **[W7]** 193:39 positive/negative imbalance unreported — BPR pair quality unaddressed
3. **[W8]** Pear absent from PyBullet table with no explanation
4. **[W9]** Conclusion future-work directions lacked concrete technical specifics

### Actions Taken
- **W2'**: Changed "We attribute" → "A plausible explanation involves … post-hoc analysis"
- **W7**: Added 2-sentence data imbalance note to §3.4 Dataset construction
- **W8**: Added Pear exclusion note to Table 2 caption
- **W9**: Added concrete technical hints to each of 3 successor requirements (superquadric fitting, conformal prediction, contact-normal feature)

### Results
- LaTeX compiles cleanly (exit 0)
- Score: 5.5 → **6.5**
- All text-only fixes, no new experiments required

### Status
Score ≥ 6 + verdict ALMOST → **POSITIVE_THRESHOLD MET — loop terminates**

---
