# Refinement Report

**Date**: 2026-05-15

## Starting Direction
"世界模型对齐 + SO-ARM 真机迁移" — broad, multiple interpretations

## Triage Decision
LGGSN net=0 failure analysis is the strongest anchor. All refinement must connect back
to the two diagnosed root causes (H saturation, yaw ambiguity). "World model alignment"
is scoped to feature-level geometric distribution matching — NOT visual domain adaptation.

## Key Narrowing Decisions

| Decision | Rationale |
|----------|-----------|
| ORFN over visual adaptation | LGGSN features are geometric; visual gap is a separate, harder problem |
| ORFN over per-object models | Data cost prohibitive (need 6× more per-class data); no generalization |
| Feature-level alignment (Exp 3) over physics param estimation | Physics params only affect collision; LGGSN runs after candidate generation |
| SO-ARM as transfer testbed, not new domain | Avoids scope creep; validates the geometric transferability hypothesis directly |
| Zero-shot transfer (no real fine-tuning) | Maximum efficiency claim; distinguishes from trivial "collect real data and retrain" |

## Final Thesis Stability
ORFN is stable because:
1. It is a deterministic remap — no hyperparameters to tune
2. Both components (yaw_obj, H_rel) are motivated by explicit failure modes in logged data
3. The sim result (net=0 → net≥+3) is testable in <8h compute before any real hardware
4. If sim fails: back to drawing board before wasting robot time

## What Would Change the Method
- If PCA is unstable (std > 30°): remove yaw_obj, keep only H_rel → weakens C1 (Scissors fix)
- If H_rel is sufficient alone (ablation A2 shows net≥+3): simplify to ORFN-H, drop PCA entirely
- If GR-ConvNet domain gap is catastrophic (Exp 3 fails): real-robot claim becomes "feature alignment pending better perception"; paper focuses on sim contribution only
