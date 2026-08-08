# Diagnostic-script sanity rules (Piper investigation, 2026-08)

Rules promoted from repeated, concrete failures in this codebase. Each has
already prevented at least one wrong conclusion.

## R1 — Suspicious-value rule

**If a measured quantity comes out suspiciously close to an input parameter,
to zero, to a geometric constant, or to an obvious physical bound, assume
the instrumentation or reference frame is wrong before explaining it as
physics.**

Caught three times in one investigation:

| observation | looked like | actually was |
|---|---|---|
| descend "tracking error" ≈ 15.0mm at a ±15mm offset | controller undershoot | metric differenced against the *unoffset* target — it was measuring the injected offset |
| pear `local_width` = 0.0 in every trial | pear ungraspable / degenerate | z-band defined around the eef target, which sits below pear's mesh |
| 0.02mm "width" on a 66mm pear | near-zero support | took the *innermost* mesh surface per side; a closed mesh always has vertices near x≈0 |

## R2 — Measure before the intervention's own effect normalises it

Sampling a relative pose *after* the gripper closes measures the grasp, not
the aim: closure pulls the object into the jaws and flattens the very
variation being tested. Observed: object-relative lateral/longitudinal
placement read 1.20mm / 20.1mm at both −15mm and +15mm offsets, while
pre-close first-contact terms over the same trials varied by 4–6mm.

Sample at the last state *upstream* of the mechanism under test.

## R3 — A constant cannot mediate a varying effect

Before proposing any mediator, check it actually moves under the
intervention. This single check eliminated four candidate families for the
P2 effect (contact-local geometry, IK solvability/joint margin, joint
tracking error, frame/phase-timing semantics).

## R4 — Decompose along named axes, not norms

A Euclidean distance hides which direction changed. The finger contact
envelope is asymmetric ([−71.5, +5.0] mm along eef-local Z), so
longitudinal placement is a distinct question from closing-axis placement
and is invisible in a norm.

## R5 — n=12 per cell is not a confirmatory sample here

Established in P1.1: under an i.i.d. rate of 0.825, observing 4/12 has
probability ~1e-4, yet that occurred — outcome is strongly driven by the
spawn draw. Small paired designs across *ordered* levels (P2's 5-level
sweep, 50/0 concordant) remain valid; single n=12 comparisons do not.
