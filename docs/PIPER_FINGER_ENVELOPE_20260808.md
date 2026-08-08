Finger-envelope probe — placement varies, but not consistently (2026-08-08)

180 trials, pre-close sampling (at `descend_refresh`, upstream of closure).

| object | offset −15 → +15 | obj longitudinal (mm) | envelope fraction | first contact step L/R | success |
|---|---|---|---|---|---|
| pear | | 24.6 → **20.3** (decreasing) | 1.26 → 1.20 | ~390–406 (flat) | 3 → 12 |
| mustard | | 35.4 → **37.5** (increasing) | — | ~344–391 (flat) | 6 → 9 |
| cracker | | — → 38.4 | — | ~378 (flat) | flat |

## Not a consistent mediator

Longitudinal placement *does* move under the intervention — unlike the
geometry, IK and tracking terms, which were flat. But it moves in
**opposite directions** on the two objects whose success improves: pear's
object slides 4mm *toward* the finger root as success rises, mustard's
slides 2mm *away* as success rises.

A mediator must move consistently with respect to the outcome. It does not.
Nor do the contact-sequence terms: first-contact step is flat (~390 for
pear, ~344–391 for mustard, no trend), and first-contact longitudinal
position varies by only 1–5mm without a consistent sign.

**Excluded, by the same rule as the other five.**

## The one solid new fact: the object is never inside the envelope

`envelope_fraction` is **1.20–1.28 in every pear condition** — the object's
centre sits 15–21mm *beyond* the tip end of the finger contact envelope
([−71.5, +5.0]mm in eef-local Z), in all five offsets, successes and
failures alike.

So Piper never centres the object in its jaws. It grasps with the *edge* of
the finger contact region, on whatever part of the object protrudes into
it. This is a static property of the current targeting, unchanged by the
intervention, and therefore not the P2 mediator — but it is a real
characterisation of how this gripper actually engages objects, and it is
the kind of thing a candidate critic would want to know.

## Standing state: six families excluded, effect unexplained

| candidate mediator | verdict |
|---|---|
| contact-local geometry (width, antipodal, centring) | constant → excluded |
| IK solvability (position + orientation residual) | flat to microns → excluded |
| joint-limit margin | flat → excluded |
| joint tracking error | real but flat → excluded |
| frame / phase-timing semantics | ~0 µm, arm settled → excluded |
| finger-envelope placement + contact sequence | moves, but inconsistently → excluded |

The P2 aim-offset effect is **robustly real** (pear 50/0 concordant
seed-pairs, p=8.9e-16; mustard p=1.3e-3; replicated at n=180) and
**mechanistically unexplained by every quantity this pipeline can
currently measure**.

That is the honest state, and it is worth recording as a result rather than
a gap. Scope of the claim, stated precisely: **this audited set of static
features is insufficient to explain or separate the effect.** That is not
the same as "no static feature could" — an unexamined representation
(learned local geometry, richer contact descriptors) might still carry the
signal. What is demonstrated is that the geometric, kinematic and contact
quantities audited here do not, which P1.2 found independently (no promoted
pre-execution separator; first separation only at descend).

## What would be worth trying next, if resumed

- The effect is object-specific (strong pear, moderate mustard, null
  cracker), so a per-object characterisation may be more tractable than a
  general mechanism.
- Nothing measured so far distinguishes *why* pear's optimum is at +15mm.
  A wider sweep (±30–45mm) would show whether pear's success curve has an
  interior optimum or is still rising at the edge of the tested range —
  cheap, and it constrains the space of possible explanations.
- Per R5, any follow-up needs ordered paired designs, not n=12 cells.
