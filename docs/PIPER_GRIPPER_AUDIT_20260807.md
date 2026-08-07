# Piper gripper audit — read-only, first pass (2026-08-07)

Scope, as agreed: **audit only, no modification to anything under
`tango_robot/piper_robosuite/` or `tango_robot/piper_assets/`.** Confirmed by
`git status`/`git diff` on both directories before and after — zero diff.
Both new scripts either load `piper_gripper.xml` directly (read-only) or
build a standalone throwaway scene that includes it verbatim.

Reproduce:

```bash
conda run -n tango python scripts/audit_piper_gripper.py
conda run -n tango python scripts/microbenchmark_piper_blocked_closure.py
```

Before reading the table: `piper_robosuite/` already has substantial prior
audit work of its own (see `PIPER_FINDINGS_SUMMARY.md` and this session's
reading of `piper_gripper.py`/`piper_controller_config.py`/
`piper_real_backend.py`) — several defects in the same *class* as this
thread's SO-101 findings were already found and fixed there (a
finger-open/closed sign convention that was backwards, an opening-range
miscalibration vs. AgileX's official spec, a documented QACC/NaN at exact
full closure fixed by excluding finger-finger self-contact and a hard floor
ctrl limit rather than by softening the solver, and a double-scaling bug in
robosuite's action pipeline that silently reduced the entire commanded travel
to ~0.1mm for an unknown period). This audit does not re-derive any of that;
it answers the six specific items requested, using what's already documented
where applicable.

## Results table

| check | verdict | detail |
|---|---|---|
| TCP / grasp reference | **FAIL** | `robot0_eef_site` (what IK targets) is **65.6mm** from the true fingertip midpoint, constant across the whole opening range. No compensating offset found anywhere in `piper_pick_and_place.py`. |
| Pad geometry | **PARTIAL** | `finger7/8_collision` are the full finger mesh (109.5×57×24.5mm), not a localized pad — same class of risk SO-101 had pre-step-3. Blocked-closure test (below) shows no ill effect *in this one test*, but the geometry itself doesn't rule out off-tip contact on other objects/poses. |
| Opening semantics | **PASS** | True opening is close to exactly linear in commanded qpos (parallel-slide kinematics, unlike SO-101's hinge) — 14mm to 104mm over the actuator's own ctrlrange. Matches the real hardware's own measured 120mm full opening to within ~13%, a plausible measurement/convention gap, not a red flag. |
| Left/right symmetry | **PASS** | 0.000mm symmetry error at half-open; both fingers receive an identical commanded target by construction (`PiperGripper.format_action`), and blocked-closure penetration was symmetric to 2 decimal places (−1.79 / −1.79mm). |
| Bilateral contact / blocked penetration | **PASS** | 30mm box, isolated closure (no arm/IK): steady penetration −1.79/−1.79mm, zero solver warnings. Substantially better out of the box than SO-101's legacy proxy-sphere collider ever was, and comparable to SO-101's *tuned* S1 result — Piper's contact was already configured aggressively (`solref="0.002 1"`, right at the stability floor) specifically because the finger-finger QACC/NaN issue was already found and fixed geometrically. |
| Success semantics / weld decoupling | **PASS** | `Lift._check_success` (inherited unmodified by `PiperLiftYCB`) is purely `object_height > table + 4cm` — no weld, no kinematic attach, no `demo_attach`-style shortcut anywhere in the inherited task. A "success" here requires the object to be genuinely held up by real contact friction against gravity. Structurally safer than SO-101's history, where `GRASP_MODE_DEMO_ATTACH` existed as a documented contamination risk. |
| Real-backend mapping | **PASS (documented, not independently re-verified)** | `piper_real_backend.py`'s header already documents a verified `piper_sdk` unit mapping (joints: 0.001°, gripper: 0.001mm) and a resolved sim-ctrl-to-metres conversion. Not re-checked in this pass; taken on the existing documentation's word since re-deriving it wasn't part of the requested scope. |

## The one real finding: 65.6mm TCP/grasp-reference offset

This is the Piper-side analogue of this thread's step-3 SO-101 finding (a
52-57mm gap between the legacy IK target and the actual pad midpoint) —
same defect class, different magnitude, found by the same method (measure
the IK target site's position against the true fingertip midpoint directly,
don't assume they coincide).

`grip_site`/`robot0_eef_site` sits at the gripper module's root body origin
(the arm-attachment point). The finger bodies (`link7`/`link8`) are also
positioned at that same origin, but their MESHES extend outward from it —
the true fingertip midpoint is **65.6mm along the gripper's own Z-axis** from
that site, constant regardless of opening (a genuine constant, unlike
SO-101's angle-dependent case, since this is a parallel not a hinge
mechanism — a single fixed correction would fully address it).

Checked directly, not assumed: grepped `piper_pick_and_place.py` for any
sign of a compensating offset (`grasp_offset`, `z_offset` used near a grasp
target, etc.) — found none. Every place that reads `eef_site_id`'s position
uses it directly as "where the gripper is" for IK targeting.

**This has not been shown to explain any specific observed failure in
`PIPER_FINDINGS_SUMMARY.md`'s existing investigation** (the wrist-fix/CR-CFM
work was about joint6 orientation, not this offset) — it is a newly found,
independently-confirmed structural defect, reported here because the audit
was scoped to find exactly this class of thing, not because it's been tied
to a specific downstream failure yet.

## What this changes for the earlier migration proposal

Per the requested scope, this was audit-only — no fix applied, no decision
made about promoting Piper to the primary platform. The decision tree from
the framing message applies directly to what was found:

- Pad geometry, opening, symmetry, contact penetration, and success
  semantics: **already reasonable or already fixed**. Not "SO-101-grade
  problems requiring the full multi-week audit sequence."
- TCP/grasp reference: **one clear, real, previously-undiscovered defect**,
  same class as SO-101's worst finding, but simpler to fix (constant offset,
  not pose-dependent) once someone decides to.

This matches the "少数小问题" branch of the decision tree, not the "又出现
类似 SO-101 的基础语义错误" branch that would call for a much larger
re-audit — but the TCP offset is not a "小问题" in isolation; it's exactly
the kind of error that silently determines where every candidate grasp in
this platform's history was actually aimed.

## Not done in this pass

- Independent re-verification of the real-hardware unit mapping (taken from
  existing documentation).
- Any check inside actual robosuite task execution (candidate placement,
  controller wiring, full pick-and-place) — this audit deliberately stayed
  at the MJCF/kinematics level to avoid depending on robosuite version/API
  specifics for checks that are fundamentally about mesh geometry.
- Any fix. Per the agreed scope, this is a baseline table, not a repair
  pass.
