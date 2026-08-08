Phase 2Y — instrument legality problem (2026-08-08)

## The problem

The finger-shift instrument mutates `model.geom_pos` after model compile:

```
scripts/piper_phase2y_smoke.py:74     m.geom_pos[g] = m.geom_pos[g] + ax * (dY/1000)
scripts/piper_phase2y_qualify.py:115  (same)
```

MuJoCo documents `geom_pos` among fields that are **not** safe to modify at
runtime, since compile-time collision-acceleration structures derive from
them. (Not independently verified here — no network access this session —
but the empirical test below does not depend on the documentation.)

If that applies, Gate 3's pre-contact divergence is an **artifact of an
illegal model mutation**, not a physical effect. This is consistent with
what was actually observed and previously unexplained: Gate 4 showed *no*
new contacts and `delta_min_distance ≈ 0`, yet the EEF trajectory still
diverged before object contact. An inconsistent collision pipeline would
produce exactly that signature.

## Consequence for the current plan

The n=40 baseline recalibration (running, → `calib/phase2y_noise_floor_n40.json`)
is **demoted**. It remains useful as a characterisation of Piper rollout
reproducibility, but it is no longer the route to resolving Gate 3:
enlarging the noise envelope from 1.6e-3 to ~3e-3 would have *hidden* an
instrument defect rather than exposing it.

## Revised design

1. **P2Y-0** — decisive legality test: build one compile-time XML variant
   with finger collision geoms shifted, and compare against the
   runtime-mutated model from an identical state. Divergence ⇒ runtime
   mutation unsafe, confirmed without relying on documentation.
2. **P2Y-1** — generate 5 compile-time variants (−15, −7.5, 0, +7.5, +15mm).
3. **P2Y-2** — diff variants against baseline on everything that is *not*
   the treatment: `nq/nv/nu`, `qpos0`, joint limits, `body_mass`,
   `body_inertia`, `body_ipos/iquat`, actuator gain/bias/damping/armature,
   geom type/size/friction/solref/solimp/contype/conaffinity. Only the two
   finger geoms' local-Y pose may differ. This matters because MuJoCo can
   derive body inertia from geoms at compile time — a shifted geom could
   silently change the finger body's inertial properties, converting a
   contact-geometry treatment into a dynamics treatment.
4. **P2Y-3..5** — exact-state branching: run the common prefix once, save
   full `mjSTATE_INTEGRATION` at `descend_refresh`, then branch each
   treatment from that identical state. This removes the from-episode-start
   bifurcation entirely, so Gate 3 no longer needs a statistical envelope.
5. **P2Y-6..7** — 5-level branching sweep, then first-divergence localisation
   (per-step `Δqpos/Δqvel/Δctrl/ncon/contact pairs/nefc`).

## Why branching is the stronger design regardless

Each seed's five treatments would share a **bit-identical pre-treatment
state**, which is a stronger match than paired seeds. It is also cheaper:
40 prefixes + 200 short branches, instead of 200 full approach/transit
episodes.

Caveat to check at P2Y-4: if two `dY=0` branches from the same saved state
still diverge, the incomplete state is **not** in MuJoCo — it is in
robosuite/controller Python state (interpolator, phase, RNG). That would be
a valuable localisation in its own right.

## Status

Gate 1 PASS · Gate 2 PASS · Gate 4 PASS · **Gate 3 suspended pending
instrument legality (P2Y-0)**. Full sweep blocked.
