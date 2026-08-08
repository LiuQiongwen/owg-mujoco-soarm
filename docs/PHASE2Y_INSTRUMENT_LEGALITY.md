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

## Implementation note for P2Y-1 (variant generation point)

Variants must be generated at the **gripper XML** level, not on the composed
model. robosuite composes the final scene at runtime from the robot,
gripper, mount and arena; there is no single on-disk XML to edit.

`PiperGripper` (`tango_robot/piper_robosuite/piper_gripper.py`) is a
`GripperModel` that loads `tango_robot/piper_assets/piper_gripper.xml`. The
variant path is therefore:

1. copy `piper_gripper.xml` into a diagnostic directory, once per dY level;
2. edit only the `pos` of `finger7_collision` / `finger8_collision` along
   the gripper's local Y;
3. subclass `PiperGripper` per variant, pointing at the variant XML;
4. register each under a distinct name and select it via `robots=`/gripper
   config, leaving `tango_robot/piper_robosuite/` and
   `tango_robot/piper_assets/` untouched (zero production diff, as
   throughout this investigation).

Note the collision geoms are `<mesh>` geoms, so shifting `pos` moves the
mesh instance without altering the mesh asset — the mesh is shared across
variants, which is what makes the P2Y-2 diff meaningful.

Two things to verify at step 3, since both have bitten already: the geoms
carry `contype/conaffinity` (R7 — do not trust the `_collision` name), and
the shift must be applied along the axis that maps to eef-local Y, which
`eef_local_axis_in_body()` in `piper_phase2y_smoke.py` already computes and
which Gate 2 verified to 0.000mm.

## P2Y-3 blocker: `run_pick_and_place` state is not snapshottable

The branching design assumes the pre-treatment state can be saved and
restored. For MuJoCo that is true (`mjSTATE_INTEGRATION`). For robosuite
controller state it is awkward but tractable. **For the pipeline itself it
is not possible as currently written.**

`run_pick_and_place` is a single ~400-line function. Its execution state at
`descend_refresh` — current phase, `qpos_seed`, `grasp_mat` (reassigned at
the pre-close refresh), `retry_count`, `descend_gripper_action`, the
`while True` retry loop position — exists only as **local variables in that
function's stack frame**. No state API can capture it, and there is no
re-entry point: the function runs start to finish or not at all.

So "snapshot at descend_refresh, branch five treatments" cannot be built on
`run_pick_and_place` without one of:

- **(a)** restructuring it into resumable phase steps — a substantial
  production change, and one this investigation has explicitly avoided;
- **(b)** re-running the prefix per branch — which reintroduces the
  from-episode-start bifurcation the branching design exists to remove,
  making it pointless;
- **(c)** snapshotting MuJoCo + controller state at `descend_refresh`, then
  driving the remaining phases (close → lift → verify) from a **separate
  minimal driver** rather than resuming `run_pick_and_place`.

**(c) is the viable route**, and it is narrower than it sounds: the segment
after `descend_refresh` is close, lift, and success check. It needs no IK
and no candidate logic — the descend target is already reached and the
gripper command sequence is fixed. A short driver reproducing just that
segment is far less code than restructuring the pipeline, and it keeps
production untouched.

Consequence for the sweep: what gets compared across treatments is the
**post-snapshot segment**, not the full episode. That is arguably the more
honest comparison anyway, since the prefix is by construction identical.
But it must be stated explicitly — the resulting success rates are
"success of the closing segment from a common state", not end-to-end
episode success, and they are therefore **not** directly comparable to P2's
numbers without care.
