# SO-101 jaw metrology — Step 4 findings (2026-08-07)

Read-only measurement of what the gripper actually does. Legacy behaviour frozen:
no change to `GRIP_CLOSED`, `GRIP_OPEN`, `move_gripper`'s map, the proxy-sphere
colliders, or the success rule. Runs with `enable_jaw_metrology=True` are directly
comparable to every result recorded before it existed.

Reproduce:

```bash
conda run -n tango python scripts/audit_jaw_opening.py
conda run -n tango python scripts/diagnose_jaw_opening_failures.py \
  --objects ScissorsC HammerC MediumClampC BananaC TomatoSoupCanC \
  --seeds 0 1 2 3 4 --out outputs/jaw_opening_diagnostic.jsonl
```

## Regression guarantee

With the flag off, four deterministic grasps (ScissorsC / TomatoSoupCanC × seeds 0,1)
produce byte-identical `last_grasp_metrics` key sets and values against a clean
worktree at `352e177`. Verified twice — once after the initial integration, once
after the measurement corrections below.

## 1. The commanded window is not the range the code claims

`move_gripper(opening_m)` asserts `angle = GRIP_CLOSED + (opening_m/0.10) * (GRIP_OPEN - GRIP_CLOSED)`
and `MujocoBackend.get_gripper_opening()` inverts it. Measured against geometry:

| hinge angle | true fingertip opening | proxy-sphere gap | code claims |
|---|---|---|---|
| joint lower limit (−10°) | 2.1 mm | 19.1 mm | −23.6 mm |
| `GRIP_CLOSED` = 0.05 rad (2.9°) | **19.4 mm** | 23.9 mm | **0.0 mm** |
| `GRIP_OPEN` = 1.00 rad (57.3°) | **70.9 mm** | 39.0 mm | **100.0 mm** |
| joint upper limit (100°) | 95.7 mm | 43.2 mm | 178.5 mm |

Two consequences:

- **The jaw cannot close below ~19.4 mm.** `GRIP_CLOSED` sits 13.6° above the joint's
  real lower limit, where the jaw would nearly close (2.1 mm). Nothing thinner than
  19.4 mm is pinchable however small the request. Someone already hit this: the legacy
  `Scissors` manifest entry carries `scale: [1.0, 1.5, 2.0] # Z×2 gives 4 cm thickness
  (gripper min 4 cm)` — the object was scaled up to work around the limit.
- **The linear map is off by ~19 mm at the closed end and ~29 mm at the open end**, and
  crosses zero error near the middle of the range. That is why the error stayed invisible
  on mid-sized objects and surfaces on thin ones.

The map error is a genuine API defect (units declared "metres", variable is radians,
the geometric relation is a hinge sine, not linear) and must be fixed — but see §3 for
why it was *not* the dominant failure mechanism in this sample.

## 2. The proxy colliders are embedded inside the objects, not touching them

`_simplify_jaw_collision` replaces each finger with one 6 mm sphere at the finger mesh's
frame origin. Over the commanded window the real fingertips travel 51.5 mm while the
proxy spheres travel 15.1 mm — a 3.4× sensitivity loss. But the larger problem is where
they end up.

Exact signed distance from each proxy sphere to the object collision geometry, minimum
over the close window (negative = sphere inside the object), against the sampled distance
of the real pad it stands in for:

| object | proxy fixed | proxy moving | real pad fixed | real pad moving |
|---|---|---|---|---|
| BananaC | −7.8 mm | −10.0 mm | +5.4 mm | +0.1 mm |
| HammerC | −11.8 mm | −4.5 mm | +5.8 mm | +0.3 mm |
| MediumClampC | −2.6 mm | −2.0 mm | +11.0 mm | +7.8 mm |
| ScissorsC | −4.5 mm | +1.4 mm | +9.8 mm | +4.0 mm |
| TomatoSoupCanC | −8.3 mm | −15.7 mm | +0.7 mm | +6.6 mm |

Worst single trial: TomatoSoupCanC seed 0, moving sphere −20.6 mm — a 6 mm sphere whose
centre sits ~26 mm inside the can.

The recurring pattern is that the **moving** pad genuinely reaches the surface (0.1–0.4 mm
on Banana/Hammer) while the **fixed** pad stays 5–11 mm clear. In real geometry these are
one-sided contacts; the buried spheres report them as bilateral, the weld fires, and
`success` follows.

## 3. Failure attribution over 25 trials

Straight-down grasp at the settled centroid, requested opening 0.065 m, so candidate
selection contributes no variance.

| object | n | success | A ctrl-bound | B proxy-FN | C proxy-embedded | D other | E approach-miss |
|---|---|---|---|---|---|---|---|
| BananaC | 5 | 5 | 0 | 0 | 4 | 1 | 0 |
| HammerC | 5 | 5 | 0 | 0 | 5 | 0 | 0 |
| MediumClampC | 5 | 4 | 0 | 0 | 3 | 1 | 1 |
| ScissorsC | 5 | 2 | 0 | 0 | 2 | 3 | 0 |
| TomatoSoupCanC | 5 | 0 | 0 | 1 | 4 | 0 | 0 |
| **total** | **25** | **16** | **0** | **1** | **18** | **5** | **1** |

**18/25 trials rest on collider interpenetration.** That includes 11 of the 16 successes.

**Bucket A never fired.** The 19.4 mm floor is real as a property of the gripper, but at
these grasp points every object was 31–71 mm across along the closing axis, so the floor
was not the binding constraint in this sample. It would bind on a thin edge grasp; this
protocol grasps at the centroid. Do not read the A column as evidence the floor is harmless.

## Two measurement corrections made during this work

Both were caught by adversarial checks against MuJoCo's own numbers, and both had already
produced wrong bucket counts before being fixed. Recorded here because the same traps
apply to anyone extending this tooling.

1. **Vertex sampling is not surface sampling.** The first pass measured distance from a
   point to the nearest mesh *vertex*. On CoACD parts with large flat triangles the nearest
   vertex can be tens of mm from a touching surface, so the metric reported 53–60 mm of
   separation where MuJoCo reported contact. Fixed by area-weighted sampling over mesh faces
   (`_mesh_geom_surface_local`).
2. **Each finger has a visual and a collision geom built from the same mesh at the same pose.**
   Matching on mesh name alone binds to the visual geom, which `_simplify_jaw_collision` never
   replaced — it is still the full 105 mm finger hull. Positions coincide, so origin-to-origin
   distances came out right, but `mj_geomDistance` then measured a mesh the solver does not
   collide and reported 43 mm penetrations against a real contact depth of 0.5 mm.
   `JawMetrology._find_geom` now prefers `contype != 0`.

After the fix, `mj_geomDistance` on the bound geoms reproduces MuJoCo's `contact.dist` to
within 0.03 mm; `tests/test_jaw_metrology.py` pins that agreement.

## What this implies for existing results

Every MuJoCo number that depends on `bilateral_contact`, `check_grasped`, `weld_triggered`,
`lifted`, retention, or recovery triggers is computed from the buried-sphere contact signal.
That covers the recovery work, the temporal contact/lift/retention features, and the visual
critic's execution outcome labels.

This is not a claim those results are wrong — the labels may be mostly stable under a
corrected collider. It is a claim they are **unverified**, and cheaply checkable: re-run the
affected experiments after step 3 and compare label flip rates. Small flip rate → re-run the
main tables. Large → the data-dependent models need rebuilding.

## Next (agreed order: 4 → 3 → 1 → 2)

- **3 — fix the collider.** Keep the visual mesh; `<contact><exclude>` the finger pair (their
  convex hulls overlap 13.6 mm at the hinge at *every* joint angle, which is why the sphere
  hack exists); add symmetric box/capsule pads on the inner faces, sized from mesh geometry
  and rigged to follow the real fingertips.
- **1 — calibrate the opening.** Replace the linear map with the measured monotone curve
  (`JawMetrology.true_opening_m` is already a LUT); make `get_gripper_opening()` return the
  real fingertip separation.
- **2 — reconsider the closed limit** only after 3 and 1, since dropping `GRIP_CLOSED` toward
  the joint limit moves the jaw into a region the collider has never described correctly.
