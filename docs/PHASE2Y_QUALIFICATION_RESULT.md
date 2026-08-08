Phase 2Y instrument qualification — Gate 3 unresolved (2026-08-08, superseded run)

```
GATE 1 (dY=0 ≡ baseline)            PASS   all seeds, large margin
GATE 2 (measured shift = command)   PASS   0.000mm error at ±15mm
GATE 3 (pre-contact EEF isolation)  FAIL   3 of 6 fail, 1 not evaluable, 2 OK
GATE 4 (unintended contacts)        PASS   properly instrumented this run
```

**Full causal sweep remains blocked.**

## Two model facts found while fixing Gate 4

Both were nearly recorded as false conclusions.

1. **`robot0_g0_vis` … `g6_vis` DO collide.** Despite the `_vis` suffix they
   carry `contype=1, conaffinity=1`. An earlier matcher looked for
   `robot0_link*`, found nothing, and reported arm distance as `inf` — which
   I was one step from writing up as "the Piper arm has no collision
   geometry". Group membership must be decided by contype/conaffinity, never
   by name suffix.
2. **The gripper genuinely has no palm collision geometry.** Its only
   colliding geoms are `finger7/8_collision`. So the earlier
   `palm = −0.023 m` reading was distance to `right_eef_target_*`, a
   *visualisation marker*, not physical geometry. An empty `palm` group is
   therefore N/A (a real model property), while an empty `arm` or `table`
   group is a matcher fault — the two must be distinguished, not both
   treated as failure.

Correct group sizes: `table=1, arm=10, palm=N/A`.

## Gate 3 — failures are real, but not explained by collisions

```
dY=-15 s5001  OK                       max 1.05e-3  rms 1.60e-4  ori 0.118°
dY=-15 s5002  NOT_EVALUABLE_PRECONTACT (no object contact detected)
dY=-15 s5003  FAIL: rms                max 8.72e-4  rms 4.60e-4  ori 0.198°
dY=+15 s5001  FAIL: max, rms, ori      max 3.04e-3  rms 8.98e-4  ori 0.458°
dY=+15 s5002  OK                       max 3.53e-4  rms 1.29e-4  ori 0.146°
dY=+15 s5003  FAIL: max, rms, ori      max 1.81e-3  rms 1.06e-3  ori 0.377°
```

Gate 4 is clean on every one of these: **no new contacts, and
`delta_min_distance` ≈ 0** for table and arm (largest +0.0003 m). So the
hypothesised mechanism — shifted fingers interacting differently with
table/palm and feeding back into the arm — is **not** supported. The
pre-contact deviation is not caused by changed collision geometry.

## The threshold itself is now the prime suspect

The frozen threshold (1.638e-3) is `observed max × 1.25` from **n=10**
baseline pairs of a distribution already characterised as **bifurcating**:
7/10 pairs reproduce `descend_refresh` to ~1e-11 m while 3 jump to
1e-5…5e-4. A max-based bound from n=10 of a heavy-tailed, bimodal
distribution is not a reliable bound — the failures here are only ~2× the
threshold, well within what an under-sampled tail could produce.

**This must be resolved before the treatment is implicated.** Required:
re-measure the baseline floor at n=30–50 pairs and re-derive the threshold.
If baseline-vs-baseline itself reaches 3e-3, Gate 3's failures are noise and
the gate passes. If baseline stays bounded near 1.3e-3, the treatment
genuinely perturbs the arm pre-contact by some route other than collision
geometry, and the instrument needs redesign.

Note this is *not* a threshold relaxation after seeing results: the
n=10 sample was always the weak point (flagged in Amendment 1, which
required ≥10 and got exactly 10), and the re-measurement is treatment-blind.

## Status

| gate | state |
|---|---|
| 1 null-intervention equivalence | PASS |
| 2 shift mechanism | PASS, exact |
| 3 pre-contact isolation | **unresolved — blocked on n=30–50 baseline floor** |
| 4 unintended contacts | PASS, properly instrumented |
| full sweep | **blocked** |
