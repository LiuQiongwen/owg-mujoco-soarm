Phase 2Y instrument qualification — NOT QUALIFIED (2026-08-08)

Four gates, thresholds frozen in `calib/phase2y_noise_floor.json`, not
recomputed. Pear, seeds 5001–5003, dY ∈ {−15, 0, +15}.

```
GATE 1 (dY=0 ≡ baseline)          PASS
GATE 2 (measured shift = command) PASS   0.000mm error at ±15mm
GATE 3 (pre-contact EEF equiv.)   FAIL   4 of 6 arms exceed threshold
GATE 4 (no new contacts)          PASS*  *incomplete — see below
```

**Full causal sweep remains blocked.**

## Gate 1 — PASS

dY=0 reproduces baseline within the frozen floor on all four metrics, with
large margin (max_pos 3.7e-5…1.1e-4 vs threshold 1.638e-3; refresh
2.7e-12…6.9e-8 vs 6.4e-4). Success outcomes match. The virtual-shift
machinery itself does not perturb the simulation.

## Gate 2 — PASS

Commanded vs measured finger displacement: **0.000mm error** at both ±15mm,
all seeds. The shift does exactly what it claims.

## Gate 3 — FAIL, and the cause is not yet established

Pre-contact EEF trajectory deviation, windowed to before first
finger-object contact:

```
dY=-15  seed 5001  1.59e-3  OK      window 401
dY=-15  seed 5002  8.64e-3  FAIL    window 1210   <- window fallback, see below
dY=-15  seed 5003  8.72e-4  FAIL    window 408    <- max_pos is UNDER threshold
dY=+15  seed 5001  3.05e-3  FAIL    window 451
dY=+15  seed 5002  4.04e-4  OK      window 392
dY=+15  seed 5003  1.80e-3  FAIL    window 404
```

Two defects in this gate's own implementation must be resolved before the
FAIL can be interpreted:

1. **Window fallback contaminates seed 5002.** When no finger-object
   contact is detected, the window falls back to the whole episode
   (1210 samples vs ~400 typical), so the comparison includes lift,
   transit and place. That is not a pre-contact measurement. The fallback
   must be an explicit failure/exclusion, not a silent widening.
2. **The failing metric is not reported.** Seed 5003 at dY=−15 shows
   max_pos = 8.72e-4, which is *under* the 1.638e-3 threshold — so it
   failed on RMS or orientation, which the output does not print. Gate 3
   must report which of the three metrics failed.

If those are fixed and Gate 3 still fails, the likely physical cause is
visible in Gate 4's distance probes (below): laterally shifted fingers
interact differently with the table and palm, feeding back through contact
dynamics into the arm before the object is ever touched. That would make
the treatment genuinely non-isolating and would require a redesign, not a
threshold change.

## Gate 4 — incomplete, reported as PASS only for the check that ran

No *new* contact categories appeared (finger↔table / palm / opposite-finger
/ arm) relative to baseline. But the gate as specified also requires
proximity comparison, and two things are unresolved:

- **Distances are negative in baseline and treatment alike**:
  `table −0.002 m`, `palm −0.023 m`. Negative `mj_geomDistance` is
  interpenetration. The palm figure is plausibly a pre-existing modelling
  artifact (fingers seated into the palm housing), but this was **not
  compared against baseline**, so no claim is made either way.
- **`arm` distance is `inf`**, meaning the arm geom group matched nothing —
  the check is vacuous, not passing. The name-matching for arm geoms needs
  verification.

Per R1, a distance of exactly `inf` and an unverified group membership are
instrumentation faults until shown otherwise.

## Status

| item | state |
|---|---|
| shift mechanism (Gate 2) | verified exact |
| null-intervention equivalence (Gate 1) | verified within frozen floor |
| pre-contact isolation (Gate 3) | **FAIL — 2 implementation defects to fix first** |
| unintended-contact check (Gate 4) | **incomplete — vacuous arm group, no baseline distance comparison** |
| full causal sweep | **blocked** |

Next: fix the Gate 3 window fallback and per-metric reporting; fix the
Gate 4 arm group and add baseline-relative distance comparison; re-run.
Only if Gate 3 still fails after that is the treatment itself implicated.
