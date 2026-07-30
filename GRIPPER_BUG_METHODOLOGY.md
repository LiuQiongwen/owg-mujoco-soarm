# Gripper-Controller Double-Scaling Bug — Methodology Section Draft

This is the project's headline finding. It should be written up as a dedicated methodology
subsection, not buried in results — the discovery process is itself the contribution (a
generalizable diagnostic methodology for silent unit/scaling bugs in composite robot
controllers), independent of the specific fix.

## The bug

`robosuite`'s `SimpleGripController` applies its own `bias + weight * desired_qvel` rescaling
from a normalized `[-1, 1]` action space to real actuator units, gated behind a
`use_action_scaling` flag that defaults to `True`. `PiperGripper.format_action`
(`piper_gripper.py`), however, does **not** emit a normalized value like other robosuite
grippers — it already tracks and returns an **absolute real-units joint position** (−0.05 to
−0.004 m, clipped to the gripper's actual `ctrlrange`), because it was ported from a different
convention than the one `SimpleGripController` assumes. With `use_action_scaling` left at its
default, the controller silently re-scaled an already-real-units value a second time, as if it
still needed converting from normalized space.

**Measured impact**: at a verified, fully-converged `current_action=-0.05` ("fully open"), the
actual `data.ctrl` written to the finger actuators was `-0.02815` — nowhere near either range
endpoint. At `current_action=-0.004` ("fully closed"), it computed to `-0.0271`. **The real
commanded travel between fully-open and fully-closed was ~0.0001 m (a tenth of a millimetre)**,
against a documented/assumed range of 0.076–0.10 m. Every finger-width and grasp-precision
investigation earlier in the project's history was operating on top of this bug without knowing
it — the gripper was, for practically all purposes, never actually opening or closing.

## How it was found

1. An earlier investigation into poor cross-object grasp generalization traced the failure
   symptom (objects not being gripped despite converged IK) one level deeper than "where does
   the arm end up" into "what does the gripper actually do when told to open" — i.e., stopped
   trusting the high-level action command and started inspecting the actual low-level
   `data.ctrl` value written to the simulator.
2. That inspection revealed the ~0.1mm real travel figure above, which was inconsistent with
   every prior assumption in the codebase (7.6cm and later 10cm documented opening widths).
3. Traced the discrepancy to `SimpleGripController`'s double application of a real-units-to-
   real-units "rescaling" that should never have run at all, given `PiperGripper.format_action`
   already outputs real units.

## Two fix attempts, one that actually worked (a genuinely useful negative-result-inside-a-fix)

**First attempt (did not work)**: set `input_min`/`input_max` on the gripper sub-config to match
`format_action`'s real output range, hypothesizing the rescaling step just needed the correct
bounds. Traced further and found `robosuite/robots/robot.py` (~lines 958–969) **rebuilds**
`part_controller_config[gripper_name]` from scratch when wiring up the composite controller,
copying only `"type"` and `"use_action_scaling"` from the config dict — `input_min`/`input_max`
are silently dropped regardless of what's set. Confirmed empirically: `ctrl` stayed at
`-0.02815` after adding them. This dead end is worth keeping in the writeup: it demonstrates the
bug was in the rescaling logic being applied at all, not merely in its parameters, and shows the
diagnostic process ruling out the more "obvious" fix before finding the real one.

**Actual fix**: `"use_action_scaling": False` in the gripper sub-config. `SimpleGripController`'s
`set_goal()` and `run_controller()` both gate their entire rescaling logic behind
`if self.use_action_scaling:` — with it off, the action passes straight through to `data.ctrl`
unchanged, which is exactly correct since `format_action` already computed the real-units
target itself. Confirmed this key survives `robot.py`'s config-rebuild step (unlike
`input_min`/`input_max` above) — a one-line config change
(`tango_robot/piper_robosuite/piper_controller_config.py`).

## Validated impact

**Initial n=20/object pilot (trial_id 400-419)**: Cracker 0%→45% (9/20), Mustard 0%→65%
(13/20), Pear already-working→65% (13/20). All three objects moved from a hard 0% or an
already-working baseline to real, substantially-above-zero success rates with **no other code
change** than this one controller-config fix.

**Final confirmatory pilot for the paper table (n=40-60/object, frozen final baseline config,
clean unused trial_id range 5000-5059, disjoint from every diagnostic/pilot batch elsewhere in
this project)**:

| Object | Final confirmed rate | n | Note |
|--------|----------------------|---|------|
| Cracker | 50% (20/40) | 40 | Replicates the 45% pilot figure cleanly across two independent 20-trial batches (50%, 50%) |
| Mustard | 70% (28/40) | 40 | Replicates the 65% pilot figure cleanly across two independent 20-trial batches (70%, 70%) |
| Pear | **43.3% (26/60)** | 60 | **Does not replicate** the original 65% pilot figure — see below |

**Pear's non-replication, reported honestly rather than kept at the flattering first number**:
the original 65% came from a single n=20 batch. Three fresh, independent n=20 batches
(trial_id 5000-5019, 5020-5039, 5040-5059) gave 40%, 40%, and 50% respectively — consistently
below 65%, never matching or exceeding it. Pooled: 26/60 (43.3%) vs. the original 13/20 (65%),
Fisher's exact p=0.12 — not formally significant at conventional thresholds, but a consistent
directional trend across three independent replications is a stronger signal than the p-value
alone suggests, and matches this project's own repeatedly-documented pattern of an early
small-batch result running hot and correcting downward on replication (see
`piper_robosuite/README.md`'s centroid-offset re-check and compliant-descend-v2 entries for two
prior instances of exactly this pattern). No code drift was found that would explain a genuine
regression (centroid offsets, controller config, and all opt-in execution-control flags checked
and confirmed unchanged/off). **The paper should report 43.3% (n=60) as Pear's confirmed rate,
not 65%** — and should note the original single-batch figure explicitly rather than silently
dropping it, consistent with this project's transparency convention.

## Honest framing for the paper

- **This does not mean the system was "finished" after the fix.** Cracker's n=20 result was not
  uniform (1/10 on the first sub-batch, 8/10 on the second), and every Cracker failure in the
  full n=20 batch had large `pre_close_drift_cm` while every success had near-zero drift —
  meaning a second, distinct bottleneck (execution-time contact dynamics during descend) remains
  and was investigated separately (see `RULED_OUT_METHODS.md` rows 7-10; still open as
  documented future work, not solved).
- **This bug was hiding in plain sight the entire time other fixes were being tried** (interpolated
  descend, pre-close refresh, centroid-offset re-measurement) — those fixes are still real and
  still validated on their own terms, and most likely still contribute now that the gripper has
  real clearance to work with instead of none, but this scaling bug was almost certainly the
  dominant lever the whole time. Worth stating plainly rather than implying the other fixes were
  wasted effort — they weren't, they just couldn't matter much against a gripper with 0.1mm of
  real travel.
- **The generalizable methodological point for the paper**: silent unit/scaling mismatches
  between a robot-specific `format_action` implementation and a generic composite controller's
  own rescaling assumptions are easy to miss because every higher-level signal (IK convergence,
  action values, even the printed "gripper opening" logs computed from the pre-rescaling
  `current_action` value) looked normal — the bug was only visible by inspecting the literal
  low-level actuator control value against the simulator's own `ctrlrange`. This is a reusable
  diagnostic lesson for anyone porting a new gripper into robosuite's composite-controller
  framework, independent of Piper specifically.
