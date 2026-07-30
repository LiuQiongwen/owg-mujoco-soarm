# Line B Experiment Plan: Real Perception Noise → Candidate Selection Re-Test

Mirrors this project's established convention (see `EXPERIMENT_PLAN.md`): staged, smoke-test-then-scale,
paired design with McNemar's exact test, pre-registered decision gates so a result can't be quietly
reframed after the fact. Blocked on camera repositioning (pending); everything not requiring live
camera data is built and ready now.

## Stage 0: Prerequisites (blocked)

- [ ] Camera repositioned overhead a real tabletop, ~0.5–0.8m working distance (D435I's accurate range).
- [ ] One real object matching an existing Piper YCB entry, placed at a fixed, undisturbed position.
      **Recommend mustard bottle or cracker box** (rigid, dimensionally stable) over a real pear
      (perishable, shape drifts over days) — either connects directly to Appendix A row 4's existing
      Pear/Mustard/Cracker data.
- [ ] One calibration capture of the *empty* table (no object) — needed for plane-fitting-based
      segmentation (Stage 1 below).

## Stage 1: Measure real noise structure

### Capture protocol
- N=100 repeated captures (real capture is <1s/frame, not the bottleneck this project's sim trials
  are — no reason to under-sample the way n=10–20 sim pilots had to for time budget reasons).
- Spaced over ~30–60s at ~1–2Hz, not a single instantaneous burst — a continuous stream's consecutive
  frames may be internally correlated (auto-exposure, temporal filtering); spacing captures samples
  genuine frame-to-frame sensor noise rather than one smoothed snapshot repeated.
- Discard the first ~10 frames after any camera (re)start — auto-exposure/auto-white-balance
  convergence transient, not representative noise.
- Object physically undisturbed throughout — any real movement between captures contaminates
  "estimation noise" with genuine position change. If in doubt mid-capture, restart the sequence.

### Pose estimation method (implemented now, tuning deferred to real data)

Deliberately classical and transparent, not a black-box learned model — this experiment's goal is
characterizing *noise*, and a method whose own failure modes are opaque would confound that.
Implemented in `cameras/noise_characterization.py::estimate_pose_from_rgbd`:

1. **Plane removal**: RANSAC-fit the table plane from the empty-table calibration capture's point
   cloud; subtract the same plane model from each object capture to isolate above-table points.
2. **ROI crop**: restrict to an expected workspace bounding box (tuned once real captures show what's
   actually in frame — camera mount, robot arm, or other clutter may need excluding).
3. **Segmentation**: largest connected component among remaining points (valid for the single-static-object
   case this stage tests; multi-object scenes are explicitly out of scope here).
4. **Position estimate**: centroid of the segmented point cloud, in the camera's own frame (a full
   camera-to-robot-base extrinsic calibration is NOT required for Stage 1 — variance/covariance
   structure is invariant to a fixed rigid transform; extrinsics only become necessary in Stage 2/3
   when porting the measured noise into the sim's coordinate frame).
5. **Orientation estimate**: PCA on the XY-projected segmented points; yaw = angle of the dominant
   horizontal axis — matches how the sim's own candidate yaw is defined (`_yaw_of` in
   `piper_candidate_selection.py`), for direct comparability.

### Analysis (tooling already built and validated)

Run `characterize_noise()` on the 100 `PoseSample`s, then `compare_to_synthetic_assumptions()` against
the exact existing parameters: `pos_jitter=0.005` (kinematic model) and `yaw_jitter_rad=radians(5.0)`
(kinematic) / `angle_jitter_rad=radians(10.0)` (perception model) from `piper_candidate_selection.py`.

### Pre-registered decision gate

**Stop here and report honestly if**: both position-std ratios (measured/assumed) fall in [0.5, 2.0]x,
the yaw-std ratio falls in [0.5, 2.0]x, |position-yaw correlation| < 0.3, and no axis's Shapiro-Wilk
p-value is below 0.05. This would mean real noise is structurally close enough to the synthetic
assumption that Stage 2/3 are unlikely to show anything new — a legitimate, citable negative result
(nobody else has quantified this comparison; the closest paper, Joyce et al. IROS 2025, argues
qualitatively that synthetic Gaussian noise doesn't match real failure modes but never measures the
actual structure) — not a reason to force Stage 2 anyway.

**Proceed to Stage 2 if**: any of the above thresholds is violated — a genuine structural mismatch
exists, worth testing whether it changes the downstream selection result.

## Stage 2: Real-noise-calibrated candidate sampler (built now, both branches)

New function in `piper_candidate_selection.py`, `sample_candidate_pool_real_noise`, decided by
Stage 1's own findings rather than assumed in advance:

- **If Stage 1's noise is well-approximated as Gaussian** (no axis rejects Shapiro-Wilk): fit a
  multivariate Gaussian (mean 0, covariance = measured empirical covariance including the
  cross-position-yaw terms if correlated) and sample via `rng.multivariate_normal`.
- **If Stage 1 rejects Gaussianity on any axis**: empirical bootstrap — resample directly from the
  measured Stage 1 residuals (with replacement) rather than fitting a parametric distribution,
  more faithful to the actual measured shape than forcing a Gaussian that's already been shown wrong.

Both branches produce the same `(pos, mat)` candidate-pool format `sample_candidate_pool` already
returns, so they're drop-in compatible with everything downstream (`select_best`, `select_consensus`,
`run_pick_and_place`'s `candidate_selection=` machinery) with zero changes elsewhere.

**Causal-validity check** (ties Line A and Line B together) — status as actually built and checked
2026-07-17, not as originally planned: `sample_candidate_pool_real_noise` is manually argued
PRE_EXECUTION-admissible under Definition 3 (pure function of the pool + a precomputed statistic,
no `env`/execution dependency), matching `sample_candidate_pool`'s own reasoning. Ran it through
`auto_tagger.py` directly to mechanically confirm this rather than trust the manual argument, and
found a real tool-coverage gap instead: `analyze_function` only recognizes `return {...}` dict
literals and `dict(...)` calls, and this function returns a list of `(pos, mat)` tuples — a shape
the tool doesn't parse, so it comes back with an empty (not confirming, not denying) result. Two
honest options before citing this as tool-verified in any paper material: extend
`analyze_function` to recognize list/tuple return patterns (real, scoped tool work), or wrap the
tuples in a dict shape the tool already understands for audit purposes only. Neither done yet —
tracked here rather than silently left as an unverified claim.

## Stage 3: Re-test best-vs-consensus (conditional on Stage 1's gate)

Exact mirror of the existing, already-validated protocol
(`piper_consensus_experiment_runner.py`/README's "Consensus vs best-IK candidate selection" entry,
re-verified 2026-07-17 under the fixed placement RNG):

- Same objects (Pear, Mustard, Cracker), same paired-by-trial_id design, same McNemar's exact test.
- Same smoke-test-then-scale discipline: n=3–5 first to confirm the new sampler behaves sanely
  (candidates spread plausibly, no degenerate pool), then n=10/cell matching the existing
  re-verification scale for direct comparability to the 2026-07-17 numbers (Pear p=1.0000, Mustard
  p=1.0000, Cracker 0 discordant pairs).
- Only difference: `sample_candidate_pool` swapped for `sample_candidate_pool_real_noise`.
- Report against both baselines: the original synthetic-noise null AND whatever Stage 1 measured,
  regardless of which direction the result goes.

## What "success" and "failure" both look like here

This is designed so every plausible outcome is a reportable result, not just the flattering one:
- Stage 1 gate fires (noise structurally similar) → report the measurement, close the line, cite
  it as evidence the synthetic model was adequate after all — genuinely useful to know.
- Stage 1 shows real mismatch, Stage 3 still null → real noise structure differs, but that
  difference still isn't what explains consensus's cross-embodiment failure — narrows the
  remaining hypothesis space for future work.
- Stage 1 shows mismatch, Stage 3 shows consensus becomes effective → the positive result, closes
  the loop on the mechanistic explanation from the original README entry.
