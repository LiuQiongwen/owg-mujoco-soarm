"""
Candidate-pool grasp selection for Piper -- replicates the SO-ARM101
consensus-selection mechanism (pick the candidate closest to the pool's
median instead of the one with the lowest IK error) on this second,
independently-implemented embodiment.

Why this needs an explicit noise model here (unlike SO-ARM101, where
different random seeds of a learned candidate generator naturally disagree):
Piper's grasp target (compute_grasp_orientation + object CoM) is currently
deterministic -- same object pose always gives the same nominal grasp. To
test whether consensus selection helps, we inject the kind of pose noise a
real perception pipeline would produce (multiple independent position/yaw
estimates that don't exactly agree), sample a pool around the nominal
target, and compare two ways of picking one candidate from that pool.

This is also a direct response to this session's own finding: several
objects (Cracker, Banana) fail with IK reporting full convergence at every
phase, yet the grasp is asymmetric or never actually secures -- i.e. the
failure mode is exactly "one estimate was slightly off" pose noise, which
median-of-pool selection is designed to average out.
"""
import numpy as np


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _yaw_of(mat):
    """Yaw of a grasp orientation matrix's closing axis (x-axis) in the
    world XY plane."""
    x_axis = mat[:, 0]
    return float(np.arctan2(x_axis[1], x_axis[0]))


def _yaw_rotate(mat, delta_yaw):
    """Rotate a grasp orientation matrix by delta_yaw about world Z. Leaves
    the z-axis (straight down, [0,0,-1]) invariant, only changes which
    horizontal direction the gripper's closing axis points along."""
    c, s = np.cos(delta_yaw), np.sin(delta_yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ mat


def sample_candidate_pool(nominal_pos, nominal_mat, n=10, pos_jitter=0.005,
                           yaw_jitter_rad=np.radians(5.0), rng=None):
    """Sample n candidate (pos, mat) grasp poses around a nominal target,
    modelling independent-estimate pose noise. pos_jitter/yaw_jitter_rad are
    the std-dev of Gaussian XY / yaw perturbations -- calibrate via smoke
    test: too small and best/consensus degenerate to the same candidate,
    too large and both strategies fail (no discriminative signal either
    way). Z height is NOT jittered -- GRASP_HEIGHT_OFFSET was itself tuned
    per-object; injecting height noise would conflate two different
    uncertainty sources.
    """
    rng = rng if rng is not None else np.random.default_rng()
    nominal_yaw = _yaw_of(nominal_mat)
    candidates = []
    for _ in range(n):
        dx, dy = rng.normal(0.0, pos_jitter, size=2)
        dyaw = rng.normal(0.0, yaw_jitter_rad)
        pos = nominal_pos + np.array([dx, dy, 0.0])
        mat = _yaw_rotate(nominal_mat, dyaw)
        candidates.append((pos, mat))
    return candidates


def sample_candidate_pool_real_noise(nominal_pos, nominal_mat, n=10, rng=None,
                                      mean_xy=None, cov_xy_yaw=None, residual_bank=None):
    """Line B (LINE_B_EXPERIMENT_PLAN.md): drop-in replacement for
    sample_candidate_pool, drawing candidate-pool jitter from an EMPIRICALLY
    MEASURED noise structure (cameras/noise_characterization.py) instead of
    the assumed-isotropic-Gaussian/independent-yaw model above. Same
    (pos, mat) return format -- compatible with select_best/select_consensus/
    run_pick_and_place's candidate_selection= machinery with no other
    changes required.

    Two mutually exclusive modes, chosen by LINE_B_EXPERIMENT_PLAN.md's
    Stage 1 measurement (not assumed in advance):

    Gaussian mode (pass mean_xy, cov_xy_yaw): draws from a fitted
    multivariate Gaussian over (dx, dy, dyaw) -- use only if Stage 1's
    Shapiro-Wilk test does NOT reject normality on any axis. cov_xy_yaw is
    the full 3x3 empirical covariance (position-yaw correlation included,
    unlike sample_candidate_pool's independence assumption).

    Bootstrap mode (pass residual_bank): resamples WITH REPLACEMENT
    directly from the measured (dx, dy, dyaw) residuals themselves, no
    parametric distribution fitted at all -- use if Stage 1 rejects
    Gaussianity on any axis, since forcing a Gaussian already shown wrong
    would just be a different, still-unjustified assumption.

    Causal validity: this function is a pure computation over the nominal
    target + a precomputed statistic (covariance or residual bank), with
    no dependency on env/execution state -- PRE_EXECUTION admissible under
    causal_validity_audit's criterion by the same manual reasoning as
    sample_candidate_pool above (Definition 3: computable from the pool +
    static inputs alone). NOT mechanically verified by auto_tagger.py --
    checked directly (2026-07-17) and confirmed the tool returns an empty
    result here, because analyze_function only recognizes `return {...}`
    dict literals and `dict(...)` calls as field-defining patterns, and
    this function returns a list of (pos, mat) tuples, a shape the tool
    doesn't parse. This is an honest tool-coverage gap, not a marker or
    execution-boundary issue -- worth closing (extend analyze_function to
    recognize list/tuple return patterns) before citing this function as
    tool-verified rather than manually-argued in any paper material."""
    if (mean_xy is None) == (residual_bank is None):
        raise ValueError(
            "sample_candidate_pool_real_noise: pass exactly one of "
            "(mean_xy, cov_xy_yaw) for Gaussian mode or residual_bank for "
            "bootstrap mode -- not both, not neither. Which mode is valid "
            "is Stage 1's empirical finding, not a default to guess."
        )
    rng = rng if rng is not None else np.random.default_rng()
    candidates = []

    if mean_xy is not None:
        if cov_xy_yaw is None:
            raise ValueError("Gaussian mode requires cov_xy_yaw alongside mean_xy")
        mean = np.zeros(3) if mean_xy is None else np.concatenate([mean_xy, [0.0]])
        deltas = rng.multivariate_normal(mean=np.zeros(3), cov=cov_xy_yaw, size=n)
    else:
        residual_bank = np.asarray(residual_bank)
        if residual_bank.ndim != 2 or residual_bank.shape[1] != 3:
            raise ValueError("residual_bank must be (M, 3): columns (dx, dy, dyaw)")
        idx = rng.integers(0, len(residual_bank), size=n)
        deltas = residual_bank[idx]

    for dx, dy, dyaw in deltas:
        pos = nominal_pos + np.array([dx, dy, 0.0])
        mat = _yaw_rotate(nominal_mat, dyaw)
        candidates.append((pos, mat))
    return candidates


def _axis_angle_to_quat(axis, angle):
    """(w,x,y,z) quaternion for a rotation of `angle` about a unit `axis`."""
    half = angle / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def sample_perception_noisy_candidates(true_pos, true_quat, obj_name, n=10,
                                        pos_jitter=0.005, angle_jitter_rad=np.radians(10.0),
                                        rng=None):
    """Alternative to sample_candidate_pool: perturb the TRUE object POSE
    (position + full 3D orientation, not just the already-computed grasp
    target's yaw) to simulate independent noisy 6D pose estimates from a
    perception pipeline, then recompute what grasp target EACH noisy
    estimate would imply via grasp_orientation_from_quat.

    This is a more faithful reproduction of SO-ARM101's actual noise source
    than sample_candidate_pool's direct kinematic jitter on the final grasp
    target: candidate diversity here comes from uncertainty in WHERE and
    HOW the object is believed to be oriented, which can (for objects whose
    narrow-axis choice depends on orientation) produce qualitatively
    different candidates -- e.g. a large enough orientation error could
    make the projected "narrow axis" computation pick a different apparent
    direction, not just a slightly-rotated version of the same one. That
    qualitative-divergence possibility is exactly the kind of noise
    consensus selection is meant to protect against, and direct kinematic
    jitter (small, symmetric, around an already-good estimate) cannot
    produce it by construction.

    angle_jitter_rad perturbs around a RANDOM 3D axis (not constrained to
    yaw/world-Z), since real 6D pose estimation error isn't confined to
    the horizontal plane the way sample_candidate_pool's yaw-only jitter
    assumes.
    """
    from tango_robot.piper_robosuite.piper_pick_and_place import grasp_orientation_from_quat
    rng = rng if rng is not None else np.random.default_rng()
    candidates = []
    for _ in range(n):
        dx, dy = rng.normal(0.0, pos_jitter, size=2)
        pos = true_pos + np.array([dx, dy, 0.0])

        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.normal(0.0, angle_jitter_rad)
        noisy_quat = _quat_multiply(_axis_angle_to_quat(axis, angle), true_quat)
        noisy_quat /= np.linalg.norm(noisy_quat)

        mat = grasp_orientation_from_quat(noisy_quat, obj_name)
        candidates.append((pos, mat))
    return candidates


def select_best(candidates, scores):
    """ikmargin-style baseline: index of the candidate with the lowest
    score (IK error). scores must be computed by the caller (needs a live
    ArmIK instance) and aligned index-for-index with candidates."""
    return int(np.argmin(scores))


def select_consensus(candidates, pos_jitter=0.005, yaw_jitter_rad=np.radians(5.0)):
    """Pick the pool candidate closest to the pool's own median pose --
    median XY position (elementwise) + circular median yaw (unwrapped
    relative to the pool's circular mean first, since the SO-ARM101
    analogue explicitly must not use a plain arithmetic median on angles).
    Returns an existing candidate's index (not an interpolated new pose),
    keeping the selection discrete and reproducible.

    pos_jitter/yaw_jitter_rad should match whatever was passed to
    sample_candidate_pool for this same call -- they're used only to
    normalize position (metres) and yaw (radians) onto a common scale
    before combining into one distance, so that neither axis dominates
    just because metres and radians have different natural magnitudes.
    """
    positions = np.array([c[0][:2] for c in candidates])  # XY only
    yaws = np.array([_yaw_of(c[1]) for c in candidates])

    median_xy = np.median(positions, axis=0)

    circ_mean = np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))
    unwrapped = np.array([_wrap_to_pi(y - circ_mean) for y in yaws])
    median_yaw = _wrap_to_pi(circ_mean + np.median(unwrapped))

    yaw_scale = pos_jitter / yaw_jitter_rad  # metres per radian, so both axes contribute on the same footing
    best_idx, best_dist = None, None
    for i, (pos, mat) in enumerate(candidates):
        pos_dist = np.linalg.norm(pos[:2] - median_xy)
        yaw_dist = abs(_wrap_to_pi(_yaw_of(mat) - median_yaw))
        dist = pos_dist + yaw_scale * yaw_dist
        if best_dist is None or dist < best_dist:
            best_idx, best_dist = i, dist
    return best_idx
