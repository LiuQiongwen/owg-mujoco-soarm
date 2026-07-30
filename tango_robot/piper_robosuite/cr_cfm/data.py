"""CR-CFM Stage A: data pipeline -- extract the descend segment from
recorded PiperTrajectory files, resample to a fixed horizon, and build
flow-matching training pairs with synthetic drift injection (per this
project's own design: since Piper's grasp target is deterministic and
descend trajectories rarely fail from clean, near-target starts, we
synthesize the "before correction" distribution rather than relying on
naturally-drifted trajectories to supply it).

Action representation: JOINT SPACE (6D joint_pos), not task-space eef pose.
Chosen over end-effector 6-DoF pose because Piper's controller is
JOINT_POSITION (absolute) -- joint_pos is exactly what gets commanded, so
generating in joint space needs no IK step at inference time (a real
latency/complexity saving for a >=50Hz closed-loop deployment target).
Deviation from the original design's "task-space 6-DoF" framing, noted
explicitly rather than silently changed.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from tango_robot.piper_robosuite.piper_trajectory import PiperTrajectory

TRAJ_DIR = "/lena/projects/OWG-main/tango_robot/piper_robosuite/piper_trajs"
ACTION_DIM = 6  # joint_pos, matches Piper's 6-DoF joint-position controller
HORIZON = 16    # action-chunk length H, matches the design doc's 8-16 range


DESCEND_PHASES = {"descend", "descend_refresh"}


def _extract_descend_segment(traj: PiperTrajectory) -> Optional[np.ndarray]:
    """Slice out the descend phase using the real phase tag (2026-07-18,
    piper_trajectory.py's PiperTrajectoryRecorder.set_phase) -- replaces an
    earlier z-height heuristic that imprecisely picked up some of the
    transit_high->approach transition at the segment's start (visible as
    one anomalously large leading joint-delta in early smoke-test data).
    Falls back to the z-height heuristic ONLY for trajectories recorded
    before phase tagging existed (phase=="" on every point), so old data
    isn't silently discarded, but new data gets the precise cut.
    """
    phases = [p.phase for p in traj.points]
    if any(ph in DESCEND_PHASES for ph in phases):
        idxs = [i for i, ph in enumerate(phases) if ph in DESCEND_PHASES]
        joint_pos = np.stack([traj.points[i].joint_pos for i in idxs])
        return joint_pos.astype(np.float32)
    return _extract_descend_segment_zheuristic(traj)


def _extract_descend_segment_zheuristic(traj: PiperTrajectory) -> Optional[np.ndarray]:
    """Fallback for trajectories recorded before phase tagging existed --
    see _extract_descend_segment's docstring."""
    z = np.array([p.eef_pos[2] for p in traj.points])
    if len(z) < 4:
        return None
    z_max_idx = int(np.argmax(z[: len(z) // 2 + 1]))
    end_idx = z_max_idx
    tol = 0.002
    for i in range(z_max_idx + 1, len(z)):
        if z[i] > z[end_idx] + tol:
            break
        end_idx = i
    if end_idx - z_max_idx < 3:
        return None
    joint_pos = np.stack([p.joint_pos for p in traj.points[z_max_idx : end_idx + 1]])
    return joint_pos.astype(np.float32)


def _resample_to_horizon(segment: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Linear-interpolate a (T, D) segment to exactly `horizon` waypoints --
    T varies per trajectory (PD convergence speed differs slightly trial to
    trial), but the flow-matching model needs a fixed action-chunk shape."""
    t_src = np.linspace(0.0, 1.0, len(segment))
    t_dst = np.linspace(0.0, 1.0, horizon)
    out = np.stack([np.interp(t_dst, t_src, segment[:, d]) for d in range(segment.shape[1])], axis=1)
    return out.astype(np.float32)  # (horizon, D)


@dataclass
class DescendDataset:
    """Loaded, resampled descend segments -- the x_1 (clean target) pool
    flow-matching training pairs are built from."""
    segments: np.ndarray  # (N, horizon, ACTION_DIM)
    obj_names: List[str]

    @classmethod
    def load(cls, obj_name: Optional[str] = None, horizon: int = HORIZON,
             augment_subsegments: bool = False, samples_per_traj: int = 6,
             min_frac: float = 0.3, angle_range: Optional[tuple] = None) -> "DescendDataset":
        """angle_range (2026-07-19, IMPROVEMENT_PLAN.md Stage 1): if given, e.g. (-30, 60), keep
        only raw trajectories whose descend-phase start->end XY displacement angle (SAME definition
        used throughout the approach-angle audits in README.md) falls inside this range -- tests
        Geometric Entropy's (arXiv:2606.20871) finding that a small model/small dataset may perform
        BETTER on a geometrically narrower training set, not worse, contrary to the intuitive
        "collect more sparse-angle data" instinct."""
        """augment_subsegments (2026-07-19): fixes a real training-
        distribution gap found via direct audit -- every training example
        previously had x0 = a FULL trajectory's own (lightly perturbed)
        start, so the "remaining-distance-to-target" conditioning value
        was ALWAYS large (measured: joint-space L2 norm 0.277-1.29 across
        all 155 real examples, never below 0.277) -- but RHC's later
        iterations present the model with small remaining-distance inputs
        as the arm approaches the target, values the model never saw in
        training and had to extrapolate to. Fix: additionally resample
        SUB-segments starting at `samples_per_traj` intermediate points
        (from `min_frac` of the way through to near the very end) of each
        raw trajectory, always still ending at that trajectory's own true
        final waypoint -- teaches the model what small remaining-distance
        conditioning should produce, not just large ones. Does not need
        new data collection; purely resamples the same n=155 recordings
        differently."""
        pattern = f"{TRAJ_DIR}/{obj_name}_*.json" if obj_name else f"{TRAJ_DIR}/*.json"
        segments, names = [], []
        n_angle_filtered = 0
        n_kept_trajectories = 0
        for path in sorted(glob.glob(pattern)):
            traj = PiperTrajectory.load(path)
            if not traj.metadata.get("success"):
                continue  # belt-and-suspenders -- collect_seed_trajs.py already filters this

            if angle_range is not None:
                descend_pts = [p for p in traj.points if p.phase in DESCEND_PHASES]
                if len(descend_pts) < 4:
                    continue
                xy_disp = descend_pts[-1].eef_pos[:2] - descend_pts[0].eef_pos[:2]
                if np.linalg.norm(xy_disp) < 1e-4:
                    continue
                angle = float(np.degrees(np.arctan2(xy_disp[1], xy_disp[0])))
                if not (angle_range[0] <= angle <= angle_range[1]):
                    n_angle_filtered += 1
                    continue

            raw = _extract_descend_segment(traj)  # (T, 6), T varies, NOT yet resampled
            if raw is None:
                continue
            n_kept_trajectories += 1
            segments.append(_resample_to_horizon(raw, horizon))
            names.append(traj.metadata.get("obj_name", "?"))

            if augment_subsegments:
                T = len(raw)
                start_idxs = np.linspace(int(T * min_frac), T - 2, samples_per_traj, dtype=int)
                for start_idx in sorted(set(start_idxs.tolist())):
                    sub_raw = raw[start_idx:]
                    if len(sub_raw) < 4:
                        continue  # too few points to resample meaningfully
                    segments.append(_resample_to_horizon(sub_raw, horizon))
                    names.append(traj.metadata.get("obj_name", "?"))

        if not segments:
            raise RuntimeError(f"DescendDataset.load: no valid descend segments found in {pattern}")
        if angle_range is not None:
            print(f"DescendDataset.load: angle_range={angle_range} kept {n_kept_trajectories} trajectories "
                  f"(filtered out {n_angle_filtered})")
        return cls(segments=np.stack(segments), obj_names=names)

    def __len__(self):
        return len(self.segments)

    def mean_template(self) -> np.ndarray:
        """Per-timestep mean trajectory shape (horizon, D) -- since every
        segment is already time-normalized to the same `horizon` via
        _resample_to_horizon, a plain elementwise mean across trajectories
        gives a real, physically-shaped "typical descend" progress curve
        (fast start, damped convergence) for inference.py's
        build_template_x0 to affinely re-target, instead of a straight
        line."""
        return self.segments.mean(axis=0).astype(np.float32)

    def mean_start(self) -> np.ndarray:
        """Dataset mean of clean trajectories' FIRST waypoint -- the
        inference-time proxy for x1_start used by inference.py's
        conditioning (see that module's docstring for why: the true
        per-trajectory x1_start used at training time isn't available at
        inference, since predicting it is the whole point)."""
        return self.segments[:, 0, :].mean(axis=0).astype(np.float32)


def contact_proximity_weight(horizon: int = HORIZON, low: float = 0.15, high: float = 1.0,
                              zero_frac: float = 0.3) -> np.ndarray:
    """Per-timestep TCR weight w_i (see the design doc's "asymmetric TCR"):
    HIGH weight (strong jerk/velocity penalty, suppress high-frequency
    wobble) during free-space descent, ramping down toward the contact-risk
    zone, then HARD ZERO for the final `zero_frac` of the horizon
    (2026-07-18, v2 -- see README's "TCR overreach" finding: the original
    soft ramp-to-`low` version was found, via a direct n=15-vs-n=41
    trend check, to still suppress real deceleration near contact -- mean
    terminal velocity roughly DOUBLED with more training data instead of
    dropping, and win rate stayed completely flat (1/8 both), ruling out
    "just needs more data" and pointing at the loss weighting itself.
    Zeroing the final zero_frac outright (not just down-weighting) removes
    ANY penalty on deceleration/jerk in exactly the window where a real
    successful grasp needs a sharp velocity drop, instead of relying on a
    small residual weight to not matter in practice."""
    n_zero = int(round(horizon * zero_frac))
    n_ramp = horizon - n_zero
    ramp = np.linspace(high, low, max(n_ramp, 1)).astype(np.float32)
    zero = np.zeros(n_zero, dtype=np.float32)
    return np.concatenate([ramp[: horizon - n_zero], zero])[:horizon]


def sample_flow_pairs(dataset: DescendDataset, batch_size: int, rng: np.random.Generator,
                       drift_std_m: float = 0.02, noise_std: float = 0.05):
    """Build one training batch of (x0, x1, t) for conditional flow
    matching: x1 = a real clean descend segment (the target); x0 = x1 with
    synthetic pose drift injected on the FIRST waypoint only (the model
    must learn to converge back to the clean trajectory's shape over the
    horizon, matching this project's own diagnosed failure mode -- a
    miscentered START that current motion doesn't correct for) plus small
    Gaussian noise throughout (standard CFM training-time perturbation).

    drift_std_m is in JOINT SPACE radians here despite the "_m" suffix
    inherited from the design doc's task-space framing -- TODO if this
    moves to task-space actions: convert via a real per-joint Jacobian
    sensitivity, not a flat guess, matching this project's established
    "don't guess unit conversions" convention.
    """
    idx = rng.integers(0, len(dataset), size=batch_size)
    x1 = dataset.segments[idx].copy()  # (B, H, D)
    B, H, D = x1.shape

    drift = rng.normal(0.0, drift_std_m, size=(B, D)).astype(np.float32)
    x0 = x1.copy()
    # Injected only at the start, then linearly faded to zero by waypoint
    # H//2 -- models "the object/pose was off at the top of descend, but
    # nothing all the way through" rather than a trajectory-wide constant
    # bias (which the model could satisfy trivially by just copying x0).
    fade = np.clip(1.0 - np.arange(H, dtype=np.float32) / (H // 2), 0.0, 1.0)
    x0 += drift[:, None, :] * fade[None, :, None]
    x0 += rng.normal(0.0, noise_std, size=x0.shape).astype(np.float32)

    t = rng.uniform(0.0, 1.0, size=(B,)).astype(np.float32)
    return x0, x1, t
