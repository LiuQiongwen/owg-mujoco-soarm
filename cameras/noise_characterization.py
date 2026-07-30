"""
Compare real perception noise (once real RGB-D captures exist) against the
synthetic noise models this project's whole candidate-selection/consensus
investigation has relied on (piper_candidate_selection.py's
sample_candidate_pool / sample_perception_noisy_candidates).

Why this exists: the synthetic models bake in specific, never-empirically-
checked assumptions -- independent isotropic Gaussian XY jitter (sigma=0.5cm),
independent yaw jitter (sigma=5-10 deg), and ZERO jitter on Z height. The
best-vs-consensus README entry's own mechanistic explanation for why
consensus selection doesn't transfer to Piper is that this synthetic jitter
lacks SO-ARM101's real candidate generator's multi-modal outlier structure --
a claim that has never been checked against actual measured perception noise,
because until the camera calibration in this session, there was no real
perception pipeline to measure. This module is the tool to finally check it,
once real repeated-capture data exists.

Status (2026-07-17): built and validated against the EXISTING synthetic
generators (a closed-loop sanity check: feed known-parameter synthetic
samples in, confirm this module recovers those known parameters) -- NOT yet
run on real camera data, which requires the camera repositioning step
(pending) and a real pose-estimation function (estimate_pose_from_rgbd
below, deliberately left unimplemented rather than guessed -- see its
docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class PoseSample:
    """A single (position, yaw) pose estimate -- either from a real
    perception pipeline (repeated captures of a static object) or from one
    of this project's synthetic noise generators, for direct comparison."""
    x: float
    y: float
    z: float
    yaw: float  # radians


@dataclass
class NoiseCharacterization:
    n: int
    mean_xyz: np.ndarray        # (3,)
    std_xyz: np.ndarray         # (3,) -- per-axis, NOT assuming isotropy
    mean_yaw: float
    std_yaw: float              # circular std, radians
    xy_yaw_correlation: float   # correlation between planar position-error magnitude and |yaw error|
    normality_p_xyz: np.ndarray  # (3,) Shapiro-Wilk p-value per axis (low p = reject Gaussian)
    normality_p_yaw: float

    def report(self) -> str:
        lines = [
            f"n={self.n}",
            f"position std (x,y,z) = ({self.std_xyz[0]*1000:.2f}, "
            f"{self.std_xyz[1]*1000:.2f}, {self.std_xyz[2]*1000:.2f}) mm",
            f"yaw std = {np.degrees(self.std_yaw):.2f} deg",
            f"position-yaw error correlation = {self.xy_yaw_correlation:+.3f} "
            f"(0 = independent, matching the synthetic model's assumption; "
            f"|r|>0.3 would be a real structural mismatch worth reporting)",
            f"Shapiro-Wilk normality p-values: x={self.normality_p_xyz[0]:.3f} "
            f"y={self.normality_p_xyz[1]:.3f} z={self.normality_p_xyz[2]:.3f} "
            f"yaw={self.normality_p_yaw:.3f} (p<0.05 = reject Gaussian at that axis)",
        ]
        return "\n".join(lines)


def characterize_noise(samples: List[PoseSample]) -> NoiseCharacterization:
    """Compute the empirical noise structure of a set of repeated pose
    estimates of what should be the SAME static object -- works identically
    whether `samples` came from a real camera or a synthetic generator,
    which is the point: the same function can validate itself against known
    synthetic parameters and later be pointed at real data with no changes."""
    from scipy import stats

    xyz = np.array([[s.x, s.y, s.z] for s in samples])
    yaws = np.array([s.yaw for s in samples])

    mean_xyz = xyz.mean(axis=0)
    std_xyz = xyz.std(axis=0, ddof=1)

    # circular mean/std for yaw (angles wrap at +-pi)
    mean_yaw = float(np.arctan2(np.sin(yaws).mean(), np.cos(yaws).mean()))
    yaw_err = np.arctan2(np.sin(yaws - mean_yaw), np.cos(yaws - mean_yaw))
    std_yaw = float(yaw_err.std(ddof=1))

    xy_err_mag = np.linalg.norm(xyz[:, :2] - mean_xyz[:2], axis=1)
    yaw_err_mag = np.abs(yaw_err)
    if xy_err_mag.std() > 0 and yaw_err_mag.std() > 0:
        xy_yaw_corr = float(np.corrcoef(xy_err_mag, yaw_err_mag)[0, 1])
    else:
        xy_yaw_corr = 0.0

    normality_p_xyz = np.array([
        stats.shapiro(xyz[:, i]).pvalue if len(samples) >= 3 else float("nan")
        for i in range(3)
    ])
    normality_p_yaw = stats.shapiro(yaw_err).pvalue if len(samples) >= 3 else float("nan")

    return NoiseCharacterization(
        n=len(samples), mean_xyz=mean_xyz, std_xyz=std_xyz,
        mean_yaw=mean_yaw, std_yaw=std_yaw,
        xy_yaw_correlation=xy_yaw_corr,
        normality_p_xyz=normality_p_xyz, normality_p_yaw=normality_p_yaw,
    )


def compare_to_synthetic_assumptions(
    real: NoiseCharacterization,
    assumed_pos_jitter: float,
    assumed_yaw_jitter_rad: float,
) -> str:
    """Report whether measured (real) noise matches the synthetic model's
    assumptions closely enough that results obtained under synthetic jitter
    should be expected to transfer, or whether the structural mismatch is
    large enough to explain non-transfer independent of any other cause."""
    lines = ["Comparison: measured noise vs. synthetic model's assumptions"]
    for axis, val in zip(("x", "y"), real.std_xyz[:2]):
        ratio = val / assumed_pos_jitter if assumed_pos_jitter > 0 else float("inf")
        lines.append(
            f"  {axis}: measured std={val*1000:.2f}mm vs. assumed={assumed_pos_jitter*1000:.2f}mm "
            f"(ratio={ratio:.2f}x)"
        )
    z_ratio_note = (
        "synthetic model assumes ZERO z jitter -- any measured z variance here "
        "is entirely unmodeled by the existing candidate-pool generators"
    )
    lines.append(f"  z: measured std={real.std_xyz[2]*1000:.2f}mm ({z_ratio_note})")
    yaw_ratio = real.std_yaw / assumed_yaw_jitter_rad if assumed_yaw_jitter_rad > 0 else float("inf")
    lines.append(
        f"  yaw: measured std={np.degrees(real.std_yaw):.2f}deg vs. "
        f"assumed={np.degrees(assumed_yaw_jitter_rad):.2f}deg (ratio={yaw_ratio:.2f}x)"
    )
    lines.append(f"  position-yaw correlation: {real.xy_yaw_correlation:+.3f} "
                 f"(synthetic model assumes exactly 0.0, i.e. independence)")
    return "\n".join(lines)


def _depth_to_point_cloud(depth_image, intrinsics, depth_scale=0.001, roi=None):
    """Deproject a raw depth image (uint16, mm, RealSense convention) into a
    (N,3) point cloud in the camera's own frame. intrinsics: dict with
    fx, fy, ppx, ppy (matches the values read directly from the RealSense
    SDK's own get_intrinsics(), see cameras/captures/ capture script --
    not re-derived or guessed). roi: optional (row_min,row_max,col_min,col_max)
    pixel crop applied before deprojection."""
    depth = depth_image
    if roi is not None:
        r0, r1, c0, c1 = roi
        depth = depth[r0:r1, c0:c1]
        row_offset, col_offset = r0, c0
    else:
        row_offset, col_offset = 0, 0

    rows, cols = np.nonzero(depth > 0)
    z = depth[rows, cols].astype(np.float64) * depth_scale
    x = (cols + col_offset - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (rows + row_offset - intrinsics["ppy"]) * z / intrinsics["fy"]
    return np.stack([x, y, z], axis=1)


def fit_table_plane(empty_table_depth_image, camera_intrinsics, depth_scale=0.001):
    """Calibration step: RANSAC-fit the table plane from an empty-table
    capture (no object present). Returns (plane_model, inlier_cloud) where
    plane_model is open3d's (a,b,c,d) for ax+by+cz+d=0. Run this ONCE per
    camera setup, before any object-noise captures."""
    import open3d as o3d
    points = _depth_to_point_cloud(empty_table_depth_image, camera_intrinsics, depth_scale)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.005, ransac_n=3, num_iterations=1000
    )
    return plane_model, pcd.select_by_index(inliers)


def estimate_pose_from_rgbd(depth_image, camera_intrinsics, table_plane_model,
                             roi=None, depth_scale=0.001, plane_distance_threshold=0.01,
                             min_cluster_points=50):
    """Classical, transparent pose estimate for a single static object on a
    calibrated table: (1) deproject depth to a point cloud, (2) remove
    points close to the pre-calibrated table plane, (3) take the largest
    connected component among what remains (valid for the single-object
    case this experiment tests -- NOT for cluttered/multi-object scenes),
    (4) centroid -> position, (5) PCA on the XY projection -> yaw, matching
    piper_candidate_selection.py's own _yaw_of definition for direct
    comparability with the sim's candidate-pool yaw.

    roi: optional (row_min,row_max,col_min,col_max) pixel crop -- tune once
    real captures show what's actually in frame (camera mount, robot arm,
    or other clutter may need excluding); left as None (no crop) until then.

    Returns a PoseSample. Raises RuntimeError (not a silent bad guess) if
    segmentation finds no plausible object cluster -- this is meant to fail
    loudly during the calibration/tuning phase, not produce a confidently
    wrong pose."""
    import open3d as o3d

    points = _depth_to_point_cloud(depth_image, camera_intrinsics, depth_scale, roi=roi)
    if len(points) == 0:
        raise RuntimeError("estimate_pose_from_rgbd: no valid depth points in ROI")

    a, b, c, d = table_plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    dist_to_plane = points @ normal + d
    # abs(), not a signed threshold: RANSAC's plane normal direction is
    # arbitrary (segment_plane doesn't guarantee it points "toward the
    # camera" vs "away") -- found via a synthetic-scene test that a signed
    # check silently returned zero points because the fitted normal
    # happened to point the opposite way expected. A downward-looking
    # camera physically cannot see points below the table anyway (occluded
    # by the table itself), so abs() is safe here, not just convenient.
    above_table = points[np.abs(dist_to_plane) > plane_distance_threshold]
    if len(above_table) < min_cluster_points:
        raise RuntimeError(
            f"estimate_pose_from_rgbd: only {len(above_table)} points above table "
            f"(need >= {min_cluster_points}) -- check ROI/plane_distance_threshold "
            f"before trusting this as 'no object detected'"
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(above_table)
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=min_cluster_points))
    if labels.max() < 0:
        raise RuntimeError("estimate_pose_from_rgbd: DBSCAN found no cluster above min_cluster_points")
    counts = np.bincount(labels[labels >= 0])
    largest_label = np.argmax(counts)
    cluster = above_table[labels == largest_label]

    centroid = cluster.mean(axis=0)
    xy_centered = cluster[:, :2] - centroid[:2]
    cov_xy = np.cov(xy_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov_xy)
    dominant_axis = eigvecs[:, np.argmax(eigvals)]  # largest-variance direction in-plane
    yaw = float(np.arctan2(dominant_axis[1], dominant_axis[0]))

    return PoseSample(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2]), yaw=yaw)
