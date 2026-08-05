#!/usr/bin/env python3
"""
Independent hand-eye calibration validation via held-out sample split.

Read-only, software-only check. Does NOT move the robot and does NOT modify
any existing calib/ file -- it only reads calib/handeye_samples.json and
writes a new file, calib/handeye_validation.json, in the format
scripts/preflight_real_act_recovery.py expects.

Why a held-out split instead of calib/charuco_robot_points.jsonl:
scripts/collect_charuco_robot_point.py records the END EFFECTOR'S CURRENT FK
POSITION alongside a camera-observed charuco corner position, but never moves
or verifies the gripper is actually AT that corner -- eef_fk_base_m and
point_camera_m are two independently-sampled quantities, not a touch-point
ground-truth correspondence pair. (Confirmed by inspecting the collection
script and the raw data: entries sharing point_id="p01" have eef_fk_base_m
values that differ by ~10cm across captures, which would be impossible if
point_id denoted one physically re-touched location.) Computing a
"reprojection RMSE" from that file would silently compare two unrelated
positions and could produce a falsely reassuring number for a safety-relevant
go/no-go decision. This script does not use that file.

What this script actually computes instead:
scripts/solve_handeye.py fits T_cam2base with cv2.calibrateHandEye (TSAI,
eye-to-hand inversion trick) on the first 12 of 26 collected samples
(calib/handeye_samples_fit12.json == calib/handeye_samples.json[:12], verified
by direct comparison) and reports the *spread* of the reconstructed
target-in-base position across those same 12 training samples as an in-sample
consistency diagnostic -- not a true externally-referenced reprojection error,
and not evaluated on unseen data.

This script re-runs the identical fit (same TSAI/eye-to-hand math, copied
verbatim from solve_handeye.py) on those same first 12 samples, sanity-checks
that the refit reproduces the existing calib/cam_to_base.json, and then
applies the fitted transform to the remaining 14 samples (indices 12-25),
which were never used to fit anything. Those 14 raw samples are NOT 14
independent poses: index 12 duplicates the exact joint pose of training
sample 11 (repeat of an already-fit configuration, not new information), and
indices 21-25 are five repeated captures of one single pose. So there are
only 9 genuinely distinct held-out physical poses. Samples are grouped by
joint pose before computing spread, matching the cluster-aware treatment
already applied to the LGGSN pairwise analysis in this project, to avoid
pseudoreplication silently understating the true error.

The output's reprojection_rmse_m is the RMS radial spread (sqrt of the sum of
per-axis variances) of the held-out target-in-base estimate across the 9
distinct novel poses -- a held-out generalization proxy using the project's
own already-trusted diagnostic formula, NOT a vision-style reprojection error
against known ground truth. That distinction is preserved in the output JSON
so it is not misread as one.

Usage:
  python3 scripts/validate_handeye_holdout.py
  python3 scripts/validate_handeye_holdout.py --out calib/handeye_validation.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

SAMPLES_PATH = Path("calib/handeye_samples.json")
EXISTING_FIT_PATH = Path("calib/cam_to_base.json")
N_FIT = 12
POSE_ROUND_DECIMALS = 4


def fit_cam2base(samples):
    """Identical math to scripts/solve_handeye.py's main() fit step."""
    R_base2gripper_list, t_base2gripper_list = [], []
    R_target2cam_list, t_target2cam_list = [], []
    for s in samples:
        R_g2b = np.array(s["R_gripper2base"])
        t_g2b = np.array(s["gripper_pos"])
        R_b2g = R_g2b.T
        t_b2g = -R_b2g @ t_g2b
        R_base2gripper_list.append(R_b2g)
        t_base2gripper_list.append(t_b2g)
        R_target2cam_list.append(np.array(s["R_target2cam"]))
        t_target2cam_list.append(np.array(s["t_target2cam"]))

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper_list, t_base2gripper_list,
        R_target2cam_list, t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base.flatten()
    return T_cam2base


def target_in_base(sample, T_cam2base):
    R_cam2base = T_cam2base[:3, :3]
    t_cam2base = T_cam2base[:3, 3]
    p_cam = np.array(sample["t_target2cam"])
    return R_cam2base @ p_cam + t_cam2base


def pose_key(sample):
    return tuple(round(a, POSE_ROUND_DECIMALS) for a in sample["joint_angles_rad"])


def rms_radial_spread(points):
    points = np.asarray(points)
    std = points.std(axis=0)
    return float(np.sqrt(np.sum(std ** 2))), std.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(SAMPLES_PATH))
    ap.add_argument("--existing-fit", default=str(EXISTING_FIT_PATH))
    ap.add_argument("--out", default="calib/handeye_validation.json")
    args = ap.parse_args()

    samples = json.loads(Path(args.samples).read_text())
    if len(samples) <= N_FIT:
        raise SystemExit(
            f"Need more than {N_FIT} samples to hold any out; found {len(samples)}.")

    fit_samples = samples[:N_FIT]
    holdout_samples = samples[N_FIT:]
    fit_pose_keys = {pose_key(s) for s in fit_samples}

    T_cam2base_refit = fit_cam2base(fit_samples)

    existing = json.loads(Path(args.existing_fit).read_text())
    T_cam2base_existing = np.asarray(existing["T_cam2base"], dtype=float)
    max_abs_diff = float(np.max(np.abs(T_cam2base_refit - T_cam2base_existing)))
    refit_matches_existing = max_abs_diff < 1e-6

    # Group held-out samples by physical pose (joint angles) to avoid
    # pseudoreplication from repeated captures at the same configuration.
    clusters = {}
    for s in holdout_samples:
        clusters.setdefault(pose_key(s), []).append(s)

    novel_clusters = {k: v for k, v in clusters.items() if k not in fit_pose_keys}
    repeat_of_fit_clusters = {k: v for k, v in clusters.items() if k in fit_pose_keys}

    def cluster_mean_target_in_base(cluster_samples):
        pts = np.array([target_in_base(s, T_cam2base_refit) for s in cluster_samples])
        return pts.mean(axis=0)

    novel_cluster_means = [cluster_mean_target_in_base(v) for v in novel_clusters.values()]
    rmse_clustered, std_clustered = rms_radial_spread(novel_cluster_means)

    all_holdout_points = [target_in_base(s, T_cam2base_refit) for s in holdout_samples]
    rmse_naive, std_naive = rms_radial_spread(all_holdout_points)

    result = {
        "schema": "handeye_holdout_validation_v1",
        "generated_by": "scripts/validate_handeye_holdout.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Held-out split of calib/handeye_samples.json: T_cam2base refit on "
            "the same first 12 samples used by scripts/solve_handeye.py "
            "(cv2.calibrateHandEye, TSAI, eye-to-hand inversion trick, "
            "identical math), then applied to the remaining 14 samples that "
            "were never used in any fit. NOT a vision-style reprojection RMSE "
            "against independently-known ground truth -- no such ground-truth "
            "correspondence data currently exists in this project (see "
            "docstring: calib/charuco_robot_points.jsonl's point_id/charuco_id "
            "does not denote a verified touch-point correspondence). This is a "
            "held-out generalization check using the same target-in-base "
            "spread diagnostic scripts/solve_handeye.py already reports "
            "in-sample."
        ),
        "sanity_check": {
            "refit_reproduces_existing_cam_to_base_json": refit_matches_existing,
            "max_abs_element_diff": max_abs_diff,
            "note": (
                "Confirms this script's fit math matches scripts/solve_handeye.py "
                "before trusting the held-out extension below."
            ),
        },
        "n_fit_samples": len(fit_samples),
        "n_holdout_samples_raw": len(holdout_samples),
        "n_holdout_distinct_novel_poses": len(novel_clusters),
        "n_holdout_samples_repeating_a_fit_pose": sum(
            len(v) for v in repeat_of_fit_clusters.values()),
        "pseudoreplication_note": (
            "Of the 14 held-out samples, 1 duplicates the exact joint pose of "
            "a training sample (no new information) and 5 are repeated "
            "captures of a single other pose. Only 9 held-out samples "
            "correspond to genuinely distinct, never-fit physical "
            "configurations. reprojection_rmse_m below is computed over "
            "per-pose cluster means (n=9) to avoid repeated captures at one "
            "pose silently deflating the spread estimate."
        ),
        "reprojection_rmse_m": rmse_clustered,
        "reprojection_rmse_m_caveat": (
            "Held-out target-in-base RMS radial spread across 9 distinct "
            "novel poses, NOT an externally-referenced ground-truth "
            "reprojection error. Treat as a generalization consistency proxy, "
            "not a calibrated accuracy bound."
        ),
        "reprojection_rmse_m_std_per_axis": std_clustered,
        "reprojection_rmse_m_naive_all_14_raw_samples": rmse_naive,
        "reprojection_rmse_m_naive_std_per_axis": std_naive,
        "existing_in_sample_target_in_base_std_m": existing.get("target_in_base_std"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
