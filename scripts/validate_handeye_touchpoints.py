#!/usr/bin/env python3
"""
Ground-truth hand-eye validation from confirmed touch-point correspondences.

Read-only, software-only. Does not move the robot, does not modify any
existing calib/ file. Reads calib/charuco_robot_points.jsonl and writes a new
file, calib/handeye_touchpoint_validation.json.

This is the consumer half of the fix in scripts/collect_charuco_robot_point.py
(2026-08-05): that script now requires an explicit --confirm-touch attestation
before appending a record, and stamps confirmed records with
touch_confirmed_by_operator=true and schema_version>=1.1. This script only
trusts those records -- the 7 pre-fix records currently in
calib/charuco_robot_points.jsonl are schema_version 1.0 with no attestation
and are excluded (see scripts/validate_handeye_holdout.py's docstring for why
they cannot be used: eef_fk_base_m in those records is just wherever the arm
happened to be, not verified as touching the detected corner).

As of 2026-08-05 there are zero confirmed touch-point records yet -- this
script is forward-looking infrastructure, written ahead of data collection so
the analysis is ready the moment real touch-point sessions are run. With zero
or few confirmed points it will say so plainly rather than compute a
misleading number from too little data.

Two independent things this script computes once confirmed points exist:

1. reprojection_rmse_m: true reprojection RMSE of the EXISTING
   calib/cam_to_base.json transform against these ground-truth
   correspondences -- for each confirmed point, project the camera-observed
   corner into base frame with the existing T_cam2base and compare to the
   FK-measured end-effector position. This is what
   scripts/validate_handeye_holdout.py's reprojection_rmse_m field is a
   proxy for; once enough confirmed points exist, THIS number should replace
   that proxy as the one preflight_real_act_recovery.py gates on.

2. independent_refit: a Kabsch/Procrustes rigid-transform fit of T_cam2base
   directly from the confirmed correspondences (closed-form SVD alignment),
   completely independent of cv2.calibrateHandEye's AX=XB formulation used
   by scripts/solve_handeye.py. Needs >=3 non-collinear confirmed points.
   Agreement between this and the AX=XB solve is a strong cross-check;
   disagreement would indicate a problem in one method or the other. The
   Kabsch implementation is self-tested against synthetic data with a known
   transform on every run (see kabsch_self_test()) so a math bug can't
   silently produce a wrong "independent" cross-check.

Usage:
  python3 scripts/validate_handeye_touchpoints.py
  python3 scripts/validate_handeye_touchpoints.py --min-points-for-refit 3
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

POINTS_PATH = Path("calib/charuco_robot_points.jsonl")
CAM_TO_BASE_PATH = Path("calib/cam_to_base.json")
MIN_SCHEMA_VERSION = "1.1"


def kabsch(source_points, target_points):
    """Closed-form rigid transform (R, t) minimizing sum ||R@s_i + t - t_i||^2.

    source_points, target_points: (N, 3) arrays of corresponding points.
    Returns (R (3,3), t (3,)).
    """
    src = np.asarray(source_points, dtype=float)
    tgt = np.asarray(target_points, dtype=float)
    centroid_src = src.mean(axis=0)
    centroid_tgt = tgt.mean(axis=0)
    X = src - centroid_src
    Y = tgt - centroid_tgt
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = centroid_tgt - R @ centroid_src
    return R, t


def kabsch_self_test():
    """Verify kabsch() recovers a known synthetic transform exactly."""
    rng = np.random.default_rng(0)
    true_R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(true_R) < 0:
        true_R[:, 0] *= -1
    true_t = rng.normal(size=3)
    src = rng.normal(size=(6, 3))
    tgt = (true_R @ src.T).T + true_t
    R, t = kabsch(src, tgt)
    r_err = float(np.max(np.abs(R - true_R)))
    t_err = float(np.max(np.abs(t - true_t)))
    ok = r_err < 1e-8 and t_err < 1e-8
    return ok, r_err, t_err


def rotation_angle_deg(R):
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=str(POINTS_PATH))
    ap.add_argument("--existing-fit", default=str(CAM_TO_BASE_PATH))
    ap.add_argument("--out", default="calib/handeye_touchpoint_validation.json")
    ap.add_argument("--min-points-for-refit", type=int, default=3)
    args = ap.parse_args()

    self_test_ok, r_err, t_err = kabsch_self_test()
    if not self_test_ok:
        raise SystemExit(
            f"Kabsch self-test FAILED (rotation err={r_err}, translation "
            f"err={t_err}) -- refusing to compute an independent refit with "
            f"unverified math. This is a bug in this script, not the data.")

    points_path = Path(args.points)
    all_records = []
    if points_path.is_file():
        all_records = [json.loads(line) for line in points_path.read_text().splitlines() if line.strip()]

    def is_confirmed(r):
        return bool(r.get("touch_confirmed_by_operator")) and \
            str(r.get("schema_version", "1.0")) >= MIN_SCHEMA_VERSION

    confirmed = [r for r in all_records if is_confirmed(r)]

    result = {
        "schema": "handeye_touchpoint_validation_v1",
        "generated_by": "scripts/validate_handeye_touchpoints.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kabsch_self_test_passed": self_test_ok,
        "n_total_records_in_file": len(all_records),
        "n_confirmed_touch_points": len(confirmed),
        "min_schema_version_trusted": MIN_SCHEMA_VERSION,
    }

    if not confirmed:
        result["status"] = "no_confirmed_touch_points_yet"
        result["note"] = (
            "No records with touch_confirmed_by_operator=true found. All "
            f"{len(all_records)} existing record(s) predate the "
            "--confirm-touch fix to scripts/collect_charuco_robot_point.py "
            "and are not usable as ground truth. Run that script with "
            "--confirm-touch after physically touching the gripper to a "
            "known charuco corner to start building real ground-truth data. "
            "reprojection_rmse_m is intentionally omitted -- computing one "
            "from zero points would be meaningless."
        )
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        print(f"\nSaved -> {args.out}")
        print(
            "\nNo confirmed touch points yet. This is expected until a real "
            "touch-point session is run with the fixed collection script."
        )
        return

    existing = json.loads(Path(args.existing_fit).read_text())
    T_cam2base = np.asarray(existing["T_cam2base"], dtype=float)

    p_cam = np.array([r["point_camera_m"] for r in confirmed])
    p_base_fk = np.array([r["eef_fk_base_m"] for r in confirmed])
    p_base_predicted = (T_cam2base[:3, :3] @ p_cam.T).T + T_cam2base[:3, 3]
    residuals = np.linalg.norm(p_base_predicted - p_base_fk, axis=1)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    result["reprojection_rmse_m"] = rmse
    result["reprojection_rmse_m_note"] = (
        "True ground-truth reprojection RMSE of the existing "
        "calib/cam_to_base.json transform, computed from operator-confirmed "
        "touch-point correspondences. Small n -- treat with appropriate "
        "caution until more confirmed points are collected."
    )
    result["per_point_residuals_m"] = residuals.tolist()
    result["per_point_ids"] = [r.get("point_id") for r in confirmed]

    if len(confirmed) >= args.min_points_for_refit:
        R_indep, t_indep = kabsch(p_cam, p_base_fk)
        T_indep = np.eye(4)
        T_indep[:3, :3] = R_indep
        T_indep[:3, 3] = t_indep
        rot_diff_deg = rotation_angle_deg(R_indep.T @ T_cam2base[:3, :3])
        trans_diff_m = float(np.linalg.norm(t_indep - T_cam2base[:3, 3]))
        result["independent_refit"] = {
            "method": "Kabsch/Procrustes SVD rigid alignment, independent of "
                      "cv2.calibrateHandEye's AX=XB formulation",
            "T_cam2base": T_indep.tolist(),
            "rotation_diff_deg_vs_existing_fit": rot_diff_deg,
            "translation_diff_m_vs_existing_fit": trans_diff_m,
            "n_points_used": len(confirmed),
            "note": (
                "Cross-check only. Agreement with the existing AX=XB solve "
                "is reassuring; disagreement flags a problem in one method "
                "or the other, not automatically which one."
            ),
        }
    else:
        result["independent_refit"] = {
            "status": "insufficient_points",
            "n_points_used": len(confirmed),
            "min_required": args.min_points_for_refit,
        }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
