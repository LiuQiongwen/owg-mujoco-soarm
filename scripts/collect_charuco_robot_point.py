#!/usr/bin/env python3
"""Read-only robot/camera correspondence capture for ChArUco calibration.

IMPORTANT correspondence contract (fixed 2026-08-05): earlier versions of this
script recorded the camera-observed charuco-corner position and the robot's
CURRENT forward-kinematics end-effector position with nothing verifying the
two were ever co-located -- eef_fk_base_m was just "wherever the arm happens
to be right now," independent of the detected corner. Existing records
sharing the same point_id showed eef_fk_base_m varying by ~10cm across
captures, which would be impossible if the gripper were actually re-touching
one fixed physical location each time; see
scripts/validate_handeye_holdout.py's module docstring for the full
investigation. That made calib/charuco_robot_points.jsonl unusable as a
ground-truth correspondence source.

This version requires an explicit --confirm-touch flag: the operator must
physically position the gripper reference point at the exact charuco corner
BEFORE running the script, and the flag is the recorded attestation that this
was done. The script cannot verify this itself (no independent ground truth
exists to check against -- that is exactly the thing this data collection is
trying to build). It also reports (informational only, not a gate) how far
the camera-observed corner position lands from the FK position under the
CURRENT (known-inaccurate, ~8cm RMSE per validate_handeye_holdout.py)
calibration, so a wildly implausible capture is visible immediately rather
than silently trusted.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path[:] = [item for item in sys.path if item not in ("", _PROJECT_ROOT)]
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: F401
sys.path.insert(0, _PROJECT_ROOT)
from paperA_data.scripts.real_hw_connect import connect_backend

CAM_TO_BASE_PATH = Path("calib/cam_to_base.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-id", required=True)
    ap.add_argument("--charuco-id", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", type=Path, default=Path("calib/charuco_robot_points.jsonl"))
    ap.add_argument(
        "--confirm-touch", action="store_true",
        help=(
            "Required. Attests that the gripper reference point (see "
            "--gripper-reference-point) is CURRENTLY physically positioned "
            "at the charuco corner named by --charuco-id, before this "
            "script is run. The script has no way to verify this itself; "
            "omitting this flag aborts before touching the robot or camera."
        ))
    ap.add_argument(
        "--gripper-reference-point", default="eef_site",
        help=(
            "Free-text description of what physical point on the gripper "
            "is assumed to coincide with the charuco corner (e.g. "
            "'eef_site' if fk_eef_pose's site origin itself was aligned to "
            "the corner, or a description of a probe/fingertip offset if "
            "not). Recorded verbatim so downstream analysis knows what "
            "correspondence was actually attested, rather than assuming."
        ))
    ap.add_argument("--notes", default="", help="Optional free-text operator notes.")
    args = ap.parse_args()

    if not args.confirm_touch:
        print(
            "Refusing to capture: --confirm-touch was not passed.\n"
            "Physically position the gripper reference point "
            f"({args.gripper_reference_point!r}) at charuco corner "
            f"{args.charuco_id} FIRST, then re-run with --confirm-touch. "
            "This script cannot verify touch itself -- see the module "
            "docstring for why earlier captures without this attestation "
            "are not usable as ground truth.",
            file=sys.stderr,
        )
        return 1

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((7, 5), 0.030, 0.022, dictionary)
    detector = cv2.aruco.ArucoDetector(dictionary) if hasattr(cv2.aruco, "ArucoDetector") else None
    charuco_detector = cv2.aruco.CharucoDetector(board) if hasattr(cv2.aruco, "CharucoDetector") else None
    pipeline = rs.pipeline(); config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    backend = connect_backend()
    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        intr = color_profile.get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
        dist = np.array(intr.coeffs, dtype=np.float64)
        detected = None
        seen_ids = set()
        depth_frame = None
        for _ in range(20):
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame(); depth_frame = frames.get_depth_frame()
            if not color or not depth_frame: continue
            image = np.asanyarray(color.get_data())
            if charuco_detector is not None:
                cc, ci, _, ids = charuco_detector.detectBoard(image)
                ok = cc is not None and ci is not None
            elif detector is not None:
                corners, ids, _ = detector.detectMarkers(image)
                ok, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, image, board)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
                ok, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, image, board)
            if not ok: continue
            seen_ids.update(int(x) for x in ci.reshape(-1))
            if ok and cc is not None and ci is not None:
                for corner, cid in zip(cc.reshape(-1, 2), ci.reshape(-1)):
                    if int(cid) == args.charuco_id:
                        detected = corner; break
            if detected is not None: break
        if detected is None:
            raise RuntimeError(
                f"ChArUco corner {args.charuco_id} not detected; "
                f"detected IDs={sorted(seen_ids)}"
            )
        u, v = map(float, detected)
        z = float(depth_frame.get_distance(int(round(u)), int(round(v))))
        if z <= 0:
            raise RuntimeError("invalid depth at detected ChArUco corner")
        p_cam = [(u - intr.ppx) * z / intr.fx, (v - intr.ppy) * z / intr.fy, z]
        q = np.asarray(backend.get_joint_positions(), dtype=float)
        from scripts.capture_handeye_sample import fk_eef_pose
        p_base, r_base = fk_eef_pose(q)
    finally:
        pipeline.stop(); backend.close()

    # Informational only, NOT a capture gate: how far apart the camera
    # observation and the FK position land under the CURRENT calibration.
    # This uses the same T_cam2base whose accuracy this correspondence data
    # is meant to eventually assess/improve, so a large residual here is
    # expected (validate_handeye_holdout.py already found ~8cm RMSE) and
    # must not be read as "the touch was wrong."
    camera_predicted_base_m = None
    residual_vs_fk_m = None
    if CAM_TO_BASE_PATH.is_file():
        T_cam2base = np.asarray(
            json.loads(CAM_TO_BASE_PATH.read_text())["T_cam2base"], dtype=float)
        camera_predicted_base_m = (
            T_cam2base[:3, :3] @ np.asarray(p_cam) + T_cam2base[:3, 3]).tolist()
        residual_vs_fk_m = float(np.linalg.norm(
            np.asarray(camera_predicted_base_m) - np.asarray(p_base)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": "1.1", "run_id": args.run_id, "point_id": args.point_id,
              "charuco_id": args.charuco_id, "pixel": [u, v], "point_camera_m": p_cam,
              "eef_fk_base_m": np.asarray(p_base).tolist(),
              "eef_rotation_base": np.asarray(r_base).tolist(),
              "touch_confirmed_by_operator": True,
              "gripper_reference_point": args.gripper_reference_point,
              "notes": args.notes,
              "camera_predicted_base_m_using_current_calibration": camera_predicted_base_m,
              "residual_vs_fk_m_using_current_calibration": residual_vs_fk_m,
              "residual_caveat": (
                  "Computed with the current T_cam2base, which "
                  "validate_handeye_holdout.py measured at ~8cm RMSE. A "
                  "large residual here reflects calibration error, not "
                  "necessarily a bad touch -- informational only."
              ),
              "timestamp": datetime.now(timezone.utc).isoformat()}
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
