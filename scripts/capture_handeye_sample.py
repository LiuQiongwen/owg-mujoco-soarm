#!/usr/bin/env python3
"""
Hand-eye calibration, step 1: capture one (robot pose, checkerboard-in-camera
pose) sample pair. Run this once per arm pose -- move the real SO-ARM101
(checkerboard taped rigidly to the gripper/end-effector) to a new pose,
then run this script; it appends one sample to a running JSON file. Collect
12-15 samples across varied positions/orientations (don't just translate --
vary orientation too, otherwise the rotation part of the solve is
underdetermined), then run scripts/solve_handeye.py.

Checkerboard: 8x6 internal corners, 15mm squares (confirmed by user 2026-07-13).

Eye-to-hand setup: the D435i is FIXED in the world overlooking the
workspace, NOT mounted on the arm -- the checkerboard moves with the
gripper instead. cv2.calibrateHandEye is formulated for eye-in-hand by
default; solve_handeye.py inverts the robot-pose leg (uses base<-gripper
instead of gripper<-base) to solve the equivalent eye-to-hand problem for
camera<-base directly, per the standard trick.

Usage (run once per arm pose, from the tango conda env):
  python3 scripts/capture_handeye_sample.py
  python3 scripts/capture_handeye_sample.py --reset   # clear samples and start over
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

CHECKERBOARD = (8, 6)   # internal corners (cols, rows)
SQUARE_SIZE_M = 0.015
SAMPLES_PATH = Path("calib/handeye_samples.json")


def get_real_joint_angles() -> np.ndarray:
    """Read current 5-DoF arm joint angles (radians) from the real follower."""
    from robots.soarm_real_backend import SOARMRealBackend
    backend = SOARMRealBackend(port="/dev/ttyACM0", robot_id="my_follower")
    backend.connect()
    try:
        q = backend.get_joint_positions()
    finally:
        backend.close()
    return q


def fk_eef_pose(q: np.ndarray):
    """Forward kinematics: 5 arm joint angles (radians) -> (pos (3,), R (3,3))
    of the EEF site, in sim-world/robot-base frame. Reuses the same MJCF
    model as HeadlessIKSolver, just reads site_xpos/site_xmat directly
    instead of running IK."""
    import mujoco
    from tango_robot.headless_ik import _build_scene_xml
    from tango_robot.env_soarm import ARM_JOINTS, EEF_SITE, GRIP_JOINT, GRIP_OPEN

    xml = _build_scene_xml([])
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    arm_qpos_adr = [model.joint(n).qposadr[0] for n in ARM_JOINTS]
    grip_qpos_adr = model.joint(GRIP_JOINT).qposadr[0]
    eef_site_id = model.site(EEF_SITE).id

    for adr, val in zip(arm_qpos_adr, q):
        data.qpos[adr] = val
    data.qpos[grip_qpos_adr] = GRIP_OPEN
    mujoco.mj_forward(model, data)

    pos = data.site_xpos[eef_site_id].copy()
    R = data.site_xmat[eef_site_id].reshape(3, 3).copy()
    return pos, R


def capture_checkerboard_pose():
    """Capture one RealSense color frame, detect the checkerboard, return
    (R_target2cam (3,3), t_target2cam (3,)) or None if not detected."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)

    try:
        for _ in range(15):
            pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        img = np.asanyarray(color.get_data())
    finally:
        pipeline.stop()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    debug_path = Path("/tmp/claude-1000/-lena/7288f7ab-dc84-4b44-a682-e7d1d9c85e05/scratchpad/handeye_debug.png")
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, found)
    cv2.imwrite(str(debug_path), img)

    if not found:
        print(f"[FAIL] Checkerboard not detected. Debug image saved -> {debug_path}")
        return None

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_M

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ok:
        print("[FAIL] solvePnP failed")
        return None
    R, _ = cv2.Rodrigues(rvec)
    print(f"[OK] Checkerboard detected. Debug image -> {debug_path}")
    return R, tvec.flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    if SAMPLES_PATH.exists() and not args.reset:
        samples = json.loads(SAMPLES_PATH.read_text())
    if args.reset:
        print("Resetting sample list.")

    q = get_real_joint_angles()
    pos, R_gripper2base = fk_eef_pose(q)

    result = capture_checkerboard_pose()
    if result is None:
        print("No sample added (checkerboard not detected). Adjust the board/camera and retry.")
        return
    R_target2cam, t_target2cam = result

    samples.append({
        "joint_angles_rad": q.tolist(),
        "gripper_pos": pos.tolist(),
        "R_gripper2base": R_gripper2base.tolist(),
        "R_target2cam": R_target2cam.tolist(),
        "t_target2cam": t_target2cam.tolist(),
    })
    SAMPLES_PATH.write_text(json.dumps(samples, indent=2))
    print(f"Sample {len(samples)} added -> {SAMPLES_PATH}")
    print(f"  gripper pos (robot-base frame): {pos.round(4)}")
    if len(samples) < 12:
        print(f"  Need >= 12 samples with varied position AND orientation for a good solve "
              f"({len(samples)}/12 so far).")
    else:
        print(f"  {len(samples)} samples collected -- enough to run scripts/solve_handeye.py, "
              f"though more (varied) samples improve accuracy.")


if __name__ == "__main__":
    main()
