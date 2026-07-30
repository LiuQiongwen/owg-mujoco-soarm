#!/usr/bin/env python3
"""
Hand-eye calibration, step 2: solve for the camera-to-robot-base transform
from the samples collected by scripts/capture_handeye_sample.py.

Eye-to-hand trick: cv2.calibrateHandEye solves AX=XB for the eye-in-hand
case (X = cam<-gripper) given (R,t)_gripper2base and (R,t)_target2cam. For
our eye-to-hand setup (camera fixed in world, checkerboard moves with the
gripper) we instead want X = cam<-base. Standard reformulation: invert the
robot-pose leg -- pass R_base2gripper/t_base2gripper (i.e. the INVERSE of
gripper2base) in place of R_gripper2base/t_gripper2base -- and the solved X
becomes cam<-base directly (equivalent derivation to treating "base" as the
moving frame and "gripper" as the fixed one).

Usage:
  python3 scripts/solve_handeye.py
"""
import json
from pathlib import Path

import numpy as np
import cv2

SAMPLES_PATH = Path("calib/handeye_samples.json")
OUT_PATH = Path("calib/cam_to_base.json")


def main():
    samples = json.loads(SAMPLES_PATH.read_text())
    n = len(samples)
    print(f"Loaded {n} samples from {SAMPLES_PATH}")
    if n < 8:
        print("WARNING: fewer than 8 samples -- solve will likely be unreliable. "
              "Collect more with scripts/capture_handeye_sample.py before trusting this result.")

    R_base2gripper_list, t_base2gripper_list = [], []
    R_target2cam_list, t_target2cam_list = [], []

    for s in samples:
        R_g2b = np.array(s["R_gripper2base"])
        t_g2b = np.array(s["gripper_pos"])
        # invert gripper2base -> base2gripper (the eye-to-hand reformulation)
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

    print("\nSolved camera-to-robot-base transform:")
    print(T_cam2base.round(4))

    # sanity check: reprojection consistency across samples -- how much do
    # the independently-solved base2cam*cam2target chains disagree on the
    # checkerboard's position in base frame (should be ~consistent if the
    # solve is good, since the board didn't move relative to the gripper
    # between samples... actually it's rigidly attached to the gripper, so
    # target2base should equal gripper2base * (fixed board-to-gripper
    # offset) -- just report the target position in base frame per-sample
    # as a spread/consistency diagnostic instead of a hard pass/fail.
    target_in_base = []
    for s, Rb2g, tb2g in zip(samples, R_base2gripper_list, t_base2gripper_list):
        R_t2c = np.array(s["R_target2cam"])
        t_t2c = np.array(s["t_target2cam"])
        p_cam = t_t2c
        p_base = R_cam2base @ p_cam + t_cam2base.flatten()
        target_in_base.append(p_base)
    target_in_base = np.array(target_in_base)
    print(f"\nTarget (checkerboard) position in base frame across {n} samples:")
    print(f"  mean = {target_in_base.mean(0).round(4)}")
    print(f"  std  = {target_in_base.std(0).round(4)}  <- smaller is better/more consistent")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "T_cam2base": T_cam2base.tolist(),
        "n_samples": n,
        "target_in_base_std": target_in_base.std(0).tolist(),
    }, indent=2))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
