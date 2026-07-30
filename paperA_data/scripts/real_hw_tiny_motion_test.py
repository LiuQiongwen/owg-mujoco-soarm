"""
Phase 3 first-ever motion test on the physical SO-ARM101 (2026-07-10).
Minimal, conservative: moves ONLY wrist_roll (last joint, lowest collision
risk) by +10 degrees from its current position, holds briefly, reads back
to confirm the move happened as commanded, then returns to the original
position. All other joints and the gripper are left untouched throughout.

max_relative_target=30.0 (set in real_hw_connect.connect_backend) clamps
any single command to at most 30 degrees of joint travel, well above this
test's 10-degree move but a real safety backstop regardless.
"""
import sys
import time
sys.path.insert(0, "/lena/projects/OWG-main/paperA_data/scripts")
from real_hw_connect import connect_backend
import numpy as np

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
WRIST_ROLL_IDX = 4
DELTA_DEG = 10.0

backend = connect_backend()
try:
    q0 = backend.get_joint_positions()
    print("Start position (deg):", np.degrees(q0))

    q_target = q0.copy()
    q_target[WRIST_ROLL_IDX] += np.radians(DELTA_DEG)
    print(f"Commanding wrist_roll {DELTA_DEG:+.1f} deg -> target (deg):", np.degrees(q_target))
    backend.move_joints(q_target, blocking=True)
    time.sleep(0.5)

    q1 = backend.get_joint_positions()
    print("Position after move (deg):", np.degrees(q1))
    print(f"wrist_roll actual delta: {np.degrees(q1[WRIST_ROLL_IDX] - q0[WRIST_ROLL_IDX]):+.2f} deg "
          f"(commanded {DELTA_DEG:+.1f} deg)")

    print("Returning to start position...")
    backend.move_joints(q0, blocking=True)
    time.sleep(0.5)

    q2 = backend.get_joint_positions()
    print("Final position (deg):", np.degrees(q2))
    print("Residual offset from start (deg):", np.degrees(q2 - q0))
finally:
    backend.close()
    print("Disconnected cleanly (torque disabled).")
