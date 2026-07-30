"""
Phase 3 diagnostic (round 4 follow-up, 2026-07-10): mirrors
real_hw_check_shoulder_pan_zero.py but targets shoulder_pan=97 degrees --
the actual angle the recorded consensus trajectory closes the gripper at
(see paperA_data/README.md, round 4 entry). HOME (pan=0) reaches sim world
+X; the object's real spawn is along sim world -Y, ~90 degrees away from
HOME's own reach direction, which this trajectory's IK resolves via a large
pan rotation. This test finds out which PHYSICAL direction that rotation
actually points to, so the marker can be placed on the correct axis instead
of guessing again.

Moves ONLY shoulder_pan to 97 degrees, leaving all other joints untouched at
HOME.

Usage:
    conda run -n tango python paperA_data/scripts/real_hw_check_shoulder_pan_97.py
"""
import time
import sys
sys.path.insert(0, "/lena/projects/OWG-main/paperA_data/scripts")
from real_hw_connect import connect_backend
import numpy as np

SHOULDER_PAN_IDX = 0
TARGET_DEG = 97.0

# connect_backend()'s default max_relative_target=30 deg clamps any single
# move_joints() command to 30 deg of travel -- a direct 0->97 deg jump would
# silently only move 30 deg. Step there in three small, safe increments
# instead of raising the clamp.
STEPS_DEG = [33.0, 66.0, TARGET_DEG]

backend = connect_backend()
try:
    backend.reset()
    q0 = backend.get_joint_positions()
    print("Position after reset (deg):", np.degrees(q0))

    q_target = q0.copy()
    for step_deg in STEPS_DEG:
        q_target[SHOULDER_PAN_IDX] = np.radians(step_deg)
        print(f"Moving shoulder_pan to {step_deg} deg (others unchanged)...")
        backend.move_joints(q_target, blocking=True)
        time.sleep(0.3)

    q1 = backend.get_joint_positions()
    print("Position after move (deg):", np.degrees(q1))
    print("\n>>> Look at the arm now: which physical direction is it pointing?")
    print(">>> (compare to where the marker/pear currently sits)")
finally:
    backend.close()
    print("Disconnected cleanly (torque disabled).")
