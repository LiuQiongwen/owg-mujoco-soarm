"""
Phase 3 diagnostic: isolate calibration/coordinate-frame correctness from
the object-placement-mismatch explanation. Moves ONLY shoulder_pan to 0
degrees (the simulation's HOME_QPOS zero for this joint), leaving all other
joints untouched at their current position -- if this base rotation visibly
points the arm somewhere other than "straight ahead" relative to the table,
that's evidence of a calibration/coordinate-frame mismatch, not just the
already-documented open-loop object-placement gap.

Usage:
    conda run -n tango python paperA_data/scripts/real_hw_check_shoulder_pan_zero.py
"""
import sys
sys.path.insert(0, "/lena/projects/OWG-main/paperA_data/scripts")
from real_hw_connect import connect_backend
import numpy as np

SHOULDER_PAN_IDX = 0

backend = connect_backend()
try:
    q0 = backend.get_joint_positions()
    print("Current position (deg):", np.degrees(q0))

    q_target = q0.copy()
    q_target[SHOULDER_PAN_IDX] = 0.0
    print(f"Moving ONLY shoulder_pan to 0 deg (others unchanged) -> target (deg):",
          np.degrees(q_target))
    backend.move_joints(q_target, blocking=True)

    q1 = backend.get_joint_positions()
    print("Position after move (deg):", np.degrees(q1))
    print("\n>>> Look at the arm now: does shoulder_pan=0 point it straight ahead")
    print(">>> (aligned with the table's centre line), or visibly rotated left/right?")
finally:
    backend.close()
    print("Disconnected cleanly (torque disabled).")
