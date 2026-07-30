"""
Phase 3 replay v2 calibration helper (2026-07-10): closes the real gripper
from fully open to fully closed with NOTHING between the jaws (table must be
clear), printing Present_Load at every step. Used to see what a "closing on
nothing" load profile looks like before trusting --load-threshold in
real_hw_replay_feedback_close.py on a live grasp.

Reuses feedback_close()/decode_load_magnitude() from that script directly --
no new closing logic, this is purely a calibration wrapper around it with a
near-disabled threshold (so it always closes fully) and verbose always on.

SAFETY: table must be clear under/between the jaws before running this --
it WILL close the gripper fully.

Usage:
    conda run -n tango python paperA_data/scripts/real_hw_calibrate_load.py
"""
import sys
import time

sys.path.insert(0, "/lena/projects/OWG-main/paperA_data/scripts")
from real_hw_connect import connect_backend
from real_hw_replay_feedback_close import feedback_close

backend = connect_backend()
try:
    print("[calibrate-load] resetting to home (gripper fully open)...")
    backend.reset()
    time.sleep(0.5)
    start_opening = backend.get_gripper_opening()
    print(f"[calibrate-load] starting opening={start_opening:.4f}m -- "
          f"closing fully with load_threshold=900 (effectively disabled) "
          f"so it runs to the end regardless of load readings.")
    print("[calibrate-load] confirm the table/jaws are CLEAR before this proceeds.")
    time.sleep(2)

    final_opening = feedback_close(
        backend, start_opening,
        load_threshold=900, step_m=0.003,
        stall_tol=0.004, stall_steps=3,
        min_opening=0.0, settle_s=0.15, verbose=True,
    )
    print(f"[calibrate-load] done. final_opening={final_opening:.4f}m "
          f"(should be ~0.0 if it closed fully on nothing, as expected)")
finally:
    backend.close()
    print("[calibrate-load] disconnected cleanly (torque disabled).")
