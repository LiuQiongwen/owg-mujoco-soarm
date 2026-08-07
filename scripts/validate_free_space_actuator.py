"""Step ② of the 2026-08-07 calibration order: does the gripper actuator reach
its OWN commanded target in free space (no object, no contact)?

This is deliberately narrower than fixing move_gripper()'s API. It answers one
question, cleanly isolated from both known confounds:

  - NOT the linear-map bug (opening_m claims meters, is actually radians) --
    that error is in what move_gripper() computes FOR the target; this script
    measures whether the actuator reaches whatever target it computed, using
    the legacy computation unmodified.
  - NOT contact/solver softness (the ~12-14mm free compression measured on
    Hammer/MediumClamp/Banana in the pad-fidelity diagnostic) -- there is no
    object in this scene, so nothing can push back on the jaw.

If target_q ~= settled_q here, any future gap seen WITH an object is
attributable to contact, not to actuator dynamics (gain/damping/settle time)
having its own free-space tracking error. If it does NOT hold here, that is a
third, previously uninvestigated defect, independent of both known ones, and
changes the calibration order below it.

Uses move_gripper() (legacy, unmodified) so the target-vs-settled comparison
is against exactly what the current control path actually commands, not a
hypothetical corrected one.

Run:  conda run -n tango python scripts/validate_free_space_actuator.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tango_robot.env_soarm import EnvironmentSoArm, GRASP_MODE_PHYSICS_WELD

REQUESTED_OPENINGS_M = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                        0.08, 0.09, 0.10]
SETTLE_STEPS = 200   # generous vs move_gripper's own default of 80, so a
                     # short settle window cannot masquerade as a real gap
# "Still moving" is judged on POSITION error, not velocity: a lightly-damped
# position controller has small residual qvel oscillation around its setpoint
# indefinitely (observed here: up to ~0.003 rad/s) without that meaning
# anything is actually unresolved. The physically meaningful question is
# whether the joint reached its target angle, and by how much it's off if not.
STILL_MOVING_TRACKING_ERROR_RAD = 0.001   # ~0.06 deg; well under the LUT's
                                          # resolution over the joint's 110 deg span
OUT = Path(__file__).resolve().parent.parent / "calib" / "jaw_free_space_actuator.json"


def main():
    env = EnvironmentSoArm(obj_names=[], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True)
    jm = env._jaw_metrology
    act_id = env._grip_act_id
    qadr = env._grip_qpos_adr

    rows = []
    try:
        for req in REQUESTED_OPENINGS_M:
            env.reset_robot()   # opens fully, zero velocity, clean start each time
            env.move_gripper(req, step=SETTLE_STEPS)
            target_q = float(env.data.ctrl[act_id])
            settled_q = float(env.data.qpos[qadr])
            settled_qvel = float(env.data.qvel[env.model.joint("gripper").dofadr[0]])
            actuator_force = float(env.data.actuator_force[act_id])
            forcerange = env.model.actuator_forcerange[act_id].tolist()
            saturated = abs(actuator_force) >= 0.98 * max(abs(f) for f in forcerange)
            row = {
                "requested_opening_m": req,
                "target_q_rad": target_q,
                "settled_q_rad": settled_q,
                "q_tracking_error_rad": settled_q - target_q,
                "settled_qvel_rad_s": settled_qvel,
                "still_moving": abs(settled_q - target_q) > STILL_MOVING_TRACKING_ERROR_RAD,
                "measured_opening_m": jm.true_opening_m(settled_q),
                "actuator_force": actuator_force,
                "forcerange": forcerange,
                "actuator_saturated": saturated,
            }
            rows.append(row)
            print(f"req={req*1000:5.1f}mm  target_q={target_q:8.4f}  "
                  f"settled_q={settled_q:8.4f}  "
                  f"err={row['q_tracking_error_rad']*1000:+7.2f}mrad  "
                  f"measured_opening={row['measured_opening_m']*1000:6.1f}mm  "
                  f"force={actuator_force:+6.3f} (range {forcerange})  "
                  f"{'SATURATED' if saturated else ''}"
                  f"{'  STILL MOVING' if row['still_moving'] else ''}")
    finally:
        env.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"settle_steps": SETTLE_STEPS, "rows": rows}, indent=1))

    max_err_mrad = max(abs(r["q_tracking_error_rad"]) for r in rows) * 1000
    any_saturated = any(r["actuator_saturated"] for r in rows)
    any_moving = any(r["still_moving"] for r in rows)
    print(f"\nmax |target_q - settled_q| = {max_err_mrad:.2f} mrad over "
          f"{len(rows)} free-space commands")
    print(f"any actuator saturation: {any_saturated}")
    print(f"any still-moving at settle horizon ({SETTLE_STEPS} steps): {any_moving}")
    if max_err_mrad < 1.0 and not any_saturated and not any_moving:
        print("VERDICT: actuator tracks its own free-space target cleanly. "
              "A future object-present gap is attributable to contact, not "
              "actuator dynamics.")
    else:
        print("VERDICT: actuator does NOT cleanly track its own free-space "
              "target -- a third, independent defect exists and should be "
              "investigated before attributing object-present gaps to "
              "contact alone.")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
