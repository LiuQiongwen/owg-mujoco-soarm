"""Closure trajectory instrumentation: when, by which side, and in what
direction does the object get pushed off-centre during blocked closure?

Zero production-code diff. Reuses (imports, does not modify)
scripts/microbenchmark_blocked_closure_codac.py's frozen-IK/deterministic-
placement machinery -- same object, same frozen arm pose, same S1 config that
showed near-perfect moving-side contact and zero fixed-side engagement on
Hammer/Banana in docs/BLOCKED_CLOSURE_CODAC_EXTENSION_20260807.md.

That doc confirmed the INITIAL setup is symmetric (both pads ~44mm from the
object centroid at freeze time) and the asymmetry develops during the 400-step
closing simulation. This script records fine-grained per-step data through
that window to find the mechanism, rather than guess at it.

Per step (every step, not sub-sampled -- 400 steps is cheap to log in full):
  q, dist_fixed, dist_moving (mj_geomDistance, exact),
  contact_fixed, contact_moving (bool, body-based, matches production
    bilateral_contact semantics),
  obj_pos, obj_quat, obj_lin_vel, obj_ang_vel,
  jaw_mid, obj_along_closing (signed, along the closing axis, relative to
    jaw_mid), obj_perp (lateral drift, perpendicular to closing axis).

Phases: PRE_CONTACT (neither pad touching) -> MOVING_ONLY (moving touches,
fixed doesn't) -> BILATERAL (both touching, if ever) -> POST_BILATERAL.

Three headline numbers per the diagnostic question:
  1. first-contact asymmetry: which side first, how far the other side was
  2. lateral drift before bilateral (or before the window ends, if bilateral
     never happens): how far the object moved off the closing axis
  3. rotation before bilateral: angular displacement of the object

Run:  conda run -n tango python scripts/instrument_closure_trajectory.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np

from tango_robot.env_soarm import (  # noqa: E402
    EnvironmentSoArm,
    GRASP_MODE_PHYSICS_WELD,
    GRIP_OPEN,
    JAW_CONTACT_MEASURED_PADS_AIMED,
)
from scripts.microbenchmark_blocked_closure_codac import (  # noqa: E402
    CONFIGS,
    SPAWN_POS,
    apply_contact_config,
    obj_collision_geom_ids,
    settle_and_freeze,
)

OUT_TRACE = Path(__file__).resolve().parent.parent / "outputs" / "closure_trajectory_trace.jsonl"
OUT_SUMMARY = Path(__file__).resolve().parent.parent / "outputs" / "closure_trajectory_summary.jsonl"
CLOSE_STEPS = 400
OBJECTS = ["HammerC", "BananaC"]
CONFIG_NAME = "S1_stiff_pads"   # the case that showed the asymmetry cleanly


def quat_angle_diff_deg(q0, q1):
    """Angular distance (deg) between two world-frame quaternions."""
    dq = np.zeros(4)
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    mujoco.mju_mulQuat(dq, q0, q1_inv)
    ang = np.zeros(3)
    mujoco.mju_quat2Vel(ang, dq, 1.0)
    return float(np.degrees(np.linalg.norm(ang)))


def trace_object(logical_name: str) -> dict:
    env = EnvironmentSoArm(obj_names=[logical_name], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True,
                           jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED)
    try:
        qpos, pose = settle_and_freeze(env, logical_name)
        oid = env.load_obj(logical_name, name=logical_name, pos=SPAWN_POS)
        apply_contact_config(env, CONFIGS[CONFIG_NAME])

        for adr, q in zip(env._arm_qpos_adr, qpos):
            env.data.qpos[adr] = q
        for act_id, q in zip(env._arm_act_ids, qpos):
            env.data.ctrl[act_id] = q
        env.data.qpos[env._grip_qpos_adr] = GRIP_OPEN
        env.data.ctrl[env._grip_act_id] = GRIP_OPEN

        slot = env._obj_pool_slot(oid)
        jnt = env.model.joint(f"obj_joint_{slot}")
        adr, vadr = jnt.qposadr[0], jnt.dofadr[0]
        env.data.qpos[adr:adr + 7] = pose
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        gids = obj_collision_geom_ids(env, oid)
        jm = env._jaw_metrology
        pf_gid, pm_gid = env._jaw_pad_geom_ids
        obj_bid = env.model.body(f"obj_{slot}").id
        obj_bodies = {obj_bid}

        # jaw midpoint / closing axis at the FROZEN pose (arm never moves
        # again, so these are constant for the whole trace -- computed once).
        jaw_mid0 = 0.5 * (env.data.geom_xpos[pf_gid] + env.data.geom_xpos[pm_gid])
        closing_axis = jm.closing_axis(env.data)
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(up @ closing_axis)) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        perp_axis = up - closing_axis * float(up @ closing_axis)
        perp_axis /= np.linalg.norm(perp_axis)
        perp_axis2 = np.cross(closing_axis, perp_axis)

        obj_quat0 = env.data.qpos[adr + 3:adr + 7].copy()

        trace = []
        for step in range(CLOSE_STEPS):
            env.data.ctrl[env._grip_act_id] = 0.0
            env.step_simulation()

            d = env._pad_to_obj_dist(gids) if gids else {}
            lc = rc = 0
            for ci in range(env.data.ncon):
                c = env.data.contact[ci]
                b1 = env.model.geom_bodyid[c.geom1]
                b2 = env.model.geom_bodyid[c.geom2]
                if not ({b1, b2} & obj_bodies):
                    continue
                if env._jaw_body_id in (b1, b2):
                    lc += 1
                if env._jaw_mv_body_id in (b1, b2):
                    rc += 1

            obj_pos = env.data.qpos[adr:adr + 3].copy()
            obj_quat = env.data.qpos[adr + 3:adr + 7].copy()
            rel = obj_pos - jaw_mid0
            along = float(rel @ closing_axis)
            perp = float(np.hypot(rel @ perp_axis, rel @ perp_axis2))

            trace.append({
                "step": step,
                "q": float(env.data.qpos[env._grip_qpos_adr]),
                "dist_fixed": d.get("fixed"),
                "dist_moving": d.get("moving"),
                "contact_fixed": bool(lc > 0),
                "contact_moving": bool(rc > 0),
                "obj_along_closing_m": along,
                "obj_perp_m": perp,
                "obj_ang_deg_from_start": quat_angle_diff_deg(obj_quat0, obj_quat),
                "obj_speed_mps": float(np.linalg.norm(env.data.cvel[obj_bid][3:6])),
            })

        return {"object": logical_name, "trace": trace}
    finally:
        env.close()


def summarize_trace(rec: dict) -> dict:
    trace = rec["trace"]
    fc_moving = next((t["step"] for t in trace if t["contact_moving"]), None)
    fc_fixed = next((t["step"] for t in trace if t["contact_fixed"]), None)
    fc_bilateral = next((t["step"] for t in trace
                         if t["contact_fixed"] and t["contact_moving"]), None)

    baseline_perp = trace[0]["obj_perp_m"]
    end_step = (fc_bilateral if fc_bilateral is not None else len(trace) - 1)
    drift = trace[end_step]["obj_perp_m"] - baseline_perp
    rotation = trace[end_step]["obj_ang_deg_from_start"]

    asym_note = None
    if fc_moving is not None and fc_fixed is None:
        # moving touched, fixed never did within the window -- report the
        # fixed-side distance AT the moment moving first touched.
        d_fixed_at_moving_contact = trace[fc_moving]["dist_fixed"]
        asym_note = {"first_side": "moving", "step": fc_moving,
                    "other_side_dist_m": d_fixed_at_moving_contact}
    elif fc_fixed is not None and fc_moving is None:
        d_moving_at_fixed_contact = trace[fc_fixed]["dist_moving"]
        asym_note = {"first_side": "fixed", "step": fc_fixed,
                    "other_side_dist_m": d_moving_at_fixed_contact}
    elif fc_moving is not None and fc_fixed is not None:
        first = "moving" if fc_moving < fc_fixed else "fixed"
        other_step = fc_moving if first == "fixed" else fc_fixed
        d_other = (trace[fc_fixed]["dist_moving"] if first == "fixed"
                  else trace[fc_moving]["dist_fixed"])
        asym_note = {"first_side": first, "step": min(fc_moving, fc_fixed),
                    "other_side_dist_m": d_other}

    return {
        "object": rec["object"],
        "first_contact_moving_step": fc_moving,
        "first_contact_fixed_step": fc_fixed,
        "first_contact_bilateral_step": fc_bilateral,
        "first_contact_asymmetry": asym_note,
        "lateral_drift_m": drift,
        "rotation_deg": rotation,
        "final_dist_fixed_m": trace[-1]["dist_fixed"],
        "final_dist_moving_m": trace[-1]["dist_moving"],
    }


def main():
    OUT_TRACE.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with OUT_TRACE.open("w") as ftr, OUT_SUMMARY.open("w") as fsum:
        for obj in OBJECTS:
            print(f"\n=== {obj} (config={CONFIG_NAME}) ===")
            rec = trace_object(obj)
            ftr.write(json.dumps(rec) + "\n")
            s = summarize_trace(rec)
            summaries.append(s)
            fsum.write(json.dumps(s) + "\n")

            print(f"  first_contact_moving_step   = {s['first_contact_moving_step']}")
            print(f"  first_contact_fixed_step    = {s['first_contact_fixed_step']}")
            print(f"  first_contact_bilateral_step= {s['first_contact_bilateral_step']}")
            if s["first_contact_asymmetry"]:
                a = s["first_contact_asymmetry"]
                od = a["other_side_dist_m"]
                print(f"  ASYMMETRY: {a['first_side']} touches first at step {a['step']}, "
                      f"other side is {od*1000 if od is not None else float('nan'):.1f}mm away")
            print(f"  lateral_drift (perp to closing axis) = {s['lateral_drift_m']*1000:+.2f}mm")
            print(f"  rotation from start                  = {s['rotation_deg']:+.2f}deg")
            print(f"  final dist (fixed/moving)            = "
                  f"{s['final_dist_fixed_m']*1000:+.2f}/{s['final_dist_moving_m']*1000:+.2f}mm")

            # A compact phase timeline: sample every 20 steps.
            print("  timeline (every 20 steps): step  q_deg  dist_f  dist_m  perp_mm  rot_deg  contact(F/M)")
            for t in rec["trace"][::20]:
                print(f"    {t['step']:4d}  {np.degrees(t['q']):6.1f}  "
                      f"{(t['dist_fixed'] or float('nan'))*1000:7.2f}  "
                      f"{(t['dist_moving'] or float('nan'))*1000:7.2f}  "
                      f"{t['obj_perp_m']*1000:7.2f}  {t['obj_ang_deg_from_start']:6.2f}  "
                      f"{int(t['contact_fixed'])}/{int(t['contact_moving'])}")

    print(f"\nwrote full trace to {OUT_TRACE}, summary to {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
