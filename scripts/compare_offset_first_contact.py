"""Decisive check: does the +10mm offset that fixes Hammer work by changing
WHICH surface the moving pad first contacts (and the resulting force/torque
direction), or by more precisely locating a static centre?

Zero production-code diff. Reuses (imports, does not modify)
scripts/microbenchmark_blocked_closure_codac.py and
scripts/sweep_aiming_offset.py's machinery.

docs/CAPTURE_REFERENCE_DERIVATION_20260807.md already showed the static
geometric answer (-1.2mm) and the working manual answer (+10mm) differ by an
order of magnitude and sign, ruling out "correcting a mis-centred aim point"
as the mechanism. This script tests the alternative directly: run the SAME
Hammer scene at offset=0 (fails) and offset=+10mm (succeeds), and compare,
step by step, WHERE the moving pad first touches the object, what direction
the contact force points, and how the object's rotation trajectory differs
from that point on.

If the first-contact point/normal differ substantially between the two runs,
and the object's subsequent rotation direction differs accordingly -- that is
direct evidence the closure outcome is contact-SEQUENCE dependent, not just a
matter of aim-point precision. If first contact is nearly identical between
the two runs (same point, same normal) and only the LATER dynamics differ,
that would point elsewhere.

Run:  conda run -n tango python scripts/compare_offset_first_contact.py
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
    CLOSE_STEPS,
    CONFIGS,
    apply_contact_config,
    obj_collision_geom_ids,
)
from scripts.sweep_aiming_offset import settle_and_freeze_with_offset  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs" / "offset_first_contact_compare.jsonl"
OBJECT = "HammerC"
OFFSETS_MM = [0.0, 10.0]
CONFIG_NAME = "S1_stiff_pads"


def quat_angle_diff_deg(q0, q1):
    dq = np.zeros(4)
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    mujoco.mju_mulQuat(dq, q0, q1_inv)
    ang = np.zeros(3)
    mujoco.mju_quat2Vel(ang, dq, 1.0)
    return float(np.degrees(np.linalg.norm(ang)))


def trace_with_offset(offset_mm: float) -> dict:
    env = EnvironmentSoArm(obj_names=[OBJECT], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True,
                           jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED)
    try:
        frozen_qpos, frozen_pose = settle_and_freeze_with_offset(env, OBJECT, offset_mm / 1000.0)
        oid = env.load_obj(OBJECT, name=OBJECT, pos=[0.0, -0.40, env.Z_TABLE_TOP + 0.10])
        apply_contact_config(env, CONFIGS[CONFIG_NAME])

        for adr, q in zip(env._arm_qpos_adr, frozen_qpos):
            env.data.qpos[adr] = q
        for act_id, q in zip(env._arm_act_ids, frozen_qpos):
            env.data.ctrl[act_id] = q
        env.data.qpos[env._grip_qpos_adr] = GRIP_OPEN
        env.data.ctrl[env._grip_act_id] = GRIP_OPEN

        slot = env._obj_pool_slot(oid)
        jadr = env.model.joint(f"obj_joint_{slot}").qposadr[0]
        env.data.qpos[jadr:jadr + 7] = frozen_pose
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        gids = set(obj_collision_geom_ids(env, oid))
        obj_bid = env.model.body(f"obj_{slot}").id
        obj_quat0 = env.data.qpos[jadr + 3:jadr + 7].copy()

        # pad geom ids for identifying which side a contact belongs to
        pf_gid, pm_gid = env._jaw_pad_geom_ids

        trace = []
        first_contact = {"moving": None, "fixed": None}
        for step in range(CLOSE_STEPS):
            env.data.ctrl[env._grip_act_id] = 0.0
            env.step_simulation()

            # Scan live contacts involving the object AND a pad geom.
            row = {"step": step,
                  "obj_ang_deg": quat_angle_diff_deg(obj_quat0, env.data.qpos[jadr + 3:jadr + 7]),
                  "obj_pos": env.data.qpos[jadr:jadr + 3].copy().tolist()}
            for ci in range(env.data.ncon):
                c = env.data.contact[ci]
                if c.geom1 not in gids and c.geom2 not in gids:
                    continue
                pad_gid = c.geom1 if c.geom1 in (pf_gid, pm_gid) else (
                    c.geom2 if c.geom2 in (pf_gid, pm_gid) else None)
                if pad_gid is None:
                    continue
                side = "moving" if pad_gid == pm_gid else "fixed"
                if first_contact[side] is None:
                    force6 = np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, ci, force6)
                    normal_world = c.frame[0:3]   # first row of the contact frame = normal
                    first_contact[side] = {
                        "step": step,
                        "pos": c.pos.tolist(),
                        "normal_world": np.array(normal_world).tolist(),
                        "normal_force": float(force6[0]),   # contact-frame x = normal
                        "tangential_force": float(np.linalg.norm(force6[1:3])),
                        "dist": float(c.dist),
                    }
            trace.append(row)

        return {
            "offset_mm": offset_mm,
            "first_contact_moving": first_contact["moving"],
            "first_contact_fixed": first_contact["fixed"],
            "final_obj_ang_deg": trace[-1]["obj_ang_deg"],
            "obj_ang_at_step_50": trace[min(50, len(trace) - 1)]["obj_ang_deg"],
            "obj_ang_at_step_100": trace[min(100, len(trace) - 1)]["obj_ang_deg"],
            "obj_pos_start": trace[0]["obj_pos"],
            "obj_pos_end": trace[-1]["obj_pos"],
        }
    finally:
        env.close()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with OUT.open("w") as fh:
        for off in OFFSETS_MM:
            print(f"\n=== offset = {off:+.1f}mm ===")
            r = trace_with_offset(off)
            results.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            for side in ("moving", "fixed"):
                fc = r[f"first_contact_{side}"]
                if fc is None:
                    print(f"  {side:7s}: never contacted")
                else:
                    print(f"  {side:7s}: step={fc['step']:4d}  pos={np.round(fc['pos'],4)}  "
                          f"normal={np.round(fc['normal_world'],3)}  "
                          f"normal_force={fc['normal_force']:.3f}N  "
                          f"tangential_force={fc['tangential_force']:.3f}N")
            print(f"  rotation: step50={r['obj_ang_at_step_50']:.2f}deg  "
                  f"step100={r['obj_ang_at_step_100']:.2f}deg  "
                  f"final={r['final_obj_ang_deg']:.2f}deg")

    print("\n" + "=" * 90)
    print("DIRECT COMPARISON: offset 0 vs +10mm")
    print("=" * 90)
    r0, r10 = results[0], results[1]
    for side in ("moving", "fixed"):
        fc0, fc10 = r0[f"first_contact_{side}"], r10[f"first_contact_{side}"]
        if fc0 is not None and fc10 is not None:
            pos_delta = np.linalg.norm(np.array(fc0["pos"]) - np.array(fc10["pos"]))
            normal_dot = float(np.array(fc0["normal_world"]) @ np.array(fc10["normal_world"]))
            print(f"  {side:7s} first-contact position shift: {pos_delta*1000:.2f}mm  "
                  f"normal alignment (dot product, 1=identical): {normal_dot:+.3f}  "
                  f"step {fc0['step']} vs {fc10['step']}")
        else:
            print(f"  {side:7s}: contact presence differs (0mm={fc0 is not None}, "
                  f"10mm={fc10 is not None}) -- itself a first-contact-topology change")
    print(f"\n  rotation direction/magnitude at step 100: "
          f"offset=0 -> {r0['obj_ang_at_step_100']:.2f}deg, "
          f"offset=+10mm -> {r10['obj_ang_at_step_100']:.2f}deg")

    print(f"\nwrote {len(results)} traces to {OUT}")


if __name__ == "__main__":
    main()
