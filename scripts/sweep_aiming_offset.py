"""Cheap, decisive check: does shifting the IK aim point along the closing
axis fix bilateral engagement on Hammer/Banana?

Zero production-code diff. Reuses (imports, does not modify)
scripts/microbenchmark_blocked_closure_codac.py's machinery.

Why this is the right next question
------------------------------------
scripts/instrument_closure_trajectory.py found the moving pad reaches contact
almost immediately (step 5 on Hammer, step 2 on Banana) while the fixed side
is already 26-32mm away at that instant -- too early to be explained by
mid-closing dynamics alone, pointing at least partly to object-surface
asymmetry relative to a purely centroid-targeted IK reference (the aiming
point, not the object's true geometric centre relative to the two pads). But
`obj_perp_m` also kept growing well after that first contact (5mm at step 20
-> 8mm at step 380), which a pure aiming-offset fix would NOT explain -- that
looks like continuing slip/rotation under sustained one-sided contact,
possibly a friction/closure-kinematics effect specific to a single-hinge
(non-parallel) jaw.

This sweep separates the two: shift the ONE-TIME IK target by a fixed offset
along the closing axis (toward the fixed side, since that's the side that
undershoots) before freezing, and see whether SOME offset brings both final
pad distances near zero simultaneously.

  If yes at some offset  -> the problem is substantially the static aiming
                            reference (per this thread's contact onset audit,
                            plausible: the aiming point derivation has already
                            been shown once today to not match physically
                            meaningful reference surfaces).
  If no offset works      -> friction/closure-kinematics dominates; aiming
                            alone can't fix it, and the continuing drift
                            after first contact is the real story.

Run:  conda run -n tango python scripts/sweep_aiming_offset.py
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
    SPAWN_POS,
    apply_contact_config,
    obj_collision_geom_ids,
)

OUT = Path(__file__).resolve().parent.parent / "outputs" / "aiming_offset_sweep.jsonl"
OFFSETS_M = [-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015]
OBJECTS = ["HammerC", "BananaC"]
CONFIG_NAME = "S1_stiff_pads"


def settle_and_freeze_with_offset(env, logical_name: str, offset_m: float):
    """Same as microbenchmark_blocked_closure_codac.settle_and_freeze, except
    the IK target is the settled object's centroid SHIFTED by `offset_m`
    along the fixed-to-moving closing axis measured at the (undisturbed,
    GRIP_OPEN) pre-IK pose -- i.e. a positive offset moves the aim point
    toward the moving side, negative toward the fixed side.
    """
    oid = env.load_obj(logical_name, name=logical_name, pos=SPAWN_POS)
    env._steps(240)
    env.wait_until_all_still(max_wait_epochs=200)
    centroid = env.get_obj_pos(oid).copy()

    # Closing axis direction is well-defined at any pose (finger geometry is
    # rigid); take it at the current (pre-IK, arm-at-whatever-it-starts-at)
    # pose purely as a direction reference for the offset.
    jm = env._jaw_metrology
    axis = jm.closing_axis(env.data)
    target = centroid + offset_m * axis

    ok, pe, _ = env._solve_ik_jaw_pos_only(target, reset_to_home=True, silent=True)
    if not ok:
        raise RuntimeError(f"{logical_name} offset={offset_m}: IK did not converge, pe={pe}")
    qpos = np.array([env.data.qpos[a] for a in env._arm_qpos_adr])
    slot = env._obj_pool_slot(oid)
    jadr = env.model.joint(f"obj_joint_{slot}").qposadr[0]
    pose = env.data.qpos[jadr:jadr + 7].copy()
    return qpos, pose


def run_trial(logical_name, offset_m) -> dict:
    env = EnvironmentSoArm(obj_names=[logical_name], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True,
                           jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED)
    try:
        frozen_qpos, frozen_pose = settle_and_freeze_with_offset(env, logical_name, offset_m)

        oid = env.load_obj(logical_name, name=logical_name, pos=SPAWN_POS)
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

        gids = obj_collision_geom_ids(env, oid)
        obj_bid = env.model.body(f"obj_{slot}").id
        p0 = env.get_obj_pos(oid).copy()

        min_fixed = min_moving = float("inf")
        for _ in range(CLOSE_STEPS):
            env.data.ctrl[env._grip_act_id] = 0.0
            env.step_simulation()
            d = env._pad_to_obj_dist(gids) if gids else {}
            if d:
                min_fixed = min(min_fixed, d["fixed"])
                min_moving = min(min_moving, d["moving"])

        d_final = env._pad_to_obj_dist(gids) if gids else {}
        p1 = env.get_obj_pos(oid)

        return {
            "object": logical_name,
            "offset_mm": offset_m * 1000,
            "final_dist_fixed_m": d_final.get("fixed"),
            "final_dist_moving_m": d_final.get("moving"),
            "min_dist_fixed_m": min_fixed if min_fixed != float("inf") else None,
            "min_dist_moving_m": min_moving if min_moving != float("inf") else None,
            "obj_displacement_m": float(np.linalg.norm(p1 - p0)),
            "bilateral_final": bool(d_final.get("fixed", 1) < 0.002
                                    and d_final.get("moving", 1) < 0.002),
        }
    finally:
        env.close()


def mm(v):
    return "n/a" if v is None else f"{v*1000:+7.2f}mm"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with OUT.open("w") as fh:
        for obj in OBJECTS:
            print(f"\n=== {obj} ===")
            for off in OFFSETS_M:
                r = run_trial(obj, off)
                records.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                tag = " <-- BILATERAL" if r["bilateral_final"] else ""
                print(f"  offset={off*1000:+5.1f}mm  final=({mm(r['final_dist_fixed_m'])},"
                      f"{mm(r['final_dist_moving_m'])})  "
                      f"min=({mm(r['min_dist_fixed_m'])},{mm(r['min_dist_moving_m'])}){tag}")

    print("\n" + "=" * 90)
    print("did any offset achieve bilateral contact (both sides < 2mm) at settle?")
    print("=" * 90)
    for obj in OBJECTS:
        rs = [r for r in records if r["object"] == obj]
        hits = [r["offset_mm"] for r in rs if r["bilateral_final"]]
        print(f"  {obj:16s} bilateral at offsets: {hits if hits else '(none)'}")

    print(f"\nwrote {len(records)} trials to {OUT}")


if __name__ == "__main__":
    main()
