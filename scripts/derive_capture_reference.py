"""Geometry-derived closure-aware capture reference, replacing COM-targeted aim.

Zero production-code diff. Reuses (imports, does not modify)
scripts/microbenchmark_blocked_closure_codac.py's machinery, plus
`tango_robot.jaw_metrology.object_collision_verts` and `JawMetrology.closing_axis`
(production, read-only).

Why this exists
----------------
docs/BILATERAL_ENGAGEMENT_MECHANISM_20260807.md found that a hand-searched
+10mm offset along the closing axis flips Hammer from persistent one-sided
contact to clean bilateral engagement, and explicitly warned against turning
that single number into an object-specific lookup table -- the open-world
goal needs the offset DERIVED from object geometry, not searched per object.

Method (first version, per the agreed design -- static two-support-surface
midpoint, not yet closure-trajectory-aware)
------------------------------------------------------------------------------
  1. bootstrap: solve IK once targeting the object's raw centroid (the OLD
     reference this whole mechanism investigation was built on).
  2. at that bootstrap pose, read the actual pad midpoint and closing axis,
     and find the object's local collision-vertex support along that axis in
     a slab near the jaw (same slab convention as
     tango_robot.jaw_metrology.object_local_thickness_m, but centred on the
     PAD geom midpoint -- the contact onset audit found tip_points()-centred
     slabs use the wrong reference surface, so this recomputes it correctly
     rather than reusing the production method as-is).
  3. capture_point = midpoint of the two along-axis extremes of that local
     support (NOT the object's overall centroid) -- this is "the point
     bilaterally between the two support surfaces the jaw would actually
     meet," derived from geometry, not searched.
  4. solve IK a second time, now targeting capture_point instead of the
     centroid, and freeze that as the trial's arm pose -- same downstream
     mechanics as every other benchmark in this thread (frozen pose, no
     candidate/approach/park-restore/weld, deterministic).

Run:  conda run -n tango python scripts/derive_capture_reference.py
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
from tango_robot.jaw_metrology import object_collision_verts  # noqa: E402
from scripts.microbenchmark_blocked_closure_codac import (  # noqa: E402
    CLOSE_STEPS,
    CONFIGS,
    SPAWN_POS,
    apply_contact_config,
    obj_collision_geom_ids,
)

OUT = Path(__file__).resolve().parent.parent / "outputs" / "derived_capture_reference.jsonl"
OBJECTS = ["HammerC", "BananaC", "TomatoSoupCanC"]
CONFIG_NAME = "S1_stiff_pads"
SLAB_HALF_M = 0.015


def local_support_extremes(env, obj_verts, slab_half_m=SLAB_HALF_M):
    """Two-support-surface geometry along the CURRENT pose's closing axis, in
    a slab centred on the PAD geom midpoint (corrected reference -- see
    module docstring). Returns (capture_point_world, width_m, centre_pt) or
    (None, None, None) if no surface falls in any tried slab width."""
    jm = env._jaw_metrology
    pf, pm = env._jaw_pad_geom_ids
    jaw_mid = 0.5 * (env.data.geom_xpos[pf] + env.data.geom_xpos[pm])
    axis = jm.closing_axis(env.data)
    if not len(obj_verts):
        return None, None, None
    centre_pt = obj_verts[int(np.argmin(np.linalg.norm(obj_verts - jaw_mid, axis=1)))]
    rel = obj_verts - centre_pt
    along = rel @ axis
    perp = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    for half in (slab_half_m, 2 * slab_half_m, 4 * slab_half_m):
        inslab = perp <= half
        if inslab.sum() >= 3:
            a = along[inslab]
            mid_along = 0.5 * (float(a.min()) + float(a.max()))
            capture_point = centre_pt + axis * mid_along
            return capture_point, float(a.max() - a.min()), centre_pt
    return None, None, None


def derive_and_freeze(env, logical_name: str):
    """Two-stage IK: bootstrap at centroid, then re-target the geometry-
    derived capture point. Returns (frozen_qpos, frozen_obj_pose, diagnostics)."""
    oid = env.load_obj(logical_name, name=logical_name, pos=SPAWN_POS)
    env._steps(240)
    env.wait_until_all_still(max_wait_epochs=200)
    centroid = env.get_obj_pos(oid).copy()

    ok0, pe0, _ = env._solve_ik_jaw_pos_only(centroid, reset_to_home=True, silent=True)
    if not ok0:
        raise RuntimeError(f"{logical_name}: bootstrap IK did not converge, pe={pe0}")

    gids = obj_collision_geom_ids(env, oid)
    verts = object_collision_verts(env.model, env.data, gids) if gids else np.empty((0, 3))
    capture_point, width, centre_pt = local_support_extremes(env, verts)

    if capture_point is None:
        # No local support found (shouldn't happen once bootstrapped near the
        # object) -- fall back to the centroid-only result rather than fail.
        capture_point = centroid
        width = None

    offset_from_centroid = capture_point - centroid
    ok1, pe1, _ = env._solve_ik_jaw_pos_only(capture_point, reset_to_home=True, silent=True)
    if not ok1:
        raise RuntimeError(f"{logical_name}: refined IK did not converge, pe={pe1}")

    qpos = np.array([env.data.qpos[a] for a in env._arm_qpos_adr])
    slot = env._obj_pool_slot(oid)
    jadr = env.model.joint(f"obj_joint_{slot}").qposadr[0]
    pose = env.data.qpos[jadr:jadr + 7].copy()

    jm = env._jaw_metrology
    axis_at_bootstrap = jm.closing_axis(env.data)   # for reporting the offset
                                                    # in closing-axis units
    diag = {
        "centroid": centroid.tolist(),
        "capture_point": capture_point.tolist(),
        "offset_from_centroid_m": offset_from_centroid.tolist(),
        "offset_along_closing_axis_mm": float(offset_from_centroid @ axis_at_bootstrap) * 1000,
        "local_support_width_m": width,
        "ik_pe_bootstrap": pe0,
        "ik_pe_refined": pe1,
    }
    return qpos, pose, diag


def run_trial(logical_name, frozen_qpos, frozen_obj_pose) -> dict:
    env = EnvironmentSoArm(obj_names=[logical_name], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True,
                           jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED)
    try:
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
        env.data.qpos[jadr:jadr + 7] = frozen_obj_pose
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
            env = EnvironmentSoArm(obj_names=[obj], vis=False,
                                   grasp_mode=GRASP_MODE_PHYSICS_WELD,
                                   enable_jaw_metrology=True,
                                   jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED)
            try:
                qpos, pose, diag = derive_and_freeze(env, obj)
            finally:
                env.close()
            print(f"  derived offset along closing axis: "
                  f"{diag['offset_along_closing_axis_mm']:+.2f}mm  "
                  f"(local support width: "
                  f"{diag['local_support_width_m']*1000 if diag['local_support_width_m'] else float('nan'):.1f}mm)")

            r_derived = run_trial(obj, qpos, pose)
            r_derived["method"] = "derived_capture_reference"
            r_derived.update(diag)
            records.append(r_derived)
            fh.write(json.dumps(r_derived) + "\n")
            fh.flush()

            tag = " <-- BILATERAL" if r_derived["bilateral_final"] else ""
            print(f"  [derived]  final=({mm(r_derived['final_dist_fixed_m'])},"
                  f"{mm(r_derived['final_dist_moving_m'])})  "
                  f"min=({mm(r_derived['min_dist_fixed_m'])},"
                  f"{mm(r_derived['min_dist_moving_m'])}){tag}")

    print("\n" + "=" * 90)
    print("summary: did the DERIVED (not searched) capture reference achieve bilateral contact?")
    print("=" * 90)
    for r in records:
        print(f"  {r['object']:16s} offset={r['offset_along_closing_axis_mm']:+6.2f}mm  "
              f"bilateral={r['bilateral_final']}")

    print(f"\nwrote {len(records)} trials to {OUT}")


if __name__ == "__main__":
    main()
