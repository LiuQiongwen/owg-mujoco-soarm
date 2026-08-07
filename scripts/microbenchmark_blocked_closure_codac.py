"""Blocked-closure microbenchmark extended to Hammer/TomatoSoupCan/Banana.

Zero production-code diff. Same pattern as every experiment in this thread:
patches an already-compiled MjModel's numeric fields; move_gripper,
GRIP_CLOSED/GRIP_OPEN, register_primitive_geom, _build_scene_xml,
_solve_ik_jaw_pos_only all used unmodified.

Extends scripts/microbenchmark_blocked_closure.py's design (IK solved once
and frozen, no candidate/approach/park-restore/weld per trial) from the two
symmetric fixtures to the three real CoACD objects already implicated
throughout this thread. Per the diagnosis in
docs/CONTACT_ONSET_AUDIT_20260807.md, none of these objects has a single
well-defined "thickness" the way a 30mm box does -- Hammer's head and handle
differ by centimetres -- so instead of a global bounding-box thickness, this
uses **local collision support width along the closing axis**, sampled in a
slab around wherever the jaw actually is, which is what
tango_robot/jaw_metrology.py's `object_local_thickness_m` already computes
for exactly this reason.

One correction applied on top of that production method, NOT by editing it
(out of scope: "不改任何正式 API"): `object_local_thickness_m` centres its
measurement slab on `tip_points()`, the SAME mesh-tip point set the contact
onset audit found to disagree with the actual pad collision geometry by up to
several mm depending on joint angle. This script recomputes the equivalent
quantity centred on the PAD geom midpoint instead
(`local_collision_support_width` below), so this extension does not inherit
today's bug into new results. `closing_axis()` (a direction, not a position)
is reused as-is from JawMetrology -- the mesh-tip and pad-box surfaces are
close to parallel since both derive from the same finger geometry, so the
DIRECTION is low-risk even though the POSITION reference disagrees.

Per-object procedure (no RNG anywhere -- fully deterministic):
  1. Drop the object at a fixed table position, settle under gravity.
  2. Solve IK ONCE (per object, since each settles to a different pose) to
     bring the frozen jaw-midpoint to the settled object's centroid.
  3. Freeze that qpos; every subsequent trial (every config, every repeat)
     reuses it verbatim -- IK never resolves again, so config-to-config
     variation cannot come from IK convergence noise, exactly as in the
     fixture benchmark.
  4. Command closure; record pad_obj_dist (steady-state and min) and the
     local collision support width at the settled contact point.

Run:  conda run -n tango python scripts/microbenchmark_blocked_closure_codac.py
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
    TABLE_TOP_Z,
)
from tango_robot.jaw_metrology import object_collision_verts  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs" / "microbenchmark_codac.jsonl"
CLOSE_STEPS = 400
OBJECTS = ["HammerC", "TomatoSoupCanC", "BananaC"]
SPAWN_POS = [0.0, -0.40, TABLE_TOP_Z + 0.10]   # fixed, no RNG

CONFIGS = {
    "S0_baseline": dict(solref=(0.02, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2), priority=0),
    "S1_stiff_pads": dict(solref=(0.005, 1.0), solimp=(0.95, 0.999, 0.0001, 0.5, 2), priority=1),
    "S1b_7_5ms": dict(solref=(0.0075, 1.0), solimp=(0.95, 0.999, 0.0001, 0.5, 2), priority=1),
}
WARNING_TYPES = [mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL,
                 mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADCTRL]


def apply_contact_config(env, cfg):
    m = env.model
    for gid in env._jaw_pad_geom_ids:
        m.geom_solref[gid] = list(cfg["solref"])
        m.geom_solimp[gid] = list(cfg["solimp"])
        m.geom_priority[gid] = cfg["priority"]
    gid0 = env._jaw_pad_geom_ids[0]
    return {"compiled_pad_solref": m.geom_solref[gid0].tolist(),
           "compiled_pad_priority": int(m.geom_priority[gid0])}


def local_collision_support_width(env, obj_verts: np.ndarray, slab_half_m: float = 0.015):
    """object_local_thickness_m's logic, corrected to centre on the PAD
    midpoint (the real contact surface) instead of tip_points() (the surface
    the contact onset audit found to disagree with it by several mm)."""
    if not len(obj_verts):
        return None
    jm = env._jaw_metrology
    pf, pm = env._jaw_pad_geom_ids
    mid = 0.5 * (env.data.geom_xpos[pf] + env.data.geom_xpos[pm])
    centre = obj_verts[int(np.argmin(np.linalg.norm(obj_verts - mid, axis=1)))]
    axis = jm.closing_axis(env.data)
    rel = obj_verts - centre
    along = rel @ axis
    perp = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    for half in (slab_half_m, 2 * slab_half_m, 4 * slab_half_m):
        inslab = perp <= half
        if inslab.sum() >= 3:
            a = along[inslab]
            return float(a.max() - a.min())
    return None


def obj_collision_geom_ids(env, obj_id):
    slot = env._obj_pool_slot(obj_id)
    bid = env.model.body(f"obj_{slot}").id
    return [gi for gi in range(env.model.ngeom)
           if env.model.geom_bodyid[gi] == bid and env.model.geom_contype[gi] != 0]


def settle_and_freeze(env, logical_name) -> tuple:
    """Deterministic settle + one-time IK freeze. Returns (frozen_qpos,
    settled_obj_pose) for reuse across every trial on this object."""
    oid = env.load_obj(logical_name, name=logical_name, pos=SPAWN_POS)
    env._steps(240)
    env.wait_until_all_still(max_wait_epochs=200)
    centroid = env.get_obj_pos(oid).copy()
    ok, pe, _ = env._solve_ik_jaw_pos_only(centroid, reset_to_home=True, silent=True)
    if not ok:
        raise RuntimeError(f"{logical_name}: one-time IK freeze did not converge, pe={pe}")
    qpos = np.array([env.data.qpos[a] for a in env._arm_qpos_adr])
    pose = np.concatenate([env.data.qpos[env.model.joint(f'obj_joint_{env._obj_pool_slot(oid)}').qposadr[0]:
                                        env.model.joint(f'obj_joint_{env._obj_pool_slot(oid)}').qposadr[0] + 7]])
    return qpos, pose


def run_trial(logical_name, frozen_qpos, frozen_obj_pose, cfg_name, cfg, repeat_idx) -> dict:
    env = EnvironmentSoArm(obj_names=[logical_name], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True,
                           jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED)
    try:
        oid = env.load_obj(logical_name, name=logical_name, pos=SPAWN_POS)
        if len(env._jaw_pad_geom_ids) != 2:
            raise RuntimeError("pad geoms not found")
        fingerprint = apply_contact_config(env, cfg)

        for adr, q in zip(env._arm_qpos_adr, frozen_qpos):
            env.data.qpos[adr] = q
        for act_id, q in zip(env._arm_act_ids, frozen_qpos):
            env.data.ctrl[act_id] = q
        env.data.qpos[env._grip_qpos_adr] = GRIP_OPEN
        env.data.ctrl[env._grip_act_id] = GRIP_OPEN

        slot = env._obj_pool_slot(oid)
        jnt = env.model.joint(f"obj_joint_{slot}")
        adr, vadr = jnt.qposadr[0], jnt.dofadr[0]
        env.data.qpos[adr:adr + 7] = frozen_obj_pose
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        gids = obj_collision_geom_ids(env, oid)
        obj_verts = object_collision_verts(env.model, env.data, gids) if gids else np.empty((0, 3))
        support_width = local_collision_support_width(env, obj_verts)

        obj_bid = env.model.body(f"obj_{slot}").id
        p0 = env.get_obj_pos(oid).copy()
        for w in WARNING_TYPES:
            env.data.warning[w].number = 0

        pf_gid, pm_gid = env._jaw_pad_geom_ids
        trace = {"min_fixed": float("inf"), "min_moving": float("inf"),
                 "max_obj_speed": 0.0, "first_contact_step": None,
                 "steady_fixed": [], "steady_moving": []}
        n_steps = 0

        def probe():
            nonlocal n_steps
            n_steps += 1
            d = env._pad_to_obj_dist(gids) if gids else {}
            if d:
                trace["min_fixed"] = min(trace["min_fixed"], d["fixed"])
                trace["min_moving"] = min(trace["min_moving"], d["moving"])
                if trace["first_contact_step"] is None and d["fixed"] < 0 and d["moving"] < 0:
                    trace["first_contact_step"] = n_steps
                if n_steps > CLOSE_STEPS - 50:
                    trace["steady_fixed"].append(d["fixed"])
                    trace["steady_moving"].append(d["moving"])
            v = float(np.linalg.norm(env.data.cvel[obj_bid][3:6]))
            trace["max_obj_speed"] = max(trace["max_obj_speed"], v)

        for _ in range(CLOSE_STEPS):
            env.data.ctrl[env._grip_act_id] = 0.0
            env.step_simulation()
            probe()

        p1 = env.get_obj_pos(oid)
        warning_counts = {str(w).split(".")[-1]: int(env.data.warning[w].number)
                          for w in WARNING_TYPES}

        return {
            "config": cfg_name,
            "config_fingerprint": fingerprint,
            "object": logical_name,
            "repeat_idx": repeat_idx,
            "local_support_width_m": support_width,
            "obj_displacement_m": float(np.linalg.norm(p1 - p0)),
            "min_pad_dist_fixed_m": (trace["min_fixed"] if trace["min_fixed"] != float("inf") else None),
            "min_pad_dist_moving_m": (trace["min_moving"] if trace["min_moving"] != float("inf") else None),
            "steady_pad_dist_fixed_m": (float(np.mean(trace["steady_fixed"])) if trace["steady_fixed"] else None),
            "steady_pad_dist_moving_m": (float(np.mean(trace["steady_moving"])) if trace["steady_moving"] else None),
            "first_contact_step": trace["first_contact_step"],
            "max_obj_speed_mps": trace["max_obj_speed"],
            "warning_counts": warning_counts,
            "any_warning": any(warning_counts.values()),
        }
    finally:
        env.close()


def mm(v):
    return "n/a" if v is None else f"{v*1000:+7.2f}mm"


def main():
    frozen = {}
    for obj in OBJECTS:
        env = EnvironmentSoArm(obj_names=[obj], vis=False,
                               grasp_mode=GRASP_MODE_PHYSICS_WELD,
                               enable_jaw_metrology=True,
                               jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED)
        try:
            qpos, pose = settle_and_freeze(env, obj)
        finally:
            env.close()
        frozen[obj] = (qpos, pose)
        print(f"{obj}: frozen arm qpos={qpos.round(4).tolist()}  "
              f"settled obj pose={pose.round(4).tolist()}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with OUT.open("w") as fh:
        # Repeatability check for one (object, config) cell before trusting
        # the rest -- same discipline as the fixture benchmark's Phase 1.
        print("\nrepeatability check: HammerC / S1_stiff_pads x 5")
        qpos, pose = frozen["HammerC"]
        rep_vals = []
        for i in range(5):
            r = run_trial("HammerC", qpos, pose, "S1_stiff_pads",
                         CONFIGS["S1_stiff_pads"], i)
            rep_vals.append(r["steady_pad_dist_fixed_m"])
            records.append(r)
            fh.write(json.dumps(r) + "\n")
        spread = (max(rep_vals) - min(rep_vals)) if all(v is not None for v in rep_vals) else None
        print(f"  steady_pad_dist_fixed_m across 5 repeats: {rep_vals}")
        print(f"  spread: {spread}")

        print("\nconfig comparison, 3 objects x 3 configs x 2 repeats")
        for cfg_name, cfg in CONFIGS.items():
            for obj in OBJECTS:
                qpos, pose = frozen[obj]
                for rep in range(2):
                    r = run_trial(obj, qpos, pose, cfg_name, cfg, rep)
                    records.append(r)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    warn = "  WARN" if r["any_warning"] else ""
                    sw = r["local_support_width_m"]
                    print(f"  [{cfg_name:16s} {obj:16s} rep{rep}] "
                          f"support_width={mm(sw)}  "
                          f"steady=({mm(r['steady_pad_dist_fixed_m'])},"
                          f"{mm(r['steady_pad_dist_moving_m'])})  "
                          f"min=({mm(r['min_pad_dist_fixed_m'])},"
                          f"{mm(r['min_pad_dist_moving_m'])}){warn}")

    summarize(records)
    print(f"\nwrote {len(records)} trials to {OUT}")


def summarize(records):
    import statistics
    print("\n" + "=" * 100)
    print("median steady pad distance (mm) by config x object")
    print("=" * 100)
    configs = list(CONFIGS.keys())
    objs = OBJECTS
    print(f"{'config':16s} " + " ".join(f"{o[:16]:>18}" for o in objs))
    for c in configs:
        row = []
        for o in objs:
            vs = [r["steady_pad_dist_fixed_m"] for r in records
                 if r["config"] == c and r["object"] == o
                 and r["steady_pad_dist_fixed_m"] is not None]
            row.append(f"{statistics.median(vs)*1000:18.2f}" if vs else f"{'n/a':>18}")
        print(f"{c:16s} " + " ".join(row))

    print("\nmedian local collision support width (mm) by object -- ground truth "
          "at the frozen contact point, contact-config-independent")
    for o in objs:
        vs = [r["local_support_width_m"] for r in records
             if r["object"] == o and r["local_support_width_m"] is not None]
        print(f"  {o:16s} {statistics.median(vs)*1000:.2f}mm" if vs else f"  {o:16s} n/a")


if __name__ == "__main__":
    main()
