"""SO-101 blocked-closure contact microbenchmark.

Zero production-code diff (same pattern as every experiment in this thread:
patches an already-compiled MjModel's numeric fields; move_gripper,
GRIP_CLOSED/GRIP_OPEN, register_primitive_geom, _build_scene_xml,
_solve_ik_jaw_pos_only all used unmodified).

Why this exists
----------------
The prior solref sweep tried to find a "stability plateau" using full grasp
success rate and found the signal was too noisy to trust -- non-monotonic,
scene-dependent, most likely dominated by IK/approach convergence chaos (tiny
floating-point differences from different contact parameters, propagated
through ~240 settle steps and iterative IK, occasionally landing on a
different local solution) rather than genuine contact-stiffness effects.

This benchmark removes that confound by construction rather than trying to
average it out:

    candidate generation   REMOVED
    IK / approach           SOLVED ONCE, then FROZEN and reused verbatim
    park / restore cycle    REMOVED (object is placed directly, once)
    weld                    REMOVED (nothing to weld -- arm never lifts)
    ACT / recovery          not in scope

The arm never moves except the gripper closing. The object is placed at a
deterministic pose derived from the frozen arm pose's pad geometry, not
sampled from a seed. This is the fully-isolated jaw-closing dynamics ONLY.

Fixtures only (not Hammer/TomatoSoupCan/Banana): getting an asymmetric CoACD
mesh into a known, reproducible orientation relative to the closing axis
without a settle/drop is its own small project. The 30mm box and 40mm
cylinder are exactly the objects this thread already has known-thickness
ground truth for, and are what "证明可重复" and "对比几个配置" need.

Two phases:
  1. Repeatability: run the SAME trial 20 times, check whether outputs are
     bit-identical or vary -- directly answers the "is the rollout
     deterministic" question this thread's cross-run discrepancy left open,
     for the first time with a design that isn't itself the source of the
     nondeterminism question (no RNG-sampled spawn).
  2. Config comparison: repeat across a few contact configs, now that any
     variation within a config's repeats has already been characterized.

Fingerprinting: each record includes the ACTUALLY COMPILED contact
parameters read back from the model (not the requested config), plus enough
of the initial state to distinguish "these two runs used different physics"
from "these two runs used identical physics and genuinely diverged" -- the
distinction the sibling solref-sweep doc could not make for its cross-run
discrepancy.

Run:  conda run -n tango python scripts/microbenchmark_blocked_closure.py
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
    register_primitive_geom,
)

OUT_REPEAT = Path(__file__).resolve().parent.parent / "outputs" / "microbenchmark_repeatability.jsonl"
OUT_CONFIG = Path(__file__).resolve().parent.parent / "outputs" / "microbenchmark_config_compare.jsonl"

FIXTURE_BOX = dict(shape="box", size=(0.05, 0.05, 0.015), mass=0.15, name="FixtureBox30mm")
FIXTURE_CYL = dict(shape="cylinder", size=(0.02, 0.05), mass=0.15, name="FixtureCyl40mm")
KNOWN_THICKNESS_M = {"FixtureBox30mm": 0.030, "FixtureCyl40mm": 0.040}

CLOSE_STEPS = 400   # generous; closing settles well before this at any config

CONFIGS = {
    "S0_baseline": dict(solref=(0.02, 1.0),
                        solimp=(0.9, 0.95, 0.001, 0.5, 2), priority=0),
    "S1_stiff_pads": dict(solref=(0.005, 1.0),
                          solimp=(0.95, 0.999, 0.0001, 0.5, 2), priority=1),
    "S1b_7_5ms": dict(solref=(0.0075, 1.0),
                      solimp=(0.95, 0.999, 0.0001, 0.5, 2), priority=1),
}
WARNING_TYPES = [mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL,
                 mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADCTRL]


def apply_contact_config(env, cfg):
    m = env.model
    for gid in env._jaw_pad_geom_ids:
        m.geom_solref[gid] = list(cfg["solref"])
        m.geom_solimp[gid] = list(cfg["solimp"])
        m.geom_priority[gid] = cfg["priority"]
    # Read back what actually landed, not what was requested -- MuJoCo modeling
    # docs: unequal geom priority means the higher-priority geom's own
    # solref/solimp/friction win outright; equal priority means they're mixed
    # via solmix. Recording the compiled result, not the intent, is the point.
    gid0 = env._jaw_pad_geom_ids[0]
    return {
        "requested": cfg,
        "compiled_pad_solref": m.geom_solref[gid0].tolist(),
        "compiled_pad_solimp": m.geom_solimp[gid0].tolist(),
        "compiled_pad_priority": int(m.geom_priority[gid0]),
        "compiled_pad_friction": m.geom_friction[gid0].tolist(),
        "compiled_solver": int(m.opt.solver),
        "compiled_cone": int(m.opt.cone),
        "compiled_impratio": float(m.opt.impratio),
        "compiled_timestep": float(m.opt.timestep),
    }


def freeze_arm_qpos(env) -> np.ndarray:
    """Solve IK exactly once, at a fixed target, and return the resulting arm
    qpos to be reused verbatim by every trial. This is the ONLY IK solve in
    the whole benchmark -- it happens before any contact config is applied and
    is never repeated, so it cannot be a source of per-trial or per-config
    variance."""
    target = np.array([0.0, -0.40, TABLE_TOP_Z + 0.06])
    ok, pe, _ = env._solve_ik_jaw_pos_only(target, reset_to_home=True, silent=True)
    if not ok:
        raise RuntimeError(f"one-time IK freeze did not converge, pe={pe}")
    return np.array([env.data.qpos[a] for a in env._arm_qpos_adr])


def place_object_at_pad_gap(env, oid, fixture_name: str):
    """Deterministically place the object centred at the pad midpoint, oriented
    so its known thickness axis aligns with the jaw's closing axis. The
    gripper starts fully open (GRIP_OPEN, ~70-95mm true opening depending on
    the jaw's mechanical range) around an object whose known thickness is
    30-40mm, so an initial clearance on each side falls out automatically --
    no separate gap offset needed, and closing dynamics/contact timing are
    genuinely exercised rather than starting pre-contacted.

    No RNG. Pose is a deterministic function of the (frozen) arm pose and the
    fixture's known geometry.
    """
    jm = env._jaw_metrology
    mujoco.mj_forward(env.model, env.data)
    pf, pm = env._jaw_pad_geom_ids
    mid = 0.5 * (env.data.geom_xpos[pf] + env.data.geom_xpos[pm])
    closing_axis = jm.closing_axis(env.data)   # unit vector, fixed <-> moving

    up = np.array([0.0, 0.0, 1.0])
    if abs(float(up @ closing_axis)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    axis2 = up - closing_axis * float(up @ closing_axis)
    axis2 /= np.linalg.norm(axis2)
    axis3 = np.cross(closing_axis, axis2)

    half_thick = KNOWN_THICKNESS_M[fixture_name] / 2.0
    # Fixture-local frame: box's local z is its 15mm half-extent (thickness);
    # cylinder's local z is its axial (height) direction, so its DIAMETER
    # (not its height) faces the closing direction -- align cylinder z with
    # axis3 (perpendicular to closing), box z with closing_axis directly.
    if fixture_name == "FixtureBox30mm":
        R = np.column_stack([axis2, axis3, closing_axis])   # local z -> closing
    elif fixture_name == "FixtureCyl40mm":
        R = np.column_stack([closing_axis, axis3, axis2])   # local z -> axis2 (perp)
    else:
        raise ValueError(fixture_name)
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.T.ravel())   # mj wants row-major; R is column-major here

    pos = mid   # object centre AT the pad midpoint; the requested gap is
               # expressed by how far OPEN the gripper starts (see run_trial),
               # not by offsetting the object off-centre.

    slot = env._obj_pool_slot(oid)
    jnt = env.model.joint(f"obj_joint_{slot}")
    adr, vadr = jnt.qposadr[0], jnt.dofadr[0]
    env.data.qpos[adr:adr + 3] = pos
    env.data.qpos[adr + 3:adr + 7] = quat
    env.data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    return half_thick


def run_trial(pool_key, logical_name, cfg_name, cfg, frozen_qpos, repeat_idx) -> dict:
    env = EnvironmentSoArm(obj_names=[pool_key], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True,
                           jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED)
    try:
        oid = env.load_obj(pool_key, name=logical_name, pos=[0, -0.40, TABLE_TOP_Z + 0.5])
        if len(env._jaw_pad_geom_ids) != 2:
            raise RuntimeError("pad geoms not found")
        config_fingerprint = apply_contact_config(env, cfg)

        # Freeze the arm: set once, hold via ctrl for the whole trial.
        for adr, q in zip(env._arm_qpos_adr, frozen_qpos):
            env.data.qpos[adr] = q
        for act_id, q in zip(env._arm_act_ids, frozen_qpos):
            env.data.ctrl[act_id] = q
        env.data.qpos[env._grip_qpos_adr] = GRIP_OPEN
        env.data.ctrl[env._grip_act_id] = GRIP_OPEN
        env.data.qvel[:] = 0.0

        place_object_at_pad_gap(env, oid, logical_name)

        obj_bid = env.model.body(f"obj_{env._obj_pool_slot(oid)}").id
        p0 = env.get_obj_pos(oid).copy()
        for w in WARNING_TYPES:
            env.data.warning[w].number = 0

        jm = env._jaw_metrology
        pf_gid, pm_gid = env._jaw_pad_geom_ids
        gids = obj_collision_geom_ids(env, oid)

        trace = {"min_dist_fixed": float("inf"), "min_dist_moving": float("inf"),
                 "max_obj_speed": 0.0, "first_contact_step": None,
                 "steady_dist_fixed": [], "steady_dist_moving": [],
                 "max_actuator_force": 0.0}
        n_steps = 0

        def probe():
            nonlocal n_steps
            n_steps += 1
            d = env._pad_to_obj_dist(gids) if gids else {}
            if d:
                trace["min_dist_fixed"] = min(trace["min_dist_fixed"], d["fixed"])
                trace["min_dist_moving"] = min(trace["min_dist_moving"], d["moving"])
                if trace["first_contact_step"] is None and d["fixed"] < 0 and d["moving"] < 0:
                    trace["first_contact_step"] = n_steps
                if n_steps > CLOSE_STEPS - 50:   # last ~50 steps = "steady state"
                    trace["steady_dist_fixed"].append(d["fixed"])
                    trace["steady_dist_moving"].append(d["moving"])
            v = float(np.linalg.norm(env.data.cvel[obj_bid][3:6]))
            trace["max_obj_speed"] = max(trace["max_obj_speed"], v)
            f = abs(float(env.data.actuator_force[env._grip_act_id]))
            trace["max_actuator_force"] = max(trace["max_actuator_force"], f)

        for _ in range(CLOSE_STEPS):
            # arm ctrl held fixed throughout; only the gripper actuator moves.
            env.data.ctrl[env._grip_act_id] = 0.0   # GRIP_CLOSED-equivalent target;
                                                    # legacy move_gripper() maps its
                                                    # own opening_m through this same
                                                    # ctrl channel -- setting it
                                                    # directly here is the same
                                                    # mechanism, just held constant
                                                    # rather than ramped, since the
                                                    # position actuator ramps its
                                                    # OWN response regardless.
            env.step_simulation()
            probe()

        pf = env.data.geom_xpos[pf_gid]
        pmv = env.data.geom_xpos[pm_gid]
        q_final = float(env.data.qpos[env._grip_qpos_adr])
        p1 = env.get_obj_pos(oid)

        warning_counts = {str(w).split(".")[-1]: int(env.data.warning[w].number)
                          for w in WARNING_TYPES}

        return {
            "config": cfg_name,
            "config_fingerprint": config_fingerprint,
            "object": logical_name,
            "repeat_idx": repeat_idx,
            "known_thickness_m": KNOWN_THICKNESS_M[logical_name],
            "frozen_arm_qpos": frozen_qpos.tolist(),
            "initial_obj_pos": p0.tolist(),
            "final_obj_pos": p1.tolist(),
            "obj_displacement_m": float(np.linalg.norm(p1 - p0)),
            "final_true_opening_m": jm.true_opening_m(q_final),
            "final_grip_qpos_rad": q_final,
            "min_pad_dist_fixed_m": (trace["min_dist_fixed"]
                                     if trace["min_dist_fixed"] != float("inf") else None),
            "min_pad_dist_moving_m": (trace["min_dist_moving"]
                                      if trace["min_dist_moving"] != float("inf") else None),
            "steady_pad_dist_fixed_m": (float(np.mean(trace["steady_dist_fixed"]))
                                        if trace["steady_dist_fixed"] else None),
            "steady_pad_dist_moving_m": (float(np.mean(trace["steady_dist_moving"]))
                                         if trace["steady_dist_moving"] else None),
            "first_contact_step": trace["first_contact_step"],
            "max_obj_speed_mps": trace["max_obj_speed"],
            "max_actuator_force": trace["max_actuator_force"],
            "warning_counts": warning_counts,
            "any_warning": any(warning_counts.values()),
        }
    finally:
        env.close()


def obj_collision_geom_ids(env, obj_id):
    """Resolve by body membership, not name -- register_primitive_geom's inline
    <geom .../> has no `name`, so production `_obj_collision_geom_ids` (which
    resolves via `ycb_col_geom_{slot}`-style names) returns empty for
    fixtures. Same fix as scripts/experiment_solver_contact_attribution.py."""
    slot = env._obj_pool_slot(obj_id)
    bid = env.model.body(f"obj_{slot}").id
    return [gi for gi in range(env.model.ngeom)
           if env.model.geom_bodyid[gi] == bid and env.model.geom_contype[gi] != 0]


def mm(v):
    return "n/a" if v is None else f"{v*1000:+7.2f}mm"


def main():
    fixture_box_pool = register_primitive_geom(
        FIXTURE_BOX["shape"], FIXTURE_BOX["size"], FIXTURE_BOX["mass"])
    fixture_cyl_pool = register_primitive_geom(
        FIXTURE_CYL["shape"], FIXTURE_CYL["size"], FIXTURE_CYL["mass"])
    fixtures = [(fixture_box_pool, FIXTURE_BOX["name"]),
               (fixture_cyl_pool, FIXTURE_CYL["name"])]

    # One-time IK freeze -- uses a throwaway env at legacy defaults, unrelated
    # to whatever contact config is later swept. Cached and reused verbatim.
    probe_env = EnvironmentSoArm(obj_names=[fixtures[0][0]], vis=False,
                                 grasp_mode=GRASP_MODE_PHYSICS_WELD,
                                 enable_jaw_metrology=True,
                                 jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED)
    try:
        probe_env.load_obj(fixtures[0][0], name=fixtures[0][1], pos=[0, -0.40, TABLE_TOP_Z + 0.5])
        frozen_qpos = freeze_arm_qpos(probe_env)
    finally:
        probe_env.close()
    print(f"frozen arm qpos: {frozen_qpos.round(4).tolist()}")

    # ── Phase 1: repeatability ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("PHASE 1: repeatability -- same trial, 20 repeats, S1_stiff_pads, FixtureBox30mm")
    print("=" * 100)
    OUT_REPEAT.parent.mkdir(parents=True, exist_ok=True)
    repeat_records = []
    with OUT_REPEAT.open("w") as fh:
        for i in range(20):
            rec = run_trial(fixture_box_pool, FIXTURE_BOX["name"], "S1_stiff_pads",
                            CONFIGS["S1_stiff_pads"], frozen_qpos, i)
            repeat_records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  rep {i:2d}  min_fixed={mm(rec['min_pad_dist_fixed_m'])} "
                  f"min_moving={mm(rec['min_pad_dist_moving_m'])} "
                  f"final_opening={rec['final_true_opening_m']*1000:6.2f}mm "
                  f"disp={rec['obj_displacement_m']*1000:6.3f}mm")

    keys = ["min_pad_dist_fixed_m", "min_pad_dist_moving_m", "final_true_opening_m",
           "obj_displacement_m", "max_obj_speed_mps"]
    print("\nrepeatability summary (should be ~0 spread if deterministic):")
    for k in keys:
        vs = [r[k] for r in repeat_records if r[k] is not None]
        if vs:
            print(f"  {k:26s} min={min(vs):+.6e}  max={max(vs):+.6e}  "
                  f"spread={max(vs)-min(vs):.3e}")
    bit_identical = all(r["min_pad_dist_fixed_m"] == repeat_records[0]["min_pad_dist_fixed_m"]
                        for r in repeat_records)
    print(f"  bit-identical across all 20 repeats: {bit_identical}")

    # ── Phase 2: config comparison ───────────────────────────────────────────
    print("\n" + "=" * 100)
    print("PHASE 2: config comparison, 2 fixtures x 3 configs x 3 repeats")
    print("=" * 100)
    config_records = []
    with OUT_CONFIG.open("w") as fh:
        for cfg_name, cfg in CONFIGS.items():
            for pool_key, logical_name in fixtures:
                for rep in range(3):
                    rec = run_trial(pool_key, logical_name, cfg_name, cfg,
                                    frozen_qpos, rep)
                    config_records.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    gap = rec["final_true_opening_m"] - rec["known_thickness_m"]
                    warn = "  WARN" if rec["any_warning"] else ""
                    print(f"  [{cfg_name:16s} {logical_name:16s} rep{rep}] "
                          f"gap_vs_known={gap*1000:+6.2f}mm  "
                          f"min_pad=({mm(rec['min_pad_dist_fixed_m'])},"
                          f"{mm(rec['min_pad_dist_moving_m'])})  "
                          f"steady=({mm(rec['steady_pad_dist_fixed_m'])},"
                          f"{mm(rec['steady_pad_dist_moving_m'])})  "
                          f"contact_step={rec['first_contact_step']}{warn}")

    summarize_config(config_records)
    print(f"\nwrote {len(repeat_records)} repeatability + {len(config_records)} "
          f"config-comparison trials")


def summarize_config(records):
    import statistics
    print("\n" + "=" * 100)
    print("config comparison summary (median over repeats)")
    print("=" * 100)
    configs = list(CONFIGS.keys())
    objs = sorted({r["object"] for r in records})
    print(f"{'config':16s} " + " ".join(f"{o[:20]:>22}" for o in objs) + "  warnings")
    for c in configs:
        row = []
        n_warn = 0
        for o in objs:
            rs = [r for r in records if r["config"] == c and r["object"] == o]
            n_warn += sum(r["any_warning"] for r in rs)
            gaps = [r["final_true_opening_m"] - r["known_thickness_m"] for r in rs]
            row.append(f"{statistics.median(gaps)*1000:+22.2f}")
        print(f"{c:16s} " + " ".join(row) + f"  {n_warn}")


if __name__ == "__main__":
    main()
