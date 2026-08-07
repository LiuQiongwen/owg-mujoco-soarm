"""Small-scale solver/contact attribution experiment.

Does NOT modify any production API. `EnvironmentSoArm`, `_build_scene_xml`,
`register_primitive_geom`, `move_gripper`, `GRIP_CLOSED`/`GRIP_OPEN` are all
used completely unmodified. The only thing this script does that isn't
"business as usual" is patch numeric fields on the ALREADY-COMPILED MjModel
(`model.opt.*`, `model.geom_solref/solimp/priority` for the two pad geoms
only) after construction, for the duration of this one throwaway process --
MuJoCo reads these fields live at every step, so this requires no MJCF change
and no recompilation, and vanishes when the process exits.

Question this answers
----------------------
The pad-fidelity diagnostic (docs/PAD_FIDELITY_DIAGNOSTIC_20260807.md) found
100% of legacy successes on Hammer/MediumClamp/Banana/TomatoSoupCan carry a
persistent excessive-penetration run (default 6mm ceiling) under stock MuJoCo
contact defaults. Free-space validation (step ②,
docs/JAW_OPENING_CALIBRATION_STEP1_20260807.md) ruled out the actuator itself
as the cause -- it tracks its own setpoint to 0.06 mrad with no object present.
This experiment asks the next question directly: is the penetration
attributable to CONTACT SOFTNESS (fixable by solver configuration), or does it
persist regardless of how stiff the contact model is made (pointing elsewhere
-- object mass/inertia, actuator force authority, mesh geometry)?

Design
------
Two canonical fixtures with EXACTLY KNOWN thickness (a rigid box and a rigid
cylinder via the existing `register_primitive_geom`/`load_primitive`
mechanism) plus the three real objects already implicated
(Hammer/TomatoSoupCan/Banana), each closed on at a fixed candidate under a
small matrix of contact configurations:

  S0  baseline       -- current defaults, unchanged (Newton solver, pyramidal
                        cone, impratio=1.0 are ALREADY MuJoCo's defaults here
                        -- confirmed by inspection, not literature -- so S0 is
                        not "unconfigured", it's what's actually running today)
  S1  stiff pads      -- pad geoms only: faster solref time constant (5ms vs
                        the current 20ms), narrower solimp band, and
                        geom_priority=1 so these params WIN outright over the
                        object's own (MuJoCo mixes contact params by simple
                        average unless one geom has higher priority) instead
                        of being diluted toward the object's softer defaults
  S2  S1 + elliptic cone
  S3  S2 + impratio=10 (biases the solver toward the frictional/normal
                        constraints being satisfied more exactly, MuJoCo's
                        own documented lever for reducing slip/penetration)

Only the pad geoms' contact parameters are touched, not every object's
collision geometry -- this experiment is scoped to what THIS PROJECT
controls (its own gripper), not a general re-tuning of every asset.

For known-thickness fixtures, penetration has an unambiguous ground truth:
settled true opening should never go below the object's actual thickness if
contact were rigid. For the CoACD objects, this experiment reuses the
pad-fidelity diagnostic's own classifier unmodified.

Run:  conda run -n tango python scripts/experiment_solver_contact_attribution.py
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
    JAW_CONTACT_MEASURED_PADS_AIMED,
    TABLE_TOP_Z,
    register_primitive_geom,
)

SEEDS = [0, 1]
OUT = Path(__file__).resolve().parent.parent / "outputs" / "solver_contact_attribution.jsonl"

# Known-thickness canonical fixtures. Sizes are MuJoCo half-extents (see
# EnvironmentSoArm.load_primitive's docstring).
FIXTURE_BOX = dict(shape="box", size=(0.05, 0.05, 0.015), mass=0.15,
                   name="FixtureBox30mm")           # full thickness 30mm
FIXTURE_CYL = dict(shape="cylinder", size=(0.02, 0.05), mass=0.15,
                   name="FixtureCyl40mm")            # full diameter 40mm
KNOWN_THICKNESS_M = {"FixtureBox30mm": 0.030, "FixtureCyl40mm": 0.040}

CODAC_OBJECTS = ["HammerC", "TomatoSoupCanC", "BananaC"]

CONFIGS = {
    "S0_baseline": {},
    "S1_stiff_pads": {
        "pad_solref": (0.005, 1.0),
        "pad_solimp": (0.95, 0.999, 0.0001, 0.5, 2),
        "pad_priority": 1,
    },
    "S2_elliptic_cone": {
        "pad_solref": (0.005, 1.0),
        "pad_solimp": (0.95, 0.999, 0.0001, 0.5, 2),
        "pad_priority": 1,
        "cone": int(mujoco.mjtCone.mjCONE_ELLIPTIC),
    },
    "S3_high_impratio": {
        "pad_solref": (0.005, 1.0),
        "pad_solimp": (0.95, 0.999, 0.0001, 0.5, 2),
        "pad_priority": 1,
        "cone": int(mujoco.mjtCone.mjCONE_ELLIPTIC),
        "impratio": 10.0,
    },
}


def apply_config(env: EnvironmentSoArm, cfg: dict) -> dict:
    """Patch the already-compiled model in place. Returns the pre-patch state
    so it can be logged (never assume readers trust the config name alone)."""
    m = env.model
    before = {
        "solver": int(m.opt.solver), "cone": int(m.opt.cone),
        "impratio": float(m.opt.impratio), "timestep": float(m.opt.timestep),
    }
    # Reset to a known baseline before applying this config's deltas, so
    # configs don't accumulate state from whichever ran before them.
    m.opt.cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    m.opt.impratio = 1.0
    for gid in env._jaw_pad_geom_ids:
        m.geom_solref[gid] = [0.02, 1.0]           # MuJoCo stock default
        m.geom_solimp[gid] = [0.9, 0.95, 0.001, 0.5, 2]
        m.geom_priority[gid] = 0

    if "cone" in cfg:
        m.opt.cone = cfg["cone"]
    if "impratio" in cfg:
        m.opt.impratio = cfg["impratio"]
    for gid in env._jaw_pad_geom_ids:
        if "pad_solref" in cfg:
            m.geom_solref[gid] = cfg["pad_solref"]
        if "pad_solimp" in cfg:
            m.geom_solimp[gid] = cfg["pad_solimp"]
        if "pad_priority" in cfg:
            m.geom_priority[gid] = cfg["pad_priority"]

    after = {
        "solver": int(m.opt.solver), "cone": int(m.opt.cone),
        "impratio": float(m.opt.impratio), "timestep": float(m.opt.timestep),
        "pad_solref": m.geom_solref[env._jaw_pad_geom_ids[0]].tolist(),
        "pad_solimp": m.geom_solimp[env._jaw_pad_geom_ids[0]].tolist(),
        "pad_priority": int(m.geom_priority[env._jaw_pad_geom_ids[0]]),
    }
    return {"before": before, "after": after}


def obj_collision_geom_ids(env: EnvironmentSoArm, obj_id: int) -> list:
    """Collision geom IDs for one loaded object, resolved by body membership.

    Production `_col_geom_names` resolves by the naming convention
    `ycb_col_geom_{slot}[_p{i}]`, which `_ycb_asset_tag`'s primitive branch
    never assigns -- register_primitive_geom's inline `<geom .../>` has no
    `name` at all, so a primitive's pad-distance fields come back None through
    the production path. Rather than touch env_soarm.py (out of scope here:
    "不改任何正式 API"), this resolves by body membership instead, which needs
    no name and works for primitives, single-mesh, and multi-part CoACD
    objects alike -- confirmed to agree with the production path on the three
    real (named-geom) objects in this experiment, see the printed cross-check.
    """
    slot = env._obj_pool_slot(obj_id)
    bid = env.model.body(f"obj_{slot}").id
    return [gi for gi in range(env.model.ngeom)
           if env.model.geom_bodyid[gi] == bid and env.model.geom_contype[gi] != 0]


def fixed_spawn(seed: int):
    rng = np.random.default_rng(seed)
    return [float(rng.uniform(-0.06, 0.06)),
            -0.40 + float(rng.uniform(-0.04, 0.04)),
            TABLE_TOP_Z + 0.12]


def run_trial(pool_key, logical_name, cfg_name, cfg, seed) -> dict:
    """One fresh EnvironmentSoArm, one object, one grasp attempt.

    Fresh construction per trial is deliberate, not an efficiency default: an
    earlier version of this script shared one env across every object and
    config in sequence and hit a >200mm nonsense pad-distance reading on
    (FixtureCyl40mm, S1, seed=0) that did NOT reproduce when that exact
    (object, config, seed) was re-run in isolation -- i.e. a cross-trial state
    leak (most likely stale weld/equality-constraint state from a DIFFERENT
    pool slot's earlier attach/detach cycle in the shared pool; not chased to
    ground given this experiment's scope). A fresh env per trial cannot suffer
    that class of bug by construction, at the cost of ~5 objects x 4 configs x
    2 seeds = 40 constructions instead of 1 -- acceptable for "small scale."
    """
    env = EnvironmentSoArm(obj_names=[pool_key], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True,
                           jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED,
                           enable_pad_fidelity_trace=True)
    try:
        # NOTE: EnvironmentSoArm.__init__'s `obj_names` kwarg is swallowed by
        # its **kwargs catch-all and never read -- every object actually
        # enters the pool via load_obj(), which calls _rebuild_model() the
        # first time it sees a new pool name. This is harmless for normal
        # callers (they never touch model state between construction and
        # load_obj), but it matters here: apply_config() MUST run AFTER
        # load_obj(), or _rebuild_model() discards the patched model and
        # every config silently collapses to defaults. An earlier version of
        # this script got this ordering backwards and every config produced
        # byte-identical results -- the tell that gave it away.
        oid = env.load_obj(pool_key, name=logical_name, pos=fixed_spawn(seed))
        if len(env._jaw_pad_geom_ids) != 2:
            raise RuntimeError("pad geoms not found")
        patch_log = apply_config(env, cfg)

        env._steps(240)
        env.wait_until_all_still(max_wait_epochs=200)
        p = env.get_obj_pos(oid).copy()

        ok, _ = env._execute_grasp(
            pos=(float(p[0]), float(p[1]), float(p[2])), roll=0.0,
            gripper_opening_length=0.065, obj_height=float(p[2] - TABLE_TOP_Z))
        m = env.last_grasp_metrics or {}
        pf = m.get("pad_fidelity_summary", {})

        # Own resolution (works for primitives too; see
        # obj_collision_geom_ids's docstring). Snapshot right after
        # _execute_grasp returns -- NOTE this is after any lift attempt inside
        # it, whereas pad_fidelity_summary's "final" sample is from the end of
        # the close+settle window, BEFORE the lift. The two are different
        # instants in time (a lift can shift the arm/object slightly), which
        # is why the cross-check below tolerates a few mm rather than treating
        # any nonzero delta as disagreement.
        gids = obj_collision_geom_ids(env, oid)
        own_pad_d = env._pad_to_obj_dist(gids) if gids else {}

        cross_check_delta = None
        if pf.get("final_pad_dist_fixed_m") is not None and own_pad_d:
            cross_check_delta = own_pad_d["fixed"] - pf["final_pad_dist_fixed_m"]

        return {
            "config": cfg_name,
            "config_applied": patch_log["after"],
            "object": logical_name,
            "seed": seed,
            "known_thickness_m": KNOWN_THICKNESS_M.get(logical_name),
            "success": bool(ok),
            "geometric_verdict": pf.get("geometric_verdict"),
            "final_pad_dist_fixed_m": pf.get("final_pad_dist_fixed_m"),
            "final_pad_dist_moving_m": pf.get("final_pad_dist_moving_m"),
            "excessive_penetration_samples": pf.get("excessive_penetration_samples"),
            "bilateral_engagement_samples": pf.get("bilateral_engagement_samples"),
            "settled_true_opening_m": m.get("true_opening_m"),
            "own_pad_dist_fixed_m": own_pad_d.get("fixed"),
            "own_pad_dist_moving_m": own_pad_d.get("moving"),
            "cross_check_delta_m": cross_check_delta,
        }
    finally:
        env.close()


def main():
    fixture_box_pool = register_primitive_geom(
        FIXTURE_BOX["shape"], FIXTURE_BOX["size"], FIXTURE_BOX["mass"])
    fixture_cyl_pool = register_primitive_geom(
        FIXTURE_CYL["shape"], FIXTURE_CYL["size"], FIXTURE_CYL["mass"])

    trials = ([(fixture_box_pool, FIXTURE_BOX["name"], s) for s in SEEDS]
             + [(fixture_cyl_pool, FIXTURE_CYL["name"], s) for s in SEEDS]
             + [(o, o, s) for o in CODAC_OBJECTS for s in SEEDS])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with OUT.open("w") as fh:
        for cfg_name, cfg in CONFIGS.items():
            print(f"\n=== {cfg_name} ===")
            for pool_key, logical_name, seed in trials:
                rec = run_trial(pool_key, logical_name, cfg_name, cfg, seed)
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                thick_note = ""
                if rec["known_thickness_m"] is not None and rec["settled_true_opening_m"]:
                    gap = rec["settled_true_opening_m"] - rec["known_thickness_m"]
                    thick_note = f"  vs_known_thickness={gap*1000:+.1f}mm"
                cc_note = (f"  (cross-check delta={rec['cross_check_delta_m']*1000:+.2f}mm)"
                          if rec["cross_check_delta_m"] is not None else "")
                print(f"  [{logical_name}/s{seed}] succ={str(rec['success']):5s} "
                      f"verdict={str(rec['geometric_verdict']):28} "
                      f"final_pad=({_mm(rec['own_pad_dist_fixed_m'])},"
                      f"{_mm(rec['own_pad_dist_moving_m'])}){thick_note}{cc_note}")

    summarize(records)
    print(f"\nwrote {len(records)} trials to {OUT}")


def _mm(v):
    return "  n/a" if v is None else f"{v*1000:+6.1f}mm"


def summarize(records):
    print("\n" + "=" * 100)
    print("geometric_verdict counts by config")
    print("=" * 100)
    configs = list(CONFIGS.keys())
    verdicts = ["NO_ENGAGEMENT", "PLAUSIBLE_ENGAGEMENT",
               "EXCESSIVE_PENETRATION_DOMINANT", "AMBIGUOUS"]
    from collections import Counter
    print(f"{'config':18s} {'n':>3} " + " ".join(f"{v.split('_')[0]:>10}" for v in verdicts))
    for c in configs:
        rs = [r for r in records if r["config"] == c]
        counts = Counter(r["geometric_verdict"] for r in rs)
        print(f"{c:18s} {len(rs):3d} " +
              " ".join(f"{counts.get(v, 0):10d}" for v in verdicts))

    print("\n" + "=" * 100)
    print("median FINAL pad-distance, fixed side (mm) by config x object "
          "(own body-based resolution -- works uniformly for fixtures and "
          "named objects, cross-checked against the production path where "
          "both are available)")
    print("=" * 100)
    objs = sorted({r["object"] for r in records})
    print(f"{'config':18s} " + " ".join(f"{o[:14]:>16}" for o in objs))
    import statistics
    for c in configs:
        row = []
        for o in objs:
            vs = [r["own_pad_dist_fixed_m"] for r in records
                 if r["config"] == c and r["object"] == o
                 and r["own_pad_dist_fixed_m"] is not None]
            row.append(f"{statistics.median(vs)*1000:16.1f}" if vs else f"{'n/a':>16}")
        print(f"{c:18s} " + " ".join(row))

    deltas = [r["cross_check_delta_m"] for r in records if r["cross_check_delta_m"] is not None]
    if deltas:
        print(f"\ncross-check (own resolution vs production pad_fidelity_summary, "
              f"named objects only): max |delta| = {max(abs(d) for d in deltas)*1000:.3f} mm "
              f"over {len(deltas)} comparisons")

    print("\n" + "=" * 100)
    print("fixtures only: settled opening vs KNOWN thickness (mm; positive = "
          "pads stopped at or outside the true surface, negative = compressed "
          "through it)")
    print("=" * 100)
    fixture_names = list(KNOWN_THICKNESS_M)
    print(f"{'config':18s} " + " ".join(f"{n[:16]:>18}" for n in fixture_names))
    for c in configs:
        row = []
        for n in fixture_names:
            vs = [r["settled_true_opening_m"] - r["known_thickness_m"]
                 for r in records if r["config"] == c and r["object"] == n
                 and r["settled_true_opening_m"] is not None]
            row.append(f"{statistics.median(vs)*1000:+18.1f}" if vs else f"{'n/a':>18}")
        print(f"{c:18s} " + " ".join(row))


if __name__ == "__main__":
    main()
