#!/usr/bin/env python3
"""
NOTE (2026-08-05): getting a physically meaningful result out of this script
required a temporary one-line probe inside env_soarm.py's
_execute_grasp_physics_topdown (a self._pilot_probe_close_active flag
bracketing the real self._close_with_contact_servo(opening) call), because
naive whole-trajectory velocity monitoring is contaminated by legitimate
internal teleport/park/restore cycles that function performs (parking the
object at z=-100 while the arm teleports to its IK solution, at least once,
possibly twice for hover-then-descend) -- see the conversation this script
came from for the two failed attempts this caught. That probe has been
REVERTED after use per instruction -- env_soarm.py currently has no such
flag, so run_one_trial's `getattr(env, "_pilot_probe_close_active", False)`
check will always be False if this script is re-run as-is, silently
excluding every step and reporting degenerate all-zero metrics. Re-add the
same two-line probe (see git history / the conversation log around
2026-08-05) before trusting this script's numeric output again.

Phase 2 diagnostic (2026-08-05): can tuning MuJoCo contact-solver parameters
keep the SO-ARM101 gripper's REAL (un-simplified) jaw collision mesh stable
during a grasp close, instead of relying on env_soarm.py's
_simplify_jaw_collision() 6mm-sphere proxy?

Motivation: _simplify_jaw_collision's own docstring states the real jaw mesh
convex hulls "create 2.8-3.9cm penetrations that generate explosive contact
impulses and send the object flying" under this project's current (default)
solver settings -- solref/solimp/iterations have never been tuned anywhere
in this codebase (verified: no override exists in tango_robot/assets/so101/
so101.xml's <default> blocks, nor in _build_scene_xml's <option> element).
This script tests whether that's a genuine, solver-parameter-independent
geometry problem, or a solver-tuning gap that was never actually explored
before reaching for the sphere-proxy workaround.

Method: reuses the REAL EnvironmentSoArm pipeline unmodified (real spawn,
real position-controlled gripper actuator, real _execute_grasp_physics_topdown
closing sequence) -- the only things toggled are (a) reverting
_simplify_jaw_collision's sphere substitution back to the real mesh geoms
right after construction (geom_dataid is untouched by the simplification, so
this is a straightforward, verified-safe type flip, not a hack), and (b)
global solver option overrides (env.model.opt.*) applied before the grasp
runs. Nothing in env_soarm.py itself is modified -- all toggling happens on
the already-built model instance, from outside, in this script only.

Per-step object linear velocity and per-step minimum jaw-object contact
distance (most negative = deepest penetration) are recorded via a local
monkey-patch of env.step_simulation for the duration of one grasp attempt
only, then restored -- this script's own instrumentation, not a change to
the class.

Usage:
  python3 scripts/contact_solver_tuning_pilot.py
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from tango_robot.env_soarm import EnvironmentSoArm, GRASP_MODE_PHYSICS_WELD, TABLE_TOP_Z

_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)

# 2026-08-05: the first version of this script hand-picked GRASP_POS/ROLL
# without validating them against the real candidate generator. Diagnosis
# revealed that pose caused the object to fall through the world (z reaching
# -70m) even under the CURRENT PRODUCTION-DEFAULT sphere-proxy config --
# meaning the earlier sweep results were an artifact of a badly-posed grasp
# attempt, not evidence about jaw-mesh collision at all. Replaced with the
# actual top-scored candidate from grasp_6dof/grasp_generator_6dof.py run
# against the real drill collision mesh, targeting this exact spawn position
# (score=0.9985, the highest of 2232 candidates that passed scoring):
#   conda run -n tango python3 grasp_6dof/grasp_generator_6dof.py \
#     --obj tango_robot/assets/ycb_objects/YcbPowerDrill/collision_vhacd.obj \
#     --out /tmp/drill_candidates_pilot.json --world-pos 0.35,-0.10,0.785 --n 300 --topk 20
OBJECT = "PowerDrill"
GRASP_POS = (0.31884504275032083, -0.11949964575979366, 0.7643616850897855)
GRASP_ROLL = -2.4506045597468424  # world_yaw from the candidate (env.grasp's "roll" arg is really yaw)
GRIPPER_OPENING = 0.09
OBJ_HEIGHT = 0.06

# Chosen relative to MuJoCo's ACTUAL defaults (verified empirically, not
# assumed): iterations=100, ls_iterations=50, o_solref=[0.02,1.0],
# o_solimp=[0.9,0.95,0.001,0.5,2.0], impratio=1.0, noslip_iterations=0,
# cone=0 (pyramidal). An earlier draft of this sweep picked solref/solimp
# values nearly identical to (or in the wrong direction from) these real
# defaults -- fixed after checking, so every setting below is a genuine,
# literature-motivated delta, not a no-op.
SETTINGS = {
    "baseline_default": {},
    "more_iterations": {"iterations": 300, "ls_iterations": 150},
    "softer_solref": {"o_solref": (0.05, 1.5)},   # longer time-const + more overdamped -> spreads impulse over more time, less "explosive"
    "forgiving_solimp": {"o_solimp": (0.5, 0.9, 0.005, 0.5, 2)},  # lower dmin -> much less stiff at small penetration depths
    "noslip_and_elliptic": {"noslip_iterations": 10, "cone": 1},  # elliptic friction cone + noslip pass, both literature-recommended for grasp contact
    "high_impratio": {"impratio": 10.0},
    "combined_tuned": {
        "iterations": 300, "ls_iterations": 150,
        "o_solref": (0.05, 1.5),
        "o_solimp": (0.5, 0.9, 0.005, 0.5, 2),
        "noslip_iterations": 10, "cone": 1,
        "impratio": 10.0,
    },
}


GRASP_POS_SPAWN = (GRASP_POS[0], GRASP_POS[1], TABLE_TOP_Z + 0.02)


def build_env(use_real_jaw_mesh: bool, opt_overrides: dict):
    env = EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD)
    obj_id = env.load_obj(OBJECT, pos=list(GRASP_POS_SPAWN), yaw=0.0)

    if use_real_jaw_mesh:
        env.model.geom_type[env._jaw_fixed_geom_id] = _MESH
        env.model.geom_type[env._jaw_mv_geom_id] = _MESH

    for key, val in opt_overrides.items():
        if key.startswith("o_"):
            setattr(env.model.opt, key, np.asarray(val, dtype=np.float64))
        else:
            setattr(env.model.opt, key, val)

    return env, obj_id


def run_one_trial(use_real_jaw_mesh: bool, opt_overrides: dict):
    env, obj_id = build_env(use_real_jaw_mesh, opt_overrides)
    slot = env._obj_pool_slot(obj_id)
    obj_body_id = env.model.body(f"obj_{slot}").id
    obj_geom_ids = set()
    for gname in (f"ycb_vis_geom_{slot}", f"ycb_col_geom_{slot}"):
        try:
            obj_geom_ids.add(env.model.geom(gname).id)
        except Exception:
            pass

    monitor = {"max_speed": 0.0, "min_contact_dist": 0.0, "n_steps": 0,
               "max_speed_at_step": -1, "max_speed_at_obj_pos": None,
               "n_steps_excluded_parked": 0}
    orig_step = env.step_simulation

    def monitored_step():
        orig_step()
        monitor["n_steps"] += 1
        obj_z = float(env.data.xpos[obj_body_id][2])
        # _execute_grasp_physics_topdown deliberately parks the object at
        # z=-100 for up to 200 steps while the arm teleports to its IK
        # solution (env_soarm.py:2018-2029, "so the arm joints can be
        # teleported ... without jaw meshes penetrating the object") --
        # legitimate free-fall while intentionally out of the workspace, not
        # a physics failure. An earlier version of this script tracked raw
        # velocity across the WHOLE trajectory including this window and
        # reported a ~1776 m/s "explosion" that was entirely this artifact,
        # not a real finding -- excluding it here (obj_z < 0 is unambiguous:
        # no legitimate in-workspace object position is ever negative).
        if obj_z < 0 or not getattr(env, "_pilot_probe_close_active", False):
            monitor["n_steps_excluded_parked"] += 1
            return
        v = float(np.linalg.norm(env.data.cvel[obj_body_id][3:6]))
        if v > monitor["max_speed"]:
            monitor["max_speed"] = v
            monitor["max_speed_at_step"] = monitor["n_steps"]
            monitor["max_speed_at_obj_pos"] = env.data.xpos[obj_body_id].copy().tolist()
        ncon = env.data.ncon
        if ncon > 0 and obj_geom_ids:
            for i in range(ncon):
                c = env.data.contact[i]
                if c.geom1 in obj_geom_ids or c.geom2 in obj_geom_ids:
                    if c.dist < monitor["min_contact_dist"]:
                        monitor["min_contact_dist"] = float(c.dist)

    env.step_simulation = monitored_step
    try:
        success, grasped = env.grasp(GRASP_POS, GRASP_ROLL, GRIPPER_OPENING, OBJ_HEIGHT)
    except Exception as e:
        return {"error": str(e), **monitor}
    finally:
        env.step_simulation = orig_step

    return {
        "success": bool(success),
        "grasped": grasped,
        "max_object_speed_mps": monitor["max_speed"],
        "min_contact_signed_dist_m": monitor["min_contact_dist"],
        "n_steps": monitor["n_steps"],
        "n_steps_excluded_parked": monitor["n_steps_excluded_parked"],
        "max_speed_at_step": monitor["max_speed_at_step"],
        "max_speed_at_obj_pos": monitor["max_speed_at_obj_pos"],
    }


def main():
    print(f"Object: {OBJECT}  |  grasp pos: {GRASP_POS}\n")

    print("=" * 100)
    print("Sanity check: sphere-proxy jaw (current production default) -- should be stable")
    print("=" * 100)
    r = run_one_trial(use_real_jaw_mesh=False, opt_overrides={})
    print(r)
    print()

    print("=" * 100)
    print("Real (un-simplified) jaw mesh collision, sweeping solver settings")
    print("=" * 100)
    header = f"{'setting':<20} {'success':<8} {'max_speed(m/s)':<16} {'min_contact_dist(mm)':<22} {'n_steps'}"
    print(header)
    results = {}
    for name, overrides in SETTINGS.items():
        r = run_one_trial(use_real_jaw_mesh=True, opt_overrides=overrides)
        results[name] = r
        if "error" in r:
            print(f"{name:<20} ERROR: {r['error']}")
            continue
        print(f"{name:<20} {str(r['success']):<8} {r['max_object_speed_mps']:<16.3f} "
              f"{r['min_contact_signed_dist_m']*1000:<22.2f} {r['n_steps']}")

    print()
    print("Interpretation: max_object_speed_mps spiking to several m/s or more during")
    print("closing (well beyond a controlled ~settle) is the 'explosion' signature")
    print("_simplify_jaw_collision's docstring describes. min_contact_signed_dist_m more")
    print("negative than roughly -0.01 to -0.02m reproduces the documented 2.8-3.9cm")
    print("(-0.028 to -0.039m) penetration this workaround was built to avoid.")

    return results


if __name__ == "__main__":
    main()
