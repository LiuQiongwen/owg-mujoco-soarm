"""1D solref stiffness sweep: find a stable plateau, not the single hardest number.

Zero production-code diff (same pattern as
scripts/experiment_solver_contact_attribution.py: patches the already-compiled
MjModel's numeric fields from inside this throwaway script; move_gripper,
GRIP_CLOSED/GRIP_OPEN, _build_scene_xml, register_primitive_geom all used
unmodified).

Why this run exists
--------------------
The prior attribution experiment's S1 (pad solref time constant 5ms, priority
override) took EXCESSIVE_PENETRATION_DOMINANT from 6/10 to 0/10 across a small
sample. But that headline metric has a blind spot: a grasp that misses the
object entirely ALSO reports 0 excessive-penetration samples, because there's
no penetration if there's no contact. Checking HammerC/seed=1 under S1
directly (not part of the aggregate table) found exactly this: a genuine
numerical blowup during the grasp itself --

    WARNING: Nan, Inf or huge value in QACC at DOF 10.
    max object speed observed: 1876 m/s (physically absurd)

-- which MuJoCo's autoreset mechanism silently recovered from (object position
snapped back, displacement read as ~0), but which left the arm's kinematic
chain in a broken state for the rest of the trial: the grasp reports a clean
"NO_ENGAGEMENT, pads 190mm from the object" rather than the blowup that
actually happened. This DIRECTLY CONTRADICTS an earlier assessment (in
docs/SOLVER_CONTACT_ATTRIBUTION_20260807.md) that a QACC warning seen at the
end of that run's log was harmless because it appeared after all data had
already been written -- that characterization was correct for THAT specific
warning's position in the log, but wrong as a general claim; this is a
distinct, reproducible instability triggered by the same S1 pad solref value
on a specific scene, not a one-off cleanup artifact.

So: 5ms was never validated for stability, only for penetration reduction.
This sweep holds solimp/priority/cone/impratio fixed at S1's values and varies
ONLY the pad solref time constant, looking for where penetration reduction and
stability both hold -- a plateau, not a single "best" number.

Warnings are read from `data.warning[...].number` (MuJoCo's own counters,
reset at the start of each trial via mj_resetData), not by scraping stdout --
exact and doesn't depend on what happens to get printed.

Run:  conda run -n tango python scripts/experiment_solref_stability_sweep.py
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

OUT = Path(__file__).resolve().parent.parent / "outputs" / "solref_stability_sweep.jsonl"

# S1's fixed values -- only solref[0] (time constant) varies.
FIXED_SOLIMP = (0.95, 0.999, 0.0001, 0.5, 2)
FIXED_PRIORITY = 1
FIXED_CONE = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
FIXED_IMPRATIO = 1.0

SOLREF_TIMECONST_S = [0.005, 0.0075, 0.010, 0.015, 0.020]   # 5/7.5/10/15/20 ms
SOLREF_DAMPRATIO = 1.0

FIXTURE_BOX = dict(shape="box", size=(0.05, 0.05, 0.015), mass=0.15,
                   name="FixtureBox30mm")
FIXTURE_CYL = dict(shape="cylinder", size=(0.02, 0.05), mass=0.15,
                   name="FixtureCyl40mm")
KNOWN_THICKNESS_M = {"FixtureBox30mm": 0.030, "FixtureCyl40mm": 0.040}

# Includes HammerC/seed=1 deliberately -- the known trigger case, not just a
# convenient sample. A sweep that only used "easy" scenes couldn't answer the
# stability question this exists to answer.
WARNING_TYPES = [mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL,
                 mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADCTRL]


def assert_config_applied(env, timeconst_s):
    """Self-proving check: read the config back from the model and fail loudly
    if it doesn't match what was requested, rather than silently reporting
    results from whatever config actually happened to be active. This is the
    exact class of bug the previous experiment hit (a config patch silently
    discarded by an intervening _rebuild_model call) -- codified as a runtime
    assertion instead of something that has to be re-noticed by inspection.
    """
    m = env.model
    for gid in env._jaw_pad_geom_ids:
        sr = m.geom_solref[gid]
        si = m.geom_solimp[gid]
        pr = int(m.geom_priority[gid])
        assert abs(sr[0] - timeconst_s) < 1e-9, (sr, timeconst_s)
        assert abs(sr[1] - SOLREF_DAMPRATIO) < 1e-9, sr
        assert all(abs(a - b) < 1e-9 for a, b in zip(si, FIXED_SOLIMP)), (si, FIXED_SOLIMP)
        assert pr == FIXED_PRIORITY, (pr, FIXED_PRIORITY)
    assert int(m.opt.cone) == FIXED_CONE
    assert abs(float(m.opt.impratio) - FIXED_IMPRATIO) < 1e-9


def apply_solref(env, timeconst_s):
    m = env.model
    for gid in env._jaw_pad_geom_ids:
        m.geom_solref[gid] = [timeconst_s, SOLREF_DAMPRATIO]
        m.geom_solimp[gid] = list(FIXED_SOLIMP)
        m.geom_priority[gid] = FIXED_PRIORITY
    m.opt.cone = FIXED_CONE
    m.opt.impratio = FIXED_IMPRATIO
    assert_config_applied(env, timeconst_s)


def fixed_spawn(seed: int):
    rng = np.random.default_rng(seed)
    return [float(rng.uniform(-0.06, 0.06)),
            -0.40 + float(rng.uniform(-0.04, 0.04)),
            TABLE_TOP_Z + 0.12]


def run_trial(pool_key, logical_name, timeconst_s, seed) -> dict:
    """
    Velocity/contact-depth are read from `close_window_max_speed_mps` /
    `close_window_min_contact_dist_m` -- env_soarm.py's OWN
    `enable_close_window_diagnostics` feature, which scopes measurement to
    exactly the real close+settle window
    (`_close_with_contact_servo` + the following `_steps(120)`).

    An earlier version of this script wrapped `step_simulation` globally
    across the WHOLE `_execute_grasp` call, which is precisely the mistake
    `enable_close_window_diagnostics`'s own docstring in env_soarm.py warns
    against: "measuring raw object velocity across a WHOLE grasp attempt is
    contaminated by legitimate internal teleport/park/restore cycles
    _execute_grasp_physics_topdown performs while positioning the arm ...
    an external ad hoc probe caught this only after producing two false
    'explosion' readings." That is exactly what happened here -- a ~1876 m/s
    reading, IDENTICAL to 3+ significant figures across every one of the 5
    solref values tested including the STOCK 20ms default, which is the tell
    that it was a deterministic artifact of the teleport mechanism and had
    nothing to do with pad solref at all. Re-checked with the properly-scoped
    diagnostic: the close-window speed for that exact trial is ~1e-14 m/s
    (machine zero) with a perfectly ordinary -1mm contact depth. Fixed by
    using the production-scoped fields instead of a hand-rolled probe.
    """
    env = EnvironmentSoArm(obj_names=[pool_key], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True,
                           jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED,
                           enable_close_window_diagnostics=True)
    try:
        oid = env.load_obj(pool_key, name=logical_name, pos=fixed_spawn(seed))
        if len(env._jaw_pad_geom_ids) != 2:
            raise RuntimeError("pad geoms not found")
        apply_solref(env, timeconst_s)   # after load_obj -- see the ordering
                                         # bug documented in the sibling script

        env._steps(240)
        env.wait_until_all_still(max_wait_epochs=200)
        p0 = env.get_obj_pos(oid).copy()

        # `data.warning[...].number` accumulates over the whole data lifetime;
        # zero it right before the grasp attempt. NOTE: a warning firing here
        # can still originate from the teleport/park/restore machinery (OUTSIDE
        # the close window measured above) -- report both independently rather
        # than assuming a warning implies a close-window problem.
        for w in WARNING_TYPES:
            env.data.warning[w].number = 0

        ok, _ = env._execute_grasp(
            pos=(float(p0[0]), float(p0[1]), float(p0[2])), roll=0.0,
            gripper_opening_length=0.065, obj_height=float(p0[2] - TABLE_TOP_Z))

        m = env.last_grasp_metrics or {}
        warning_counts = {str(w).split(".")[-1]: int(env.data.warning[w].number)
                          for w in WARNING_TYPES}
        any_warning = any(warning_counts.values())

        return {
            "timeconst_ms": timeconst_s * 1000,
            "object": logical_name,
            "seed": seed,
            "known_thickness_m": KNOWN_THICKNESS_M.get(logical_name),
            "success": bool(ok),
            "settled_true_opening_m": m.get("true_opening_m"),
            "final_pad_dist_fixed_m": m.get("pad_obj_dist_fixed_m"),
            "final_pad_dist_moving_m": m.get("pad_obj_dist_moving_m"),
            "close_window_max_speed_mps": m.get("close_window_max_speed_mps"),
            "close_window_min_contact_dist_m": m.get("close_window_min_contact_dist_m"),
            "warning_counts": warning_counts,
            "any_warning": any_warning,
        }
    finally:
        env.close()


def main():
    fixture_box_pool = register_primitive_geom(
        FIXTURE_BOX["shape"], FIXTURE_BOX["size"], FIXTURE_BOX["mass"])
    fixture_cyl_pool = register_primitive_geom(
        FIXTURE_CYL["shape"], FIXTURE_CYL["size"], FIXTURE_CYL["mass"])

    scenes = [
        (fixture_box_pool, FIXTURE_BOX["name"], 0),
        (fixture_cyl_pool, FIXTURE_CYL["name"], 0),
        ("HammerC", "HammerC", 0),
        ("HammerC", "HammerC", 1),   # showed config-correlated success variation
        ("HammerC", "HammerC", 2),   # third seed to separate real signal from
                                     # per-seed IK-convergence noise
        ("TomatoSoupCanC", "TomatoSoupCanC", 0),
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with OUT.open("w") as fh:
        for timeconst_s in SOLREF_TIMECONST_S:
            print(f"\n=== solref timeconst = {timeconst_s*1000:.1f} ms ===")
            for pool_key, logical_name, seed in scenes:
                rec = run_trial(pool_key, logical_name, timeconst_s, seed)
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                thick_note = ""
                if rec["known_thickness_m"] is not None and rec["settled_true_opening_m"]:
                    gap = rec["settled_true_opening_m"] - rec["known_thickness_m"]
                    thick_note = f"  vs_thickness={gap*1000:+.1f}mm"
                warn_note = ("  WARNINGS:" + str(rec["warning_counts"])
                             if rec["any_warning"] else "")
                cw = rec["close_window_max_speed_mps"]
                cw_str = f"{cw:.2e}m/s" if cw is not None else "n/a"
                print(f"  [{logical_name}/s{seed}] succ={str(rec['success']):5s} "
                      f"close_window_max_speed={cw_str}"
                      f"{thick_note}{warn_note}")

    summarize(records)
    print(f"\nwrote {len(records)} trials to {OUT}")


def summarize(records):
    print("\n" + "=" * 100)
    print("stability by solref time constant")
    print("=" * 100)
    tcs = sorted({r["timeconst_ms"] for r in records})
    print(f"{'timeconst_ms':14s} {'n':>3} {'any_warning':>12} "
          f"{'max_close_win_speed(m/s)':>26} {'success':>8}")
    for tc in tcs:
        rs = [r for r in records if r["timeconst_ms"] == tc]
        n_warn = sum(r["any_warning"] for r in rs)
        speeds = [r["close_window_max_speed_mps"] for r in rs
                 if r["close_window_max_speed_mps"] is not None]
        max_speed = max(speeds) if speeds else float("nan")
        n_succ = sum(r["success"] for r in rs)
        flag = "  <-- warning fired (see per-scene detail: in/outside close window)" if n_warn else ""
        print(f"{tc:14.1f} {len(rs):3d} {n_warn:12d} {max_speed:26.2e} "
              f"{n_succ:8d}/{len(rs)}{flag}")

    print("\nper-scene detail")
    print(f"{'timeconst_ms':14s} {'scene':20s} {'succ':>6} "
          f"{'close_win_max_speed':>20} {'warnings':>10}")
    for r in records:
        tag = f"{r['object']}/s{r['seed']}"
        cw = r["close_window_max_speed_mps"]
        cw_str = f"{cw:.2e}" if cw is not None else "n/a"
        print(f"{r['timeconst_ms']:14.1f} {tag:20s} {str(r['success']):>6} "
              f"{cw_str:>20} "
              f"{'YES' if r['any_warning'] else '.':>10}")

    print("\nsuccess rate per scene across the sweep (n=5 timeconst values each) "
          "-- is the variation clustered on specific scenes, or spread evenly?")
    scenes = sorted({(r["object"], r["seed"]) for r in records})
    for obj, seed in scenes:
        rs = [r for r in records if r["object"] == obj and r["seed"] == seed]
        succs = [tc for tc in tcs
                for r in rs if r["timeconst_ms"] == tc and r["success"]]
        fails = [tc for tc in tcs
                for r in rs if r["timeconst_ms"] == tc and not r["success"]]
        print(f"  {obj}/s{seed}: succeeds at {succs} ms, fails at {fails} ms")


if __name__ == "__main__":
    main()
