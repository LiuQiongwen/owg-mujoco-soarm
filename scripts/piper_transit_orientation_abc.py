"""transit_high orientation A/B/C -- read-only w.r.t. tango_robot/piper_robosuite/
and tango_robot/piper_assets/, zero diff confirmed by git status/diff.

Follow-up to docs/PIPER_TRANSIT_HIGH_AND_BILATERAL_AUDIT_20260807.md's
finding: every transit_high IK failure pins some joint exactly at its
REAL_JOINT_LIMITS bound on all 7 solve attempts (primary + 6 diverse
fallback seeds), because solve_and_move defaults transit_high's target_mat
to the candidate's (measured 4-57deg tilted) grasp_mat instead of a neutral
hover orientation -- forcing a hard 6D pose match onto what's meant to be a
simple safe-height transit waypoint.

Scope, exactly as requested: this ONLY changes which target_mat gets passed
to non-descend-phase solve() calls. TCP correction, contact/bilateral
mechanics, and joint6 rules are untouched -- all three arms here use the
LEGACY (uncorrected) target_pos semantics throughout; the only thing that
varies between T0/T1/T2 is orientation, and only away from the final
grasp-commit phases.

  T0 legacy:          every phase's target_mat = candidate's grasp_mat (as
                       today, unmodified pass-through)
  T1 neutral-hover:    ONLY transit_high's target_mat forced to
                       DOWN_ORIENTATION; every other phase (approach, lift,
                       tray, descend) keeps today's grasp_mat default
  T2 relaxed-hover:    every phase EXCEPT descend/descend_refresh forced to
                       DOWN_ORIENTATION; grasp_mat is restored only at the
                       final grasp-commit phases. Tests whether T1's
                       abrupt transit_high(DOWN) -> approach(tilted)
                       reorientation (approach uses an uninterpolated
                       move_to, so a large one-shot PD move) is itself a
                       problem T1 doesn't fix.

Seeds: the 7 (object, seed) pairs where transit_high failed in the original
A/B, PLUS 6 representative pairs where the ORIGINAL (uncorrected) pipeline
already succeeded -- to check both "does this rescue the failures" and "does
this damage what already worked," per the two questions this ablation was
scoped to answer.

Run:  conda run -n tango python scripts/piper_transit_orientation_abc.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite import piper_pick_and_place as ppp
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from scripts.piper_tcp_correction_ab import (
    _SolveRecorder, _ORIGINAL_ARMIK_SOLVE, _is_capture_phase, _failure_stage,
    PhaseTracker, scene_objects_for,
)

OUT = Path(__file__).resolve().parent.parent / "outputs" / "piper_transit_orientation_abc.jsonl"

FAILING_PAIRS = [
    ("cracker", 1045), ("cracker", 1048),
    ("pear", 1042), ("pear", 1044), ("pear", 1045), ("pear", 1046), ("pear", 1049),
]
SUCCESS_PAIRS = [
    ("cracker", 1041), ("cracker", 1043), ("cracker", 1047),
    ("pear", 1041), ("pear", 1043), ("pear", 1047),
]
ALL_PAIRS = FAILING_PAIRS + SUCCESS_PAIRS


class T0LegacyOrientation(_SolveRecorder, ppp.ArmIK):
    """Unmodified target_mat semantics at every phase -- the baseline this
    ablation is measured against."""

    def _solve_impl(self, target_pos, seed_qpos, target_mat, iters, phase):
        return _ORIGINAL_ARMIK_SOLVE(self, target_pos, seed_qpos, target_mat=target_mat, iters=iters)


class T1NeutralHoverOnly(_SolveRecorder, ppp.ArmIK):
    """Only transit_high's target_mat forced to DOWN_ORIENTATION."""

    def _solve_impl(self, target_pos, seed_qpos, target_mat, iters, phase):
        use_mat = ppp.DOWN_ORIENTATION if phase == "transit_high" else target_mat
        return _ORIGINAL_ARMIK_SOLVE(self, target_pos, seed_qpos, target_mat=use_mat, iters=iters)


class T2RelaxedHoverBroad(_SolveRecorder, ppp.ArmIK):
    """Every non-capture (non-descend) phase forced to DOWN_ORIENTATION;
    grasp_mat restored only at descend/descend_refresh."""

    def _solve_impl(self, target_pos, seed_qpos, target_mat, iters, phase):
        use_mat = target_mat if _is_capture_phase(phase) else ppp.DOWN_ORIENTATION
        return _ORIGINAL_ARMIK_SOLVE(self, target_pos, seed_qpos, target_mat=use_mat, iters=iters)


ARMS = [("T0_legacy", T0LegacyOrientation), ("T1_neutral_hover", T1NeutralHoverOnly),
        ("T2_relaxed_hover", T2RelaxedHoverBroad)]


def run_one(obj_name, seed, arm_cls):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for(obj_name),
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20,
    )
    try:
        env.reset()
        original_armik = ppp.ArmIK
        ppp.ArmIK = arm_cls
        tracker = PhaseTracker(env=env, obj_name=obj_name)
        real_init = arm_cls.__init__

        def _capturing_init(self, env, _orig=real_init, _tracker=tracker):
            _orig(self, env)
            self._phase_tracker = _tracker

        arm_cls.__init__ = _capturing_init
        try:
            result = ppp.run_pick_and_place(
                env, obj_name, use_oriented_grasp=True, verbose=False,
                candidate_selection=None, wrist_friendly_orientation=True,
                step_hook=tracker,
            )
        finally:
            arm_cls.__init__ = real_init
            ppp.ArmIK = original_armik

        phases = result.get("phases") or {}
        transit = phases.get("transit_high", {})
        return {
            "object": obj_name, "seed": seed, "arm": arm_cls.__name__,
            "success": bool(result.get("success")),
            "failure_stage": _failure_stage(phases, result.get("success")),
            "transit_high_converged": transit.get("converged"),
            "transit_high_err_cm": transit.get("err_cm"),
        }
    finally:
        env.close()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with OUT.open("w") as fh:
        for obj, seed in ALL_PAIRS:
            print(f"\n=== {obj} seed={seed} ===")
            for label, arm_cls in ARMS:
                rec = run_one(obj, seed, arm_cls)
                rec["condition"] = label
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"  [{label:17s}] success={rec['success']!s:5s} "
                      f"transit_high_converged={rec['transit_high_converged']!s:5s} "
                      f"err_cm={rec['transit_high_err_cm']:.2f}  stage={rec['failure_stage']}")

    summarize(records)
    print(f"\nwrote {len(records)} records to {OUT}")


def summarize(records):
    print("\n" + "=" * 90)
    print("transit_high convergence rate by condition")
    print("=" * 90)
    for label, _ in ARMS:
        vs = [r["transit_high_converged"] for r in records if r["condition"] == label]
        print(f"  {label:17s} {sum(bool(v) for v in vs)}/{len(vs)} converged")

    print("\n" + "=" * 90)
    print("did T1/T2 rescue the 7 known-failing pairs?")
    print("=" * 90)
    for obj, seed in FAILING_PAIRS:
        row = {r["condition"]: r["success"] for r in records if r["object"] == obj and r["seed"] == seed}
        print(f"  {obj:8s} seed={seed}  T0={row.get('T0_legacy')!s:5s} "
              f"T1={row.get('T1_neutral_hover')!s:5s} T2={row.get('T2_relaxed_hover')!s:5s}")

    print("\n" + "=" * 90)
    print("did T1/T2 damage the 6 known-successful pairs?")
    print("=" * 90)
    for obj, seed in SUCCESS_PAIRS:
        row = {r["condition"]: r["success"] for r in records if r["object"] == obj and r["seed"] == seed}
        print(f"  {obj:8s} seed={seed}  T0={row.get('T0_legacy')!s:5s} "
              f"T1={row.get('T1_neutral_hover')!s:5s} T2={row.get('T2_relaxed_hover')!s:5s}")

    print("\n" + "=" * 90)
    print("success rate summary")
    print("=" * 90)
    for label, _ in ARMS:
        vs = [r["success"] for r in records if r["condition"] == label]
        print(f"  {label:17s} {sum(vs)}/{len(vs)}")


if __name__ == "__main__":
    main()
