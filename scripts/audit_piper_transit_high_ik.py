"""transit_high IK reachability audit -- read-only, zero diff on
tango_robot/piper_robosuite/ and tango_robot/piper_assets/.

Follow-up to scripts/piper_tcp_correction_ab.py: `ik_no_converge:transit_high`
was the dominant failure mode in that A/B (7 of the 8 tracked failure-seed
pairs), and it fires at a phase the TCP correction never touches
(transit_high isn't a "descend"-prefixed phase, so CorrectedArmIK passes its
target through unmodified) -- so it's a genuinely separate blocker, not a
downstream effect of the capture-frame reference.

Two things this audit answers per failing seed, per arm:
  1. WHY does transit_high fail to converge? -- dumps target_pos, the
     actual target_mat transit_high was solved against (recorded directly
     from the real ArmIK.solve() call -- solve_and_move defaults target_mat
     to whatever grasp_mat currently holds, which wrist_friendly_orientation
     may have already adjusted, so recomputing it separately from the raw
     object quaternion would silently measure the wrong thing), its angular
     distance from DOWN_ORIENTATION, every solve attempt's (primary + all
     fallback seeds) position AND orientation error (ArmIK.solve only
     returns pos_err_final -- orientation convergence was invisible in the
     A/B's own reporting until piper_tcp_correction_ab.py's _SolveRecorder
     was extended to also record target_mat/ori_err_deg for this audit),
     and per-joint distance to REAL_JOINT_LIMITS on each attempt's final
     qpos.
  2. Is transit_high's outcome actually identical between legacy/corrected
     at the same seed, as it should be (transit_high isn't a capture-frame
     call, so CorrectedArmIK passes it through unmodified)? The A/B run
     showed two counter-examples (cracker/1045, pear/1044) where legacy
     converged but corrected didn't, at the SAME seed -- that should be
     impossible if candidate selection and IK inputs are byte-identical
     between arms up to that point (same np.random.seed(seed) before an
     equivalent env.reset()). Reruns the SAME arm (legacy) twice at the
     same seed to separate "legacy vs corrected genuinely differ" from
     "this environment/solve isn't bit-reproducible run to run at all".

Reuses LegacyArmIK/CorrectedArmIK/PhaseTracker/scene_objects_for/
_failure_stage from the already-committed, already-validated A/B harness
rather than duplicating it.

Run:  conda run -n tango python scripts/audit_piper_transit_high_ik.py
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
    LegacyArmIK, CorrectedArmIK, PhaseTracker, scene_objects_for, _failure_stage,
)

OUT = Path(__file__).resolve().parent.parent / "outputs" / "piper_transit_high_audit.jsonl"

# From outputs/piper_tcp_correction_ab.jsonl: every (object, seed) pair
# where failure_stage mentioned transit_high, for either arm.
FAILING_SEEDS = [
    ("cracker", 1045), ("cracker", 1048),
    ("pear", 1042), ("pear", 1044), ("pear", 1045), ("pear", 1046), ("pear", 1049),
]


def run_and_capture(obj_name, seed, arm_cls):
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
        ik_holder = {}
        tracker = PhaseTracker(env=env, obj_name=obj_name)
        real_init = arm_cls.__init__

        def _capturing_init(self, env, _orig=real_init, _holder=ik_holder, _tracker=tracker):
            _orig(self, env)
            self._phase_tracker = _tracker
            _holder["ik"] = self

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

        ik = ik_holder.get("ik")
        calls = getattr(ik, "_calls", [])
        transit_calls = [c for c in calls if c["phase"] == "transit_high"]

        model = ik.model._model
        jnt_ids = ik.jnt_ids
        joint_ranges = [model.jnt_range[j].tolist() for j in jnt_ids]

        down_angle_deg = None
        diagnostics = []
        for c in transit_calls:
            qpos = np.array(c["qpos"])
            target_mat = np.array(c["target_mat"])
            if down_angle_deg is None:
                down_angle_deg = float(np.degrees(np.linalg.norm(
                    ppp.ArmIK._ori_error(target_mat, ppp.DOWN_ORIENTATION))))
            diagnostics.append({
                "target_pos": c["target_pos"],
                "converged": c["converged"],
                "pos_err_cm": c["err_cm"],
                "ori_err_deg": c["ori_err_deg"],
                "qpos": qpos.tolist(),
                "joint_margin_rad": [
                    min(qpos[i] - lo, hi - qpos[i]) for i, (lo, hi) in enumerate(joint_ranges)
                ],
            })

        return {
            "object": obj_name, "seed": seed, "arm": arm_cls.__name__,
            "success": bool(result.get("success")),
            "failure_stage": _failure_stage(result.get("phases"), result.get("success")),
            "target_mat_angle_from_down_deg": down_angle_deg,
            "joint_ranges": joint_ranges,
            "n_transit_attempts": len(transit_calls),
            "transit_attempts": diagnostics,
        }
    finally:
        env.close()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_records = []
    with OUT.open("w") as fh:
        for obj, seed in FAILING_SEEDS:
            print(f"\n=== {obj} seed={seed} ===")
            by_arm = {}
            for arm_cls in (LegacyArmIK, CorrectedArmIK):
                rec = run_and_capture(obj, seed, arm_cls)
                by_arm[arm_cls.__name__] = rec
                all_records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()

                print(f"  [{arm_cls.__name__:14s}] success={rec['success']!s:5s} "
                      f"target_mat_vs_down={rec['target_mat_angle_from_down_deg']:.1f}deg "
                      f"n_attempts={rec['n_transit_attempts']}")
                for i, a in enumerate(rec["transit_attempts"]):
                    tight = min(a["joint_margin_rad"])
                    tight_j = int(np.argmin(a["joint_margin_rad"]))
                    print(f"      attempt {i}: converged={a['converged']!s:5s} "
                          f"pos_err={a['pos_err_cm']:6.2f}cm  ori_err={a['ori_err_deg']:6.1f}deg  "
                          f"tightest_margin=joint{tight_j+1}:{tight:+.3f}rad")

            # Do legacy and corrected actually see the SAME transit_high
            # target/target_mat at this seed, as they must (transit_high
            # isn't a capture-frame call)?
            leg, cor = by_arm["LegacyArmIK"], by_arm["CorrectedArmIK"]
            if leg["transit_attempts"] and cor["transit_attempts"]:
                same_target = (leg["transit_attempts"][0]["target_pos"]
                              == cor["transit_attempts"][0]["target_pos"])
                print(f"  same transit_high target in both arms: {same_target}")

            # Rerun legacy a second time at the same seed to separate
            # "arms differ" from "this isn't reproducible run to run". Exact
            # `==` on the recorded dicts is the WRONG check here -- DLS IK
            # (np.linalg.solve inside a 3000-iter loop, backed by
            # multithreaded BLAS) exhibits ~1e-13 relative floating-point
            # noise between runs from non-associative summation order, which
            # trips exact float equality on every single field without ever
            # being large enough to cross the 5mm/0.02rad convergence
            # tolerances -- confirmed by hand before trusting this check.
            # What actually matters is whether that noise floor ever flips
            # the converged/not-converged OUTCOME (it could, in principle,
            # for a solve sitting exactly on the tolerance boundary).
            rerun = run_and_capture(obj, seed, LegacyArmIK)
            qpos_diffs = [
                float(np.max(np.abs(np.array(a1["qpos"]) - np.array(a2["qpos"]))))
                for a1, a2 in zip(leg["transit_attempts"], rerun["transit_attempts"])
            ]
            converged_diffs = [
                a1["converged"] != a2["converged"]
                for a1, a2 in zip(leg["transit_attempts"], rerun["transit_attempts"])
            ]
            max_qpos_diff = max(qpos_diffs) if qpos_diffs else None
            any_converged_flip = any(converged_diffs)
            print(f"  reproducibility (legacy re-run twice, same seed): "
                  f"max|qpos_diff|={max_qpos_diff:.2e}  "
                  f"{'OUTCOME FLIPPED' if any_converged_flip else 'outcome stable'}  "
                  f"success1={leg['success']} success2={rerun['success']}")

    print(f"\nwrote {len(all_records)} records to {OUT}")


if __name__ == "__main__":
    main()
