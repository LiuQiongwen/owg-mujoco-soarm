"""Cracker contact-sequence trace -- read-only, zero diff on
tango_robot/piper_robosuite/ and tango_robot/piper_assets/.

Follow-up to scripts/piper_tcp_correction_ab.py: Cracker's bilateral-contact
rate dropped from 9/10 (legacy) to 5/10 (corrected) despite near-identical
final success rate (Lift._check_success is purely height-based, so a
one-sided pin/wedge hold can still count as "success"). That result alone
doesn't say WHY -- this traces the actual closing sequence, step by step,
for both arms on the same seeds, to see which side touches first, whether
the object gets pushed/rotated during closure, and whether the whole-finger
collision mesh (piper_gripper.xml's finger7/8_collision -- confirmed in
docs/PIPER_GRIPPER_AUDIT_20260807.md to be the FULL finger mesh, not a
localized pad, same defect class SO-101 had pre-step-3) is a plausible
contributor.

Seeds: cracker 1041/1042/1046 -- chosen because the CORRECTED arm SUCCEEDED
(object reached the tray) at all three despite bilateral_contact=False, which
isolates "does a converged, accurate descend still produce one-sided
contact" from upstream IK-failure noise (a different, already-tracked
problem -- see docs/PIPER_TCP_CORRECTION_AB_20260807.md and the
transit_high audit).

Traces the window from "descend_refresh" (fresh re-read of the object pose,
immediately before the final approach+close) through "lift" (the same
phase-transition boundary piper_tcp_correction_ab.py's PhaseTracker already
uses for its bilateral_contact_post_close snapshot) -- covering the final
approach motion AND the 250-step close command, so contact onset is visible
in its actual context, not just a single before/after snapshot.

Run:  conda run -n tango python scripts/trace_piper_cracker_contact_sequence.py
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
    LegacyArmIK, CorrectedArmIK, PhaseTracker, scene_objects_for,
)

OUT = Path(__file__).resolve().parent.parent / "outputs" / "piper_cracker_contact_trace.jsonl"
SEEDS = [1041, 1042, 1046]


def _finger_geoms(model):
    def matching(substrings):
        return {i for i in range(model.ngeom)
               if all(s in (model.geom(i).name or "") for s in substrings)}
    return matching(["finger7"]), matching(["finger8"])


def _contact_state(env, obj_name, left_geoms, right_geoms):
    obj_geoms = ppp._object_contact_geoms(env, obj_name)
    data = env.sim.data._data
    left_dist = right_dist = float("inf")
    left_touch = right_touch = False
    left_is_whole_mesh_geom = right_is_whole_mesh_geom = None
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if not (pair & obj_geoms):
            continue
        if pair & left_geoms:
            left_touch = True
            left_dist = min(left_dist, float(c.dist))
        if pair & right_geoms:
            right_touch = True
            right_dist = min(right_dist, float(c.dist))
    body_id = env.object_body_ids[obj_name]
    return {
        "left_touch": left_touch, "left_dist_m": left_dist if left_dist != float("inf") else None,
        "right_touch": right_touch, "right_dist_m": right_dist if right_dist != float("inf") else None,
        "obj_pos": data.xpos[body_id].tolist(),
        "obj_quat": data.xquat[body_id].tolist(),
    }


class ContactTracingPhaseTracker(PhaseTracker):
    """Extends the A/B's own PhaseTracker (unmodified, imported not copied)
    with a per-step contact/pose trace active only between "descend_refresh"
    and "lift" -- the final-approach-and-close window."""

    def __init__(self, env, obj_name, left_geoms, right_geoms):
        super().__init__(env=env, obj_name=obj_name)
        self._left_geoms = left_geoms
        self._right_geoms = right_geoms
        self.tracing = False
        self.trace = []
        self._step_idx = 0

    def set_phase(self, name):
        super().set_phase(name)
        if name == "descend_refresh":
            self.tracing = True
        elif name == "lift":
            self.tracing = False

    def __call__(self, env):
        super().__call__(env)
        if self.tracing:
            self._step_idx += 1
            state = _contact_state(env, self._obj_name, self._left_geoms, self._right_geoms)
            state["step"] = self._step_idx
            self.trace.append(state)


def run_traced(obj_name, seed, arm_cls):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for(obj_name),
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20,
    )
    try:
        env.reset()
        left_geoms, right_geoms = _finger_geoms(env.sim.model._model)
        original_armik = ppp.ArmIK
        ppp.ArmIK = arm_cls
        tracker = ContactTracingPhaseTracker(env, obj_name, left_geoms, right_geoms)
        # CorrectedArmIK._solve_impl reads self._phase_tracker off the ik
        # INSTANCE (not a module global) to decide whether to apply the
        # correction -- without wiring it up here, phase would read as None
        # on every call and the "corrected" arm would silently run
        # uncorrected. Same capturing-init pattern as piper_tcp_correction_ab.py.
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

        trace = tracker.trace
        first_left = next((s["step"] for s in trace if s["left_touch"]), None)
        first_right = next((s["step"] for s in trace if s["right_touch"]), None)
        if trace:
            q0 = np.array(trace[0]["obj_quat"])
            q_end = np.array(trace[-1]["obj_quat"])
            # quaternion angular distance (w,x,y,z convention, MuJoCo's own)
            dot = float(np.clip(abs(np.dot(q0, q_end)), -1.0, 1.0))
            obj_rotation_deg = float(np.degrees(2 * np.arccos(dot)))
            p0 = np.array(trace[0]["obj_pos"])
            p_end = np.array(trace[-1]["obj_pos"])
            obj_translation_m = float(np.linalg.norm(p_end - p0))
        else:
            obj_rotation_deg = obj_translation_m = None

        return {
            "object": obj_name, "seed": seed, "arm": arm_cls.__name__,
            "success": bool(result.get("success")),
            "bilateral_contact_post_close": tracker.bilateral_contact_post_close,
            "n_trace_steps": len(trace),
            "first_left_contact_step": first_left,
            "first_right_contact_step": first_right,
            "which_side_first": (
                "left" if (first_left is not None and (first_right is None or first_left < first_right))
                else "right" if first_right is not None
                else "neither"
            ),
            "obj_rotation_deg_during_window": obj_rotation_deg,
            "obj_translation_m_during_window": obj_translation_m,
            "trace": trace,
        }
    finally:
        env.close()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with OUT.open("w") as fh:
        for seed in SEEDS:
            print(f"\n=== cracker seed={seed} ===")
            for arm_cls in (LegacyArmIK, CorrectedArmIK):
                rec = run_traced("cracker", seed, arm_cls)
                records.append(rec)
                # Don't dump the full per-step trace to stdout -- write it
                # to the jsonl and print only the summary.
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"  [{arm_cls.__name__:14s}] success={rec['success']!s:5s} "
                      f"bilateral_post_close={rec['bilateral_contact_post_close']!s:5s} "
                      f"first_touch={rec['which_side_first']:7s} "
                      f"(left@step={rec['first_left_contact_step']}, right@step={rec['first_right_contact_step']})  "
                      f"obj_rot={rec['obj_rotation_deg_during_window']:.1f}deg "
                      f"obj_trans={rec['obj_translation_m_during_window']*1000:.1f}mm"
                      if rec["obj_rotation_deg_during_window"] is not None else "  (no trace steps)")

    print(f"\nwrote {len(records)} records (with full per-step traces) to {OUT}")


if __name__ == "__main__":
    main()
