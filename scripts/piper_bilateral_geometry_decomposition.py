"""Geometric decomposition of T_eef_capture's local offset for Cracker's
traced seeds -- read-only, zero diff on tango_robot/piper_robosuite/ and
tango_robot/piper_assets/.

Follow-up to docs/PIPER_TRANSIT_HIGH_AND_BILATERAL_AUDIT_20260807.md's
"leading hypothesis" (not yet confirmed there): that T_eef_capture's local
offset picks up a world-frame horizontal component under Cracker's tilted
grasp_mat, and that this explains the observed one-sided contact.

IMPORTANT CORRECTION caught while building this: the hypothesis as
originally phrased ("the offset leaks into the jaw-closing axis") is
mathematically impossible for a proper rigid rotation. LOCAL_OFFSET is a
PURE translation along the eef frame's own local Z (approach axis):
  delta_world = R_grasp @ [0, 0, -0.0656]
Projected back into the SAME local frame: R_grasp^T @ delta_world =
R_grasp^T @ R_grasp @ [0,0,-0.0656] = [0,0,-0.0656] exactly, for ANY
rotation R_grasp (R^T R = I is what "rotation matrix" means) -- so the
offset has EXACTLY ZERO component along the local X (jaw-closing) or local
Y axis, in the gripper's own frame, always, regardless of tilt. That part
of the prior hypothesis is not "unconfirmed," it's false as stated.

What tilt DOES change is the WORLD-FRAME direction of that purely-local-Z
offset -- and the actual mechanism this script checks is different and more
precise: under LEGACY, the true fingertip midpoint ends up
`P + R_grasp @ [0,0,-0.0656]` for whatever point P the candidate computed
(NOT exactly at P -- that's the whole 65.6mm defect). Under CORRECTED, the
true fingertip midpoint lands almost exactly AT P (already verified,
median 0.24mm). So the real question is: was P (the candidate's intended
grasp point, e.g. object centroid + GRASP_HEIGHT_OFFSET) actually the right
place to center the fingers for a stable bilateral grasp on a tilted box --
or did LEGACY's uncorrected 65.6mm mis-aim happen to accidentally displace
the true contact point somewhere that worked better for this object's
geometry, in which case CORRECTED's accuracy at reaching P exposes that P
itself isn't well-centered for tilted grasps? This computes, per seed, the
actual displacement legacy's error introduced (gravity vs horizontal split,
and horizontal direction relative to the object), and checks whether that
historical, "wrong," displacement is what was actually keeping bilateral
contact working.

Confirms: finger7 = "left" sits at local X=-0.010, finger8 = "right" at
local X=+0.010 in the gripper's own frame (checked directly against
piper_gripper.xml's live geom positions) -- i.e. the JAW-CLOSING axis is
local X, matching grasp_mat's convention (DOWN_ORIENTATION's column 0).

Run:  conda run -n tango python scripts/piper_bilateral_geometry_decomposition.py
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
    LegacyArmIK, PhaseTracker, scene_objects_for, LOCAL_OFFSET,
)


class ObjPosSnapshotTracker(PhaseTracker):
    """PhaseTracker records phase transitions but not object pose at each
    one. The object moves substantially over a trial (it ends up at the
    tray, ~30-40cm from its spawn point) -- reading env.sim.data.xpos AFTER
    run_pick_and_place returns compares P against the WRONG object position
    entirely (caught via an implausible 300-400mm "P vs object" distance on
    the first run of this script). Snapshot obj_pos at the exact moment
    descend_refresh fires instead."""

    def __init__(self, env, obj_name):
        super().__init__(env=env, obj_name=obj_name)
        self.obj_pos_at_descend_refresh = None

    def set_phase(self, name):
        super().set_phase(name)
        if name == "descend_refresh" and self.obj_pos_at_descend_refresh is None:
            body_id = self._env.object_body_ids[self._obj_name]
            self.obj_pos_at_descend_refresh = self._env.sim.data.xpos[body_id].copy()


OUT = Path(__file__).resolve().parent.parent / "outputs" / "piper_bilateral_geometry_decomposition.json"

# From outputs/piper_cracker_contact_trace.jsonl (already collected,
# committed): which side made contact under the CORRECTED arm, and the
# object's translation during the descend_refresh->lift window.
OBSERVED = {
    1041: {"which_side_first": "right", "obj_translation_mm": 7.0},
    1042: {"which_side_first": "left", "obj_translation_mm": 5.3},
    1046: {"which_side_first": "right", "obj_translation_mm": 8.8},
}


def get_descend_refresh_context(obj_name, seed):
    """Fresh LegacyArmIK run (no TCP correction -- target_mat is identical
    under corrected, since the correction only ever touches target_pos) to
    recover target_mat and the candidate target P at the descend_refresh
    call, plus the object's actual centroid/quat at that moment."""
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for(obj_name),
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20,
    )
    try:
        env.reset()
        original_armik = ppp.ArmIK
        ppp.ArmIK = LegacyArmIK
        ik_holder = {}
        tracker = ObjPosSnapshotTracker(env=env, obj_name=obj_name)
        real_init = LegacyArmIK.__init__

        def _capturing_init(self, env, _orig=real_init, _holder=ik_holder, _tracker=tracker):
            _orig(self, env)
            self._phase_tracker = _tracker
            _holder["ik"] = self

        LegacyArmIK.__init__ = _capturing_init
        try:
            result = ppp.run_pick_and_place(
                env, obj_name, use_oriented_grasp=True, verbose=False,
                candidate_selection=None, wrist_friendly_orientation=True,
                step_hook=tracker,
            )
        finally:
            LegacyArmIK.__init__ = real_init
            ppp.ArmIK = original_armik

        ik = ik_holder["ik"]
        calls = [c for c in ik._calls if c["phase"] == "descend_refresh"]
        c = calls[-1]  # final commit target for this phase
        P = np.array(c["target_pos"])
        R_grasp = np.array(c["target_mat"])
        legacy_actual_capture = np.array(c["capture_pos"])

        assert tracker.obj_pos_at_descend_refresh is not None, \
            "descend_refresh phase never fired -- trial diverged before reaching it"

        return {
            "P_candidate_target": P, "R_grasp": R_grasp,
            "legacy_actual_capture_pos": legacy_actual_capture,
            "obj_pos_at_descend_refresh": tracker.obj_pos_at_descend_refresh,
            "success": bool(result.get("success")),
        }
    finally:
        env.close()


def main():
    records = []
    for seed, obs in OBSERVED.items():
        ctx = get_descend_refresh_context("cracker", seed)
        P = ctx["P_candidate_target"]
        R = ctx["R_grasp"]
        obj_pos = ctx["obj_pos_at_descend_refresh"]

        delta_world = R @ LOCAL_OFFSET  # legacy's actual displacement from P
        gravity_component_m = float(delta_world[2])
        horizontal_xy = delta_world[:2]
        horizontal_mag_m = float(np.linalg.norm(horizontal_xy))
        horizontal_angle_deg = float(np.degrees(np.arctan2(horizontal_xy[1], horizontal_xy[0])))

        # Sanity check on the "always zero in local frame" claim.
        local_check = R.T @ delta_world
        assert np.allclose(local_check, LOCAL_OFFSET, atol=1e-9), \
            f"local-frame decomposition should reproduce LOCAL_OFFSET exactly, got {local_check}"

        # Which side of P (in world XY) did legacy's error actually land the
        # true capture point on, relative to the object's true position?
        # And how does that compare to which side later showed contact
        # under CORRECTED (which lands almost exactly at P)?
        legacy_capture_xy = ctx["legacy_actual_capture_pos"][:2]
        p_to_obj = obj_pos[:2] - P[:2]
        legacy_capture_to_obj = obj_pos[:2] - legacy_capture_xy

        rec = {
            "seed": seed,
            "P_candidate_target": P.tolist(),
            "obj_pos_at_descend_refresh": obj_pos.tolist(),
            "gravity_component_mm": gravity_component_m * 1000,
            "horizontal_mag_mm": horizontal_mag_m * 1000,
            "horizontal_angle_deg": horizontal_angle_deg,
            "P_to_obj_xy_dist_mm": float(np.linalg.norm(p_to_obj)) * 1000,
            "legacy_capture_to_obj_xy_dist_mm": float(np.linalg.norm(legacy_capture_to_obj)) * 1000,
            "observed_corrected_touch_side": obs["which_side_first"],
            "observed_corrected_obj_translation_mm": obs["obj_translation_mm"],
            "local_frame_decomposition_check_passed": True,
        }
        records.append(rec)
        print(f"\n=== cracker seed={seed} ===")
        print(f"  T_eef_capture displacement (legacy's actual error from P):")
        print(f"    gravity(world Z) component: {rec['gravity_component_mm']:+7.2f}mm")
        print(f"    horizontal(world XY) magnitude: {rec['horizontal_mag_mm']:7.2f}mm  "
              f"direction: {rec['horizontal_angle_deg']:+7.1f}deg")
        print(f"  |P - obj_centroid| (xy): {rec['P_to_obj_xy_dist_mm']:.2f}mm  "
              f"(is the candidate's OWN target already centered on the object?)")
        print(f"  |legacy_actual_capture - obj_centroid| (xy): {rec['legacy_capture_to_obj_xy_dist_mm']:.2f}mm  "
              f"(legacy's mis-aimed ACTUAL capture point vs object)")
        print(f"  under corrected: touched side={obs['which_side_first']}  "
              f"obj_translation={obs['obj_translation_mm']}mm")

    print("\n" + "=" * 90)
    print("summary")
    print("=" * 90)
    for r in records:
        print(f"  seed={r['seed']}: horizontal_mag={r['horizontal_mag_mm']:.1f}mm "
              f"P_to_obj_xy={r['P_to_obj_xy_dist_mm']:.1f}mm "
              f"legacy_capture_to_obj_xy={r['legacy_capture_to_obj_xy_dist_mm']:.1f}mm "
              f"(smaller-is-more-centered)")

    OUT.write_text(json.dumps(records, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
