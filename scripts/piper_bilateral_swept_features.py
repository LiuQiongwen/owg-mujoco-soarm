"""Deterministic swept-volume-style bilateral-engagement features -- the
"minimal version" cheap alternative to a full GraspGen-X-style learned
swept-volume representation: no learning, just direct geometry against the
Piper gripper's own measured mesh/kinematics.

Read-only, zero diff on tango_robot/piper_robosuite/ and
tango_robot/piper_assets/.

Idea: instead of treating a candidate grasp as a static SE(3) point, model
what each finger's closing sweep actually intersects. Piper's gripper is a
parallel/prismatic mechanism (fingers translate along local X, not a
hinge -- confirmed in docs/PIPER_GRIPPER_AUDIT_20260807.md), so each
finger's swept path during closing is fully described by its INNER FACE's
X position moving from the OPEN value to the CLOSED value (measured
directly below: -0.0500 -> -0.0040 for finger7/left, +0.0500 -> +0.0040
for finger8/right -- these numbers are close to but not identical to the
existing 14mm/104mm tip-gap LUT in calib/piper_gripper_audit.json, which
used a stricter tip-only vertex quantile filter; this script uses the raw
mesh AABB extent along the closing axis, a coarser but simpler measure,
documented as a limitation below.)

For a candidate capture pose (position P, orientation R = grasp_mat --
CORRECTED semantics: since T_eef_capture is confirmed rigid and accurate to
~0.2mm, evaluating features AT P is evaluating what CORRECTED actually
does), transforms the object's mesh vertices into the candidate's own local
frame and computes, per side:

  gap = distance the finger must travel from its OPEN position to reach
        the object's nearest surface point in a plausible contact band
        (restricted to points near the candidate's local Y=0, Z=0 --
        finger7/8_collision's own measured extent, not the whole object)

Predicts which side touches first (smaller gap) and by how much (gap
imbalance), directly comparable against
outputs/piper_cracker_contact_trace.jsonl's OBSERVED first-touch side and
step count -- this is the actual test of whether this cheap feature has any
predictive value, not just a plausible-sounding computation.

Run:  conda run -n tango python scripts/piper_bilateral_swept_features.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite import piper_pick_and_place as ppp
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from scripts.piper_tcp_correction_ab import (
    LegacyArmIK, scene_objects_for, LOCAL_OFFSET,
)
from scripts.piper_bilateral_geometry_decomposition import ObjPosSnapshotTracker

OUT = Path(__file__).resolve().parent.parent / "outputs" / "piper_bilateral_swept_features.json"

# Standalone-gripper-measured inner-face X extents (open -> closed), from a
# live query against piper_gripper.xml directly.
GRIPPER_XML = Path(__file__).resolve().parent.parent / "tango_robot" / "piper_assets" / "piper_gripper.xml"
Q_OPEN, Q_CLOSED = -0.05, -0.004

# Contact-band half-extents in the candidate's local Y/Z (perpendicular to
# closing axis and to approach axis respectively) -- a heuristic
# approximating finger7/8_collision's own footprint (measured full extent
# ~109.5x57x24.5mm in docs/PIPER_GRIPPER_AUDIT_20260807.md; using half of
# the two non-closing-axis dimensions here, generously rounded). Flagged as
# a heuristic, not a re-derivation of the exact mesh footprint.
BAND_HALF_Y_M = 0.03
BAND_HALF_Z_M = 0.02

OBSERVED = {
    1041: {"which_side_first": "right", "first_step": 1},
    1042: {"which_side_first": "left", "first_step": 1},
    1046: {"which_side_first": "right", "first_step": 2},
}


def measure_finger_inner_x():
    model = mujoco.MjModel.from_xml_path(str(GRIPPER_XML))
    data = mujoco.MjData(model)
    j7, j8 = model.joint("joint7").qposadr[0], model.joint("joint8").qposadr[0]

    def inner_x(q):
        data.qpos[j7] = q
        data.qpos[j8] = q
        mujoco.mj_forward(model, data)

        def verts(gname, bname):
            gid = model.geom(gname).id
            mid = model.geom_dataid[gid]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            vl = model.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)
            R = data.geom_xmat[gid].reshape(3, 3)
            return vl @ R.T + data.geom_xpos[gid]

        v7 = verts("finger7_collision", "link7")
        v8 = verts("finger8_collision", "link8")
        return float(v7[:, 0].max()), float(v8[:, 0].min())

    open_l, open_r = inner_x(Q_OPEN)
    closed_l, closed_r = inner_x(Q_CLOSED)
    return {"open_left_x": open_l, "open_right_x": open_r,
            "closed_left_x": closed_l, "closed_right_x": closed_r}


def object_mesh_points_world(env, obj_name):
    model = env.sim.model._model
    data = env.sim.data._data
    body_id = env.object_body_ids[obj_name]
    pts = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body_id:
            continue
        if model.geom_dataid[gid] < 0:
            continue  # not a mesh geom
        mid = model.geom_dataid[gid]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vl = model.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)
        R = data.geom_xmat[gid].reshape(3, 3)
        pts.append(vl @ R.T + data.geom_xpos[gid])
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))


def compute_features(obj_name, seed, finger_x):
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
            ppp.run_pick_and_place(
                env, obj_name, use_oriented_grasp=True, verbose=False,
                candidate_selection=None, wrist_friendly_orientation=True,
                step_hook=tracker,
            )
        finally:
            LegacyArmIK.__init__ = real_init
            ppp.ArmIK = original_armik

        ik = ik_holder["ik"]
        calls = [c for c in ik._calls if c["phase"] == "descend_refresh"]
        c = calls[-1]
        P = np.array(c["target_pos"])  # == corrected arm's true fingertip midpoint target
        R = np.array(c["target_mat"])

        # Object mesh points AT the descend_refresh moment -- need to
        # re-set sim state there, but the trial has already run to
        # completion by now. Use the object's LOCAL mesh (mesh_vert is
        # geometry-only, doesn't depend on current pose) transformed by
        # the SNAPSHOT pos/quat the tracker captured, not live sim state.
        model = env.sim.model._model
        body_id = env.object_body_ids[obj_name]
        obj_quat_at_call = None
        # xquat wasn't snapshotted by ObjPosSnapshotTracker (only xpos) --
        # re-derive geom points using the object's mesh in its OWN local
        # frame, rotated by a freshly-read quat. Since the object is a
        # free body and largely static during this brief window (confirmed
        # low translation/rotation in the earlier trace, <1deg/<9mm), using
        # the CURRENT (post-trial) quat as an approximation of orientation
        # (not position, which the tracker did snapshot) is reasonable for
        # a first-pass feature -- documented as a limitation.
        obj_quat_now = env.sim.data.xquat[body_id].copy()
        obj_pos_snapshot = tracker.obj_pos_at_descend_refresh

        # Build object points from LOCAL mesh vertices, transformed by the
        # snapshot position and current-read orientation.
        local_pts = []
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != body_id or model.geom_dataid[gid] < 0:
                continue
            mid = model.geom_dataid[gid]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            local_pts.append(model.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64))
        local_pts = np.concatenate(local_pts, axis=0) if local_pts else np.zeros((0, 3))
        R_obj = np.zeros((3, 3))
        mujoco.mju_quat2Mat(R_obj.reshape(-1), obj_quat_now)
        world_pts = local_pts @ R_obj.T + obj_pos_snapshot

        # Transform into the candidate's local frame (origin P, rotation R).
        cand_pts = (world_pts - P) @ R

        band = ((np.abs(cand_pts[:, 1]) <= BAND_HALF_Y_M) &
               (np.abs(cand_pts[:, 2]) <= BAND_HALF_Z_M))
        band_pts = cand_pts[band]

        left_pts = band_pts[band_pts[:, 0] < 0]
        right_pts = band_pts[band_pts[:, 0] > 0]

        open_l, open_r = finger_x["open_left_x"], finger_x["open_right_x"]
        left_surface_x = float(left_pts[:, 0].max()) if len(left_pts) else None
        right_surface_x = float(right_pts[:, 0].min()) if len(right_pts) else None
        left_gap = (left_surface_x - open_l) if left_surface_x is not None else None
        right_gap = (open_r - right_surface_x) if right_surface_x is not None else None

        if left_gap is None and right_gap is None:
            predicted_side = "neither"
        elif right_gap is None or (left_gap is not None and left_gap < right_gap):
            predicted_side = "left"
        else:
            predicted_side = "right"
        imbalance = (abs(left_gap - right_gap)
                    if (left_gap is not None and right_gap is not None) else None)

        return {
            "object": obj_name, "seed": seed,
            "n_band_points": int(len(band_pts)),
            "left_gap_m": left_gap, "right_gap_m": right_gap,
            "predicted_first_touch_side": predicted_side,
            "predicted_imbalance_m": imbalance,
        }
    finally:
        env.close()


def main():
    finger_x = measure_finger_inner_x()
    print("finger inner-face X extents (open -> closed):")
    print(f"  left (finger7):  {finger_x['open_left_x']:+.4f} -> {finger_x['closed_left_x']:+.4f}")
    print(f"  right (finger8): {finger_x['open_right_x']:+.4f} -> {finger_x['closed_right_x']:+.4f}")

    records = []
    for seed, obs in OBSERVED.items():
        rec = compute_features("cracker", seed, finger_x)
        rec["observed"] = obs
        records.append(rec)
        match = (rec["predicted_first_touch_side"] == obs["which_side_first"])
        print(f"\n=== cracker seed={seed} ===")
        print(f"  n_band_points={rec['n_band_points']}  "
              f"left_gap={rec['left_gap_m']}  right_gap={rec['right_gap_m']}")
        print(f"  predicted_first_touch={rec['predicted_first_touch_side']}  "
              f"imbalance={rec['predicted_imbalance_m']}")
        print(f"  observed_first_touch={obs['which_side_first']} (step {obs['first_step']})  "
              f"{'MATCH' if match else 'MISMATCH'}")

    n_match = sum(r["predicted_first_touch_side"] == r["observed"]["which_side_first"] for r in records)
    print(f"\n{n_match}/{len(records)} predictions matched observed first-touch side")

    OUT.write_text(json.dumps({"finger_x": finger_x, "records": records}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
