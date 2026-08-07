"""Calibrate T_eef_capture: is the 65.6mm eef_site-to-fingertip-midpoint gap
found in scripts/audit_piper_gripper.py a genuine RIGID transform (constant
in the eef_site's own local frame), or does it drift with arm pose /
gripper opening?

READ-ONLY. Constructs a standard PiperLiftYCB env (same class every existing
Piper script in this repo already uses) and reads state -- does not modify
any file under tango_robot/piper_robosuite/ or tango_robot/piper_assets/.

Why local frame, not world frame
---------------------------------
A fixed WORLD-frame offset ("+65.6mm along world Z") would only be correct
for one specific arm orientation. The physically correct object is a rigid
transform expressed in the eef_site's OWN frame: local_offset = R_eef^T @
(fingertip_mid_world - eef_pos_world). If that local vector is the same
regardless of how the arm (and therefore the eef_site) is oriented, it's a
genuine fixed transform and a single T_eef_capture can be composed with any
future candidate pose correctly, orientation included.

Sweeps 5 arm joint configurations (spanning a realistic range, not just
READY_QPOS) x 3 gripper openings (open/mid/closed) = 15 samples, and reports
the local offset's magnitude and direction stability across all of them.

Run:  conda run -n tango python scripts/calibrate_piper_capture_frame.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa: registers Piper/PiperGripper
from tango_robot.piper_robosuite.piper_lift_ycb import PiperLiftYCB

JOINTS = [f"robot0_joint{i}" for i in range(1, 7)]
EEF_SITE = "robot0_eef_site"
GRIP_OPENINGS = [-0.05, -0.025, -0.004]   # open / mid / closed, per piper_gripper.xml's ctrlrange

# 5 arm configurations spanning a realistic range around READY_QPOS
# (piper_pick_and_place.py's own constant), not just one pose.
READY_QPOS = np.array([0.0, 0.2, 0.42, 1.6, 0.0, 0.0])
ARM_CONFIGS = [
    READY_QPOS,
    READY_QPOS + np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0]),
    READY_QPOS + np.array([-0.3, 0.1, -0.1, 0.0, 0.2, 0.5]),
    READY_QPOS + np.array([0.0, -0.15, 0.2, -0.3, 0.0, -0.8]),
    READY_QPOS + np.array([0.5, 0.05, -0.15, 0.4, -0.3, 1.2]),
]

OUT = Path(__file__).resolve().parent.parent / "calib" / "piper_capture_frame_calibration.json"


def _find_by_substring(model, kind, substrings):
    """Resolve a body/geom id by substring match on its name -- robosuite
    prefixes/renames everything (e.g. "robot0_right_gripper_link7"), so exact
    names from the standalone piper_gripper.xml don't apply here."""
    getter = model.body if kind == "body" else model.geom
    n = model.nbody if kind == "body" else model.ngeom
    for i in range(n):
        name = getter(i).name or ""
        if all(s in name for s in substrings):
            return i
    raise ValueError(f"no {kind} matching {substrings} found")


def fingertip_midpoint_world(model, data):
    g7 = _find_by_substring(model, "geom", ["finger7", "collision"])
    g8 = _find_by_substring(model, "geom", ["finger8", "collision"])
    b7 = model.geom_bodyid[g7]
    b8 = model.geom_bodyid[g8]

    def tip(gid, bid):
        mid = model.geom_dataid[gid]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vl = model.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)
        R = data.geom_xmat[gid].reshape(3, 3)
        vw = vl @ R.T + data.geom_xpos[gid]
        bpos = data.xpos[bid]
        r = np.linalg.norm(vw - bpos, axis=1)
        return vw[r >= np.quantile(r, 0.75)]

    t7 = tip(g7, b7)
    t8 = tip(g8, b8)
    return 0.5 * (t7.mean(0) + t8.mean(0))


def main():
    env = PiperLiftYCB(robots="Piper", ycb_object="pear", has_renderer=False,
                       has_offscreen_renderer=False, use_camera_obs=False,
                       control_freq=20)
    env.reset()
    model = env.sim.model._model
    data = env.sim.data._data

    qpos_adr = [model.joint(n).qposadr[0] for n in JOINTS]
    # Resolve gripper joint names as actually registered (robosuite prefixes
    # with the gripper's own naming, not necessarily "joint7"/"joint8").
    j7_id = next(j for j in range(model.njnt) if "joint7" in (model.joint(j).name or ""))
    j8_id = next(j for j in range(model.njnt) if "joint8" in (model.joint(j).name or ""))
    j7adr = model.joint(j7_id).qposadr[0]
    j8adr = model.joint(j8_id).qposadr[0]
    eef_site_id = model.site(EEF_SITE).id

    print(f"resolved gripper joints: {model.joint(j7_id).name} / {model.joint(j8_id).name}")

    results = []
    for ai, arm_q in enumerate(ARM_CONFIGS):
        for gi, gq in enumerate(GRIP_OPENINGS):
            for adr, q in zip(qpos_adr, arm_q):
                data.qpos[adr] = q
            data.qpos[j7adr] = gq
            data.qpos[j8adr] = gq
            mujoco.mj_forward(model, data)

            eef_pos = data.site_xpos[eef_site_id].copy()
            eef_R = data.site_xmat[eef_site_id].reshape(3, 3).copy()
            cap = fingertip_midpoint_world(model, data)

            world_offset = cap - eef_pos
            local_offset = eef_R.T @ world_offset

            results.append({
                "arm_config": ai, "grip_opening": gq,
                "eef_pos": eef_pos.tolist(),
                "capture_pos": cap.tolist(),
                "world_offset_m": world_offset.tolist(),
                "local_offset_m": local_offset.tolist(),
                "local_offset_norm_m": float(np.linalg.norm(local_offset)),
            })
            print(f"  arm={ai} grip={gq:+.3f}  world_offset={np.round(world_offset,4)}  "
                  f"local_offset={np.round(local_offset,4)}  "
                  f"norm={np.linalg.norm(local_offset)*1000:.2f}mm")

    norms = np.array([r["local_offset_norm_m"] for r in results])
    local_offsets = np.array([r["local_offset_m"] for r in results])
    mean_local = local_offsets.mean(0)
    max_dev = np.max(np.linalg.norm(local_offsets - mean_local, axis=1))

    print("\n" + "=" * 90)
    print("rigidity check")
    print("=" * 90)
    print(f"  local_offset_norm: mean={norms.mean()*1000:.2f}mm  "
          f"std={norms.std()*1000:.3f}mm  range=[{norms.min()*1000:.2f}, {norms.max()*1000:.2f}]mm")
    print(f"  mean local_offset vector: {np.round(mean_local, 5)}")
    print(f"  max deviation from mean (any sample): {max_dev*1000:.3f}mm")
    verdict = "RIGID (PASS)" if max_dev < 0.001 else ("MOSTLY RIGID" if max_dev < 0.005 else "NOT RIGID (FAIL)")
    print(f"  verdict: {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "results": results,
        "mean_local_offset_m": mean_local.tolist(),
        "max_deviation_m": float(max_dev),
        "verdict": verdict,
    }, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
