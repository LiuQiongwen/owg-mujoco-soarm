"""Audit each Piper object's actual graspable geometry AT the real grasp
height -- read-only.

Two questions, both raised by earlier findings that turned out to rest on
stale or cross-platform numbers:

1. WIDTH. The 106.7mm "local support width" that condemned TomatoSoupCan as
   mechanically ungraspable came from the SO-101 line
   (scripts/derive_capture_reference.py, compared against SO-101's 95.7mm
   jaw) -- a different platform, asset variant, and jaw range. Piper's own
   file separately claims "~8.6cm diameter" against a "~7.6cm max opening",
   but the live composed model measures a 100.0mm inner-face opening. None
   of those numbers can be trusted for Piper without re-measuring here.

2. HEIGHT. OBJECT_TOP_OFFSET (top surface above the object's reference
   position) lists can=43.7mm and banana=34.0mm, but the validated grasp
   height puts the FINGERTIP MIDPOINT at +65.6mm above that same reference
   (GRASP_CAPTURE_HEIGHT_OFFSET -- see
   docs/PIPER_CORRECTION_AND_INTEGRATION_20260807.md). If those numbers are
   right, the gripper closes ABOVE those objects entirely -- which would be
   a far better explanation for their failures than width, and would be a
   candidate failure separator. Checked directly here rather than inferred
   from the table.

Measures, per object, from the live composed robosuite model: the mesh's
true z extent relative to the grasp reference, whether the grasp height
falls inside the object at all, and the cross-sectional width along the
gripper's closing axis in a band at that height.

Run:  conda run -n tango python scripts/audit_piper_object_grasp_geometry.py
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
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene, ALL_OBJECTS

OUT = Path(__file__).resolve().parent.parent / "outputs" / "piper_object_grasp_geometry.json"

# Measured inner-face gap of the live composed model at the actuator's own
# ctrlrange floor (-0.05). NOT the stale ~7.6cm in the narrow-axis comments,
# and not REAL_GRIP_OPEN_M=0.12 (a real-hardware figure).
MEASURED_MAX_OPENING_M = 0.100

BAND_HALF_M = 0.005   # +/-5mm slice around the grasp height


def measure(obj_name):
    np.random.seed(0)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=[obj_name],
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        m = env.sim.model._model
        d = env.sim.data._data
        bid = env.object_body_ids[obj_name]

        # Grasp reference position, exactly as run_pick_and_place computes it.
        body_origin = d.xpos[bid].copy()
        quat = d.xquat[bid].copy()
        ref = ppp.true_centroid_xy(body_origin, quat, obj_name)

        # Object mesh points in WORLD frame.
        pts = []
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] != bid or m.geom_dataid[gid] < 0:
                continue
            mid = m.geom_dataid[gid]
            adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            vl = m.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)
            pts.append(vl @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid])
        pts = np.concatenate(pts, axis=0)

        z_rel = pts[:, 2] - ref[2]
        z_min, z_max = float(z_rel.min()), float(z_rel.max())

        grasp_h = ppp.GRASP_CAPTURE_HEIGHT_OFFSET
        table_top_offset = ppp.OBJECT_TOP_OFFSET.get(obj_name)

        inside = z_min <= grasp_h <= z_max
        headroom_mm = (z_max - grasp_h) * 1000   # +ve => grasp height is below the top

        # Cross-sectional width along the gripper's closing axis, in a band
        # at the grasp height. The closing axis is grasp_mat's column 0
        # (local X, confirmed against finger7/8 geometry).
        grasp_mat = ppp.compute_grasp_orientation(env, obj_name)
        closing_axis = grasp_mat[:, 0]
        band = pts[np.abs(z_rel - grasp_h) <= BAND_HALF_M]
        if len(band):
            proj = (band - ref) @ closing_axis
            width = float(proj.max() - proj.min())
        else:
            width = None

        # Also the width at the object's own mid-height, for comparison --
        # what a "grasp the middle of the object" reference would see.
        mid_h = 0.5 * (z_min + z_max)
        band_mid = pts[np.abs(z_rel - mid_h) <= BAND_HALF_M]
        width_mid = None
        if len(band_mid):
            proj_mid = (band_mid - ref) @ closing_axis
            width_mid = float(proj_mid.max() - proj_mid.min())

        return {
            "object": obj_name,
            "z_min_rel_m": z_min, "z_max_rel_m": z_max,
            "object_height_m": z_max - z_min,
            "OBJECT_TOP_OFFSET_m": table_top_offset,
            "grasp_height_rel_m": grasp_h,
            "grasp_height_inside_object": bool(inside),
            "headroom_below_top_mm": headroom_mm,
            "width_at_grasp_height_m": width,
            "width_at_mid_height_m": width_mid,
            "max_opening_m": MEASURED_MAX_OPENING_M,
            "width_fits_at_grasp_height": (None if width is None
                                            else bool(width < MEASURED_MAX_OPENING_M)),
            "width_fits_at_mid_height": (None if width_mid is None
                                          else bool(width_mid < MEASURED_MAX_OPENING_M)),
        }
    finally:
        env.close()


def main():
    rows = []
    for obj in ALL_OBJECTS:
        try:
            rows.append(measure(obj))
        except Exception as e:  # placement sampler can reject some solo scenes
            print(f"  {obj}: SKIPPED ({type(e).__name__}: {e})")

    print(f"\n{'object':9s} {'height':>8s} {'top_off':>8s} {'grasp_h':>8s} {'headroom':>9s} "
          f"{'w@grasp':>9s} {'w@mid':>8s}  verdict")
    print("-" * 88)
    for r in rows:
        w = r["width_at_grasp_height_m"]
        wm = r["width_at_mid_height_m"]
        if not r["grasp_height_inside_object"]:
            verdict = "GRASP HEIGHT ABOVE OBJECT"
        elif w is not None and w >= MEASURED_MAX_OPENING_M:
            verdict = "too wide at grasp height"
        else:
            verdict = "ok"
        print(f"{r['object']:9s} {r['object_height_m']*1000:7.1f} "
              f"{(r['OBJECT_TOP_OFFSET_m'] or 0)*1000:7.1f} {r['grasp_height_rel_m']*1000:7.1f} "
              f"{r['headroom_below_top_mm']:8.1f} "
              f"{(w*1000 if w else float('nan')):8.1f} {(wm*1000 if wm else float('nan')):7.1f}  {verdict}")

    print(f"\n(measured max opening = {MEASURED_MAX_OPENING_M*1000:.1f}mm; "
          f"headroom = mm the grasp height sits BELOW the object's top; negative = above it)")

    OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
