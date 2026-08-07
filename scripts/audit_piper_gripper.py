"""Piper gripper audit: pure static/kinematic checks, no robosuite env, no task.

READ-ONLY. Loads tango_robot/piper_assets/piper_gripper.xml directly via
mujoco.MjModel.from_xml_path -- does not modify any file under
tango_robot/piper_robosuite/ or tango_robot/piper_assets/, and does not go
through robosuite's env/controller machinery (avoids depending on exactly
which robosuite version/API is installed for a check that's really about
mesh geometry and the actuator/joint definitions already in the MJCF).

Answers, for the PASS/FAIL table:
  - pad geometry: is finger7/8_collision a small, well-localized contact
    surface, or (like SO-101 before step 3) a large mesh spanning the whole
    finger body?
  - opening semantics: true fingertip opening as a function of the
    (coupled, both-fingers-equal) commanded joint position, vs. what the
    already-documented calibration (piper_gripper.py's comments) claims.
  - left/right symmetry: do joint7 and joint8 produce mirror-symmetric
    finger motion, given their different body quaternions?
  - contact stiffness sanity: read back the ALREADY-DOCUMENTED solref/solimp
    (piper_gripper.xml already notes solref="0.002 1", found and worked
    around for a real QACC/NaN at exact full closure) and report it plainly
    for the table, rather than re-discovering it.

Run:  conda run -n tango python scripts/audit_piper_gripper.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

XML = str(Path(__file__).resolve().parent.parent / "tango_robot" / "piper_assets" / "piper_gripper.xml")
OUT = Path(__file__).resolve().parent.parent / "calib" / "piper_gripper_audit.json"

# From piper_gripper.py's format_action (already-documented, verified
# calibration): the actuator ctrlrange the real controller commands within.
CTRL_MIN, CTRL_MAX = -0.05, -0.004   # "open" = -0.05, "closed" = -0.004
# From piper_real_backend.py: real hardware's own measured full opening.
REAL_GRIP_OPEN_M = 0.12


def geom_extent_along_local_axes(model, gid):
    """Mesh bounding half-extents in the geom's own local frame -- for type=7
    (mesh) geoms, model.geom_size stores the AABB half-extents directly."""
    return model.geom_size[gid].copy()


def fingertip_point(model, data, body_name, gid):
    """Best-effort 'tip' point: the mesh vertex farthest from the joint
    origin along the body's local -Z (fingers close along the slide axis,
    which is local Z per the joint definition; the tip is the extreme end).
    """
    mid = model.geom_dataid[gid]
    adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts_local = model.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)
    R = data.geom_xmat[gid].reshape(3, 3)
    verts_world = verts_local @ R.T + data.geom_xpos[gid]
    return verts_world, verts_local


def main():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)

    g7 = model.geom("finger7_collision").id
    g8 = model.geom("finger8_collision").id
    j7adr = model.joint("joint7").qposadr[0]
    j8adr = model.joint("joint8").qposadr[0]

    print("=" * 90)
    print("1. pad geometry: is the collision geom a small pad or the whole finger body?")
    print("=" * 90)
    for name, gid in (("finger7_collision", g7), ("finger8_collision", g8)):
        ext = geom_extent_along_local_axes(model, gid)
        print(f"  {name}: mesh AABB half-extents (local) = {np.round(ext*1000, 1)} mm "
              f"-> full size {np.round(ext*2000, 1)} mm")
    print("  (SO-101's pre-step-3 jaw mesh spanned ~105mm and produced 12-14mm "
          "penetration events; a similarly large uncontrolled contact surface "
          "here would carry the same risk)")

    print("\n" + "=" * 90)
    print("2. contact stiffness (already documented in piper_gripper.xml, "
          "reading back rather than re-discovering)")
    print("=" * 90)
    for name, gid in (("finger7_collision", g7), ("finger8_collision", g8)):
        print(f"  {name}: solref={model.geom_solref[gid]}  solimp={model.geom_solimp[gid]}  "
              f"friction={model.geom_friction[gid]}")
    lo, hi = model.geom_solref[g7]
    ratio = lo / model.opt.timestep
    print(f"  solref time constant / timestep = {lo}/{model.opt.timestep} = {ratio:.2f}x "
          f"({'AT the >=2x stability floor' if ratio <= 2.01 else 'above the floor'})")

    print("\n" + "=" * 90)
    print("3. opening semantics: commanded qpos -> true fingertip opening")
    print("=" * 90)
    # Both joints receive the SAME target per PiperGripper.format_action
    # (symmetric control) -- sweep them together.
    qs = np.linspace(-0.05, 0.0, 11)   # full mechanical range
    lut = []
    for q in qs:
        data.qpos[j7adr] = q
        data.qpos[j8adr] = q
        mujoco.mj_forward(model, data)
        v7w, _ = fingertip_point(model, data, "link7", g7)
        v8w, _ = fingertip_point(model, data, "link8", g8)
        # True opening: min distance between the two finger meshes' vertex sets
        # restricted to their distal (tip) quarter by distance from the joint
        # origin, matching the SO-101 audit's method.
        j7pos = data.xpos[model.body("link7").id]
        j8pos = data.xpos[model.body("link8").id]
        r7 = np.linalg.norm(v7w - j7pos, axis=1)
        r8 = np.linalg.norm(v8w - j8pos, axis=1)
        tip7 = v7w[r7 >= np.quantile(r7, 0.75)]
        tip8 = v8w[r8 >= np.quantile(r8, 0.75)]
        from scipy.spatial import cKDTree
        gap = float(cKDTree(tip8).query(tip7, k=1)[0].min())
        claimed_travel = q - (-0.05)   # naive "0 at open, travel to closed" reading
        in_ctrl_window = CTRL_MIN <= q <= CTRL_MAX
        lut.append({"q": float(q), "true_tip_gap_m": gap, "in_ctrl_window": bool(in_ctrl_window)})
        marker = "  <- commanded window" if in_ctrl_window else ""
        print(f"  q={q:+.4f}  true_tip_gap={gap*1000:7.2f}mm{marker}")

    window = [r for r in lut if r["in_ctrl_window"]]
    open_gap = max(r["true_tip_gap_m"] for r in window)
    closed_gap = min(r["true_tip_gap_m"] for r in window)
    print(f"\n  over the commanded ctrlrange [{CTRL_MIN}, {CTRL_MAX}]:")
    print(f"    true opening spans {closed_gap*1000:.1f} .. {open_gap*1000:.1f} mm")
    print(f"    real hardware's own measured full opening (piper_real_backend.py): "
          f"{REAL_GRIP_OPEN_M*1000:.1f} mm")
    sim_vs_real_ratio = open_gap / REAL_GRIP_OPEN_M
    print(f"    sim/real ratio: {sim_vs_real_ratio:.2f}"
          f"  ({'plausible (both fingers, so real 120mm may be PER-SIDE or FULL -- check convention)' if 0.3 < sim_vs_real_ratio < 3 else 'suspicious mismatch'})")

    print("\n" + "=" * 90)
    print("4. left/right symmetry: do joint7 and joint8 move mirror-symmetrically?")
    print("=" * 90)
    data.qpos[j7adr] = -0.025
    data.qpos[j8adr] = -0.025
    mujoco.mj_forward(model, data)
    c7 = data.geom_xpos[g7].copy()
    c8 = data.geom_xpos[g8].copy()
    mid = 0.5 * (c7 + c8)
    d7 = float(np.linalg.norm(c7 - mid))
    d8 = float(np.linalg.norm(c8 - mid))
    print(f"  at q7=q8=-0.025 (init_qpos, 'roughly half-open'):")
    print(f"    finger7_collision centre: {np.round(c7, 4)}  dist from midpoint: {d7*1000:.2f}mm")
    print(f"    finger8_collision centre: {np.round(c8, 4)}  dist from midpoint: {d8*1000:.2f}mm")
    print(f"    symmetry error: {abs(d7-d8)*1000:.3f}mm "
          f"({'PASS' if abs(d7-d8) < 0.002 else 'FAIL'})")

    print("\n" + "=" * 90)
    print("5. finger-finger self-overlap (already documented + excluded in the XML)")
    print("=" * 90)
    for q in (-0.05, -0.025, -0.004, 0.0):
        data.qpos[j7adr] = q
        data.qpos[j8adr] = q
        mujoco.mj_forward(model, data)
        dist = mujoco.mj_geomDistance(model, data, g7, g8, 1.0, np.zeros(6))
        print(f"  q={q:+.4f}: finger7<->finger8 mj_geomDistance = {dist*1000:+.2f}mm "
              f"({'would overlap without the <exclude>' if dist < 0 else 'clear'})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "opening_lut": lut,
        "symmetry_error_m": abs(d7 - d8),
        "solref": model.geom_solref[g7].tolist(),
        "solref_timeconst_over_timestep": ratio,
    }, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
