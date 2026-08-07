"""Piper blocked-closure microbenchmark. READ-ONLY w.r.t. piper_robosuite/ --
no file under tango_robot/piper_robosuite/ or tango_robot/piper_assets/ is
modified. Builds a small standalone scene combining piper_gripper.xml
(included verbatim, not edited) with a free-floating fixture box, entirely
outside robosuite -- no arm, no IK, no candidate/task machinery, matching the
frozen/isolated design validated on SO-101
(scripts/microbenchmark_blocked_closure.py).

Since the gripper's two fingers are commanded to an IDENTICAL target
(PiperGripper.format_action, symmetric control) and are geometrically mirror-
symmetric (confirmed in scripts/audit_piper_gripper.py), there is no
approach/IK step to freeze here at all -- the gripper body itself is fixed in
world frame (no freejoint added), and the fixture is placed directly between
the fingers at their measured tip midpoint. This sidesteps the
grip_site-vs-fingertip 65.6mm offset entirely (that offset matters for WHERE
the arm positions the gripper in the real pipeline, not for this isolated
closing test).

Run:  conda run -n tango python scripts/microbenchmark_piper_blocked_closure.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.spatial import cKDTree

GRIPPER_XML = Path(__file__).resolve().parent.parent / "tango_robot" / "piper_assets" / "piper_gripper.xml"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "piper_blocked_closure.jsonl"

KNOWN_THICKNESS_M = 0.030   # 30mm box, same as the SO-101 fixture for direct comparability
CLOSE_STEPS = 400
Q_OPEN = -0.05
Q_CLOSED = -0.004   # actuator ctrlrange floor, per piper_gripper.xml (0.0 causes documented QACC/NaN)


def build_scene_xml(box_half_extents_m, box_pos_m):
    """Wrap piper_gripper.xml verbatim (read, not modified) plus a free box."""
    gripper_body = GRIPPER_XML.read_text()
    # Extract just the <worldbody> children and <asset>/<actuator>/<equality>/
    # <contact>/<sensor> blocks by simple string ops (the file is small and
    # fixed-format; avoids adding an XML-diffing dependency for one script).
    import re
    def extract(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", gripper_body, re.S)
        return m.group(1) if m else ""

    asset = extract("asset")
    actuator = extract("actuator")
    worldbody = extract("worldbody")
    equality = extract("equality")
    contact = extract("contact")

    hx, hy, hz = box_half_extents_m
    bx, by, bz = box_pos_m
    return f"""<mujoco model="piper_blocked_closure">
  <compiler meshdir="{GRIPPER_XML.parent}" angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <asset>{asset}</asset>
  <worldbody>
    {worldbody}
    <body name="fixture" pos="{bx} {by} {bz}">
      <freejoint name="fixture_joint"/>
      <geom name="fixture_geom" type="box" size="{hx} {hy} {hz}" mass="0.15"
            rgba="0.7 0.3 0.2 1" friction="1.5 0.005 0.0001"/>
    </body>
  </worldbody>
  <equality>{equality}</equality>
  <contact>{contact}</contact>
  <actuator>{actuator}</actuator>
</mujoco>"""


def measure_fingertip_midpoint(model, data, g7, g8, q):
    j7 = model.joint("joint7").qposadr[0]
    j8 = model.joint("joint8").qposadr[0]
    data.qpos[j7] = q
    data.qpos[j8] = q
    mujoco.mj_forward(model, data)

    def tip(gid, bname):
        mid = model.geom_dataid[gid]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vl = model.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)
        R = data.geom_xmat[gid].reshape(3, 3)
        vw = vl @ R.T + data.geom_xpos[gid]
        bpos = data.xpos[model.body(bname).id]
        r = np.linalg.norm(vw - bpos, axis=1)
        return vw[r >= np.quantile(r, 0.75)]

    t7, t8 = tip(g7, "link7"), tip(g8, "link8")
    return 0.5 * (t7.mean(0) + t8.mean(0))


def run_trial():
    # First pass: measure the open-gripper fingertip midpoint using the
    # standalone gripper model (matches audit_piper_gripper.py's method).
    probe = mujoco.MjModel.from_xml_path(str(GRIPPER_XML))
    pdata = mujoco.MjData(probe)
    g7p, g8p = probe.geom("finger7_collision").id, probe.geom("finger8_collision").id
    mid_open = measure_fingertip_midpoint(probe, pdata, g7p, g8p, Q_OPEN)

    xml = build_scene_xml((0.05, 0.05, KNOWN_THICKNESS_M / 2), mid_open.tolist())
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    g7 = model.geom("finger7_collision").id
    g8 = model.geom("finger8_collision").id
    j7 = model.joint("joint7").qposadr[0]
    j8 = model.joint("joint8").qposadr[0]
    act7 = model.actuator("gripper_finger_joint7").id
    act8 = model.actuator("gripper_finger_joint8").id

    data.qpos[j7] = Q_OPEN
    data.qpos[j8] = Q_OPEN
    data.ctrl[act7] = Q_OPEN
    data.ctrl[act8] = Q_OPEN
    mujoco.mj_forward(model, data)

    fbid = model.body("fixture").id
    p0 = data.xpos[fbid].copy()

    fgid = model.geom("fixture_geom").id

    warn_types = [mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL,
                 mujoco.mjtWarning.mjWARN_BADQPOS]
    for w in warn_types:
        data.warning[w].number = 0

    min_d7 = min_d8 = float("inf")
    steady_d7, steady_d8 = [], []
    for step in range(CLOSE_STEPS):
        data.ctrl[act7] = Q_CLOSED
        data.ctrl[act8] = Q_CLOSED
        mujoco.mj_step(model, data)
        d7 = float(mujoco.mj_geomDistance(model, data, g7, fgid, 1.0, np.zeros(6)))
        d8 = float(mujoco.mj_geomDistance(model, data, g8, fgid, 1.0, np.zeros(6)))
        min_d7, min_d8 = min(min_d7, d7), min(min_d8, d8)
        if step > CLOSE_STEPS - 50:
            steady_d7.append(d7)
            steady_d8.append(d8)

    p1 = data.xpos[fbid].copy()
    warning_counts = {str(w).split(".")[-1]: int(data.warning[w].number) for w in warn_types}

    return {
        "known_thickness_m": KNOWN_THICKNESS_M,
        "min_dist_finger7_m": min_d7,
        "min_dist_finger8_m": min_d8,
        "steady_dist_finger7_m": float(np.mean(steady_d7)),
        "steady_dist_finger8_m": float(np.mean(steady_d8)),
        "obj_displacement_m": float(np.linalg.norm(p1 - p0)),
        "warning_counts": warning_counts,
        "any_warning": any(warning_counts.values()),
    }


def main():
    r = run_trial()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=1) + "\n")

    print(f"known thickness: {r['known_thickness_m']*1000:.1f}mm")
    print(f"steady dist (finger7/finger8): "
          f"{r['steady_dist_finger7_m']*1000:+.2f} / {r['steady_dist_finger8_m']*1000:+.2f} mm")
    print(f"min dist (finger7/finger8): "
          f"{r['min_dist_finger7_m']*1000:+.2f} / {r['min_dist_finger8_m']*1000:+.2f} mm")
    print(f"object displacement: {r['obj_displacement_m']*1000:.2f}mm")
    print(f"warnings: {r['warning_counts']}")
    verdict = "PASS" if (abs(r['steady_dist_finger7_m']) < 0.005
                        and abs(r['steady_dist_finger8_m']) < 0.005
                        and not r['any_warning']) else "FAIL"
    print(f"\nblocked-closure verdict: {verdict}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
