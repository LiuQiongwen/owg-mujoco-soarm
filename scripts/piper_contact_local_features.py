"""Phase A: recompute P2's geometry features on the FINGER CONTACT REGION
instead of the eef target, then test them as mediators of the P2 effect.

Why: P2's width feature was defined as a +/-15mm z-band around the eef
aim target. That is the wrong reference -- the fingers do not straddle the
eef site, they extend upward from roughly it. The band therefore sat below
pear's mesh entirely and returned 0.0 for every pear trial, leaving the
first link of the chain untested on the object with the strongest effect.

This measures the finger contact region directly from the model (finger
collision geoms, expressed in the eef site's own frame) and defines the
band from that, following the contact-local rather than object-origin
framing.

Cheap by construction: the aim point and the geometry around it are
PRE-EXECUTION quantities, so each row needs only env.reset() plus candidate
computation -- no rollout. The 180 P2 outcomes are joined in from
outputs/piper_cross_section_intervention.jsonl.

Run:  conda run -n tango python scripts/piper_contact_local_features.py
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
from scripts.piper_tcp_correction_ab import scene_objects_for
from scripts.piper_execution_trace import _mesh_world

ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "outputs" / "piper_cross_section_intervention.jsonl"
OUT = ROOT / "outputs" / "piper_contact_local_features.jsonl"

MAX_OPENING_M = 0.100
BAND_HALF_Y_M = 0.020


def finger_contact_zrange(env):
    """Finger collision geoms' extent along the eef site's own local Z,
    measured at the open position. This is the region that can actually
    touch the object -- the reference P2 should have used."""
    m, d = env.sim.model._model, env.sim.data._data
    eef = m.site("robot0_eef_site").id
    R = d.site_xmat[eef].reshape(3, 3)
    p = d.site_xpos[eef]
    zs = []
    for gid in range(m.ngeom):
        n = m.geom(gid).name or ""
        if ("finger7" in n or "finger8" in n) and "collision" in n:
            mid = m.geom_dataid[gid]
            if mid < 0:
                continue
            adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            vl = m.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)
            vw = vl @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
            zs.append(((vw - p) @ R)[:, 2])
    z = np.concatenate(zs)
    return float(z.min()), float(z.max())


def local_normal(pts, centre, k=60):
    """Surface normal at `centre` by PCA over its k nearest neighbours:
    the smallest-variance direction of a local patch."""
    if len(pts) < 8:
        return None
    d = np.linalg.norm(pts - centre, axis=1)
    nb = pts[np.argsort(d)[:min(k, len(pts))]]
    if len(nb) < 4:
        return None
    c = nb - nb.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[-1]


def features(env, obj_name, offset, zlo, zhi):
    m, d = env.sim.model._model, env.sim.data._data
    bid = env.object_body_ids[obj_name]
    quat = d.xquat[bid].copy()
    ref = ppp.true_centroid_xy(d.xpos[bid].copy(), quat, obj_name)
    gm = ppp.compute_grasp_orientation(env, obj_name)
    aim = ref + np.array([0.0, 0.0, ppp.GRASP_HEIGHT_OFFSET]) + gm[:, 1] * offset

    pts = _mesh_world(m, d, bid)
    loc = (pts - aim) @ gm            # x = closing axis, z = eef local Z
    band = loc[(loc[:, 2] >= zlo) & (loc[:, 2] <= zhi) &
               (np.abs(loc[:, 1]) <= BAND_HALF_Y_M)]
    out = {"n_band_points": int(len(band))}
    if len(band) < 10:
        return out

    # Parallel jaws close onto the EXTREME points along the closing axis,
    # so the relevant quantity is the band's extent. An earlier version of
    # this used the "innermost" surface on each side, which is meaningless
    # for a closed mesh: the surface wraps at the top and bottom of the
    # cross-section, so points near x~0 always exist and both innermost
    # values collapse to ~0 (observed: 0.02mm widths on a 66mm pear).
    lx = float(band[:, 0].min())      # left contact surface
    rx = float(band[:, 0].max())      # right contact surface
    out["left_surface_mm"] = lx * 1000
    out["right_surface_mm"] = rx * 1000
    out["support_width_mm"] = (rx - lx) * 1000
    out["opening_margin_mm"] = (MAX_OPENING_M - (rx - lx)) * 1000
    out["centring_error_mm"] = ((rx + lx) / 2) * 1000   # 0 = jaws centred on object

    lp = band[np.argmin(band[:, 0])]
    rp = band[np.argmax(band[:, 0])]
    nl = local_normal(band, lp)
    nr = local_normal(band, rp)
    if nl is not None and nr is not None:
        ax = np.array([1.0, 0.0, 0.0])         # closing axis in this frame
        al = abs(float(np.dot(nl / np.linalg.norm(nl), ax)))
        ar = abs(float(np.dot(nr / np.linalg.norm(nr), ax)))
        out["antipodal_left"] = al
        out["antipodal_right"] = ar
        out["antipodal_score"] = float(min(al, ar))   # both sides must be good
    return out


def main():
    p2 = [json.loads(l) for l in P2.open()]
    grid = sorted({(r["object"], r["seed"], r["offset_mm"]) for r in p2})
    outcome = {(r["object"], r["seed"], r["offset_mm"]): r for r in p2}

    rows = []
    cur_obj = None
    env = None
    zlo = zhi = None
    try:
        for obj, seed, off_mm in grid:
            np.random.seed(seed)
            if env is not None:
                env.close()
            env = PiperMultiObjectScene(
                robots="Piper", ycb_objects=scene_objects_for(obj),
                has_renderer=False, has_offscreen_renderer=False,
                use_camera_obs=False, control_freq=20)
            env.reset()
            if zlo is None:
                zlo, zhi = finger_contact_zrange(env)
                print(f"finger contact region, eef-local Z: [{zlo*1000:.1f}, {zhi*1000:.1f}] mm")
                print("(P2 used a +/-15mm band around the aim -- hence pear's zeros)\n")
            f = features(env, obj, off_mm / 1000.0, zlo, zhi)
            o = outcome[(obj, seed, off_mm)]
            f.update({"object": obj, "seed": seed, "offset_mm": off_mm,
                      "success": o["success"],
                      "gripper_q_at_close": o["gripper_q_at_close"],
                      "rel_dist_at_descend_mm": o["rel_dist_at_descend_mm"]})
            rows.append(f)
    finally:
        if env is not None:
            env.close()

    with OUT.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    keys = ["support_width_mm", "opening_margin_mm", "centring_error_mm",
            "antipodal_score", "left_surface_mm", "right_surface_mm"]
    offs = sorted({r["offset_mm"] for r in rows})
    for obj in sorted({r["object"] for r in rows}):
        print(f"\n=== {obj} ===")
        print(f"{'offset':>8s} {'succ':>6s} " + " ".join(f"{k.replace('_mm','')[:13]:>14s}" for k in keys))
        for off in offs:
            sub = [r for r in rows if r["object"] == obj and r["offset_mm"] == off]
            if not sub:
                continue
            cells = []
            for k in keys:
                v = [r[k] for r in sub if r.get(k) is not None]
                cells.append(f"{np.mean(v):14.2f}" if v else f"{'-':>14s}")
            print(f"{off:8.1f} {sum(r['success'] for r in sub):3d}/{len(sub):<2d} " + " ".join(cells))

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
