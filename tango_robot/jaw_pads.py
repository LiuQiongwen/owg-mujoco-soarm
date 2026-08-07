"""Derive box pad geoms for the SO-101 fingers from mesh geometry.

Produces the `<geom>` attributes `_so101_fragment` injects into the scene XML,
so the pads are measured rather than guessed, and the derivation is auditable.
Run `scripts/derive_jaw_pads.py` to print them.

Why the hinge's polar frame
---------------------------
The SO-101 jaw is a single-hinge scissor, not a parallel gripper, so the natural
axes are radial (along the finger), tangential (the closing direction) and axial
(along the hinge = finger width).  Measured in the `gripper` body frame the hinge
axis comes out exactly [0, -1, 0].

Selecting the gripping face
---------------------------
Per radius bin, each finger's inner face is its angular extreme on the side
facing the other finger -- max angle for the fixed finger, min angle for the
moving one.  Two approaches were tried first and rejected against measurement:

  * "points closest to the opposing mesh, averaged over closing angles" drifts
    with the angle and picks up the hinge structure;
  * a plain SVD over those points returns rms spread [43, 4, 3] mm, i.e. a long
    thin ridge rather than a plane, so its third axis is not a face normal.

The fingers taper: at the near-closed pose the two inner faces are 10.3 deg
apart at r = 60 mm but only 0.6 deg apart at r = 80 mm, so they close tip-first.
A pad perpendicular to the tangential direction would stand several mm off the
real surface; fitting a plane to the properly selected face points instead lets
the box follow that taper.

Extents use quantiles, not max: the face point set has thin tails running up the
finger shank, and sizing a box off the extreme point makes a pad several times
longer than the surface it represents.

"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from tango_robot.jaw_metrology import _mesh_geom_surface_local, _to_world

SO101_XML = str(Path(__file__).resolve().parent / "assets" / "so101" / "so101.xml")

FIXED_FINGER = ("gripper", "wrist_roll_follower_so101_v1")
MOVING_FINGER = ("moving_jaw_so101_v1", "moving_jaw_so101_v1")

BAND_LO, BAND_HI = 0.058, 0.086   # distal gripping band, radius from the hinge
N_BINS = 14
ANG_TOL_DEG = 6.0        # keep points within this of the bin's extreme angle
AXIAL_Q = (0.05, 0.95)   # robust width limits along the hinge axis
EXT_Q = 0.98             # quantile for the pad's in-plane half-extents
PAD_HALF_THICK = 0.0015  # 3 mm thick pad
PAD_PROUD = 0.0005       # sit 0.5 mm proud of the measured face


def _col_geom(model, body: str, mesh: str) -> int:
    bid = model.body(body).id
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] != bid:
            continue
        did = model.geom_dataid[gi]
        if did >= 0 and model.mesh(did).name == mesh and model.geom_contype[gi] != 0:
            return gi
    raise ValueError(f"collision geom for {mesh} not on {body}")


def derive(verbose: bool = False) -> dict:
    """Return {'fixed': {...}, 'moving': {...}} with pos/size/quat per pad,
    expressed in the frame of the body each pad attaches to."""
    m = mujoco.MjModel.from_xml_path(SO101_XML)
    d = mujoco.MjData(m)
    gf = _col_geom(m, *FIXED_FINGER)
    gm = _col_geom(m, *MOVING_FINGER)
    qadr = m.joint("gripper").qposadr[0]
    gr_bid = m.body(FIXED_FINGER[0]).id
    mv_bid = m.body(MOVING_FINGER[0]).id

    Vf = _mesh_geom_surface_local(m, gf, 30000)
    Vm = _mesh_geom_surface_local(m, gm, 30000)

    d.qpos[:] = 0.0
    d.qpos[qadr] = float(m.joint("gripper").range[0])   # near-closed
    mujoco.mj_forward(m, d)

    # Hinge polar frame, anchored in the gripper body (static w.r.t. the fixed
    # finger) so the moving finger's swing is explicit.
    Og = d.xpos[gr_bid].copy()
    Rg = d.xmat[gr_bid].reshape(3, 3)
    hinge_g = Rg.T @ (d.xpos[mv_bid] - Og)
    axis_g = Rg.T @ d.xmat[mv_bid].reshape(3, 3)[:, 2]
    axis_g /= np.linalg.norm(axis_g)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(tmp @ axis_g) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis_g, tmp); u /= np.linalg.norm(u)
    v = np.cross(axis_g, u)

    def polar(W):
        G = (W - Og) @ Rg - hinge_g
        pu, pv = G @ u, G @ v
        return np.hypot(pu, pv), np.arctan2(pv, pu), G @ axis_g

    Wf, Wm = _to_world(Vf, d, gf), _to_world(Vm, d, gm)
    rf, af, zf = polar(Wf)
    rm, am, zm = polar(Wm)
    ref = np.arctan2(np.sin(af).mean(), np.cos(af).mean())
    af = np.angle(np.exp(1j * (af - ref)))
    am = np.angle(np.exp(1j * (am - ref)))

    # Axial band shared by both fingers = the width over which they can grip.
    zlo = max(np.quantile(zf, AXIAL_Q[0]), np.quantile(zm, AXIAL_Q[0]))
    zhi = min(np.quantile(zf, AXIAL_Q[1]), np.quantile(zm, AXIAL_Q[1]))

    out = {}
    edges = np.linspace(BAND_LO, BAND_HI, N_BINS + 1)
    for tag, V, r, a, z, gid, sign in (
            ("fixed", Vf, rf, af, zf, gf, +1),     # inner face = MAX angle
            ("moving", Vm, rm, am, zm, gm, -1)):   # inner face = MIN angle
        keep = np.zeros(len(V), dtype=bool)
        inband_z = (z >= zlo) & (z <= zhi)
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = (r >= lo) & (r < hi) & inband_z
            if b.sum() < 20:
                continue
            extreme = np.quantile(a[b] * sign, 0.98) * sign
            keep |= b & (np.abs(a - extreme) < np.radians(ANG_TOL_DEG))
        if keep.sum() < 50:
            raise RuntimeError(f"{tag}: only {keep.sum()} face points selected")

        P = V[keep]                     # geom-local frame
        centre = P.mean(0)
        _, S, Vt = np.linalg.svd(P - centre, full_matrices=False)
        rms = S / np.sqrt(len(P))
        e0, e1, n = Vt[0], Vt[1], Vt[2]
        proj = (P - centre) @ np.vstack([e0, e1, n]).T
        out[tag] = dict(
            gid=gid, centre=centre, e0=e0, e1=e1, n=n, rms=rms,
            half0=float(np.quantile(np.abs(proj[:, 0]), EXT_Q)),
            half1=float(np.quantile(np.abs(proj[:, 1]), EXT_Q)),
            flatness=float(rms[2]), n_pts=int(keep.sum()))
        if verbose:
            print(f"{tag}: {keep.sum()} face pts, rms spread "
                  f"{(rms*1000).round(2)} mm, half-extents "
                  f"{out[tag]['half0']*1000:.1f} x {out[tag]['half1']*1000:.1f} mm")

    # Orient each normal at the other finger; keep each box frame right-handed.
    cf_w = _to_world(out["fixed"]["centre"][None, :], d, gf)[0]
    cm_w = _to_world(out["moving"]["centre"][None, :], d, gm)[0]
    sep = cm_w - cf_w
    sep = sep / max(float(np.linalg.norm(sep)), 1e-9)
    for tag, gid, want in (("fixed", gf, sep), ("moving", gm, -sep)):
        n_w = d.geom_xmat[gid].reshape(3, 3) @ out[tag]["n"]
        if float(n_w @ want) < 0:
            out[tag]["n"] = -out[tag]["n"]
            out[tag]["e1"] = -out[tag]["e1"]

    nf_w = d.geom_xmat[gf].reshape(3, 3) @ out["fixed"]["n"]
    nm_w = d.geom_xmat[gm].reshape(3, 3) @ out["moving"]["n"]
    out["_check"] = dict(normals_dot=float(nf_w @ nm_w),
                         centre_sep_m=float(np.linalg.norm(cm_w - cf_w)),
                         fixed_normal_dot_sep=float(nf_w @ sep),
                         axial_band_mm=(float(zlo * 1000), float(zhi * 1000)))

    # Pads attach to BODIES, not geoms; convert from the geom frame.
    for tag, body_name in (("fixed", FIXED_FINGER[0]), ("moving", MOVING_FINGER[0])):
        o = out[tag]
        gid = o["gid"]
        # geom -> body transform straight from the model
        gq = np.zeros(9)
        mujoco.mju_quat2Mat(gq, m.geom_quat[gid])
        R_gb = gq.reshape(3, 3)
        centre_body = m.geom_pos[gid] + R_gb @ o["centre"]
        n_body = R_gb @ o["n"]
        e0_body = R_gb @ o["e0"]
        e1_body = R_gb @ o["e1"]
        R = np.vstack([e0_body, e1_body, n_body]).T
        if np.linalg.det(R) < 0:
            R[:, 1] *= -1
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.T.ravel())
        o["pos"] = centre_body + n_body * (PAD_PROUD + PAD_HALF_THICK)
        o["quat"] = quat
        o["size"] = np.array([o["half0"], o["half1"], PAD_HALF_THICK])
        o["body"] = body_name
    return out


def pad_geom_xml(pads: dict) -> dict:
    """{body_name: '<geom .../>'} ready to inject into the scene XML."""
    xml = {}
    for tag in ("fixed", "moving"):
        o = pads[tag]
        p = " ".join(f"{x:.6g}" for x in o["pos"])
        s = " ".join(f"{x:.6g}" for x in o["size"])
        q = " ".join(f"{x:.6g}" for x in o["quat"])
        xml[o["body"]] = (f'<geom name="jaw_pad_{tag}" type="box" pos="{p}" '
                          f'size="{s}" quat="{q}" class="collision"/>')
    return xml
