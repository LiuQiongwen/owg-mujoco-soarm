"""Read-only metrology for the SO-101 jaw.

Measures what the gripper is *actually* doing, without changing any of it.
Nothing in this module writes to MjModel or MjData; it is safe to call from
inside a running grasp.

Why this exists
---------------
`EnvironmentSoArm.move_gripper(opening_m)` asserts the linear map

    angle = GRIP_CLOSED + (opening_m / 0.10) * (GRIP_OPEN - GRIP_CLOSED)

and `MujocoBackend.get_gripper_opening()` inverts it, but the map was never
checked against geometry.  Measured (`scripts/audit_jaw_opening.py`), over the
window `move_gripper` can command:

    commanded 0 mm   -> true fingertip gap 19.4 mm, proxy-sphere gap 23.9 mm
    commanded 100 mm -> true fingertip gap 80.1 mm, proxy-sphere gap 39.0 mm

Two distinct distortions, which this module reports separately so they can be
attributed separately:

  * the *command* distortion -- the jaw never closes below ~19 mm because
    GRIP_CLOSED (0.05 rad) sits 13.6 deg above the joint's real lower limit;
  * the *contact* distortion -- `_simplify_jaw_collision` replaces each finger
    with one 6 mm sphere at the finger mesh's frame origin (5-76 mm from that
    mesh's own vertices), so the geometry MuJoCo collides against tracks the
    command over a 15 mm range while the real fingertips track it over 60 mm.

Terms
-----
true_opening    gap between the distal quarter of the two finger collision
                MESHES -- what a physically faithful jaw would close on.
                A pure function of the gripper hinge angle (both fingers are
                rigid), so it is tabulated once and interpolated.
proxy_gap       surface separation of the two 6 mm spheres actually simulated.
claimed         what move_gripper()'s linear map says the opening is.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import mujoco
import numpy as np

# Mirrors env_soarm; duplicated rather than imported to keep this module
# importable standalone (env_soarm imports heavy scene-building machinery).
GRIP_OPEN_RAD = 1.0
GRIP_CLOSED_RAD = 0.05
CLAIMED_TRAVEL_M = 0.10
PROXY_SPHERE_R = 0.006

FIXED_FINGER = ("gripper", "wrist_roll_follower_so101_v1")
MOVING_FINGER = ("moving_jaw_so101_v1", "moving_jaw_so101_v1")

TIP_QUANTILE = 0.75    # distal quarter of each finger mesh counts as "tip"
TIP_MAX_PTS = 1500     # face-sampled points per finger before tip selection
LUT_N = 61             # angle samples across the full joint range
OBJ_SURF_PTS = 4000    # face-sampled points per object collision geom set


def claimed_opening_m(qpos_rad: float) -> float:
    """What move_gripper()'s linear map claims this hinge angle means, in metres."""
    return ((qpos_rad - GRIP_CLOSED_RAD) / (GRIP_OPEN_RAD - GRIP_CLOSED_RAD)
            * CLAIMED_TRAVEL_M)


def _mesh_geom_verts_local(model, gid: int) -> Optional[np.ndarray]:
    """Vertices of the mesh a geom was built from, in the geom's own frame.

    Still recoverable after `_simplify_jaw_collision` swaps geom_type to
    sphere: that call rewrites geom_type/geom_size but leaves geom_dataid
    pointing at the original mesh.
    """
    mid = int(model.geom_dataid[gid])
    if mid < 0:
        return None
    adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    return model.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)


def _mesh_geom_surface_local(model, gid: int, n_pts: int,
                             seed: int = 0) -> Optional[np.ndarray]:
    """Points sampled over a mesh geom's FACES, in the geom's own frame.

    Vertices alone are the wrong sample for a distance query: a CoACD part or a
    machined finger has large flat triangles, so the nearest *vertex* to a
    contact point can be tens of mm away while the *surface* is touching.
    Measuring vertex-to-vertex distance therefore reports separation where
    MuJoCo (which collides surfaces) reports contact.  Area-weighted face
    sampling keeps the error bounded by the sample spacing instead.
    """
    mid = int(model.geom_dataid[gid])
    if mid < 0:
        return None
    vadr, vnum = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fadr, fnum = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    V = model.mesh_vert[vadr:vadr + vnum].reshape(-1, 3).astype(np.float64)
    if fnum <= 0:
        return V
    F = model.mesh_face[fadr:fadr + fnum].reshape(-1, 3)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(area.sum())
    if total <= 0:
        return V
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), size=n_pts, p=area / total)
    u = rng.random(n_pts)
    v = rng.random(n_pts)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    pts = a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])
    # Keep the vertices too: they carry the extreme points sampling can miss.
    return np.vstack([pts, V]) if len(V) <= n_pts else np.vstack([pts, _subsample(V, n_pts)])


def _to_world(V: np.ndarray, data, gid: int) -> np.ndarray:
    return V @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]


def _subsample(V: np.ndarray, n: int) -> np.ndarray:
    if len(V) <= n:
        return V
    idx = np.linspace(0, len(V) - 1, n).astype(int)
    return V[idx]


class JawMetrology:
    """Per-model jaw measurements. Build once per MjModel; then query per step.

    `available` is False when the jaw geoms cannot be resolved (e.g. a scene
    without the SO-101); every query then returns an empty dict so callers
    need no special-casing.
    """

    def __init__(self, model):
        self.model = model
        self.available = False
        self._gf = self._gm = -1
        self._tip_f: Optional[np.ndarray] = None
        self._tip_m: Optional[np.ndarray] = None
        self._lut_q: Optional[np.ndarray] = None
        self._lut_gap: Optional[np.ndarray] = None
        try:
            self._build()
            self.available = True
        except Exception:
            self.available = False

    # ── construction ─────────────────────────────────────────────────────────

    def _find_geom(self, body_name: str, mesh_name: str) -> int:
        """The COLLISION geom for `mesh_name` on `body_name`.

        Each finger carries two geoms built from the same mesh -- a visual one
        (contype=0) and a collision one -- at identical poses.  Matching on mesh
        name alone returns the visual geom, which `_simplify_jaw_collision`
        never replaced: it is still the full 105 mm finger hull.  Positions
        coincide so distances between geom origins come out the same either way,
        but anything that queries the geom's SHAPE (mj_geomDistance, geom_type)
        then silently measures a mesh the solver is not colliding, reporting
        4 cm penetrations where the real contact is 1 mm.  Prefer contype != 0.
        """
        bid = self.model.body(body_name).id
        fallback = -1
        for gi in range(self.model.ngeom):
            if self.model.geom_bodyid[gi] != bid:
                continue
            did = int(self.model.geom_dataid[gi])
            if did < 0 or self.model.mesh(did).name != mesh_name:
                continue
            if self.model.geom_contype[gi] != 0:
                return gi
            if fallback < 0:
                fallback = gi
        if fallback >= 0:
            return fallback
        raise ValueError(f"geom for mesh '{mesh_name}' not found on '{body_name}'")

    def _build(self):
        self._gf = self._find_geom(*FIXED_FINGER)
        self._gm = self._find_geom(*MOVING_FINGER)
        self._jnt_qadr = int(self.model.joint("gripper").qposadr[0])
        self._jnt_range = tuple(float(v) for v in self.model.joint("gripper").range)
        Vf = _mesh_geom_surface_local(self.model, self._gf, 4 * TIP_MAX_PTS)
        Vm = _mesh_geom_surface_local(self.model, self._gm, 4 * TIP_MAX_PTS)
        if Vf is None or Vm is None:
            raise ValueError("finger geoms carry no mesh data")

        # Tip selection and the opening LUT both need a scratch MjData; this
        # never touches the caller's data.
        scratch = mujoco.MjData(self.model)
        hinge_bid = self.model.body(MOVING_FINGER[0]).id

        scratch.qpos[:] = 0.0
        scratch.qpos[self._jnt_qadr] = 0.5 * sum(self._jnt_range)
        mujoco.mj_forward(self.model, scratch)
        hinge = scratch.xpos[hinge_bid].copy()
        tips = []
        for V, gid in ((Vf, self._gf), (Vm, self._gm)):
            r = np.linalg.norm(_to_world(V, scratch, gid) - hinge, axis=1)
            tips.append(_subsample(V[r >= np.quantile(r, TIP_QUANTILE)], TIP_MAX_PTS))
        self._tip_f, self._tip_m = tips

        lo, hi = self._jnt_range
        qs = np.linspace(lo, hi, LUT_N)
        gaps = np.empty(LUT_N)
        for i, q in enumerate(qs):
            scratch.qpos[:] = 0.0
            scratch.qpos[self._jnt_qadr] = q
            mujoco.mj_forward(self.model, scratch)
            A = _to_world(self._tip_f, scratch, self._gf)
            B = _to_world(self._tip_m, scratch, self._gm)
            gaps[i] = _min_cross_dist(A, B)
        self._lut_q, self._lut_gap = qs, gaps

    # ── queries ──────────────────────────────────────────────────────────────

    def true_opening_m(self, qpos_rad: float) -> float:
        """Fingertip-pad separation (m) for a gripper hinge angle."""
        return float(np.interp(qpos_rad, self._lut_q, self._lut_gap))

    def proxy_gap_m(self, data) -> float:
        """Surface separation (m) of the two simulated 6 mm proxy spheres."""
        c = float(np.linalg.norm(data.geom_xpos[self._gf] - data.geom_xpos[self._gm]))
        return c - 2 * PROXY_SPHERE_R

    def tip_points(self, data) -> Dict[str, np.ndarray]:
        return {"fixed": _to_world(self._tip_f, data, self._gf),
                "moving": _to_world(self._tip_m, data, self._gm)}

    def closing_axis(self, data) -> np.ndarray:
        """Unit vector from the fixed to the moving finger tip centroid."""
        pts = self.tip_points(data)
        v = pts["moving"].mean(0) - pts["fixed"].mean(0)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])

    def proxy_to_object_dist_m(self, model, data,
                               obj_geom_ids: Sequence[int]) -> Dict[str, float]:
        """Exact surface distance from each proxy sphere to the object.

        Uses mj_geomDistance on the geoms MuJoCo actually collides, so this is
        ground truth for "was the collider touching" -- no sampling involved.
        Negative means interpenetration.
        """
        out = {}
        for tag, gid in (("fixed", self._gf), ("moving", self._gm)):
            best = float("inf")
            for ogid in obj_geom_ids:
                d = mujoco.mj_geomDistance(model, data, gid, int(ogid), 1.0,
                                           np.zeros(6))
                best = min(best, float(d))
            out[tag] = best
        return out

    def tip_to_object_dist_m(self, data, obj_verts: np.ndarray) -> Dict[str, float]:
        """Min distance (m) from each real fingertip pad to the object surface.

        This is the counterfactual the proxy spheres cannot answer: would a
        faithful pad have been touching here?
        """
        pts = self.tip_points(data)
        return {"fixed": _min_cross_dist(pts["fixed"], obj_verts),
                "moving": _min_cross_dist(pts["moving"], obj_verts)}

    def object_local_thickness_m(self, data, obj_verts: np.ndarray,
                                 slab_half_m: float = 0.015) -> Optional[float]:
        """Object extent along the closing axis, in a slab around the point the
        jaw is aimed at.

        The whole-object AABB is the wrong number for a hammer or scissors: the
        head is thick and the handle is not, and the jaw only ever meets one of
        them.  The slab is centred on the object-surface point NEAREST the jaw
        midpoint rather than on the midpoint itself, so the answer stays defined
        ("how thick is the object where the jaw was aimed") even on trials where
        the approach missed by centimetres -- those are exactly the trials where
        a None would hide the distinction between "too thin to pinch" and
        "never got close".  `slab_half_m` bounds the slab in the two directions
        orthogonal to closing.
        """
        if not len(obj_verts):
            return None
        pts = self.tip_points(data)
        mid = 0.5 * (pts["fixed"].mean(0) + pts["moving"].mean(0))
        centre = obj_verts[int(np.argmin(np.linalg.norm(obj_verts - mid, axis=1)))]
        axis = self.closing_axis(data)
        rel = obj_verts - centre
        along = rel @ axis
        perp = np.linalg.norm(rel - np.outer(along, axis), axis=1)
        for half in (slab_half_m, 2 * slab_half_m, 4 * slab_half_m):
            inslab = perp <= half
            if inslab.sum() >= 3:
                a = along[inslab]
                return float(a.max() - a.min())
        return None

    def jaw_to_obj_surface_m(self, data, obj_verts: np.ndarray) -> Optional[float]:
        """Distance from the jaw midpoint to the nearest object surface point.

        Separates "the jaw closed but the object was too thin" from "the jaw
        never arrived", which the opening numbers alone cannot distinguish.
        """
        if not len(obj_verts):
            return None
        pts = self.tip_points(data)
        mid = 0.5 * (pts["fixed"].mean(0) + pts["moving"].mean(0))
        return float(np.linalg.norm(obj_verts - mid, axis=1).min())

    def snapshot(self, data, obj_verts: Optional[np.ndarray] = None,
                 ctrl_id: Optional[int] = None,
                 model=None,
                 obj_geom_ids: Optional[Sequence[int]] = None) -> dict:
        """All jaw metrology for the current state. Never mutates anything.

        Pass `model` and `obj_geom_ids` to also get the EXACT signed proxy-to-
        object distance (verified equal to the solver's own contact.dist);
        `obj_verts` drives the sampled real-pad counterfactual, which is a
        non-negative proximity measure and cannot express penetration.
        """
        if not self.available:
            return {}
        q = float(data.qpos[self._jnt_qadr])
        out = {
            "grip_qpos_rad": round(q, 5),
            "true_opening_m": round(self.true_opening_m(q), 5),
            "proxy_gap_m": round(self.proxy_gap_m(data), 5),
            "claimed_opening_m": round(claimed_opening_m(q), 5),
        }
        if ctrl_id is not None:
            out["grip_ctrl_rad"] = round(float(data.ctrl[ctrl_id]), 5)
        if model is not None and obj_geom_ids:
            pd = self.proxy_to_object_dist_m(model, data, obj_geom_ids)
            out["proxy_obj_dist_fixed_m"] = round(pd["fixed"], 5)
            out["proxy_obj_dist_moving_m"] = round(pd["moving"], 5)
        if obj_verts is not None and len(obj_verts):
            d = self.tip_to_object_dist_m(data, obj_verts)
            out["true_tip_obj_dist_fixed_m"] = round(d["fixed"], 5)
            out["true_tip_obj_dist_moving_m"] = round(d["moving"], 5)
            th = self.object_local_thickness_m(data, obj_verts)
            out["object_local_thickness_m"] = round(th, 5) if th is not None else None
            js = self.jaw_to_obj_surface_m(data, obj_verts)
            out["jaw_mid_to_obj_surface_m"] = round(js, 5) if js is not None else None
        return out


def _min_cross_dist(A: np.ndarray, B: np.ndarray) -> float:
    """Min distance between two point sets. cKDTree when available, else chunked."""
    try:
        from scipy.spatial import cKDTree
        return float(cKDTree(B).query(A, k=1)[0].min())
    except Exception:
        best = np.inf
        for i in range(0, len(A), 256):
            d = np.linalg.norm(A[i:i + 256, None, :] - B[None, :, :], axis=2).min()
            best = min(best, float(d))
        return best


def object_collision_verts(model, data, geom_ids: Sequence[int],
                           max_pts: int = OBJ_SURF_PTS) -> np.ndarray:
    """World-frame SURFACE points of an object's collision geoms.

    Face-sampled, not vertex-listed -- see `_mesh_geom_surface_local` for why
    that distinction decides whether a distance query agrees with MuJoCo.
    Handles CoACD multi-part objects (one geom per convex part) and falls back
    to the geom's AABB corners for primitive (non-mesh) geoms.
    """
    chunks: List[np.ndarray] = []
    per_geom = max(64, max_pts // max(1, len(geom_ids)))
    for gid in geom_ids:
        V = _mesh_geom_surface_local(model, gid, per_geom)
        if V is None:
            s = model.geom_size[gid]
            V = np.array([[sx, sy, sz]
                          for sx in (-s[0], s[0])
                          for sy in (-s[1], s[1])
                          for sz in (-s[2], s[2])], dtype=np.float64)
        chunks.append(_to_world(V, data, gid))
    if not chunks:
        return np.empty((0, 3))
    return np.vstack(chunks)
