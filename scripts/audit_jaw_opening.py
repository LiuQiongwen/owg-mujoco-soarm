"""Audit the SO-101 jaw: command -> joint angle -> actual fingertip opening.

Nothing in the codebase currently measures how far the jaws are actually apart.
`EnvironmentSoArm.move_gripper(opening_m)` asserts a linear map

    angle = GRIP_CLOSED + (opening_m / 0.10) * (GRIP_OPEN - GRIP_CLOSED)

and `MujocoBackend.get_gripper_opening()` inverts it, so "metres" propagate
through trajectory recording and replay without ever being checked against
geometry.  This script measures the geometry directly.

Three quantities per joint angle:

  true tip gap    min distance between the distal quarter of the two finger
                  collision meshes (exact vertices, not convex hulls) -- what a
                  real jaw would close on
  proxy gap       surface separation of the two 6 mm spheres that
                  EnvironmentSoArm._simplify_jaw_collision() substitutes at
                  runtime -- what MuJoCo actually collides against
  claimed         what move_gripper()'s linear map says the opening is

Run:  conda run -n tango python scripts/audit_jaw_opening.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.spatial import cKDTree

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tango_robot", "assets", "so101", "so101.xml")

GRIP_OPEN, GRIP_CLOSED = 1.0, 0.05   # env_soarm.py, radians despite the docstrings
TRAVEL_M = 0.10                      # move_gripper's assumed full travel
SPHERE_R = 0.006                     # _simplify_jaw_collision
TIP_QUANTILE = 0.75                  # distal quarter of each finger mesh

FIXED_FINGER = ("gripper", "wrist_roll_follower_so101_v1")
MOVING_FINGER = ("moving_jaw_so101_v1", "moving_jaw_so101_v1")


def find_collision_geom(m, body_name, mesh_name):
    bid = m.body(body_name).id
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] != bid or m.geom_contype[gi] == 0:
            continue
        did = m.geom_dataid[gi]
        if did >= 0 and m.mesh(did).name == mesh_name:
            return gi
    raise ValueError(f"collision geom '{mesh_name}' not found on body '{body_name}'")


def local_verts(m, gid):
    mid = m.geom_dataid[gid]
    adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    return m.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)


def to_world(V, d, gid):
    return V @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]


def claimed_m(q):
    """Inverse of move_gripper: what the code thinks this angle means, in metres."""
    return (q - GRIP_CLOSED) / (GRIP_OPEN - GRIP_CLOSED) * TRAVEL_M


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    gf = find_collision_geom(m, *FIXED_FINGER)
    gm = find_collision_geom(m, *MOVING_FINGER)
    jadr = m.joint("gripper").qposadr[0]
    lo, hi = m.actuator("gripper").ctrlrange
    Vf, Vm = local_verts(m, gf), local_verts(m, gm)

    # Convex-hull separation is what MuJoCo would use for mesh-mesh contact.
    # Measured once: it is constant in the joint angle, i.e. the two hulls
    # permanently interpenetrate at the pivot -- the reason the sphere proxy exists.
    def hull_gap(q):
        d.qpos[:] = 0
        d.qpos[jadr] = q
        mujoco.mj_forward(m, d)
        return mujoco.mj_geomDistance(m, d, gf, gm, 1.0, np.zeros(6))

    # Tip region is selected once, at a mid-open pose, by distance from the hinge.
    d.qpos[:] = 0
    d.qpos[jadr] = 0.5
    mujoco.mj_forward(m, d)
    hinge = d.xpos[m.body("moving_jaw_so101_v1").id].copy()
    tip = {}
    for tag, V, gid in (("f", Vf, gf), ("m", Vm, gm)):
        r = np.linalg.norm(to_world(V, d, gid) - hinge, axis=1)
        tip[tag] = r >= np.quantile(r, TIP_QUANTILE)

    def measure(q):
        d.qpos[:] = 0
        d.qpos[jadr] = q
        mujoco.mj_forward(m, d)
        A = to_world(Vf, d, gf)[tip["f"]]
        B = to_world(Vm, d, gm)[tip["m"]]
        true_gap = float(cKDTree(B).query(A, k=1)[0].min())
        proxy = float(np.linalg.norm(d.geom_xpos[gf] - d.geom_xpos[gm])) - 2 * SPHERE_R
        return true_gap, proxy

    print(f"model: {XML}")
    print(f"gripper joint range (MJCF): {lo:.5f} .. {hi:.5f} rad "
          f"({np.degrees(lo):.1f} .. {np.degrees(hi):.1f} deg)")
    print(f"window move_gripper() uses: {GRIP_CLOSED:.5f} .. {GRIP_OPEN:.5f} rad "
          f"({np.degrees(GRIP_CLOSED):.1f} .. {np.degrees(GRIP_OPEN):.1f} deg)")
    print(f"finger mesh convex hulls overlap by {-hull_gap(0.5) * 1000:.1f} mm "
          f"at every joint angle (hull gap at 0.2/0.5/1.2 rad: "
          f"{hull_gap(0.2)*1000:.1f} / {hull_gap(0.5)*1000:.1f} / "
          f"{hull_gap(1.2)*1000:.1f} mm)")
    print()
    print(f"{'qpos rad':>9} {'deg':>7} | {'true tip gap':>13} | {'proxy gap':>11} "
          f"| {'claimed':>9}")
    print("-" * 62)
    for q in np.linspace(lo, hi, 17):
        t, p = measure(q)
        mark = "  <- commanded" if GRIP_CLOSED <= q <= GRIP_OPEN else ""
        print(f"{q:9.4f} {np.degrees(q):7.1f} | {t*1000:11.1f}mm | {p*1000:9.1f}mm "
              f"| {claimed_m(q)*1000:7.1f}mm{mark}")

    print()
    for label, q in (("fully closed (MJCF limit)", lo),
                     ("GRIP_CLOSED  = 0.05 rad", GRIP_CLOSED),
                     ("GRIP_OPEN    = 1.00 rad", GRIP_OPEN),
                     ("fully open   (MJCF limit)", hi)):
        t, p = measure(q)
        print(f"  {label:26s}  true tip gap {t*1000:6.1f} mm | "
              f"proxy {p*1000:5.1f} mm | claimed {claimed_m(q)*1000:6.1f} mm")


if __name__ == "__main__":
    main()
