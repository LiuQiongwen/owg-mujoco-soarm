#!/usr/bin/env python3
"""
Minimal-viable Piper sim migration, step 1: verify IK reachability on the
new workspace scene (tango_robot/piper_assets/scene_workspace.xml) -- arm +
table + tray + one test object, no grasp-execution logic yet (that's a
separate, much larger port of env_soarm.py's physics_weld_after_bilateral
machinery, deferred until this basic kinematics check passes).

Same damped-least-squares position-only IK pattern as
tango_robot/headless_ik.py (HeadlessIKSolver), just retargeted at the Piper
model's 6 joints and the newly-added `eef_site` (see piper_assets/robot.xml)
instead of SO-ARM101's 5 joints + jaw-geom-midpoint target.

Usage:
  conda run -n tango python3 tango_robot/piper_ik_check.py
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

ASSETS_DIR = Path(__file__).resolve().parent / "piper_assets"
JOINTS = [f"joint{i}" for i in range(1, 7)]
EEF_SITE = "eef_site"
IK_DAMPING = 1e-3
IK_TOL = 5e-3


class PiperIKCheck:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ASSETS_DIR / "scene_workspace.xml"))
        self.data = mujoco.MjData(self.model)
        self.jnt_ids = [self.model.joint(n).id for n in JOINTS]
        self.qpos_adr = [self.model.joint(n).qposadr[0] for n in JOINTS]
        self.dof_adr = [self.model.joint(n).dofadr[0] for n in JOINTS]
        self.eef_site_id = self.model.site(EEF_SITE).id
        mujoco.mj_forward(self.model, self.data)

    def get_eef_pos(self):
        return self.data.site_xpos[self.eef_site_id].copy()

    def _ik_step(self, target_pos):
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.eef_site_id)
        J = jacp[:, self.dof_adr]
        err = target_pos - self.get_eef_pos()
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(3), err)
        for i, adr in enumerate(self.qpos_adr):
            lo, hi = self.model.jnt_range[self.jnt_ids[i]]
            self.data.qpos[adr] = np.clip(self.data.qpos[adr] + dq[i], lo, hi)
        mujoco.mj_forward(self.model, self.data)
        return np.linalg.norm(err) < IK_TOL

    def solve_ik(self, target_pos, iters=1500):
        for adr in self.qpos_adr:
            self.data.qpos[adr] = 0.0
        mujoco.mj_forward(self.model, self.data)
        converged = False
        for _ in range(iters):
            if self._ik_step(target_pos):
                converged = True
                break
        pe = float(np.linalg.norm(self.get_eef_pos() - target_pos))
        return converged, pe


def main():
    checker = PiperIKCheck()

    # Targets spanning the intended workspace: near the test object, the
    # tray, and a couple of corners of the table region defined in
    # scene_workspace.xml (table centred at x=0.35, half-extents 0.35x0.35).
    targets = {
        "test_object":      np.array([0.35, 0.05, 0.06]),
        "tray":              np.array([0.30, -0.25, 0.03]),
        "table_near_corner": np.array([0.20, 0.20, 0.05]),
        "table_far_corner":  np.array([0.55, -0.20, 0.05]),
        "table_centre":      np.array([0.35, 0.0, 0.05]),
    }

    print(f"{'target':<20} {'requested_xyz':<28} {'converged':<10} {'pos_err_cm'}")
    for name, pos in targets.items():
        converged, pe = checker.solve_ik(pos)
        print(f"{name:<20} {str(pos.round(3)):<28} {str(converged):<10} {pe*100:.2f}")


if __name__ == "__main__":
    main()
