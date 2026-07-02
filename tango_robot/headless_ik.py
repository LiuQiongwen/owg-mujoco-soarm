# -*- coding: utf-8 -*-
"""Headless IK solver for SO-ARM101 — W1 real-hardware bring-up.

Extracts EnvironmentSoArm._solve_ik_jaw_pos_only (tango_robot/env_soarm.py)
into a standalone solver that can run on any machine with no GL/EGL context,
no viewer, and no MUJOCO_GL setting.

Design constraint (hard requirement): this module must NOT call
EnvironmentSoArm._rebuild_model() and must NOT instantiate EnvironmentSoArm.
_rebuild_model() unconditionally constructs mujoco.Renderer(...)
(env_soarm.py:554), which allocates a GL/EGL context at construction time —
regardless of the `vis` flag. That makes EnvironmentSoArm unsuitable as a
base for a machine that has no rendering backend at all. Instead this module
calls _build_scene_xml([]) directly (same MJCF-building function, empty
object pool) and builds mujoco.MjModel/MjData itself, so the only mujoco
calls made are mj_jacSite / mj_forward — the same two calls the original
_ik_step() makes.

Fidelity: every constant, threshold, and code path below is copied verbatim
from env_soarm.py (not reimplemented or "improved"), including
_simplify_jaw_collision() — which has no effect on IK correctness (it only
changes collision geom type/size for contact physics, not the geom's local
offset used by mj_forward to compute geom_xpos) but is included anyway so
Step 3's cross-check against the original class is an exact reproduction,
not an approximation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import mujoco

from tango_robot.env_soarm import (
    _build_scene_xml,
    ARM_JOINTS,
    EEF_SITE,
    GRIP_JOINT,
    GRIP_OPEN,
    HOME_QPOS,
    IK_DAMPING,
    IK_TOL,
    ROBOT_BASE_POS,
)

# robot_base_mount is placed at ROBOT_BASE_POS="0 0 0.785" with no rotation
# (env_soarm.py:415-417) — so the MuJoCo world frame and the robot-base frame
# share the same orientation and X/Y origin; they differ only by this fixed
# Z translation. Every IK target/result in this module is in **sim-world
# frame** (same frame as EnvironmentSoArm's site_xpos/geom_xpos). Convert to
# robot-base frame (what robots/base.py's RobotBackend contract expects) with
# world_to_base_frame() / base_to_world_frame() below — never subtract this
# offset by hand at a call site.
_BASE_Z_OFFSET = float(ROBOT_BASE_POS.split()[2])  # 0.785 m


def world_to_base_frame(pos_world: np.ndarray) -> np.ndarray:
    """Sim-world XYZ -> robot-base-frame XYZ (X/Y unchanged, Z -= 0.785)."""
    pos_world = np.asarray(pos_world, dtype=np.float64)
    return pos_world - np.array([0.0, 0.0, _BASE_Z_OFFSET])


def base_to_world_frame(pos_base: np.ndarray) -> np.ndarray:
    """Robot-base-frame XYZ -> sim-world XYZ (X/Y unchanged, Z += 0.785)."""
    pos_base = np.asarray(pos_base, dtype=np.float64)
    return pos_base + np.array([0.0, 0.0, _BASE_Z_OFFSET])


class HeadlessIKSolver:
    """Standalone jaw-midpoint position IK — no viewer, no renderer, no GL.

    Reproduces EnvironmentSoArm._solve_ik_jaw_pos_only exactly: same MJCF
    (via _build_scene_xml([])), same DLS damping/tolerance constants, same
    outer/inner iteration structure, same jaw-tip-geom target point.
    """

    def __init__(self) -> None:
        xml = _build_scene_xml([])   # empty object pool — arm + floor/table/tray only
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self._arm_jnt_ids = [self.model.joint(n).id for n in ARM_JOINTS]
        self._arm_qpos_adr = [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self._eef_site_id = self.model.site(EEF_SITE).id

        # The moving jaw body's kinematic position depends on the gripper
        # joint's qpos, not just the 5 arm joints — _get_jaw_geom_midpoint()
        # (the actual IK target point) is wrong unless this is initialized.
        # Matches EnvironmentSoArm._cache_ids (env_soarm.py:588-590) +
        # _rebuild_model's GRIP_OPEN seed (env_soarm.py:562).
        self._grip_qpos_adr = self.model.joint(GRIP_JOINT).qposadr[0]

        try:
            self._jaw_body_id = self.model.body("gripper").id
            self._jaw_mv_body_id = self.model.body("moving_jaw_so101_v1").id
            self._jaw_fixed_geom_id = self._find_collision_geom(
                "gripper", "wrist_roll_follower_so101_v1")
            self._jaw_mv_geom_id = self._find_collision_geom(
                "moving_jaw_so101_v1", "moving_jaw_so101_v1")
        except Exception:
            self._jaw_body_id = -1
            self._jaw_mv_body_id = -1
            self._jaw_fixed_geom_id = -1
            self._jaw_mv_geom_id = -1

        # Strict-fidelity requirement: replicate env_soarm.py's collision-geom
        # simplification even though it has no effect on IK (mj_jacSite/
        # mj_forward use kinematic geom placement, not geom_type/geom_size).
        self._simplify_jaw_collision()

        for adr, q in zip(self._arm_qpos_adr, HOME_QPOS):
            self.data.qpos[adr] = q
        self.data.qpos[self._grip_qpos_adr] = GRIP_OPEN
        mujoco.mj_forward(self.model, self.data)

    # ── copied verbatim from EnvironmentSoArm (env_soarm.py) ───────────────────

    def _find_collision_geom(self, body_name: str, mesh_name: str) -> int:
        """Return the ID of the collision geom on body_name whose mesh is mesh_name."""
        bid = self.model.body(body_name).id
        for gi in range(self.model.ngeom):
            if self.model.geom_bodyid[gi] != bid:
                continue
            if self.model.geom_contype[gi] == 0:
                continue   # visual-only geom
            dataid = self.model.geom_dataid[gi]
            if dataid >= 0 and self.model.mesh(dataid).name == mesh_name:
                return gi
        raise ValueError(f"collision geom '{mesh_name}' not found on body '{body_name}'")

    def _simplify_jaw_collision(self):
        """Replace large jaw mesh collision geoms with small spheres.

        Verbatim copy of EnvironmentSoArm._simplify_jaw_collision
        (env_soarm.py:514-546). Kept for strict fidelity with the original
        model even though it doesn't affect IK.
        """
        if self._jaw_body_id < 0:
            return
        try:
            motor_gid = self._find_collision_geom("gripper", "sts3215_03a_v1")
            self.model.geom_contype[motor_gid] = 0
            self.model.geom_conaffinity[motor_gid] = 0
        except Exception:
            pass

        _SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        _R = 0.006  # 6 mm radius — geom centres are ~1-2 mm outside object faces
        for gid in (self._jaw_fixed_geom_id, self._jaw_mv_geom_id):
            if gid >= 0:
                self.model.geom_type[gid] = _SPHERE
                self.model.geom_size[gid, 0] = _R
                self.model.geom_size[gid, 1] = 0.0
                self.model.geom_size[gid, 2] = 0.0

    def _get_eef_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._eef_site_id].copy()

    def _get_jaw_midpoint(self) -> np.ndarray:
        """World-frame midpoint between the fixed and moving jaw body centres."""
        if self._jaw_body_id < 0:
            return self._get_eef_pos()
        return 0.5 * (self.data.xpos[self._jaw_body_id] +
                      self.data.xpos[self._jaw_mv_body_id])

    def _get_jaw_geom_midpoint(self) -> np.ndarray:
        """World-frame midpoint of the actual jaw tip GEOM centres."""
        if self._jaw_fixed_geom_id < 0:
            return self._get_jaw_midpoint()
        return 0.5 * (self.data.geom_xpos[self._jaw_fixed_geom_id] +
                      self.data.geom_xpos[self._jaw_mv_geom_id])

    def _ik_step(self, target_pos: np.ndarray) -> bool:
        """Damped least-squares IK - position only (xyz). Verbatim copy of
        EnvironmentSoArm._ik_step (env_soarm.py:758-771)."""
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._eef_site_id)
        cols = [self.model.joint(n).dofadr[0] for n in ARM_JOINTS]
        J = jacp[:, cols]
        err = target_pos - self._get_eef_pos()
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(3), err)
        for i, adr in enumerate(self._arm_qpos_adr):
            lo = self.model.jnt_range[self._arm_jnt_ids[i], 0]
            hi = self.model.jnt_range[self._arm_jnt_ids[i], 1]
            self.data.qpos[adr] = np.clip(self.data.qpos[adr] + dq[i], lo, hi)
        mujoco.mj_forward(self.model, self.data)
        return np.linalg.norm(err) < IK_TOL

    def solve_ik_jaw_pos_only(
        self,
        target_jaw_mid: np.ndarray,
        iters: int = 800,
        pos_tol: float = 5e-3,
        n_outer: int = 8,
        reset_to_home: bool = True,
        silent: bool = False,
    ) -> Tuple[bool, float, float]:
        """Verbatim copy of EnvironmentSoArm._solve_ik_jaw_pos_only
        (env_soarm.py:952-993).

        target_jaw_mid : sim-world-frame XYZ (metres) of the jaw GEOM
                          midpoint. Use base_to_world_frame() first if you
                          have a robot-base-frame target.

        Returns (converged, geom_mid_pos_err, 0.0) — matches the original's
        return signature exactly (third element is always 0.0; kept only for
        API parity with the orientation-aware IK variants).
        """
        if reset_to_home:
            for adr, q in zip(self._arm_qpos_adr, HOME_QPOS):
                self.data.qpos[adr] = q
        mujoco.mj_forward(self.model, self.data)

        iters_per = max(50, iters // n_outer)
        for _ in range(n_outer):
            offset = self._get_eef_pos() - self._get_jaw_geom_midpoint()
            adjusted = target_jaw_mid + offset
            for _ in range(iters_per):
                if self._ik_step(adjusted):
                    break
        geom_pos = self._get_jaw_geom_midpoint()
        pe = float(np.linalg.norm(geom_pos - target_jaw_mid))
        if not silent:
            print(f"  [headless_ik] target={target_jaw_mid.round(4)} "
                  f"solved_geom_mid={geom_pos.round(4)} pe={pe*100:.2f}cm")
        return pe < pos_tol, pe, 0.0

    def get_arm_qpos(self) -> np.ndarray:
        """Read back the 5 solved arm joint angles (radians), ARM_JOINTS order.

        The IK writes into self.data.qpos as a side effect (matching the
        original's behavior) rather than returning angles directly — call
        this after solve_ik_jaw_pos_only() to retrieve them, exactly as
        env_soarm.py:1691 does with q_grasp.
        """
        return np.array([self.data.qpos[adr] for adr in self._arm_qpos_adr],
                        dtype=np.float64)
