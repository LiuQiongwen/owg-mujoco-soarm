# -*- coding: utf-8 -*-
"""
MuJoCo simulation environment for 6-DoF grasp validation.
Grasp format preserved: {position, rpy, width, score}
"""
import os
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "..",
                       "owg_robot", "assets", "so101", "assets")
_SO101  = os.path.join(os.path.dirname(__file__), "..", "..",
                       "owg_robot", "assets", "so101", "so101.xml")

TABLE_Z   = 0.0          # table surface in world frame
TABLE_SZ  = "0.35 0.35 0.02"
CAM_POS   = "0.0 -0.05 0.55"
CAM_EULER = "3.14159 0 0"
FOVY      = 60.0
IMG_H = IMG_W = 224


def _so101_worldbody() -> Tuple[str, str, str, str]:
    """Parse so101.xml → (defaults, assets, worldbody_inner, actuators)."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(_SO101)
    root = tree.getroot()
    s = lambda e: ET.tostring(e, encoding="unicode")
    defaults  = "".join(s(e) for e in root if e.tag == "default")
    # assets: emit inner children only (we embed inside our own <asset> block)
    asset_el  = root.find("asset")
    assets    = "".join(s(c) for c in asset_el) if asset_el is not None else ""
    wb        = root.find("worldbody")
    wb_inner  = "".join(s(c) for c in wb) if wb is not None else ""
    actuators = "".join(s(e) for e in root if e.tag == "actuator")
    return defaults, assets, wb_inner, actuators


def _build_xml(obj_mesh_vis: Optional[str], obj_mesh_col: Optional[str],
               obj_size: float = 0.04) -> str:
    """
    Build scene XML. If obj_mesh_vis is None, use a simple box as the object.
    """
    so_def, so_ast, so_wb, so_act = _so101_worldbody()

    if obj_mesh_vis and obj_mesh_col:
        obj_asset = f"""
  <mesh name="obj_vis" file="{obj_mesh_vis}"/>
  <mesh name="obj_col" file="{obj_mesh_col}"/>
"""
        obj_geoms = f"""
      <geom name="obj_vis_geom" type="mesh" mesh="obj_vis"
            contype="0" conaffinity="0" group="2" rgba="0.8 0.6 0.2 1"/>
      <geom name="obj_col_geom" type="mesh" mesh="obj_col"
            contype="1" conaffinity="1" friction="0.8 0.01 0.001"/>
"""
    else:
        obj_asset = ""
        obj_geoms = f"""
      <geom name="obj_col_geom" type="box"
            size="{obj_size} {obj_size} {obj_size}"
            contype="1" conaffinity="1" friction="0.8 0.01 0.001"
            rgba="0.8 0.5 0.2 1"/>
"""

    return f"""<mujoco model="grasp6dof_scene">
  <compiler meshdir="{_ASSETS}" autolimits="true" angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>

  {so_def}

  <asset>
    {so_ast}
    {obj_asset}
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>

    <geom name="table" type="box" size="{TABLE_SZ}"
          pos="0 0 {TABLE_Z - 0.02}" rgba="0.85 0.75 0.6 1"
          contype="1" conaffinity="1"/>

    <camera name="overhead" pos="{CAM_POS}" euler="{CAM_EULER}" fovy="{FOVY}"/>

    <!-- SO-101 arm mounted at table edge -->
    <body name="robot_mount" pos="0 -0.28 {TABLE_Z}">
      {so_wb}
    </body>

    <!-- Object (free body) -->
    <body name="object" pos="0 0.05 {TABLE_Z + obj_size}">
      <freejoint name="obj_joint"/>
      {obj_geoms}
    </body>
  </worldbody>

  {so_act}
</mujoco>"""


ARM_JOINTS = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"]
GRIP_JOINT = "gripper"
EEF_SITE   = "gripperframe"
HOME_QPOS  = np.array([0.0, -0.5, 1.0, -0.5, 0.0])
GRIP_OPEN  = 1.0
GRIP_CLOSE = 0.05
IK_ITERS   = 150
IK_DAMP    = 0.05


class MujocoGraspEnv:
    """
    Minimal MuJoCo grasp environment.

    Grasp action dict: {"position": [x,y,z], "rpy": [r,p,y], "width": w, "score": s}
    """

    def __init__(self, obj_mesh_vis=None, obj_mesh_col=None,
                 obj_size=0.04, render_mode="rgb_array", seed=0):
        self.obj_size  = obj_size
        self._xml      = _build_xml(obj_mesh_vis, obj_mesh_col, obj_size)
        self._rng      = np.random.default_rng(seed)
        self._render_mode = render_mode

        self._load_model()

    # ── model ────────────────────────────────────────────────────────────────

    def _load_model(self):
        self.model = mujoco.MjModel.from_xml_string(self._xml)
        self.data  = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)

        self._arm_qpos = [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self._arm_act  = [self.model.actuator(n).id      for n in ARM_JOINTS]
        self._arm_jdof = [self.model.joint(n).dofadr[0]  for n in ARM_JOINTS]
        self._grip_act = self.model.actuator(GRIP_JOINT).id
        self._grip_adr = self.model.joint(GRIP_JOINT).qposadr[0]
        self._eef_site = self.model.site(EEF_SITE).id

        try:
            self._jaw_fix = self.model.body("gripper").id
            self._jaw_mov = self.model.body("moving_jaw_so101_v1").id
        except Exception:
            self._jaw_fix = self._jaw_mov = -1

        self._obj_jnt  = self.model.joint("obj_joint")
        self._obj_adr  = self._obj_jnt.qposadr[0]

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self, obj_pos=None) -> dict:
        """Reset scene. Returns initial observation."""
        mujoco.mj_resetData(self.model, self.data)

        # home arm
        for adr, q in zip(self._arm_qpos, HOME_QPOS):
            self.data.qpos[adr] = q
        self.data.qpos[self._grip_adr] = GRIP_OPEN
        for act, q in zip(self._arm_act, HOME_QPOS):
            self.data.ctrl[act] = q
        self.data.ctrl[self._grip_act] = GRIP_OPEN

        # place object
        if obj_pos is None:
            x = self._rng.uniform(-0.08, 0.08)
            y = self._rng.uniform(0.01, 0.12)
            obj_pos = [x, y, TABLE_Z + self.obj_size + 0.002]
        self.data.qpos[self._obj_adr: self._obj_adr+3] = obj_pos
        self.data.qpos[self._obj_adr+3: self._obj_adr+7] = [1,0,0,0]  # w x y z

        mujoco.mj_forward(self.model, self.data)
        self._step_n(100)  # settle

        return self.get_obs()

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        """
        Execute one grasp action.

        action: {"position": [x,y,z], "rpy": [r,p,y], "width": w, "score": s}
        Returns: (obs, reward, done, info)
        """
        pos   = np.asarray(action["position"])
        rpy   = np.asarray(action.get("rpy",   [0.0, 0.0, 0.0]))
        width = float(action.get("width", 0.04))

        before = self.get_object_pose()

        # --- pre-grasp: move above target ---
        pre = pos.copy(); pre[2] = TABLE_Z + 0.18
        self._move_eef(pre)
        self._set_gripper(GRIP_OPEN)
        self._step_n(60)

        # --- descend ---
        target_z = max(pos[2], TABLE_Z + 0.01)
        self._move_eef(np.array([pos[0], pos[1], target_z]))
        self._step_n(80)

        # --- close gripper ---
        self._set_gripper(GRIP_CLOSE)
        self._step_n(80)

        # --- lift ---
        lift = pos.copy(); lift[2] = TABLE_Z + 0.25
        self._move_eef(lift)
        self._step_n(80)

        after = self.get_object_pose()
        success = bool(after["position"][2] > TABLE_Z + 0.08)
        reward  = 1.0 if success else 0.0

        obs  = self.get_obs()
        info = {"before_pos": before["position"].tolist(),
                "after_pos":  after["position"].tolist(),
                "success":    success}
        return obs, reward, True, info  # always done after one grasp

    def render(self) -> np.ndarray:
        """Return RGB image (H, W, 3) uint8."""
        self._renderer.update_scene(self.data, camera="overhead")
        return self._renderer.render().copy()

    def get_obs(self) -> dict:
        """Return {'rgb', 'depth', 'object_pose'}."""
        self._renderer.update_scene(self.data, camera="overhead")
        rgb = self._renderer.render().copy()

        self._renderer.enable_depth_rendering()
        depth_raw = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        ext  = self.model.stat.extent
        near = self.model.vis.map.znear * ext
        far  = self.model.vis.map.zfar  * ext
        depth = near / (1 - np.clip(depth_raw, 1e-6, 1-1e-6) * (1 - near/far))

        return {"rgb": rgb, "depth": depth, "object_pose": self.get_object_pose()}

    def get_object_pose(self) -> dict:
        """Return {"position": np.ndarray(3), "quaternion": np.ndarray(4)}."""
        q = self.data.qpos[self._obj_adr: self._obj_adr+7]
        return {"position": q[:3].copy(), "quaternion": q[3:7].copy()}

    def close(self):
        self._renderer.close()

    # ── internals ────────────────────────────────────────────────────────────

    def _step_n(self, n: int):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def _set_gripper(self, angle: float):
        self.data.ctrl[self._grip_act] = angle
        self._step_n(40)

    def _eef_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._eef_site].copy()

    def _move_eef(self, target: np.ndarray):
        """IK-based arm motion toward target xyz."""
        for _ in range(IK_ITERS):
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self._eef_site)
            J   = jacp[:, self._arm_jdof]
            err = target - self._eef_pos()
            if np.linalg.norm(err) < 5e-4:
                break
            dq  = J.T @ np.linalg.solve(J @ J.T + IK_DAMP * np.eye(3), err)
            for i, adr in enumerate(self._arm_qpos):
                jid = self.model.joint(ARM_JOINTS[i]).id
                lo, hi = self.model.jnt_range[jid]
                self.data.qpos[adr] = np.clip(self.data.qpos[adr] + dq[i], lo, hi)
            mujoco.mj_forward(self.model, self.data)

        for act, adr in zip(self._arm_act, self._arm_qpos):
            self.data.ctrl[act] = self.data.qpos[adr]
        self._step_n(30)
