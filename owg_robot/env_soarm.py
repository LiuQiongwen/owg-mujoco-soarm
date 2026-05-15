# -*- coding: utf-8 -*-
"""
MuJoCo SO-ARM101 environment — PyBullet-compatible API layer.

Drop-in replacement for owg_robot/env.py so that demo.py / ui.py / batch
scripts work unchanged by swapping the backend import.

Public API mirrors Environment (owg_robot/env.py):
  reset(), get_obs(), get_points(), step()
  pick_obj_by_id(), get_obj_pose(), get_obj_pos(), get_obj_orn()
  put_obj_in_tray(), put_obj_in_free_space(), put_obj_in_loc()
  load_obj(), load_isolated_obj(), remove_obj(), remove_all_obj()
  set_obj_grasps(), get_obj_grasps(), get_obj_grasp_rects()
  set_obj_state(), update_obj_states(), get_obj_states()
  reset_robot(), move_gripper(), move_ee(), get_ee_pos()
  gripper_contact(), check_grasped(), check_grasped_id()
  draw_predicted_grasp(), remove_drawing()   ← no-ops (MuJoCo viewer handles viz)
  dummy_simulation_steps(), close()

Key differences from PyBullet backend:
  - IK is xyz-only for now; orientation is accepted but ignored.
    TODO markers show exactly where 6-DoF IK should be plugged in later.
  - draw_predicted_grasp / remove_drawing are no-ops.
  - Camera is a _MockCamera shim (ui.py accesses env.camera.width).
"""
import os
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import mujoco
import mujoco.viewer

# ── paths ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
SO101_XML  = os.path.join(_DIR, "assets", "so101", "so101.xml")
SO101_MESH = os.path.join(_DIR, "assets", "so101", "assets")
YCB_ROOT   = os.path.join(_DIR, "assets", "ycb_objects")

# ── scene constants ───────────────────────────────────────────────────────────
TABLE_TOP_Z    = 0.785
ROBOT_BASE_POS = "0 0 0.785"
CAM_POS        = "0.05 -0.52 1.9"
CAM_EULER      = "0 0 0"   # with angle="radian" compiler, default orientation looks -Z (down)
TARGET_ZONE_POS = [0.5, 0.0, TABLE_TOP_Z]

ARM_JOINTS  = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIP_JOINT  = "gripper"
EEF_SITE    = "gripperframe"
HOME_QPOS   = np.array([0.0, -0.4, 0.8, -0.4, 0.0], dtype=float)
GRIP_OPEN   = 1.0
GRIP_CLOSED = 0.05

IMG_SIZE = 224
FOVY     = 55.0
PIX_CONV = 277.0
IMG_ROT  = -np.pi * 0.54
CAM_ROT  = 0.0

# IK params
IK_ITERS   = 200
IK_DAMPING = 0.05
IK_TOL     = 5e-4

# log path (matches PyBullet env)
_LOG_DIR  = "logs"
_LOG_PATH = os.path.join(_LOG_DIR, "ui_grasp_exec.jsonl")


# ── compatibility shim ───────────────────────────────────────────────────────

class FailToReachTargetError(RuntimeError):
    pass


class _MockCamera:
    """Minimal camera-like object so ui.py / GraspGenerator can access camera attributes.

    GraspGenerator uses: camera.near, camera.far
    ui.py uses:          camera.width
    """
    def __init__(self, size: int, x: float = 0.05, y: float = -0.52, z: float = 1.9,
                 znear: float = 0.01, zfar: float = 10.0, fov: float = FOVY):
        self.width  = size
        self.height = size
        self.x, self.y, self.z = x, y, z
        self.znear, self.zfar  = znear, zfar
        self.fov = fov
        # aliases expected by GraspGenerator
        self.near = znear
        self.far  = zfar


# ── scene XML builder ─────────────────────────────────────────────────────────

def _so101_fragment() -> Tuple[str, str, str, str]:
    tree = ET.parse(SO101_XML)
    root = tree.getroot()

    def _s(elem):
        return ET.tostring(elem, encoding="unicode")

    defaults    = "".join(_s(e) for e in root if e.tag == "default")
    asset_el    = root.find("asset")
    assets      = "".join(_s(c) for c in asset_el) if asset_el is not None else ""
    wbody_el    = root.find("worldbody")
    wbody_inner = "".join(_s(c) for c in wbody_el) if wbody_el is not None else ""
    actuators   = "".join(_s(e) for e in root if e.tag == "actuator")
    return defaults, assets, wbody_inner, actuators



def _find_ycb_mesh(obj_dir):
    candidates = [
        "textured_simple_reoriented.obj",
        "textured_reoriented.obj",
        "textured_simple.obj",
        "textured.obj",
        "collision_vhacd.obj",
    ]
    for name in candidates:
        path = Path(obj_dir) / name
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"No valid YCB mesh found in {obj_dir}")

def _ycb_asset_tag(obj_name: str, obj_idx: int) -> Tuple[str, str]:
    obj_dir = os.path.join(YCB_ROOT, obj_name)
    vis_mesh = _find_ycb_mesh(obj_dir)
    print(f"[INFO] Using mesh for {obj_name}: {Path(vis_mesh).name}")

    col_mesh = os.path.join(obj_dir, "collision_vhacd.obj")
    if not os.path.isfile(col_mesh):
        col_mesh = vis_mesh

    tex_file = os.path.join(obj_dir, "texture_map.png")
    if not os.path.isfile(tex_file):
        tex_file = ""

    material_block = (
        f'  <texture name="ycb_tex_{obj_idx}" type="2d" file="{tex_file}"/>\n'
        f'  <material name="ycb_mat_{obj_idx}" texture="ycb_tex_{obj_idx}"/>\n'
    ) if tex_file else f'  <material name="ycb_mat_{obj_idx}" rgba="0.8 0.8 0.8 1"/>\n'

    asset = f"""
  <mesh name="ycb_vis_{obj_idx}" file="{vis_mesh}"/>
  <mesh name="ycb_col_{obj_idx}" file="{col_mesh}"/>
{material_block}"""
    body = f"""
    <body name="obj_{obj_idx}" pos="0 0 -100">
      <freejoint name="obj_joint_{obj_idx}"/>
      <geom name="ycb_vis_geom_{obj_idx}" type="mesh" mesh="ycb_vis_{obj_idx}"
            material="ycb_mat_{obj_idx}" contype="0" conaffinity="0" group="2"/>
      <geom name="ycb_col_geom_{obj_idx}" type="mesh" mesh="ycb_col_{obj_idx}"
            contype="1" conaffinity="1" friction="0.8 0.01 0.001"/>
    </body>
"""
    return asset, body


def _build_scene_xml(obj_names: List[str]) -> str:
    so_defaults, so_assets, so_wbody, so_acts = _so101_fragment()
    ycb_assets = ""
    ycb_bodies = ""
    for i, name in enumerate(obj_names):
        a, b = _ycb_asset_tag(name, i)
        ycb_assets += a
        ycb_bodies  += b

    return f"""<mujoco model="owg_soarm_scene">
  <compiler meshdir="{SO101_MESH}" autolimits="true" angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>

  {so_defaults}

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1=".4 .6 .8" rgb2="0 0 0" width="512" height="512"/>
    <texture name="table_tex" type="2d" builtin="flat" rgb1=".8 .7 .6" width="64" height="64"/>
    <material name="table_mat" texture="table_tex" texrepeat="4 4"/>
    {so_assets}
    {ycb_assets}
  </asset>

  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0" material="table_mat"/>
    <geom name="table_top" type="box"
          size="0.45 0.45 0.02" pos="0 -0.45 {TABLE_TOP_Z - 0.02}"
          material="table_mat" contype="1" conaffinity="1"/>

    <!-- Tray: target drop zone, matches TARGET_ZONE_POS -->
    <body name="tray" pos="{TARGET_ZONE_POS[0]} {TARGET_ZONE_POS[1]} {TARGET_ZONE_POS[2]}">
      <geom name="tray_bottom" type="box" size="0.12 0.12 0.005"
            pos="0 0 0.005" rgba="0.4 0.6 0.4 0.8" contype="1" conaffinity="1"/>
      <geom name="tray_w1" type="box" size="0.005 0.12 0.02"
            pos=" 0.125 0 0.02" rgba="0.4 0.6 0.4 0.8"/>
      <geom name="tray_w2" type="box" size="0.005 0.12 0.02"
            pos="-0.125 0 0.02" rgba="0.4 0.6 0.4 0.8"/>
      <geom name="tray_w3" type="box" size="0.12 0.005 0.02"
            pos="0  0.125 0.02" rgba="0.4 0.6 0.4 0.8"/>
      <geom name="tray_w4" type="box" size="0.12 0.005 0.02"
            pos="0 -0.125 0.02" rgba="0.4 0.6 0.4 0.8"/>
    </body>

    <camera name="overhead" pos="{CAM_POS}" euler="{CAM_EULER}" fovy="{FOVY}"/>

    <body name="robot_base_mount" pos="{ROBOT_BASE_POS}">
      {so_wbody}
    </body>

    {ycb_bodies}
  </worldbody>

  {so_acts}
</mujoco>"""


# ── coordinate helpers ────────────────────────────────────────────────────────

def _get_transform_matrix(x, y, z, rot):
    return np.array([[np.cos(rot), -np.sin(rot), 0, x],
                     [np.sin(rot),  np.cos(rot), 0, y],
                     [0,            0,           1, z],
                     [0,            0,           0, 1]])


# ── main class ────────────────────────────────────────────────────────────────

class EnvironmentSoArm:
    """MuJoCo SO-ARM101 environment with PyBullet-compatible public API."""

    # class-level constants matching Environment (owg_robot/env.py)
    OBJECT_INIT_HEIGHT          = TABLE_TOP_Z + 0.10
    GRIPPER_MOVING_HEIGHT       = TABLE_TOP_Z + 0.20
    GRIPPER_GRASPED_LIFT_HEIGHT = TABLE_TOP_Z + 0.35
    FINGER_LENGTH               = 0.04
    Z_TABLE_TOP                 = TABLE_TOP_Z
    TARGET_ZONE_POS             = TARGET_ZONE_POS
    GRIP_REDUCTION              = 0.85
    IMG_SIZE                    = IMG_SIZE
    PIX_CONVERSION              = PIX_CONV
    IMG_ROTATION                = IMG_ROT
    CAM_ROTATION                = CAM_ROT
    SIMULATION_STEP_DELAY       = 0.0

    def __init__(self,
                 camera=None,          # accepted but ignored (PyBullet compat)
                 vis: bool = False,
                 debug: bool = False,
                 finger_length: float = 0.04,
                 n_grasp_attempts: int = 3,
                 **kwargs):
        self.vis   = vis
        self.debug = debug
        self.finger_length    = finger_length
        self.N_GRASP_ATTEMPTS = n_grasp_attempts

        # mock camera so ui.py can do env.camera.width
        self.camera = _MockCamera(IMG_SIZE)

        # coordinate transforms (identical to env.py)
        img_center = IMG_SIZE / 2 - 0.5
        cx, cy, cz = [float(v) for v in CAM_POS.split()]
        self.img_to_cam = _get_transform_matrix(
            -img_center / PIX_CONV, img_center / PIX_CONV, 0, IMG_ROT)
        self.cam_to_robot_base = _get_transform_matrix(cx, cy, cz, CAM_ROT)

        # object book-keeping (same attribute names as env.py)
        self._pool_names:      List[str] = []   # YCB directory names (e.g. "YcbBanana")
        self._pool_slots:      List[int] = []   # pool slot index per active obj (parallel to obj_ids)
        self.obj_ids:          List[int] = []
        self.obj_names:        List[str] = []   # logical names (e.g. "Banana") for VLM labels
        self.obj_positions:    list = []
        self.obj_orientations: list = []
        self.obj_grasps:       list = []
        self.obj_grasp_rects:  list = []

        self.gripper_open_limit = (0.0, 0.10)
        self.ee_position_limit  = ((-0.4, 0.4), (-0.4, 0.4),
                                   (TABLE_TOP_Z, TABLE_TOP_Z + 0.4))

        # MuJoCo model
        self.model: Optional[mujoco.MjModel] = None
        self.data:  Optional[mujoco.MjData]  = None
        self._renderer: Optional[mujoco.Renderer] = None
        self._viewer = None
        self._rebuild_model()

        if vis:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.reset_robot()

    # ── model lifecycle ───────────────────────────────────────────────────────

    def _rebuild_model(self):
        xml = _build_scene_xml(self._pool_names)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data  = mujoco.MjData(self.model)
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(self.model, height=IMG_SIZE, width=IMG_SIZE)
        self._cache_ids()
        # initialize arm to HOME_QPOS so _steps() after object spawn is stable
        for adr, q in zip(self._arm_qpos_adr, HOME_QPOS):
            self.data.qpos[adr] = q
        for act_id, q in zip(self._arm_act_ids, HOME_QPOS):
            self.data.ctrl[act_id] = q
        self.data.qpos[self._grip_qpos_adr] = GRIP_OPEN
        self.data.ctrl[self._grip_act_id]   = GRIP_OPEN
        mujoco.mj_forward(self.model, self.data)
        # Reattach passive viewer to new model/data when vis is on
        if getattr(self, "vis", False) and getattr(self, "_viewer", None) is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _cache_ids(self):
        self._arm_jnt_ids   = [self.model.joint(n).id for n in ARM_JOINTS]
        self._arm_qpos_adr  = [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self._arm_act_ids   = [self.model.actuator(n).id for n in ARM_JOINTS]
        self._grip_act_id   = self.model.actuator(GRIP_JOINT).id
        self._grip_jnt_id   = self.model.joint(GRIP_JOINT).id
        self._grip_qpos_adr = self.model.joint(GRIP_JOINT).qposadr[0]
        self._eef_site_id   = self.model.site(EEF_SITE).id

        try:
            self._jaw_body_id    = self.model.body("gripper").id
            self._jaw_mv_body_id = self.model.body("moving_jaw_so101_v1").id
        except Exception:
            self._jaw_body_id    = -1
            self._jaw_mv_body_id = -1

    # ── simulation step ───────────────────────────────────────────────────────

    def step_simulation(self):
        mujoco.mj_step(self.model, self.data)
        if self.vis and self._viewer is not None:
            self._viewer.sync()
            time.sleep(1 / 60)

    def step(self):
        """Alias for step_simulation() (PyBullet compatibility)."""
        self.step_simulation()

    def _steps(self, n: int = 1):
        for _ in range(n):
            self.step_simulation()

    def dummy_simulation_steps(self, n: int):
        self._steps(n)

    # ── robot control ─────────────────────────────────────────────────────────

    def reset_robot(self):
        for adr, q in zip(self._arm_qpos_adr, HOME_QPOS):
            self.data.qpos[adr] = q
        self.data.qpos[self._grip_qpos_adr] = GRIP_OPEN
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        for act_id, q in zip(self._arm_act_ids, HOME_QPOS):
            self.data.ctrl[act_id] = q
        self.data.ctrl[self._grip_act_id] = GRIP_OPEN
        self._steps(100)

    def reset(self) -> dict:
        """Reset robot to home, restore saved object positions, return obs."""
        self.reset_robot()
        self.reset_all_obj()
        return self.get_obs()

    def move_arm_away(self):
        home = HOME_QPOS.copy()
        home[0] = 1.5
        for act_id, q in zip(self._arm_act_ids, home):
            self.data.ctrl[act_id] = q
        self._steps(150)

    def move_gripper(self, opening_length: float, step: int = 80):
        t = np.clip(opening_length / 0.1, 0.0, 1.0)
        angle = GRIP_CLOSED + t * (GRIP_OPEN - GRIP_CLOSED)
        self.data.ctrl[self._grip_act_id] = angle
        self._steps(step)

    def auto_close_gripper(self, step: int = 100, check_contact: bool = False) -> bool:
        for i in range(step):
            t = 1.0 - i / step
            self.data.ctrl[self._grip_act_id] = GRIP_CLOSED + t * (GRIP_OPEN - GRIP_CLOSED)
            self.step_simulation()
            if check_contact and self.gripper_contact():
                return True
        return False

    def calc_z_offset(self, gripper_opening_length: float) -> float:
        return 0.0

    # ── IK + end-effector motion ──────────────────────────────────────────────

    def _get_eef_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._eef_site_id].copy()

    def _ik_step(self, target_pos: np.ndarray) -> bool:
        """Damped least-squares IK — position only (xyz).

        TODO(6dof-ik): extend to 6-DoF by stacking rotation Jacobian rows
        and computing orientation error from quat difference. Replace the
        3×n Jacobian with 6×n and solve for [Δpos; Δori] jointly.
        """
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._eef_site_id)
        cols = [self.model.joint(n).dofadr[0] for n in ARM_JOINTS]
        J    = jacp[:, cols]
        err  = target_pos - self._get_eef_pos()
        dq   = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(3), err)
        for i, adr in enumerate(self._arm_qpos_adr):
            lo = self.model.jnt_range[self._arm_jnt_ids[i], 0]
            hi = self.model.jnt_range[self._arm_jnt_ids[i], 1]
            self.data.qpos[adr] = np.clip(self.data.qpos[adr] + dq[i], lo, hi)
        mujoco.mj_forward(self.model, self.data)
        return np.linalg.norm(err) < IK_TOL

    def _solve_ik(self, target_pos: np.ndarray) -> bool:
        for _ in range(IK_ITERS):
            if self._ik_step(target_pos):
                return True
        return np.linalg.norm(target_pos - self._get_eef_pos()) < 5e-3

    def move_ee(self, action, max_step: int = 200, **kwargs):
        """Move EEF to (x, y, z, orn). orn is accepted but currently ignored.

        vis=True  → smooth interpolation: ctrl interpolated from current to target
                    over max_step physics steps, viewer.sync() each step.
        vis=False → fast IK teleport + settle (headless eval path, unchanged).
        """
        x, y, z = action[0], action[1], action[2]
        x = np.clip(x, *self.ee_position_limit[0])
        y = np.clip(y, *self.ee_position_limit[1])
        z = np.clip(z, *self.ee_position_limit[2])

        if self.vis:
            # Save current physical state before IK modifies qpos
            qpos_saved = self.data.qpos[self._arm_qpos_adr].copy()
            qvel_saved = self.data.qvel.copy()
            q_start    = np.array([self.data.ctrl[i] for i in self._arm_act_ids])

            ok = self._solve_ik(np.array([x, y, z]))
            q_target = np.array([self.data.qpos[adr] for adr in self._arm_qpos_adr])

            # Restore physics state so interpolation starts from actual position
            for adr, q in zip(self._arm_qpos_adr, qpos_saved):
                self.data.qpos[adr] = q
            self.data.qvel[:] = qvel_saved
            mujoco.mj_forward(self.model, self.data)

            # Drive ctrl from q_start → q_target over max_step steps (smooth motion)
            for i in range(max_step):
                t = (i + 1) / max_step
                for act_id, qs, qt in zip(self._arm_act_ids, q_start, q_target):
                    self.data.ctrl[act_id] = float(qs + t * (qt - qs))
                self.step_simulation()
        else:
            # Headless: IK teleport + physics settle (original fast path)
            ok = self._solve_ik(np.array([x, y, z]))
            for act_id, adr in zip(self._arm_act_ids, self._arm_qpos_adr):
                self.data.ctrl[act_id] = self.data.qpos[adr]
            self._steps(max_step // 4)

        pos = self._get_eef_pos()
        return ok, (pos, np.array([0, 0, 0, 1]))

    def get_ee_pos(self):
        return self._get_eef_pos(), np.array([0, 0, 0, 1])

    def move_to_init_pos(self):
        above = self._get_eef_pos().copy()
        above[2] = self.GRIPPER_MOVING_HEIGHT
        self.move_ee([above[0], above[1], above[2], None])
        self._steps(30)
        return self.move_ee([0.0, 0.2, self.GRIPPER_MOVING_HEIGHT, None])

    # ── contact / grasp detection ─────────────────────────────────────────────

    def _contacts_on_bodies(self, body_ids_a, body_ids_b=None):
        result = set()
        for c in self.data.contact[:self.data.ncon]:
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            for ba in body_ids_a:
                if b1 == ba:
                    result.add(b2)
                elif b2 == ba:
                    result.add(b1)
        if body_ids_b is not None:
            result = {b for b in result if b in body_ids_b}
        return result

    def check_grasped(self, obj_id: int) -> bool:
        slot     = self._obj_pool_slot(obj_id)
        obj_body = self.model.body(f"obj_{slot}").id
        contacts = self._contacts_on_bodies(
            [self._jaw_body_id, self._jaw_mv_body_id], {obj_body})
        return len(contacts) > 0

    def check_grasped_id(self) -> List[int]:
        finger_ids = [self._jaw_body_id, self._jaw_mv_body_id]
        # map body_id → logical obj_id using pool slots
        obj_bodies = {self.model.body(f"obj_{slot}").id: oid
                      for oid, slot in zip(self.obj_ids, self._pool_slots)}
        contacts = self._contacts_on_bodies(finger_ids, set(obj_bodies.keys()))
        return [obj_bodies[b] for b in contacts if b in obj_bodies]

    def check_contact(self, id_a: int, id_b: int) -> bool:
        ba = self.model.body(f"obj_{self._obj_pool_slot(id_a)}").id
        bb = self.model.body(f"obj_{self._obj_pool_slot(id_b)}").id
        return len(self._contacts_on_bodies([ba], {bb})) > 0

    def gripper_contact(self, bool_operator: str = 'and', force: float = 5.0) -> bool:
        obj_bodies = set(self.model.body(f"obj_{slot}").id for slot in self._pool_slots)
        cf = self._contacts_on_bodies([self._jaw_body_id],    obj_bodies)
        cm = self._contacts_on_bodies([self._jaw_mv_body_id], obj_bodies)
        if bool_operator == 'and':
            return len(cf) > 0 and len(cm) > 0
        return len(cf) > 0 or len(cm) > 0

    def check_target_reached(self, obj_id: int) -> bool:
        qpos = self._obj_qpos(self._obj_pool_slot(obj_id))
        x, y = qpos[0], qpos[1]
        tp   = self.model.body("tray").pos
        half = 0.12
        return abs(x - tp[0]) < half and abs(y - tp[1]) < half

    def check_target_loc_reached(self, obj_id: int, pos) -> bool:
        obj_pos = self.get_obj_pos(obj_id)
        return abs(pos[0] - obj_pos[0]) < 0.1 and abs(pos[1] - obj_pos[1]) < 0.1

    # ── object qpos helpers ───────────────────────────────────────────────────

    def _obj_pool_idx(self, obj_id: int) -> int:
        """Return the list index of obj_id in self.obj_ids."""
        return self.obj_ids.index(obj_id)

    def _obj_pool_slot(self, obj_id: int) -> int:
        """Return the pool slot (model index) for obj_id."""
        list_pos = self.obj_ids.index(obj_id)
        return self._pool_slots[list_pos]

    def _obj_qpos(self, pool_slot: int) -> np.ndarray:
        jnt = self.model.joint(f"obj_joint_{pool_slot}")
        return self.data.qpos[jnt.qposadr[0]: jnt.qposadr[0] + 7].copy()

    def _set_obj_pose(self, pool_slot: int, pos, orn=(1, 0, 0, 0)):
        jnt = self.model.joint(f"obj_joint_{pool_slot}")
        adr = jnt.qposadr[0]
        self.data.qpos[adr:adr+3]   = pos
        self.data.qpos[adr+3:adr+7] = orn   # MuJoCo: w x y z
        mujoco.mj_forward(self.model, self.data)

    # ── object pose access ────────────────────────────────────────────────────

    def get_obj_pos(self, obj_id: int) -> np.ndarray:
        return self._obj_qpos(self._obj_pool_slot(obj_id))[:3]

    def get_obj_orn(self, obj_id: int) -> np.ndarray:
        return self._obj_qpos(self._obj_pool_slot(obj_id))[3:7]

    def get_obj_pose(self, obj_id: int) -> Dict[str, np.ndarray]:
        """Return {'position': np.ndarray(3), 'quaternion': np.ndarray(4)}.
        Matches the interface expected by grasp_ranker and mujoco_validator."""
        qpos = self._obj_qpos(self._obj_pool_slot(obj_id))
        return {"position": qpos[:3], "quaternion": qpos[3:7]}

    def get_object_pose(self) -> Dict[str, np.ndarray]:
        """mujoco_validator interface: pose of first loaded object.
        Returns table-surface height stub when no objects are loaded."""
        if self.obj_ids:
            return self.get_obj_pose(self.obj_ids[0])
        return {"position": np.array([0.0, -0.45, TABLE_TOP_Z]),
                "quaternion": np.array([1.0, 0.0, 0.0, 0.0])}

    def get_obj_states(self) -> list:
        return [{'id': oid, 'name': name,
                 'pos': self.get_obj_pos(oid),
                 'orn': self.get_obj_orn(oid)}
                for oid, name in zip(self.obj_ids, self.obj_names)]

    def get_obj_id_by_name(self, name: str) -> Optional[int]:
        try:
            return self.obj_ids[self.obj_names.index(name)]
        except ValueError:
            return None

    # ── object state sync ─────────────────────────────────────────────────────

    def update_obj_states(self):
        """Refresh stored positions/orientations from current simulation state.
        Clears cached grasps (they depend on image, which may have changed)."""
        for i, (oid, slot) in enumerate(zip(self.obj_ids, self._pool_slots)):
            q = self._obj_qpos(slot)
            self.obj_positions[i]    = list(q[:3])
            self.obj_orientations[i] = list(q[3:7])
            self.obj_grasps[i]       = None
            self.obj_grasp_rects[i]  = None

    def set_obj_state(self, state: list):
        """Teleport objects to saved state list [{'id', 'pos', 'orn'}, ...].
        Used by ui.py reset_same() to restore the initial scene."""
        for item in state:
            oid = item['id']
            if oid in self.obj_ids:
                list_pos = self._obj_pool_idx(oid)
                slot     = self._pool_slots[list_pos]
                self._set_obj_pose(slot, item['pos'], item['orn'])
                self.obj_grasps[list_pos]      = None
                self.obj_grasp_rects[list_pos] = None
        self.wait_until_all_still()

    # ── stillness checks ──────────────────────────────────────────────────────

    def is_still(self, obj_id: int) -> bool:
        slot = self._obj_pool_slot(obj_id)
        jnt  = self.model.joint(f"obj_joint_{slot}")
        vel  = self.data.qvel[jnt.dofadr[0]: jnt.dofadr[0] + 6]
        return np.abs(vel).sum() < 1e-3

    def wait_until_still(self, obj_id: int, max_wait_epochs: int = 100):
        for _ in range(max_wait_epochs):
            self.step_simulation()
            if self.is_still(obj_id):
                return

    def wait_until_all_still(self, max_wait_epochs: int = 1000):
        for _ in range(max_wait_epochs):
            self.step_simulation()
            if all(self.is_still(oid) for oid in self.obj_ids):
                return

    # ── object management ─────────────────────────────────────────────────────

    def preload_pool(self, logical_names: list):
        """Pre-register all object types so spawn() triggers only ONE model rebuild.

        Call before the spawn loop with the full object list.  load_obj() will
        find every pool_name already registered and skip the per-object rebuild.
        """
        changed = False
        for name in logical_names:
            if os.path.isdir(os.path.join(YCB_ROOT, "Ycb" + name)):
                pool_name = "Ycb" + name
            elif os.path.isdir(os.path.join(YCB_ROOT, name)):
                pool_name = name
            else:
                pool_name = "Ycb" + name
            if pool_name not in self._pool_names:
                self._pool_names.append(pool_name)
                changed = True
        if changed:
            self._rebuild_model()

    def load_obj(self, path_or_name, name: Optional[str] = None,
                 pos=None, orn=None,
                 mod_orn: bool = False, mod_stiffness: bool = False,
                 yaw: float = 0.0, texture=None, color=None,
                 globalScaling: float = 1.0) -> int:
        """Load a YCB object.

        Accepts both PyBullet-style (path, name, ...) and MuJoCo-style (obj_name, ...) calls.

        Naming convention:
          - pool_name  : YCB directory name (e.g. "YcbBanana") — used for mesh lookup
          - logical_name: human-readable name (e.g. "Banana") — stored in obj_names for VLM

        When a path is given, pool_name is extracted from the directory segment.
        When name= is given, it becomes the logical_name (VLM label).
        """
        # --- resolve pool_name (mesh directory) ---
        if os.sep in str(path_or_name) or "/" in str(path_or_name):
            # path_or_name is a file path; extract YCB directory name from it
            parts = str(path_or_name).replace("\\", "/").split("/")
            pool_name = path_or_name   # fallback
            for part in reversed(parts):
                if part and not part.endswith(".urdf") and part != "model.urdf":
                    pool_name = part   # e.g. "YcbBanana"
                    break
        else:
            # path_or_name is already a directory name (e.g. "YcbBanana" or "Banana")
            # If it exists in ycb_objects as-is, use it; else try with "Ycb" prefix
            if os.path.isdir(os.path.join(YCB_ROOT, path_or_name)):
                pool_name = path_or_name
            elif os.path.isdir(os.path.join(YCB_ROOT, "Ycb" + path_or_name)):
                pool_name = "Ycb" + path_or_name
            else:
                pool_name = path_or_name   # use as-is, will fail at model build if wrong

        # --- resolve logical_name (for VLM labels, env.obj_names) ---
        logical_name = name if name is not None else pool_name

        # --- ensure pool slot exists (rebuild model if this is a new object type) ---
        if pool_name not in self._pool_names:
            self._pool_names.append(pool_name)
            self._rebuild_model()
            # restore already-loaded objects (rebuild resets all qpos)
            for slot, (op, oo) in enumerate(zip(self.obj_positions, self.obj_orientations)):
                self._set_obj_pose(self._pool_slots[slot], op, oo)

        pool_slot = self._pool_names.index(pool_name)

        if pos is None:
            pos = [0.0, 0.2, self.OBJECT_INIT_HEIGHT]
        if orn is None:
            import math as _math
            orn = [_math.cos(yaw / 2), 0, 0, _math.sin(yaw / 2)]  # w x y z

        self._set_obj_pose(pool_slot, pos, orn)
        self._steps(50)

        # IDs start from 1 so that 0 can unambiguously mean background in seg maps
        logical_id = (max(self.obj_ids) + 1) if self.obj_ids else 1
        self.obj_ids.append(logical_id)
        self.obj_names.append(logical_name)   # "Banana" not "YcbBanana"
        self._pool_slots.append(pool_slot)
        self.obj_positions.append(list(pos))
        self.obj_orientations.append(list(orn))
        self.obj_grasps.append(None)
        self.obj_grasp_rects.append(None)
        return logical_id

    def load_isolated_obj(self, path_or_name, name: Optional[str] = None,
                          mod_orn: bool = False, mod_stiffness: bool = False,
                          pos=None, orn=None, **kwargs) -> int:
        """PyBullet-compatible load of a single isolated object.

        PyBullet signature: load_isolated_obj(path, name, mod_orn, mod_stiffness)
        MuJoCo signature:   load_isolated_obj(obj_name, pos, orn)
        Both are handled via the load_obj resolver above.

        Does NOT clear existing objects — callers that want a fresh scene should
        call remove_all_obj() before the spawn loop.
        """
        r_x = np.random.uniform(-0.15, 0.15)
        r_y = np.random.uniform(-0.35, -0.10)   # within arm y-limit (-0.4, 0.4)
        yaw = np.random.uniform(0, np.pi)
        if pos is None:
            # spawn above table so mesh doesn't penetrate; gravity settles it
            pos = [r_x, r_y, self.OBJECT_INIT_HEIGHT]
        return self.load_obj(path_or_name, name=name, pos=pos, orn=orn,
                             mod_orn=mod_orn, mod_stiffness=mod_stiffness,
                             yaw=yaw, **kwargs)

    def remove_obj(self, obj_id: int):
        idx = self._obj_pool_idx(obj_id)
        pool_slot = self._pool_slots[idx]
        self._set_obj_pose(pool_slot, [0, 0, -100])
        self.obj_ids.pop(idx)
        self.obj_names.pop(idx)
        self._pool_slots.pop(idx)
        self.obj_positions.pop(idx)
        self.obj_orientations.pop(idx)
        self.obj_grasps.pop(idx)
        self.obj_grasp_rects.pop(idx)

    def remove_all_obj(self):
        for slot in self._pool_slots:
            self._set_obj_pose(slot, [0, 0, -100])
        # also zero out any unreferenced pool slots
        for i in range(len(self._pool_names)):
            if i not in self._pool_slots:
                self._set_obj_pose(i, [0, 0, -100])
        self.obj_ids.clear()
        self.obj_names.clear()
        self._pool_slots.clear()
        self.obj_positions.clear()
        self.obj_orientations.clear()
        self.obj_grasps.clear()
        self.obj_grasp_rects.clear()

    def reset_all_obj(self):
        for slot, pos, orn in zip(self._pool_slots, self.obj_positions, self.obj_orientations):
            self._set_obj_pose(slot, pos, orn)
        self._steps(100)

    # ── grasp library ─────────────────────────────────────────────────────────

    def set_obj_grasps(self, obj_id: int, grasps, grasp_rects):
        idx = self._obj_pool_idx(obj_id)
        self.obj_grasps[idx]      = grasps
        self.obj_grasp_rects[idx] = grasp_rects

    def get_obj_grasps(self, obj_id: int):
        idx = self._obj_pool_idx(obj_id)
        return self.obj_grasps[idx]

    def get_obj_grasp_rects(self, obj_id: int):
        idx = self._obj_pool_idx(obj_id)
        return self.obj_grasp_rects[idx]

    # ── camera / observation ──────────────────────────────────────────────────

    def _render_rgb_depth_seg(self):
        """Render RGB, metric depth, and segmentation from the overhead camera.

        Returns (rgb, depth, seg) where seg pixel values are logical obj_ids
        (matching self.obj_ids), not MuJoCo internal geom IDs.
        Background pixels are 0.
        """
        # RGB — update_scene must be called per render mode in MuJoCo 3.x
        self._renderer.update_scene(self.data, camera="overhead")
        rgb = self._renderer.render().copy()

        # Depth — MuJoCo 3.x enable_depth_rendering() returns metric depth in metres
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera="overhead")
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()

        # Segmentation — re-update scene after enabling segmentation mode so that
        # geom IDs are encoded into the scene; without this, all pixels return -1
        self._renderer.enable_segmentation_rendering()
        self._renderer.update_scene(self.data, camera="overhead")
        seg_raw = self._renderer.render()
        self._renderer.disable_segmentation_rendering()
        geom_ids = seg_raw[:, :, 0].astype(np.int32)   # MuJoCo geom IDs per pixel

        # Remap geom IDs → logical obj_ids so that (seg == obj_id) works downstream
        seg = np.zeros_like(geom_ids)
        for list_pos, (oid, pool_slot) in enumerate(zip(self.obj_ids, self._pool_slots)):
            for geom_name in (f"ycb_vis_geom_{pool_slot}", f"ycb_col_geom_{pool_slot}"):
                try:
                    gid = self.model.geom(geom_name).id
                    seg[geom_ids == gid] = oid
                except Exception:
                    pass

        return rgb, depth, seg

    def _depth_to_pointcloud(self, depth: np.ndarray) -> np.ndarray:
        """Project depth image to 3D pointcloud in robot-base frame."""
        h, w = depth.shape
        fy = (h / 2) / np.tan(np.deg2rad(FOVY / 2))
        fx = fy
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        z = depth
        xc = (xs - w / 2) / fx * z
        yc = (ys - h / 2) / fy * z
        pc = np.stack([xc, -yc, -z], axis=-1).reshape(-1, 3)
        pc_hom = np.vstack([pc.T, np.ones((1, pc.shape[0]))])
        return (self.cam_to_robot_base @ pc_hom)[:3].T

    def get_obs(self, pointcloud: bool = True) -> dict:
        """Return {'image', 'depth', 'seg', 'obj_ids', 'points'} — matches env.py API.

        Aliases available on the returned dict:
          obs['image']  ↔  obs['rgb']
          obs['seg']    ↔  obs['segmentation']
        """
        rgb, depth, seg = self._render_rgb_depth_seg()

        # filter seg to known obj_ids only (matches PyBullet env.get_obs behaviour)
        known = set(self.obj_ids)
        mask = ~np.isin(seg, list(known))
        seg[mask] = 0

        result = {
            'image': rgb,
            'depth': depth,
            'seg':   seg,
            'obj_ids': np.array(self.obj_ids),
            'object_pose': self.get_object_pose(),
            # aliases
            'rgb':          rgb,
            'segmentation': seg,
        }
        if pointcloud:
            result['points'] = self._depth_to_pointcloud(depth)
        return result

    def get_points(self) -> np.ndarray:
        """Return 3D pointcloud in robot-base frame (standalone, no full obs)."""
        _, depth, _ = self._render_rgb_depth_seg()
        return self._depth_to_pointcloud(depth)

    # ── grasp execution (internal) ────────────────────────────────────────────

    def _execute_grasp(self, pos: tuple, roll: float,
                       gripper_opening_length: float,
                       obj_height: float) -> Tuple[bool, Optional[int]]:
        """Low-level grasp primitive. xyz-only motion; roll is accepted but unused.

        TODO(6dof-ik): pass roll to move_ee once orientation IK is available.
        Returns (success, grasped_obj_id_or_None).
        """
        self.reset_robot()  # start each attempt from a known arm pose

        x, y, z = pos
        z = np.clip(z, *self.ee_position_limit[2])

        orn = None
        opening = gripper_opening_length * self.GRIP_REDUCTION

        if self.vis:
            print(f"  [grasp] approach       → xy=({x:.3f}, {y:.3f})  z={self.GRIPPER_MOVING_HEIGHT:.3f}")
        self.move_gripper(opening)
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        if self.vis:
            print(f"  [grasp] descend        → xy=({x:.3f}, {y:.3f})  z={z:.3f}")
        self.move_ee([x, y, z, orn])

        if self.vis:
            print(f"  [grasp] close_gripper  (opening={opening:.3f})")
        self.auto_close_gripper(check_contact=False)
        self._steps(60)
        contact = bool(self.check_grasped_id())

        if self.vis:
            print(f"  [grasp] lift           → z={self.GRIPPER_MOVING_HEIGHT:.3f}")
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])
        self._steps(80)

        grasped_ids = self.check_grasped_id()
        obj_z       = (self.get_obj_pos(grasped_ids[0])[2]
                       if grasped_ids else self.Z_TABLE_TOP)
        lifted = obj_z > self.Z_TABLE_TOP + 0.07

        if self.vis:
            print(f"  [grasp] result         → contact={contact}  grasped={bool(grasped_ids)}  lifted={lifted}")

        # match eval criterion: contact before lift OR still grasped after OR object rose
        if contact or grasped_ids or lifted:
            if grasped_ids:
                return True, grasped_ids[0]
            # try to attribute to the nearest object
            if self.obj_ids:
                return True, self.obj_ids[0]
            return True, None
        return False, None

    # kept as public alias for backward compat
    def grasp(self, pos, roll, gripper_opening_length, obj_height,
              debug=False, vis=False):
        return self._execute_grasp(pos, roll, gripper_opening_length, obj_height)

    # ── pick_obj_by_id — full PyBullet-compatible logic ───────────────────────

    def pick_obj_by_id(self, obj_id: int,
                       vis: bool = False,
                       grasp_indices=None) -> Tuple[bool, Optional[int], Any]:
        """Pick object by id. Returns (success_grasp, grasped_obj_id, grasp_used).

        grasp_indices can be:
          None / []         → fallback: first N_GRASP_ATTEMPTS grasps
          [int, ...]        → integer indices into obj_grasps[obj_id]
          [Grasp2D/array, ...]  → grasp objects used directly (Case A)
        Mirrors the PyBullet pick_obj_by_id signature exactly.
        """
        stored_grasps = self.get_obj_grasps(obj_id)
        if stored_grasps is None or len(stored_grasps) == 0:
            print(f"[WARN] No grasps for obj {obj_id}")
            return False, None, None

        # ── resolve grasp list ────────────────────────────────────────────────
        if not grasp_indices:
            # None or empty list → use first K
            k = min(len(stored_grasps), self.N_GRASP_ATTEMPTS)
            grasps_to_try = list(stored_grasps[:k])
        elif not isinstance(grasp_indices[0], (int, np.integer)):
            # Case A: grasp_indices already contains grasp objects
            grasps_to_try = list(grasp_indices)
        else:
            # Case B: integer indices
            valid = [int(i) for i in grasp_indices
                     if isinstance(i, (int, np.integer)) and 0 <= int(i) < len(stored_grasps)]
            if not valid:
                k = min(len(stored_grasps), self.N_GRASP_ATTEMPTS)
                valid = list(range(k))
            grasps_to_try = [stored_grasps[i] for i in valid]

        # ── attempt each grasp ────────────────────────────────────────────────
        for j, g in enumerate(grasps_to_try):
            if j >= self.N_GRASP_ATTEMPTS:
                print(f"Exceeded {self.N_GRASP_ATTEMPTS} grasping attempts.")
                return False, obj_id, None

            # grasp tuple: (x, y, z, yaw, opening_len, obj_height, ...)
            x, y, z   = float(g[0]), float(g[1]), float(g[2])
            yaw        = float(g[3]) if len(g) > 3 else 0.0
            opening    = float(g[4]) if len(g) > 4 else 0.05
            obj_height = float(g[5]) if len(g) > 5 else 0.05

            success, grasped_id = self._execute_grasp((x, y, z), yaw, opening, obj_height)
            if success:
                if grasped_id != obj_id:
                    print(f"Grasped wrong object (id={grasped_id})")
                    return False, obj_id, g
                return True, obj_id, g

            print("Grasping failed. Retrying...")

        return False, None, None

    # ── placement helpers ─────────────────────────────────────────────────────

    def get_placement_points(self, obj_id: int) -> np.ndarray:
        """Return candidate table positions for re-placing an object."""
        xs = np.linspace(-0.3, 0.3, 4)
        ys = np.linspace(-0.7, -0.3, 4)
        pts = [[x, y, self.Z_TABLE_TOP] for x in xs for y in ys]
        return np.array(pts)

    def put_obj_in_loc(self, obj_id: int, target_pos,
                       vis: bool = False,
                       grasp_indices=None) -> Tuple[bool, bool]:
        """Pick object and place it at target_pos. Returns (success_grasp, success_target)."""
        success_grasp, grasped_id, grasp = self.pick_obj_by_id(
            obj_id, vis=vis, grasp_indices=grasp_indices)
        if not success_grasp:
            print("Grasping failed. Exiting.")
            return False, False

        # move to target
        x, y = target_pos[0], target_pos[1]
        z_drop = self.GRIPPER_MOVING_HEIGHT
        orn = None
        self.move_ee([x, y, z_drop, orn])
        self._steps(60)
        self.move_ee([x, y, self.Z_TABLE_TOP + 0.06, orn])
        self._steps(40)
        self.move_gripper(0.085)
        self._steps(40)
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        for _ in range(20):
            self.step_simulation()

        success_target = self.check_target_loc_reached(obj_id, target_pos)
        return success_grasp, success_target

    def put_obj_in_tray(self, obj_id: int,
                        debug: bool = False,
                        vis: bool = False,
                        grasp_indices=None) -> Tuple[bool, bool]:
        """Pick object and deliver it to the tray. Returns (success_grasp, success_target).
        Mirrors PyBullet put_obj_in_tray signature exactly."""
        success_grasp, grasped_id, grasp = self.pick_obj_by_id(
            obj_id, vis=vis, grasp_indices=grasp_indices)

        if not success_grasp:
            print("Grasping failed. Exiting.")
            self._log_ui_grasp_exec("tray", obj_id, None, False, False)
            return False, False

        # deliver to tray
        tp   = self.model.body("tray").pos
        orn  = None
        z_drop = tp[2] + 0.15
        self.move_ee([tp[0], tp[1], self.GRIPPER_MOVING_HEIGHT, orn])
        self._steps(60)
        self.move_ee([tp[0], tp[1], float(z_drop), orn])
        self._steps(40)
        self.move_gripper(0.085)
        self.move_ee([tp[0], tp[1], self.GRIPPER_MOVING_HEIGHT, orn])

        for _ in range(20):
            self.step_simulation()

        success_target = self.check_target_reached(grasped_id or obj_id)
        self._log_ui_grasp_exec("tray", obj_id, grasp, success_grasp, success_target)
        return success_grasp, success_target

    def put_obj_in_free_space(self, obj_id: int,
                              debug: bool = False,
                              vis: bool = False,
                              grasp_indices=None) -> Tuple[bool, bool]:
        """Pick object and place it in a free table spot (second-closest candidate).
        Mirrors PyBullet put_obj_in_free_space signature."""
        candidates = self.get_placement_points(obj_id)
        obj_pos    = self.get_obj_pos(obj_id)
        dists      = np.linalg.norm(candidates - obj_pos, axis=1)
        chosen     = candidates[np.argsort(dists)[1]]  # second-closest (avoids self)
        return self.put_obj_in_loc(obj_id, chosen, vis=vis, grasp_indices=grasp_indices)

    def place(self, target_loc: list, grasp: list, vis: bool = False):
        """Place currently-held object at target_loc (PyBullet compat)."""
        x, y = target_loc[0], target_loc[1]
        orn = None
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])
        self._steps(60)
        self.move_ee([x, y, self.Z_TABLE_TOP + 0.06, orn])
        self._steps(40)
        self.move_gripper(0.08)
        self._steps(40)

    # ── visualization stubs (PyBullet compat, no-ops in MuJoCo) ──────────────

    @staticmethod
    def draw_predicted_grasp(grasps, color=None, lineIDs=None):
        """No-op: PyBullet uses debug lines; MuJoCo viewer handles visualization."""
        return lineIDs if lineIDs is not None else []

    @staticmethod
    def remove_drawing(lineIDs):
        """No-op: PyBullet-specific debug line removal."""
        pass

    # ── logging ───────────────────────────────────────────────────────────────

    def _log_ui_grasp_exec(self, mode, obj_id, grasp, success_grasp, success_target):
        os.makedirs(_LOG_DIR, exist_ok=True)
        if grasp is None:
            x = y = z = yaw = opening = height = None
        else:
            x, y, z = float(grasp[0]), float(grasp[1]), float(grasp[2])
            yaw     = float(grasp[3]) if len(grasp) > 3 else None
            opening = float(grasp[4]) if len(grasp) > 4 else None
            height  = float(grasp[5]) if len(grasp) > 5 else None
        record = dict(
            time=time.strftime("%Y-%m-%d %H:%M:%S"),
            mode=mode, obj_id=int(obj_id),
            x=x, y=y, z=z, yaw=yaw,
            opening_len=opening, obj_height=height,
            success_grasp=bool(success_grasp),
            success_target=bool(success_target),
        )
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ── misc utils ────────────────────────────────────────────────────────────

    @staticmethod
    def get_transform_matrix(x, y, z, rot):
        return _get_transform_matrix(x, y, z, rot)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
        if self._renderer is not None:
            self._renderer.close()


# ── Module-level aliases ──────────────────────────────────────────────────────
Environment     = EnvironmentSoArm   # PyBullet-compatible name
MujocoGraspEnv  = EnvironmentSoArm  # mujoco_validator expected name
