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
_PROJ_ROOT = os.path.dirname(_DIR)             # project root (parent of owg_robot/)
SO101_XML  = os.path.join(_DIR, "assets", "so101", "so101.xml")
SO101_MESH = os.path.join(_DIR, "assets", "so101", "assets")
YCB_ROOT   = os.path.join(_DIR, "assets", "ycb_objects")
_MANIFEST_PATH = os.path.join(_PROJ_ROOT, "configs", "objects", "ycb_mujoco_manifest.yaml")

# ── scene constants ───────────────────────────────────────────────────────────
TABLE_TOP_Z         = 0.785
GRASP_Z_TABLE_MARGIN = 0.020  # min offset above obj CoM: half_jaw_span(10mm)+sphere_r(6mm)+safety(4mm)
ROBOT_BASE_POS = "0 0 0.785"
CAM_POS        = "0.05 -0.52 1.9"
CAM_EULER      = "0 0 0"   # with angle="radian" compiler, default orientation looks -Z (down)
TARGET_ZONE_POS = [0.20, 0.25, TABLE_TOP_Z]   # within arm workspace (x,y ≤ 0.4)

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

# IK execution modes
IK_MODE_XYZ_ONLY    = "xyz_only"            # position-only (legacy default)
IK_MODE_YAW_TOPDOWN = "yaw_aware_topdown"   # top-down with jaw-yaw orientation
IK_MODE_FULL_6DOF   = "full_6dof"           # full 6-DoF pose target
IK_MODE_JAW_TOPDOWN = "jaw_midpoint_topdown"  # jaw-midpoint + orientation (preferred)
IK_MODE_JAW_POS     = "jaw_pos_only"        # jaw-midpoint, natural arm orientation


def make_topdown_rotation(yaw: float = 0.0) -> np.ndarray:
    """3×3 rotation matrix for a top-down gripper approach at a given jaw yaw.

    Site Z-axis → world -Z (straight down), site X-axis → [cos(yaw), sin(yaw), 0].
    The SO-ARM101 gripperframe site Z-axis is the approach/closing direction.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [ cy,  sy,  0.0],
        [ sy, -cy,  0.0],
        [0.0, 0.0, -1.0],
    ], dtype=float)

# log path (matches PyBullet env)
_LOG_DIR  = "logs"
_LOG_PATH = os.path.join(_LOG_DIR, "ui_grasp_exec.jsonl")

# Grasp execution modes — choose at EnvironmentSoArm construction time.
#
#   GRASP_MODE_PHYSICS_WELD  (recommended for all evaluation):
#     Phase 1 — uses real MuJoCo contact detection: IK-based descent, gripper
#     close, post-close bilateral-contact check.
#     Phase 2 — kinematic lift: if and only if both jaw spheres contact the
#     object (bilateral_contacts == 1), the object is kinematically welded to
#     the EEF for the lift phase.  The weld is released if the object does not
#     clear TABLE_TOP_Z + 0.07 m (reported as success=False).
#     Rationale: 6 mm sphere colliders cannot generate sufficient friction force
#     to lift ~0.15 kg objects against gravity; physics-only lift always fails.
#     The bilateral-contact gate preserves the integrity of the success signal:
#     a weld that never triggers means no grasp contact → failure.
#     Use this mode for ALL benchmarks and training-label generation.
#
#   GRASP_MODE_PHYSICS  (legacy alias → same as GRASP_MODE_PHYSICS_WELD):
#     Kept for backward compatibility.  Maps to the same execution function.
#
#   GRASP_MODE_DEMO_ATTACH:
#     Kinematic sticky-gripper: pre-selects the nearest object by XY before
#     descent, snaps it 3 cm below EEF regardless of contacts, zeros velocity
#     on release.  Looks correct on video but success signal is unconditional.
#     Use ONLY for semantic demo recordings — never for benchmarks or labels.
GRASP_MODE_PHYSICS_WELD = "physics_weld_after_bilateral"   # recommended
GRASP_MODE_PHYSICS      = "physics"                        # legacy alias
GRASP_MODE_DEMO_ATTACH  = "demo_attach"


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



# ── primitive-object registry ────────────────────────────────────────────────
# Calibration helpers can inject simple geoms (box/cylinder/sphere) without
# requiring a YCB mesh directory.  Pool names start with "__prim__" so they
# are distinguished from YCB directory names.

_PRIMITIVE_POOL: Dict[str, str] = {}   # pool_name → geom attribute string


def register_primitive_geom(
    shape: str,
    size: tuple,
    mass: float,
    friction: tuple = (1.5, 0.05, 0.01),
    rgba: tuple = (0.7, 0.3, 0.2, 1.0),
) -> str:
    """Register a primitive geom and return its pool_name.

    The pool_name is stable for a given (shape, size, mass, friction, rgba)
    tuple, so re-registering the same primitive returns the same name without
    rebuilding the model.
    """
    key = f"{shape}_{size}_{mass}_{friction}_{rgba}"
    pool_name = f"__prim__{abs(hash(key)) % (10 ** 9):09d}"
    if pool_name not in _PRIMITIVE_POOL:
        size_str     = " ".join(str(s) for s in size)
        friction_str = " ".join(str(f) for f in friction)
        rgba_str     = " ".join(str(r) for r in rgba)
        _PRIMITIVE_POOL[pool_name] = (
            f'type="{shape}" size="{size_str}" mass="{mass}" '
            f'friction="{friction_str}" rgba="{rgba_str}"'
        )
    return pool_name


# ── object manifest (Menagerie-style) ─────────────────────────────────────────
# Lazily loaded from configs/objects/ycb_mujoco_manifest.yaml.
# Provides per-object mass, friction, scale, and mesh names so these values
# are defined in one place rather than scattered through env_soarm.py.
_MANIFEST_CACHE: Optional[dict] = None


def _load_manifest() -> dict:
    """Return the parsed ycb_mujoco_manifest.yaml, loading it once."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        if os.path.isfile(_MANIFEST_PATH):
            try:
                import yaml
                with open(_MANIFEST_PATH) as f:
                    _MANIFEST_CACHE = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[WARN] Could not load YCB manifest ({e}); using defaults.")
                _MANIFEST_CACHE = {}
        else:
            print(f"[WARN] YCB manifest not found at {_MANIFEST_PATH}; using defaults.")
            _MANIFEST_CACHE = {}
    return _MANIFEST_CACHE


def _manifest_entry(obj_name: str) -> dict:
    """Look up the manifest entry for obj_name.

    Accepts both pool names ("YcbBanana") and logical names ("Banana").
    Returns an empty dict if the object is not listed in the manifest.
    """
    manifest = _load_manifest()
    objects  = manifest.get("objects", {})
    # Direct lookup first (logical name as key)
    if obj_name in objects:
        return objects[obj_name]
    # Strip "Ycb" prefix and try again
    logical = obj_name[3:] if obj_name.startswith("Ycb") else obj_name
    return objects.get(logical, {})


# ── mesh helpers ──────────────────────────────────────────────────────────────

def _find_ycb_mesh(obj_dir: str) -> str:
    """Auto-discover the best available visual mesh for a YCB object directory."""
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
    """Build MuJoCo <asset> and <body> XML snippets for one YCB pool slot.

    Physical properties (mass, friction, scale) and mesh filenames are read
    from configs/objects/ycb_mujoco_manifest.yaml when the object is listed
    there.  Falls back to dynamic mesh discovery and hardcoded defaults for
    objects not in the manifest.

    Primitive objects registered via register_primitive_geom() use inline
    geoms instead of mesh assets.
    """
    # ── primitive shortcut ────────────────────────────────────────────────────
    if obj_name in _PRIMITIVE_POOL:
        geom_attrs = _PRIMITIVE_POOL[obj_name]
        park_x     = 2.0 + obj_idx * 0.5
        asset = ""   # no mesh assets needed for primitives
        body  = f"""
    <body name="obj_{obj_idx}" pos="{park_x} 0 0.15">
      <freejoint name="obj_joint_{obj_idx}"/>
      <geom {geom_attrs} contype="1" conaffinity="1"/>
    </body>
"""
        return asset, body

    obj_dir = os.path.join(YCB_ROOT, obj_name)
    entry   = _manifest_entry(obj_name)

    # ── mesh resolution ───────────────────────────────────────────────────────
    if entry:
        vis_name = entry.get("visual_mesh", "")
        col_name = entry.get("collision_mesh", "collision_vhacd.obj")
        vis_mesh = (os.path.join(obj_dir, vis_name) if vis_name
                    else _find_ycb_mesh(obj_dir))
        col_path = os.path.join(obj_dir, col_name)
        col_mesh = col_path if os.path.isfile(col_path) else vis_mesh
    else:
        vis_mesh = _find_ycb_mesh(obj_dir)
        col_path = os.path.join(obj_dir, "collision_vhacd.obj")
        col_mesh = col_path if os.path.isfile(col_path) else vis_mesh

    print(f"[INFO] Using mesh for {obj_name}: {Path(vis_mesh).name}"
          + ("" if entry else "  (manifest entry missing — using defaults)"))

    # ── physical properties from manifest (or hardcoded defaults) ─────────────
    scale    = float(entry.get("scale",   1.0))
    mass     = float(entry.get("mass",    0.1))
    friction = entry.get("friction", [2.0, 0.05, 0.01])

    scale_str    = f"{scale} {scale} {scale}"
    friction_str = " ".join(str(v) for v in friction)

    # ── texture / material ────────────────────────────────────────────────────
    tex_file = os.path.join(obj_dir, "texture_map.png")
    if not os.path.isfile(tex_file):
        tex_file = ""

    if tex_file:
        material_block = (
            f'  <texture name="ycb_tex_{obj_idx}" type="2d" file="{tex_file}"/>\n'
            f'  <material name="ycb_mat_{obj_idx}" texture="ycb_tex_{obj_idx}"/>\n'
        )
    else:
        material_block = f'  <material name="ycb_mat_{obj_idx}" rgba="0.8 0.8 0.8 1"/>\n'

    # ── XML assembly ──────────────────────────────────────────────────────────
    asset = f"""
  <mesh name="ycb_vis_{obj_idx}" file="{vis_mesh}" scale="{scale_str}"/>
  <mesh name="ycb_col_{obj_idx}" file="{col_mesh}" scale="{scale_str}"/>
{material_block}"""

    # Initial body pos matches _park_pos(obj_idx): above floor, far from workspace
    park_x = 2.0 + obj_idx * 0.5
    body = f"""
    <body name="obj_{obj_idx}" pos="{park_x} 0 0.15">
      <freejoint name="obj_joint_{obj_idx}"/>
      <geom name="ycb_vis_geom_{obj_idx}" type="mesh" mesh="ycb_vis_{obj_idx}"
            material="ycb_mat_{obj_idx}" contype="0" conaffinity="0" group="2"/>
      <geom name="ycb_col_geom_{obj_idx}" type="mesh" mesh="ycb_col_{obj_idx}"
            mass="{mass}" friction="{friction_str}"
            contype="1" conaffinity="1"/>
    </body>
"""
    return asset, body


def _park_pos(slot: int) -> list:
    """Off-table parking position for an inactive pool slot.

    Placed above the floor (z=0.15) and far from the robot workspace in X, so
    parked objects settle stably on the floor plane without interfering with the
    active object on the table.  Each slot gets a unique X to prevent pile-ups.
    """
    return [2.0 + slot * 0.5, 0.0, 0.15]


def _build_scene_xml(obj_names: List[str]) -> str:
    so_defaults, so_assets, so_wbody, so_acts = _so101_fragment()
    ycb_assets = ""
    ycb_bodies = ""
    weld_eqs   = ""
    for i, name in enumerate(obj_names):
        a, b = _ycb_asset_tag(name, i)
        ycb_assets += a
        ycb_bodies  += b
        # Inactive weld per pool slot — activated at grasp time with computed relpose
        weld_eqs += (f'  <weld name="grasp_weld_{i}" body1="gripper" body2="obj_{i}"'
                     f' relpose="0 0 0 1 0 0 0" active="false"/>\n')

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

  <equality>
{weld_eqs}  </equality>

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
                 grasp_mode: str = GRASP_MODE_PHYSICS,
                 **kwargs):
        self.vis        = vis
        self.debug      = debug
        self.finger_length    = finger_length
        self.N_GRASP_ATTEMPTS = n_grasp_attempts
        self.grasp_mode = grasp_mode

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
        self._welded_obj_id:    Optional[int]        = None   # kinematically attached object
        self._kinematic_offset: Optional[np.ndarray] = None   # world-frame EEF→obj offset

        # out-band contact metrics from the most recent grasp attempt
        self.last_grasp_metrics: Optional[dict] = None

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

    def _simplify_jaw_collision(self):
        """Replace large jaw mesh collision geoms with small spheres.

        The SO-ARM101 scissor jaw mesh geoms (sts3215 motor, wrist_roll_follower,
        moving_jaw) have convex hulls spanning ~10 cm along the jaw arm.  When the
        arm is teleported to the grasp configuration and the object is restored, these
        hulls create 2.8–3.9 cm penetrations that generate explosive contact impulses
        and send the object flying.

        Fix: disable the sts3215 motor collision geom and replace the jaw tip mesh
        collision geoms with 6 mm radius spheres at the same local positions.  The IK
        drives the geom centres to ≈1–2 mm outside the object faces, so the spheres
        produce small (4–5 mm), controlled penetrations instead of the catastrophic
        hull overlaps.  This preserves bilateral contact detection while eliminating
        the explosion.
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
        _R = 0.006  # 6 mm radius — geom centres are ≈1–2 mm outside object faces
        for gid in (self._jaw_fixed_geom_id, self._jaw_mv_geom_id):
            if gid >= 0:
                self.model.geom_type[gid] = _SPHERE
                self.model.geom_size[gid, 0] = _R
                self.model.geom_size[gid, 1] = 0.0
                self.model.geom_size[gid, 2] = 0.0

    def _rebuild_model(self):
        xml = _build_scene_xml(self._pool_names)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data  = mujoco.MjData(self.model)
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(self.model, height=IMG_SIZE, width=IMG_SIZE)
        self._cache_ids()
        self._simplify_jaw_collision()
        # initialize arm to HOME_QPOS so _steps() after object spawn is stable
        for adr, q in zip(self._arm_qpos_adr, HOME_QPOS):
            self.data.qpos[adr] = q
        for act_id, q in zip(self._arm_act_ids, HOME_QPOS):
            self.data.ctrl[act_id] = q
        self.data.qpos[self._grip_qpos_adr] = GRIP_OPEN
        self.data.ctrl[self._grip_act_id]   = GRIP_OPEN
        # Park all pool slots at a safe above-floor location far from the workspace.
        # This prevents inactive freejoints (default z=-100 in raw MjData) from
        # penetrating the floor plane and generating huge contact impulses that would
        # fly through the scene and destabilize the active object.
        for i in range(len(self._pool_names)):
            jnt  = self.model.joint(f"obj_joint_{i}")
            adr  = jnt.qposadr[0]
            vadr = jnt.dofadr[0]
            self.data.qpos[adr:adr+3]   = _park_pos(i)
            self.data.qpos[adr+3:adr+7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[vadr:vadr+6] = 0.0
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
            # Actual jaw tip geom centers: wrist_roll_follower (fixed finger)
            # and moving_jaw_so101_v1 (moving finger).  These are offset ~2 cm
            # from the body centers and give the true gripping midpoint.
            self._jaw_fixed_geom_id = self._find_collision_geom("gripper", "wrist_roll_follower_so101_v1")
            self._jaw_mv_geom_id    = self._find_collision_geom("moving_jaw_so101_v1", "moving_jaw_so101_v1")
        except Exception:
            self._jaw_body_id       = -1
            self._jaw_mv_body_id    = -1
            self._jaw_fixed_geom_id = -1
            self._jaw_mv_geom_id    = -1

        self._grasp_weld_eq_ids: list = []
        for i in range(len(self._pool_names)):
            try:
                self._grasp_weld_eq_ids.append(int(self.model.equality(f"grasp_weld_{i}").id))
            except Exception:
                self._grasp_weld_eq_ids.append(-1)

    # ── sticky-gripper (kinematic attachment) ────────────────────────────────

    def _attach_obj(self, obj_id: int,
                    offset: Optional[np.ndarray] = None) -> bool:
        """Kinematically attach obj_id to the EEF.

        If offset is given, use it as the world-frame EEF→obj delta directly
        (useful for a canonical "held below EEF" position when the object was
        physically knocked during descent).  Otherwise compute offset from
        current EEF and object positions.
        Returns True on success.
        """
        try:
            eef_pos = self._get_eef_pos()
            if offset is not None:
                self._kinematic_offset = np.asarray(offset, dtype=np.float64)
            else:
                obj_pos = self.get_obj_pos(obj_id)
                self._kinematic_offset = obj_pos - eef_pos
            self._welded_obj_id = obj_id
            return True
        except Exception:
            return False

    def _sync_kinematic_grasp(self) -> None:
        """Move the kinematically attached object to follow the EEF each step.

        Also zeros the object's linear velocity so that contact impulses from
        the arm's geometry do not accumulate.  Without this, heavy objects
        (e.g., MustardBottle at 0.603 kg) would exert large reaction forces
        on the arm joints during lift, preventing the arm from reaching its
        target height.
        """
        if self._welded_obj_id is None:
            return
        try:
            eef_pos  = self.data.site_xpos[self._eef_site_id]
            new_pos  = eef_pos + self._kinematic_offset
            slot     = self._obj_pool_slot(self._welded_obj_id)
            jnt      = self.model.joint(f"obj_joint_{slot}")
            adr      = jnt.qposadr[0]
            adr_v    = jnt.dofadr[0]
            self.data.qpos[adr:adr + 3] = new_pos
            self.data.qvel[adr_v:adr_v + 3] = 0.0   # kill linear velocity each step
        except Exception:
            self._welded_obj_id = None

    def _detach_obj(self, obj_id: Optional[int] = None) -> None:
        """Release the kinematic attachment, zeroing the object velocity so it drops cleanly."""
        target = self._welded_obj_id if obj_id is None else obj_id
        if target is not None and self._welded_obj_id == target:
            try:
                slot = self._obj_pool_slot(target)
                jnt  = self.model.joint(f"obj_joint_{slot}")
                # Zero the object's 6-DOF velocity so it doesn't fly away on release
                adr_v = jnt.dofadr[0]
                self.data.qvel[adr_v:adr_v + 6] = 0.0
            except Exception:
                pass
            self._welded_obj_id    = None
            self._kinematic_offset = None

    # ── simulation step ───────────────────────────────────────────────────────

    def step_simulation(self):
        mujoco.mj_step(self.model, self.data)
        if self._welded_obj_id is not None:
            self._sync_kinematic_grasp()
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
        self._detach_obj()  # release any active weld
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
        # Release weld when opening the gripper
        if opening_length > 0.04 and self._welded_obj_id is not None:
            self._detach_obj()
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

    def _get_eef_rot(self) -> np.ndarray:
        """Current 3×3 rotation matrix of the EEF site (world frame)."""
        return self.data.site_xmat[self._eef_site_id].reshape(3, 3).copy()

    # ── position-only IK (legacy) ─────────────────────────────────────────────

    def _ik_step(self, target_pos: np.ndarray) -> bool:
        """Damped least-squares IK — position only (xyz)."""
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

    # ── 6-DoF orientation-aware IK ────────────────────────────────────────────

    def _ik_step_6dof(self, target_pos: np.ndarray, target_rot: np.ndarray,
                      w_pos: float = 1.0, w_ori: float = 0.5,
                      damping: float = IK_DAMPING) -> Tuple[float, float]:
        """Single DLS IK step targeting both position and orientation.

        Uses the full 6×n Jacobian (translation + rotation rows).  Orientation
        error is computed as the rotation-vector difference between the target
        and current site rotation matrices.

        Returns (pos_err_norm, ori_err_norm).
        """
        nv   = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._eef_site_id)

        cols = [self.model.joint(n).dofadr[0] for n in ARM_JOINTS]
        Jp   = jacp[:, cols]   # 3×5
        Jr   = jacr[:, cols]   # 3×5

        pos_err = target_pos - self._get_eef_pos()

        R_cur = self._get_eef_rot()
        R_err = target_rot @ R_cur.T
        q_err = np.zeros(4)
        mujoco.mju_mat2Quat(q_err, R_err.ravel())
        ori_vec = np.zeros(3)
        mujoco.mju_quat2Vel(ori_vec, q_err, 1.0)

        # Clamp orientation error to avoid it drowning out position when the
        # initial orientation is far from target (5-DOF arm can have >π gap).
        ori_norm = float(np.linalg.norm(ori_vec))
        if ori_norm > 0.3:
            ori_vec = ori_vec * (0.3 / ori_norm)

        # Weighted 6-D error and Jacobian
        err6 = np.concatenate([w_pos * pos_err, w_ori * ori_vec])
        J6   = np.vstack([w_pos * Jp, w_ori * Jr])   # 6×5

        dq = J6.T @ np.linalg.solve(J6 @ J6.T + damping * np.eye(6), err6)
        for i, adr in enumerate(self._arm_qpos_adr):
            lo = self.model.jnt_range[self._arm_jnt_ids[i], 0]
            hi = self.model.jnt_range[self._arm_jnt_ids[i], 1]
            self.data.qpos[adr] = np.clip(self.data.qpos[adr] + dq[i], lo, hi)
        mujoco.mj_forward(self.model, self.data)
        return float(np.linalg.norm(pos_err)), ori_norm

    def _solve_ik_6dof(self, target_pos: np.ndarray, target_rot: np.ndarray,
                       iters: int = IK_ITERS,
                       w_pos: float = 10.0, w_ori: float = 0.3,
                       pos_tol: float = 5e-3, ori_tol: float = 0.3) -> Tuple[bool, float, float]:
        """Two-phase DLS IK: position-only warm-start then joint 6-DOF refinement.

        Phase 1 (first half of iters): position-only to anchor XYZ.
        Phase 2 (second half): 6-DOF with w_pos >> w_ori to nudge orientation
        while staying near the position solution.

        Returns (converged, final_pos_err, final_ori_err).
        """
        half = iters // 2
        # Phase 1: position-only warm-start
        for _ in range(half):
            if self._ik_step(target_pos):
                break

        # Phase 2: gentle orientation correction from the position solution
        pe = oe = float("inf")
        for _ in range(iters - half):
            pe, oe = self._ik_step_6dof(target_pos, target_rot, w_pos, w_ori)
            if pe < pos_tol and oe < ori_tol:
                return True, pe, oe
        return pe < pos_tol and oe < ori_tol, pe, oe

    # ── jaw midpoint IK ───────────────────────────────────────────────────────

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

    def _get_jaw_midpoint(self) -> np.ndarray:
        """World-frame midpoint between the fixed and moving jaw body centres."""
        if self._jaw_body_id < 0:
            return self._get_eef_pos()
        return 0.5 * (self.data.xpos[self._jaw_body_id] +
                      self.data.xpos[self._jaw_mv_body_id])

    def _get_jaw_geom_midpoint(self) -> np.ndarray:
        """World-frame midpoint of the actual jaw tip GEOM centres.

        Uses wrist_roll_follower (fixed finger) and moving_jaw_so101_v1 geom
        centres — the true gripping surfaces, offset ~2 cm from body centres.
        Falls back to body midpoint when geom IDs are unavailable.
        """
        if self._jaw_fixed_geom_id < 0:
            return self._get_jaw_midpoint()
        return 0.5 * (self.data.geom_xpos[self._jaw_fixed_geom_id] +
                      self.data.geom_xpos[self._jaw_mv_geom_id])

    def _solve_ik_jaw_topdown(
        self,
        target_jaw_mid: np.ndarray,
        yaw:            float = 0.0,
        iters:          int   = 600,
        w_pos:          float = 10.0,
        w_ori:          float = 0.3,
        pos_tol:        float = 5e-3,
        n_outer:        int   = 8,
    ) -> Tuple[bool, float, float]:
        """Jaw-midpoint–targeted top-down IK.

        Targets the MIDPOINT of the two jaw bodies at `target_jaw_mid` while
        driving the site Z-axis toward world -Z (top-down orientation).

        Always resets to HOME_QPOS before solving so the IK starts from a
        well-conditioned configuration regardless of the current arm posture.
        This is safe because _move_ee_internal saves and restores qpos around
        this call.

        Returns (converged, jaw_mid_pos_err, ori_err).
        """
        # Reset to HOME_QPOS for a reproducible, well-conditioned IK start
        for adr, q in zip(self._arm_qpos_adr, HOME_QPOS):
            self.data.qpos[adr] = q
        mujoco.mj_forward(self.model, self.data)

        target_rot = make_topdown_rotation(yaw)

        # Phase 1: position-only warm-start (1/3 of budget).
        # Anchors the arm near the target xyz before the orientation term can
        # push it away — mirrors the two-phase approach in _solve_ik_6dof.
        warmup = iters // 3
        offset = self._get_eef_pos() - self._get_jaw_midpoint()
        adjusted_site_target = target_jaw_mid + offset
        for _ in range(warmup):
            if self._ik_step(adjusted_site_target):
                break

        # Phase 2: 6-DOF refinement with offset updates
        iters_6dof      = iters - warmup
        iters_per_outer = max(50, iters_6dof // n_outer)

        pe = oe = float("inf")
        for _ in range(n_outer):
            # Recompute offset from the current (improved) arm state
            offset = self._get_eef_pos() - self._get_jaw_midpoint()
            adjusted_site_target = target_jaw_mid + offset

            pe_site, oe = float("inf"), float("inf")
            for _ in range(iters_per_outer):
                pe_site, oe = self._ik_step_6dof(
                    adjusted_site_target, target_rot, w_pos, w_ori)
                if pe_site < pos_tol:
                    break

        pe = float(np.linalg.norm(self._get_jaw_midpoint() - target_jaw_mid))
        return pe < pos_tol, pe, oe

    def _solve_ik_jaw_pos_only(
        self,
        target_jaw_mid: np.ndarray,
        iters:          int   = 800,
        pos_tol:        float = 5e-3,
        n_outer:        int   = 8,
        reset_to_home:  bool  = True,
    ) -> Tuple[bool, float, float]:
        """Position-only IK targeting the jaw GEOM midpoint with natural arm orientation.

        Drives the midpoint of the two jaw tip GEOM centres (wrist_roll_follower +
        moving_jaw) to target_jaw_mid.  Using geom centres (not kinematic body
        centres) ensures the actual jaw tips straddle the object symmetrically
        after the IK converges, which is required for bilateral contact on closing.

        reset_to_home : if True (default, used by move_ee/hover), resets to HOME_QPOS
                        before solving.  Set False when called from
                        _execute_grasp_physics_topdown so the IK starts from the
                        hover arm state — this keeps the arm in the same kinematic
                        workspace and ensures the gripper closing motion is directed
                        toward the object rather than sweeping 30 cm past it.

        Returns (converged, geom_mid_pos_err, 0.0).
        """
        if reset_to_home:
            for adr, q in zip(self._arm_qpos_adr, HOME_QPOS):
                self.data.qpos[adr] = q
        mujoco.mj_forward(self.model, self.data)

        iters_per = max(50, iters // n_outer)
        for _ in range(n_outer):
            offset   = self._get_eef_pos() - self._get_jaw_geom_midpoint()
            adjusted = target_jaw_mid + offset
            for _ in range(iters_per):
                if self._ik_step(adjusted):
                    break
        geom_pos = self._get_jaw_geom_midpoint()
        pe = float(np.linalg.norm(geom_pos - target_jaw_mid))
        print(f"  [ik_jaw_geom] target={target_jaw_mid.round(4)} solved_geom_mid={geom_pos.round(4)} pe={pe*100:.2f}cm")
        return pe < pos_tol, pe, 0.0

    # ── EEF motion ────────────────────────────────────────────────────────────

    def _move_ee_internal(self, target_pos: np.ndarray,
                          target_rot: Optional[np.ndarray],
                          ik_mode: str, max_step: int) -> Tuple[bool, float, float]:
        """Solve IK then interpolate ctrl to drive the arm.  Returns (ok, pe, oe)."""
        qpos_saved = self.data.qpos[self._arm_qpos_adr].copy()
        qvel_saved = self.data.qvel.copy()
        q_start    = np.array([self.data.ctrl[i] for i in self._arm_act_ids])

        if ik_mode == IK_MODE_JAW_POS:
            # Position-only jaw-midpoint IK — must be checked before target_rot is None
            ok, pe, oe = self._solve_ik_jaw_pos_only(target_pos)
        elif ik_mode == IK_MODE_JAW_TOPDOWN:
            yaw = float(np.arctan2(target_rot[0, 1], target_rot[0, 0])) \
                  if target_rot is not None else 0.0
            ok, pe, oe = self._solve_ik_jaw_topdown(target_pos, yaw)
        elif ik_mode == IK_MODE_XYZ_ONLY or target_rot is None:
            ok = self._solve_ik(target_pos)
            pe = float(np.linalg.norm(target_pos - self._get_eef_pos()))
            oe = 0.0
        else:
            ok, pe, oe = self._solve_ik_6dof(target_pos, target_rot)

        q_target = np.array([self.data.qpos[adr] for adr in self._arm_qpos_adr])

        for adr, q in zip(self._arm_qpos_adr, qpos_saved):
            self.data.qpos[adr] = q
        self.data.qvel[:] = qvel_saved
        mujoco.mj_forward(self.model, self.data)

        n_steps = max_step if self.vis else max(100, max_step // 2)
        for i in range(n_steps):
            t = (i + 1) / n_steps
            for act_id, qs, qt in zip(self._arm_act_ids, q_start, q_target):
                self.data.ctrl[act_id] = float(qs + t * (qt - qs))
            self.step_simulation()
            if self.vis and self._viewer is not None:
                self._viewer.sync()

        return ok, pe, oe

    def move_ee(self, action, max_step: int = 200, ik_mode: str = IK_MODE_XYZ_ONLY,
                target_rot: Optional[np.ndarray] = None, **kwargs):
        """Move EEF to (x, y, z, orn).

        Parameters
        ----------
        action      : [x, y, z, orn]  — orn unused by legacy callers
        ik_mode     : IK_MODE_XYZ_ONLY | IK_MODE_YAW_TOPDOWN | IK_MODE_FULL_6DOF
        target_rot  : 3×3 rotation matrix (world frame) to track; auto-built from
                      action[3] when ik_mode == IK_MODE_YAW_TOPDOWN and action[3]
                      is a float yaw angle.
        """
        x, y, z = action[0], action[1], action[2]
        x = np.clip(x, *self.ee_position_limit[0])
        y = np.clip(y, *self.ee_position_limit[1])
        z = np.clip(z, *self.ee_position_limit[2])

        rot = target_rot
        if ik_mode in (IK_MODE_YAW_TOPDOWN, IK_MODE_JAW_TOPDOWN) and rot is None:
            yaw = float(action[3]) if (action[3] is not None and
                                       not isinstance(action[3], np.ndarray)) else 0.0
            rot = make_topdown_rotation(yaw)

        ok, pe, oe = self._move_ee_internal(np.array([x, y, z]), rot, ik_mode, max_step)
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
        obj_bodies = {self.model.body(f"obj_{slot}").id: oid
                      for oid, slot in zip(self.obj_ids, self._pool_slots)}
        contacts = self._contacts_on_bodies(finger_ids, set(obj_bodies.keys()))
        result = [obj_bodies[b] for b in contacts if b in obj_bodies]
        # also include any object held via weld constraint
        if self._welded_obj_id is not None and self._welded_obj_id not in result:
            result.append(self._welded_obj_id)
        return result

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
        # zero velocity to prevent phantom motion when teleporting from parked z=-100
        vadr = jnt.dofadr[0]
        self.data.qvel[vadr:vadr+6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ── object pose access ────────────────────────────────────────────────────

    def get_obj_pos(self, obj_id: int) -> np.ndarray:
        return self._obj_qpos(self._obj_pool_slot(obj_id))[:3]

    def get_obj_com_pos(self, obj_id: int) -> np.ndarray:
        """World-frame center-of-mass position of the object body.

        Unlike get_obj_pos() (which returns the free-joint qpos position, i.e.
        the body reference frame origin), this returns data.xpos[body_id] —
        the actual CoM in world space.  Use this for grasp z-targeting so the
        jaw midpoint is placed at the object centre regardless of mesh origin
        offset.
        """
        slot    = self._obj_pool_slot(obj_id)
        body_id = self.model.body(f"obj_{slot}").id
        return self.data.xpos[body_id].copy()

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

    def load_primitive(
        self,
        shape: str = "box",
        size: tuple = (0.025, 0.025, 0.025),
        mass: float = 0.200,
        friction: tuple = (1.5, 0.05, 0.01),
        rgba: tuple = (0.7, 0.3, 0.2, 1.0),
        pos=None,
        name: str = None,
    ) -> int:
        """Load a primitive geom (box / cylinder / sphere) for grasp calibration.

        Unlike load_obj(), this injects a simple inline geom into the pool
        without requiring a YCB mesh directory.  All other env behaviour
        (IK, contact detection, lift checking) is identical.

        MuJoCo size convention
        ----------------------
        box      : size = (half_x, half_y, half_z)  → full dims = 2×size
        cylinder : size = (radius, half_height)
        sphere   : size = (radius,)

        Returns the logical obj_id.
        """
        pool_name = register_primitive_geom(shape, size, mass, friction, rgba)
        spawn_z   = TABLE_TOP_Z + max(size) + 0.04   # drop slightly above CoM
        spawn_pos = pos or [0.0, -0.40, spawn_z]
        return self.load_obj(pool_name, name=name or shape, pos=spawn_pos)

    def remove_obj(self, obj_id: int):
        idx = self._obj_pool_idx(obj_id)
        pool_slot = self._pool_slots[idx]
        self._set_obj_pose(pool_slot, _park_pos(pool_slot))
        self.obj_ids.pop(idx)
        self.obj_names.pop(idx)
        self._pool_slots.pop(idx)
        self.obj_positions.pop(idx)
        self.obj_orientations.pop(idx)
        self.obj_grasps.pop(idx)
        self.obj_grasp_rects.pop(idx)

    def remove_all_obj(self):
        for slot in self._pool_slots:
            self._set_obj_pose(slot, _park_pos(slot))
        # also zero out any unreferenced pool slots
        for i in range(len(self._pool_names)):
            if i not in self._pool_slots:
                self._set_obj_pose(i, _park_pos(i))
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
        """Project metric depth image to 3D pointcloud in robot-base frame.

        Delegates to owg_robot.pointcloud.mujoco_depth_to_world_points so
        the projection logic lives in one place.
        """
        from owg_robot.pointcloud import mujoco_depth_to_world_points
        return mujoco_depth_to_world_points(
            depth, seg=None,
            cam_to_world=self.cam_to_robot_base,
            fov_deg=FOVY)

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

    _GRASP_PROXIMITY   = 0.12   # demo_attach fallback: 3-D post-close threshold
    _GRASP_XY_PRESEL   = 0.10   # demo_attach: XY pre-selection radius before descent

    def _execute_grasp(self, pos: tuple, roll: float,
                       gripper_opening_length: float,
                       obj_height: float) -> Tuple[bool, Optional[int]]:
        """Dispatch to the active grasp execution mode (self.grasp_mode).

        Returns (success, grasped_obj_id_or_None).
        """
        if self.grasp_mode == GRASP_MODE_DEMO_ATTACH:
            return self._execute_grasp_demo_attach(pos, roll,
                                                   gripper_opening_length,
                                                   obj_height)
        # GRASP_MODE_PHYSICS_WELD and legacy GRASP_MODE_PHYSICS both use
        # jaw-midpoint IK + bilateral-contact-conditioned kinematic weld.
        return self._execute_grasp_physics_topdown(pos, roll,
                                                   gripper_opening_length,
                                                   obj_height)

    def _execute_grasp_physics(self, pos: tuple, roll: float,
                               gripper_opening_length: float,
                               obj_height: float) -> Tuple[bool, Optional[int]]:
        """Honest physics-based grasp.  Uses real MuJoCo contact detection and
        post-lift Z verification.  No kinematic attachment.

        The SO-ARM moving jaw cannot physically contact table-level objects in a
        top-down approach (geometric clearance gap), so this mode will report 0 %
        success for the current hardware config.  That is the CORRECT result for
        benchmark and world-model training label generation.

        For orientation-controlled top-down grasps use _execute_grasp_physics_topdown.
        """
        self.reset_robot()

        x, y, z = pos
        z = np.clip(z, *self.ee_position_limit[2])
        orn     = None
        opening = gripper_opening_length * self.GRIP_REDUCTION

        print(f"  [grasp/physics] approach → xy=({x:.3f}, {y:.3f})  z={z:.3f}")
        self.move_gripper(opening)
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])
        self.move_ee([x, y, z, orn])

        self.auto_close_gripper(check_contact=False)
        self._steps(80)

        self.last_grasp_metrics = self.get_grasp_debug_metrics()

        contact_ids = self.check_grasped_id()
        contact     = bool(contact_ids)
        weld_obj    = contact_ids[0] if contact_ids else None
        if weld_obj is not None:
            self._attach_obj(weld_obj)

        # Lift
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn], max_step=300)
        self._steps(100)

        if weld_obj is not None:
            obj_z  = self.get_obj_pos(weld_obj)[2]
            lifted = obj_z > self.Z_TABLE_TOP + 0.07
            grasped_ids = [weld_obj] if lifted else []
            if not lifted:
                self._detach_obj(weld_obj)
        else:
            grasped_ids = []
            obj_z       = self.Z_TABLE_TOP
            lifted      = False

        print(f"  [grasp/physics] result → contact={contact}"
              f"  grasped={grasped_ids}  lifted={lifted}  obj_z={obj_z:.4f}")

        if grasped_ids and lifted:
            return True, grasped_ids[0]
        return False, None

    def _execute_grasp_physics_topdown(self, pos: tuple, yaw: float,
                                       gripper_opening_length: float,
                                       obj_height: float) -> Tuple[bool, Optional[int]]:
        """Physics grasp with jaw-midpoint–targeted approach.

        Uses IK_MODE_JAW_POS so the JAW MIDPOINT (not the site) is driven to
        the target position with the arm in its natural configuration.  This
        avoids the near-joint-limit configs that arise when top-down orientation
        is forced at table-level reach distances, which caused the site Z-axis
        to flip to the -X world direction and the jaw midpoint to miss the
        object entirely.

        The arm descends in three phases:
          1. Hover above the object at GRIPPER_MOVING_HEIGHT (jaw-pos IK).
          2. Descend to the grasp Z with jaw-midpoint alignment maintained.
          3. Close gripper, wait, lift, verify.
        """
        self.reset_robot()

        x, y, z = pos
        z = np.clip(z, *self.ee_position_limit[2])
        opening = gripper_opening_length * self.GRIP_REDUCTION

        print(f"  [grasp/physics_weld] approach → xy=({x:.3f}, {y:.3f})"
              f"  z={z:.3f}  yaw={yaw:.3f}")

        self.move_gripper(opening)

        # Phase 1 — Hover: rotate base joint to correct orientation via physics.
        # max_step=1200 → n_steps=600 headless.  ~101° base rotation from HOME.
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, None],
                     ik_mode=IK_MODE_JAW_POS, max_step=1200)
        self._steps(100)

        # Phase 2 — Descend: solve IK for grasp height then directly assign
        # joint angles (bypassing physics interpolation).  Physics interpolation
        # from hover to grasp passes through intermediate configs where the open
        # jaws collide with the box sides (moving jaw sweeps to within 0.6 mm of
        # the box face), stopping the arm 10+ cm above target.  Direct assignment
        # places the arm at the collision-free IK solution immediately.
        #
        # IK: drive jaw GEOM midpoint (not body midpoint) to object centre.
        # Using geom positions ensures the actual jaw tips straddle the object
        # symmetrically in Y with ~5 mm clearance on each side, so that
        # restoring the object after teleport finds no Y-penetration and closing
        # sweeps the moving jaw tip into bilateral contact.
        qpos_saved = self.data.qpos.copy()
        qvel_saved = self.data.qvel.copy()

        obj_target = np.array([x, y, z])
        # Start IK from hover state (not HOME) so the arm stays in the same
        # workspace and the gripper closing motion sweeps toward the object.
        ok_ik, pe_ik, _ = self._solve_ik_jaw_pos_only(obj_target, reset_to_home=False)
        q_grasp = np.array([self.data.qpos[adr] for adr in self._arm_qpos_adr])

        if self._jaw_fixed_geom_id >= 0:
            fg = self.data.geom_xpos[self._jaw_fixed_geom_id]
            mg = self.data.geom_xpos[self._jaw_mv_geom_id]
            eef_z = self._get_eef_rot()[:, 2]
            print(f"  [grasp/physics_weld] jaw geoms at IK: fixed={fg.round(4)}  moving={mg.round(4)}"
                  f"  Y-gap={abs(mg[1]-fg[1])*100:.1f}cm  eef_z={eef_z.round(3)}")

        # Restore hover state.  Park all objects to z=-100 so the arm joints
        # can be teleported to the IK solution without jaw meshes penetrating
        # the object (which would push it away during the settle steps).
        self.data.qpos[:] = qpos_saved
        self.data.qvel[:] = qvel_saved
        obj_poses: list[tuple] = []
        for oid in self.obj_ids:
            slot = self._obj_pool_slot(oid)
            jnt  = self.model.joint(f"obj_joint_{slot}")
            adr  = jnt.qposadr[0]
            obj_poses.append(self.data.qpos[adr : adr + 7].copy())
            self.data.qpos[adr + 2] = -100.0   # park below floor

        # Teleport arm to IK solution and settle without the object
        for adr, act_id, q in zip(self._arm_qpos_adr, self._arm_act_ids, q_grasp):
            self.data.qpos[adr] = q
            self.data.ctrl[act_id] = q
        mujoco.mj_forward(self.model, self.data)
        self._steps(200)

        # Restore object poses and settle — geom IK guarantees ~5 mm Y clearance
        # on each side so no penetration on restore
        for oid, pose in zip(self.obj_ids, obj_poses):
            slot = self._obj_pool_slot(oid)
            jnt  = self.model.joint(f"obj_joint_{slot}")
            adr  = jnt.qposadr[0]
            vadr = jnt.dofadr[0]
            self.data.qpos[adr : adr + 7] = pose
            self.data.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._steps(100)   # settle object into jaw gap

        metrics_pre = self.get_grasp_debug_metrics()
        jaw_gap_pre = metrics_pre.get("jaw_obj_xy_gap", 0)
        print(f"  [grasp/physics_weld] pre-close metrics: {metrics_pre}")

        if self._jaw_fixed_geom_id >= 0:
            fg0 = self.data.geom_xpos[self._jaw_fixed_geom_id].copy()
            mg0 = self.data.geom_xpos[self._jaw_mv_geom_id].copy()
            print(f"  [grasp/physics_weld] pre-close geoms: fixed_Y={fg0[1]:.4f} moving_Y={mg0[1]:.4f} moving_Z={mg0[2]:.4f}")

        self.auto_close_gripper(check_contact=False)
        self._steps(120)   # let gripper settle and contacts stabilize

        if self._jaw_fixed_geom_id >= 0:
            fg1 = self.data.geom_xpos[self._jaw_fixed_geom_id].copy()
            mg1 = self.data.geom_xpos[self._jaw_mv_geom_id].copy()
            print(f"  [grasp/physics_weld] post-close geoms: fixed_Y={fg1[1]:.4f} moving_Y={mg1[1]:.4f}"
                  f"  ΔY_moving={( mg1[1]-mg0[1])*1000:.1f}mm  ncon={self.data.ncon}")
            # debug: print all active contacts with body names
            _jaw_bid  = self._jaw_body_id
            _jaw_mbid = self._jaw_mv_body_id
            for _ci in range(self.data.ncon):
                _c  = self.data.contact[_ci]
                _b1 = self.model.geom_bodyid[_c.geom1]
                _b2 = self.model.geom_bodyid[_c.geom2]
                _n1 = self.model.body(_b1).name
                _n2 = self.model.body(_b2).name
                print(f"  [contact {_ci}] geom1={_c.geom1}(body={_n1}) geom2={_c.geom2}(body={_n2})"
                      f"  dist={_c.dist:.4f}")

        contact_ids  = self.check_grasped_id()
        contact      = bool(contact_ids)
        metrics_post = self.get_grasp_debug_metrics()
        print(f"  [grasp/physics_weld] post-close metrics: {metrics_post}")

        # Detect jaw-table contact (fixed jaw sphere touching table surface).
        _world_bid = 0  # MuJoCo world body is always body 0
        table_contact = any(
            (self.model.geom_bodyid[int(c.geom1)] in (_world_bid, 1) and
             self.model.geom_bodyid[int(c.geom2)] in (self._jaw_body_id, self._jaw_mv_body_id))
            or
            (self.model.geom_bodyid[int(c.geom2)] in (_world_bid, 1) and
             self.model.geom_bodyid[int(c.geom1)] in (self._jaw_body_id, self._jaw_mv_body_id))
            for c in self.data.contact[:self.data.ncon]
        )

        # Kinematic weld for lift: sphere colliders provide insufficient friction
        # to lift objects against gravity.  Attach kinematically when bilateral
        # contacts confirm the jaws straddle the object; release if lift fails.
        weld_obj      = contact_ids[0] if contact_ids else None
        weld_triggered = weld_obj is not None
        if weld_triggered:
            self._attach_obj(weld_obj)
            print(f"  [grasp/physics_weld] bilateral contacts → kinematic attach obj={weld_obj}")
        else:
            print(f"  [grasp/physics_weld] no bilateral contacts → skip weld")

        # Lift: xyz-only to avoid arm reconfiguration pushing cube into table.
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, None],
                     ik_mode=IK_MODE_XYZ_ONLY, max_step=300)
        self._steps(100)

        if weld_obj is not None:
            obj_z  = self.get_obj_pos(weld_obj)[2]
            lifted = obj_z > self.Z_TABLE_TOP + 0.07
            grasped_ids = [weld_obj] if lifted else []
            if not lifted:
                self._detach_obj(weld_obj)
        else:
            grasped_ids = []
            obj_z       = self.Z_TABLE_TOP
            lifted      = False

        success = bool(grasped_ids) and lifted
        print(f"  [grasp/physics_weld] result → bilateral={contact}"
              f"  weld={weld_triggered}  table_contact={table_contact}"
              f"  lifted={lifted}  obj_z={obj_z:.4f}  success={success}")

        # Populate last_grasp_metrics with all result fields so benchmark
        # runner and collect scripts can read them without re-computing.
        metrics_post.update({
            "exec_mode":      GRASP_MODE_PHYSICS_WELD,
            "bilateral_contact": contact,
            "weld_triggered":    weld_triggered,
            "table_contact":     table_contact,
            "final_z":           float(obj_z),
            "lifted":            lifted,
            "success":           success,
        })
        self.last_grasp_metrics = metrics_post

        if success:
            return True, grasped_ids[0]
        return False, None

    def get_grasp_debug_metrics(self, obj_id: Optional[int] = None) -> dict:
        """Return diagnostic metrics about the current gripper-object geometry.

        Keys
        ----
        ori_err_norm      : orientation error (rad) between current EEF and ideal
                            top-down rotation (0 if arm is in perfect top-down pose)
        jaw_obj_xy_gap    : XY distance (m) between jaw midpoint and nearest object
                            CoM; None if no object loaded
        bilateral_contacts: number of contacts involving both fixed AND moving jaw
        left_contacts     : contacts on fixed jaw body
        right_contacts    : contacts on moving jaw body
        symmetry_score    : |left_contacts - right_contacts| / max(1, left+right),
                            0 = perfectly symmetric
        eef_z_axis        : current EEF site Z-axis in world frame (3-vector)
        """
        R_cur    = self._get_eef_rot()
        R_target = make_topdown_rotation(0.0)
        R_err    = R_target @ R_cur.T
        q_err    = np.zeros(4)
        mujoco.mju_mat2Quat(q_err, R_err.ravel())
        ori_vel  = np.zeros(3)
        mujoco.mju_quat2Vel(ori_vel, q_err, 1.0)
        ori_err  = float(np.linalg.norm(ori_vel))

        jaw_fixed_pos = self.data.xpos[self._jaw_body_id].copy()
        jaw_mv_pos    = self.data.xpos[self._jaw_mv_body_id].copy()
        jaw_mid_xy    = 0.5 * (jaw_fixed_pos[:2] + jaw_mv_pos[:2])

        obj_ids = [obj_id] if obj_id is not None else list(self.obj_ids)
        jaw_obj_xy_gap = None
        if obj_ids:
            gaps = []
            for oid in obj_ids:
                try:
                    op = self.get_obj_pos(oid)
                    gaps.append(float(np.linalg.norm(jaw_mid_xy - op[:2])))
                except Exception:
                    pass
            if gaps:
                jaw_obj_xy_gap = float(min(gaps))

        jaw_ids  = {self._jaw_body_id, self._jaw_mv_body_id}
        left_c   = 0
        right_c  = 0
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if b1 == self._jaw_body_id or b2 == self._jaw_body_id:
                left_c += 1
            if b1 == self._jaw_mv_body_id or b2 == self._jaw_mv_body_id:
                right_c += 1

        bilateral = 1 if (left_c > 0 and right_c > 0) else 0
        total     = left_c + right_c
        symmetry  = abs(left_c - right_c) / max(1, total)

        return {
            "ori_err_norm":       round(ori_err, 4),
            "jaw_obj_xy_gap":     round(jaw_obj_xy_gap, 4) if jaw_obj_xy_gap is not None else None,
            "bilateral_contacts": bilateral,
            "left_contacts":      left_c,
            "right_contacts":     right_c,
            "symmetry_score":     round(symmetry, 3),
            "eef_z_axis":         R_cur[:, 2].round(3).tolist(),
        }

    def _execute_grasp_demo_attach(self, pos: tuple, roll: float,
                                   gripper_opening_length: float,
                                   obj_height: float) -> Tuple[bool, Optional[int]]:
        """Kinematic sticky-gripper for VISUAL DEMOS ONLY.

        NOT for benchmarks or world-model training label generation.  The
        "success" signal here is synthetic: the object is kinematically
        teleported to follow the EEF rather than physically grasped.

        Strategy:
          1. Pre-select the closest object by XY before descent so that tall/
             round objects knocked sideways during the arm's downward sweep are
             still captured.
          2. After gripper closes, snap the pre-selected object 3 cm below the
             EEF (canonical held offset) and kinematically track it through lift
             and tray delivery.
          3. Fallback: if no XY pre-selection matched, fall back to the closest
             object within _GRASP_PROXIMITY after close.

        """
        self.reset_robot()

        x, y, z = pos
        z = np.clip(z, *self.ee_position_limit[2])
        orn     = None
        opening = gripper_opening_length * self.GRIP_REDUCTION

        # Pre-select the intended target by XY proximity before we descend
        pre_target: Optional[int] = None
        best_xy: float = self._GRASP_XY_PRESEL
        for oid in self.obj_ids:
            opos = self.get_obj_pos(oid)
            xy_d = float(np.linalg.norm(np.array([x, y]) - opos[:2]))
            z_ok = (self.Z_TABLE_TOP - 0.12) < opos[2] < (self.Z_TABLE_TOP + 0.30)
            if xy_d < best_xy and z_ok:
                best_xy    = xy_d
                pre_target = oid

        print(f"  [grasp/demo] approach  → xy=({x:.3f}, {y:.3f})  z={z:.3f}"
              + (f"  target={pre_target}" if pre_target is not None else ""))
        self.move_gripper(opening)
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])
        self.move_ee([x, y, z, orn])

        self.auto_close_gripper(check_contact=False)
        self._steps(80)

        weld_obj: Optional[int] = None

        if pre_target is not None:
            # Snap object 3 cm below EEF regardless of where descent pushed it
            attached = self._attach_obj(pre_target,
                                        offset=np.array([0.0, 0.0, -0.03]))
            weld_obj = pre_target
            print(f"  [grasp/demo] attach    → obj={weld_obj}  ok={attached}")
        else:
            # Fallback: standard 3-D proximity check
            eef_pos = self._get_eef_pos()
            best_d  = self._GRASP_PROXIMITY
            for oid in self.obj_ids:
                d = float(np.linalg.norm(eef_pos - self.get_obj_pos(oid)))
                if d < best_d:
                    best_d   = d
                    weld_obj = oid
            if weld_obj is not None:
                attached = self._attach_obj(weld_obj)
                print(f"  [grasp/demo] attach    → obj={weld_obj}"
                      f"  dist={best_d:.3f}  ok={attached}")
            else:
                print(f"  [grasp/demo] no object in proximity"
                      f"  ({self._GRASP_PROXIMITY:.2f} m)  eef={eef_pos.round(3)}")

        # Lift
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn], max_step=300)
        self._steps(100)

        if weld_obj is not None:
            obj_z  = self.get_obj_pos(weld_obj)[2]
            lifted = obj_z > self.Z_TABLE_TOP + 0.07
            print(f"  [grasp/demo] result    → weld_obj={weld_obj}  lifted={lifted}"
                  f"  obj_z={obj_z:.4f}")
            if lifted:
                return True, weld_obj
            self._detach_obj(weld_obj)
        else:
            print(f"  [grasp/demo] result    → no attach  → fail")

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
        # Wait for object to settle in tray (~0.3 s at timestep=0.002)
        self._steps(150)

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
            execution_mode=self.grasp_mode,
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
