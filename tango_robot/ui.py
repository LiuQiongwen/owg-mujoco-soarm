import numpy as np
import time
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter, HtmlFormatter
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
from typing import *
import os, json
try:
    from tango_robot.env import *                      # PyBullet Environment (default)
except ImportError:
    pass                                             # pybullet not available (MuJoCo-only env)
from tango_robot.env_soarm import (EnvironmentSoArm,      # MuJoCo backend
                                  GRASP_MODE_PHYSICS,
                                  GRASP_MODE_DEMO_ATTACH,
                                  GRASP_Z_TABLE_MARGIN,
                                  TABLE_TOP_Z)

# XY spread around CoM when generating grasp candidates — must match
# collect_lggsn_data._SPREAD_XY so inference features lie in the training distribution.
_LGGSN_SPREAD_XY = 0.06

# ── CFM grasp generator (optional) ────────────────────────────────────────────
# Set env var CFM_CKPT to enable CFM candidate generation in _setup_grasps_mujoco.
# The matching _stats.json must exist alongside the .pt checkpoint.
_CFM_CKPT = os.environ.get("CFM_CKPT", "")

def _load_cfm_model(ckpt_path: str):
    """Load OT-CFM or DDPM generative model and inference stats.  Returns (model, stats) or (None, None)."""
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None, None
    stats_path = ckpt_path.replace(".pt", "_stats.json")
    if not os.path.isfile(stats_path):
        print(f"[CFM] stats file not found: {stats_path}")
        return None, None
    try:
        import torch, json as _json
        stats      = _json.load(open(stats_path))
        model_type = stats.get("model_type", "cfm")
        if model_type == "ddpm":
            from train_diffusion_grasp import NoiseNet
            model = NoiseNet(hidden=512)
        else:
            from train_cfm_grasp import VelocityNet
            model = VelocityNet(hidden=512)
        sd  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
        model.eval()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)
        tag = f"[{model_type.upper()}]"
        print(f"{tag} Loaded {ckpt_path} — objects: {list(stats['mean_vis_per_obj'].keys())}")
        return model, stats
    except Exception as e:
        print(f"[CFM] Failed to load: {e}")
        return None, None

def _cfm_sample_candidates(model, stats, obj_name: str, n: int,
                            gx: float, gy: float, gz: float, pe: float,
                            rng: np.random.Generator):
    """Generate n grasp candidates via CFM conditioned on obj_name's mean vis_feat.

    CFM was trained on absolute world-frame positions.  To ensure generated
    poses are centred on the *current* object CoM (not the training-set mean),
    we subtract the training global mean (x,y) and add the live CoM (gx,gy).
    This preserves the learned yaw distribution while adapting to the actual
    object location at inference time.

    Returns list of grasp dicts, or None if obj_name not in stats.
    """
    import torch
    model_type  = stats.get("model_type", "cfm")
    key = obj_name.lower().replace("ycb", "").replace(" ", "")
    # fuzzy match: "MustardBottle" → "mustard"
    vis_map = stats["mean_vis_per_obj"]
    cond_vec = None
    for k in vis_map:
        if k in key or key.startswith(k):
            cond_vec = vis_map[k]
            break
    if cond_vec is None:
        return None

    dev    = next(model.parameters()).device
    # CFM_ZERO_VIS=1: condition on zeros to ablate the visual feature contribution
    if os.environ.get("CFM_ZERO_VIS") == "1":
        cond_vec = [0.0] * len(cond_vec)
    cond_t = torch.tensor(cond_vec, dtype=torch.float32, device=dev)
    pmean  = np.array(stats["pose_mean"], dtype=np.float32)   # global training mean
    pstd   = np.array(stats["pose_std"],  dtype=np.float32)

    if model_type == "ddpm":
        from train_diffusion_grasp import sample_poses_ddpm
        infer_steps = int(os.environ.get("DDIM_STEPS", stats.get("infer_steps", 100)))
        poses_norm = sample_poses_ddpm(model, cond_t, n=n, steps=infer_steps)
    else:
        from train_cfm_grasp import sample_poses
        poses_norm = sample_poses(model, cond_t, n=n, steps=20)   # (n, 6) normalised
    poses      = poses_norm * pstd + pmean                     # denormalise → absolute

    grasps = []
    for p in poses:
        cfm_x, cfm_y, _z, roll, pitch, yaw = [float(v) for v in p]
        # Shift from training-mean centre → current object CoM
        x = cfm_x - float(pmean[0]) + gx
        y = cfm_y - float(pmean[1]) + gy
        width = float(rng.uniform(0.04, 0.09))
        grasps.append({
            "position": [x, y, gz],
            "rpy":      [roll, pitch, yaw],
            "width":    width,
            "score":    0.0,
            "_metrics": {"H": 0.05, "dz": 0.0, "dz_lift": 0.0, "need_dz": 0.0, "pe_ik": pe},
        })
    return grasps
try:
    from tango_robot.camera import Camera
except ImportError:
    Camera = None                                    # pybullet not available
from tango_robot.objects import YcbObjects
from tango.policy import OwgPolicy
from tango.utils.config import load_config
from tango.utils.grasp import Grasp2D
from third_party.grconvnet import load_grasp_generator
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "ui_grasp_exec.jsonl")

def log_exec(event: dict):
    """Append one grasp execution record to JSONL log."""
    event = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **event)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")

# GUI stuff
# Function to create a text input dialog using Tkinter
def ask_for_user_input():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    user_input = simpledialog.askstring("Input", "User Input: ")
    root.destroy()
    return input("请输入对象描述（自然语言）：")


class RobotEnvUI:

    def __init__(self, config: Union[Dict[str, Any], str], backend: str = "pybullet"):
        self.cfg = load_config(config) if isinstance(config, str) else config
        self.backend = getattr(self.cfg, "backend", backend)
        self.n_objects = self.cfg.n_objects
        self.seed = self.cfg.seed

        self.n_grasp_attempts = self.cfg.n_grasp_attempts
        # Read CFM_CKPT at __init__ time (not module level) so demo.py can set it before UI creation
        self._cfm_model, self._cfm_stats = _load_cfm_model(os.environ.get("CFM_CKPT", ""))

        if self.backend == "mujoco":
            # ── MuJoCo / SO-ARM101 backend ─────────────────────────────────
            _mj_vis = bool(getattr(self.cfg.policy, "vis", False))
            # grasp_mode: read from config; fall back to "physics" so all
            # benchmark runs are honest by default.  Set "demo_attach" in the
            # mujoco env.yaml (or pass --grasp_mode demo_attach) only for
            # semantic demo recordings — never for benchmark evaluation.
            _grasp_mode = getattr(self.cfg, "grasp_mode", GRASP_MODE_PHYSICS)
            self.env = EnvironmentSoArm(
                vis=_mj_vis,
                debug=False,
                finger_length=self.cfg.finger_length,
                n_grasp_attempts=self.cfg.n_grasp_attempts,
                grasp_mode=_grasp_mode,
            )
            print(f"[INFO] MuJoCo grasp_mode: {_grasp_mode}"
                  + (" (demo only — not for benchmarks)"
                     if _grasp_mode == GRASP_MODE_DEMO_ATTACH else ""))
            self.camera = self.env.camera   # _MockCamera shim
            self.img_size = (self.env.camera.width, self.env.camera.height)
            print("[INFO] MuJoCo backend: EnvironmentSoArm (SO-ARM101)")
        else:
            # ── Fallback: any non-mujoco backend string uses EnvironmentSoArm
            _mj_vis = bool(getattr(self.cfg.policy, "vis", False))
            _grasp_mode = getattr(self.cfg, "grasp_mode", GRASP_MODE_PHYSICS)
            self.env = EnvironmentSoArm(
                vis=_mj_vis,
                debug=False,
                finger_length=self.cfg.finger_length,
                n_grasp_attempts=self.cfg.n_grasp_attempts,
                grasp_mode=_grasp_mode,
            )
            self.camera = self.env.camera
            self.img_size = (self.env.camera.width, self.env.camera.height)
            print("[INFO] Fallback to EnvironmentSoArm (SO-ARM101)")

        # 自然语言级执行日志（query -> action -> success）
        os.makedirs("logs", exist_ok=True)
        self.nl_log_path = os.path.join("logs", "ui_nl_exec.jsonl")

        # load objects
        self.objects = YcbObjects(
            './tango_robot/assets/ycb_objects',
            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
            mod_stiffness=['Strawberry'],
            seed=self.seed)
        self.objects.shuffle_objects()

        # --object flag: keep only the requested object(s) in the pool
        pin_object = getattr(self.cfg, "object", None)
        if pin_object is not None:
            pin_lower = pin_object.lower()
            filtered = [n for n in self.objects.obj_names if n.lower() == pin_lower]
            if not filtered:
                all_names = self.objects.obj_names[:]
                raise ValueError(
                    f"--object '{pin_object}' not found in obj_list.txt. "
                    f"Valid names: {all_names}"
                )
            self.objects.obj_names = filtered

        self.env.dummy_simulation_steps(10)

        # init TANGO policy
        self.policy = OwgPolicy(
            self.cfg.policy.config_path,
            verbose=self.cfg.policy.verbose,
            vis=self.cfg.policy.vis,
            use_grasp_ranker=self.cfg.policy.use_grasp_ranker,
            lggsn_input_dim=getattr(self.cfg.policy, "lggsn_input_dim", None))
        print(f"[DEBUG] cfg.use_grasp_ranker = {self.cfg.policy.use_grasp_ranker}, "
            f"policy.use_grasp_ranker = {getattr(self.policy, 'use_grasp_ranker', None)}")
         
        # MuJoCo backend: grasps are always stored as 3D (get_obj_grasps, not grasp_rects)
        self.grasp_rank_3d = (self.backend == "mujoco")
        if self.policy.grasp_ranker is not None:
            self.grasp_rank_3d = self.policy.grasp_ranker.use_3d_prompt

        # derive stage label for per-trial logging; use runtime flag (may differ
        # from cfg when LGGSN init fails and use_grasp_ranker was cleared)
        _sem = getattr(self.cfg.policy, "enable_semantic", False)
        _rank = getattr(self.policy, "use_grasp_ranker", False)
        if _rank and _sem:
            self._stage = 4
        elif _sem:
            self._stage = 3
        elif getattr(self.cfg.policy, "enable_grasp_sampling", False):
            self._stage = 2
        else:
            self._stage = 1

        # spawn scene
        obs = self.spawn(self.n_objects)

        # GR-ConvNet grasp generator
        self.grasp_generator = load_grasp_generator(self.env.camera)
        # setup and visualize once
        self.setup_grasps(obs, visualise_grasps=True)

        self.n_action_attempts = self.cfg.n_action_attempts
        self.n_grasp_attempts = self.cfg.n_grasp_attempts

    def spawn(self, n_objects):
        self.n_objects = n_objects
        self.env.remove_all_obj()

        # MuJoCo semantic demo: cfg.scene_objects overrides the random pool
        scene_objects = getattr(self.cfg, "scene_objects", None)
        if scene_objects is not None and self.backend == "mujoco":
            obj_list = scene_objects[:n_objects]
        else:
            obj_list = self.objects.obj_names[:n_objects]
            # Guarantee the prompted object is in the scene.
            # Skip when --object already pins the pool to one name.
            _prompt = getattr(self.cfg, "prompt", None)
            _pin    = getattr(self.cfg, "object", None)
            if _prompt and n_objects > 0 and _pin is None:
                _pl = _prompt.strip().lower()
                _target = next(
                    (n for n in self.objects.obj_names
                     if n.lower() in _pl or _pl in n.lower()),
                    None,
                )
                if _target is not None and _target not in obj_list:
                    _others = [n for n in self.objects.obj_names if n != _target]
                    obj_list = [_target] + _others[:n_objects - 1]

        # Pre-register all object types → single model rebuild before spawn loop
        if self.backend == "mujoco" and hasattr(self.env, "preload_pool"):
            self.env.preload_pool(obj_list)

        # MuJoCo: assign non-overlapping spawn positions so objects don't collide on drop
        if self.backend == "mujoco":
            import math as _math
            _n = len(obj_list)
            _ys = np.linspace(-0.30, -0.12, max(_n, 1))  # evenly spaced y, within arm reach
            _xs = [0.05 * (i - _n // 2) for i in range(_n)]  # small x spread
            _spawn_pos = [[_xs[i], float(_ys[i]), self.env.OBJECT_INIT_HEIGHT]
                          for i in range(_n)]
        for i, obj_name in enumerate(obj_list):
            path, mod_orn, mod_stiffness = self.objects.get_obj_info(obj_name)
            if self.backend == "mujoco":
                self.env.load_isolated_obj(path, obj_name, mod_orn, mod_stiffness,
                                           pos=_spawn_pos[i])
            else:
                self.env.load_isolated_obj(path, obj_name, mod_orn, mod_stiffness)
            # Extra settling for MuJoCo: objects need ~150 total steps to fully land
            settle = 100 if self.backend == "mujoco" else 30
            self.env.dummy_simulation_steps(settle)
        print(f"[DEBUG] loaded objects: {list(zip(self.env.obj_ids, self.env.obj_names))}")
        self.init_obj_state = self.env.get_obj_states()
        obs = self.env.get_obs()
        return obs

    def reset_same(self):
        assert self.init_obj_state is not None, "Have to spawn once to initialize state"
        self.env.reset_robot()
        self.env.set_obj_state(self.init_obj_state)
        self.env.dummy_simulation_steps(10)
        obs = self.update()
        self.init_obj_state = self.env.get_obj_states()
        for _ in range(30):
            self.env.step_simulation()
        return obs

    def reset(self, new=False):
        if new:
            self.env.remove_all_obj()
            for _ in range(30):
                self.env.step_simulation()
            # self.objects = YcbObjects('./tango_robot/assets/ycb_objects',
            #         mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
            #         mod_stiffness=['Strawberry'],
            #         seed=self.seed
            # )
            self.seed += 100
            self.objects.set_seed(self.seed)
            self.objects.shuffle_objects()
            self.env.dummy_simulation_steps(10)
            return self.spawn(self.n_objects)
        return self.reset_same()

    def update(self):
        self.env.dummy_simulation_steps(10)
        self.env.update_obj_states()
        obs = self.env.get_obs()
        self.setup_grasps(obs)
        self.env.dummy_simulation_steps(10)
        return obs

    @staticmethod
    def _compute_obj_H(obs: dict, obj_id: int) -> float:
        """Estimate object height from segmented point cloud (H = max_z - TABLE_TOP_Z).
        Mirrors collect_lggsn_data._compute_H so inference matches training features."""
        seg = obs.get("seg") if obs else None
        pts = obs.get("points") if obs else None
        if seg is None or pts is None:
            return 0.05
        flat_pts = pts.reshape(-1, 3)
        flat_seg = seg.ravel()
        n = min(len(flat_seg), len(flat_pts))
        obj_pts = flat_pts[:n][flat_seg[:n] == obj_id]
        if len(obj_pts) < 5:
            return 0.05
        return float(max(0.005, obj_pts[:, 2].max() - TABLE_TOP_Z))

    def _setup_grasps_mujoco(self, obs: dict | None = None):
        """Generate top-down grasp candidates whose feature distribution matches
        the LGGSN training data (collect_lggsn_data._sample_candidates).

        Key alignment with training:
          - XY positions: CoM ± _LGGSN_SPREAD_XY (was 0 → all features degenerate)
          - H: computed from live point cloud (was hardcoded 0.05)
          - roll=π, pitch=0, yaw=uniform(-π/2, π/2)
          - width=uniform(0.04, 0.09), score=0, dz=dz_lift=need_dz=0
          - pe_ik: CoM-target IK error (episode-level constant, same as training)
        """
        IK_PE_THRESHOLD = float(os.environ.get("IK_PE_THRESH", "0.005"))
        rng = np.random.default_rng(self.seed)
        for obj_id in self.env.obj_ids:
            try:
                com = self.env.get_obj_com_pos(obj_id)
            except Exception:
                com = self.env.get_obj_pos(obj_id)

            gx, gy = float(com[0]), float(com[1])
            gz     = float(com[2]) + GRASP_Z_TABLE_MARGIN
            H      = self._compute_obj_H(obs, obj_id) if obs is not None else 0.05

            candidates = [
                (float(gx + rng.uniform(-_LGGSN_SPREAD_XY, _LGGSN_SPREAD_XY)),
                 float(gy + rng.uniform(-_LGGSN_SPREAD_XY, _LGGSN_SPREAD_XY)),
                 float(rng.uniform(-np.pi / 2, np.pi / 2)),
                 float(rng.uniform(0.04, 0.09)))
                for _ in range(self.n_grasp_attempts)
            ]

            # pe_ik: IK error at CoM target (episode-level, same for all candidates)
            com_target = np.array([gx, gy, gz])
            pe_iks = self.env.compute_ik_reachability([com_target] * len(candidates))
            pe = pe_iks[0]

            reachable = [i for i, e in enumerate(pe_iks) if e <= IK_PE_THRESHOLD]
            if reachable:
                use_idx = reachable
                if len(reachable) < len(candidates):
                    print(f"  [IK prefilter] pe_ik={pe*1000:.1f}mm OK "
                          f"({len(use_idx)}/{len(candidates)} retained)")
            else:
                use_idx = list(range(min(3, len(candidates))))
                print(f"  [IK prefilter] pe_ik={pe*1000:.1f}mm "
                      f"> {IK_PE_THRESHOLD*1000:.0f}mm threshold; "
                      f"fallback to first {len(use_idx)}")

            # ── CFM candidate generation (replaces random sampling when enabled) ─
            cfm_grasps = None
            if self._cfm_model is not None:
                obj_name = (self.env.obj_names[self.env.obj_ids.index(obj_id)]
                            if obj_id in self.env.obj_ids and self.env.obj_names
                            else "")
                cfm_grasps = _cfm_sample_candidates(
                    self._cfm_model, self._cfm_stats, obj_name,
                    n=len(use_idx), gx=gx, gy=gy, gz=gz, pe=pe, rng=rng,
                )
                if cfm_grasps is not None:
                    print(f"  [CFM] Generated {len(cfm_grasps)} candidates for '{obj_name}'")

            if cfm_grasps is not None:
                grasps = cfm_grasps
            else:
                grasps = []
                for i in use_idx:
                    cx, cy, yaw, opening = candidates[i]
                    grasps.append({
                        "position": [cx, cy, gz],
                        "rpy":      [np.pi, 0.0, yaw],
                        "width":    opening,
                        "score":    0.0,
                        "_metrics": {
                            "H":       H,
                            "dz":      0.0,
                            "dz_lift": 0.0,
                            "need_dz": 0.0,
                            "pe_ik":   pe,
                        },
                    })

            self.env.set_obj_grasps(obj_id, grasps, grasp_rects=[])

    def _setup_grasps_grconvnet(self, obs: dict | None = None):
        """GR-ConvNet 2D prediction lifted to 6-DoF via MuJoCo camera model.

        Replaces uniform-random CoM sampling with GR-ConvNet spatial predictions.
        All downstream logic (IK filter, LGGSN reranker) stays identical to the
        random-sampling path, so this is a clean A/B swap of the candidate generator.

        Back-projection: MuJoCo metric depth (already metres) + pinhole intrinsics
        derived from FOVY=55°/224px.  Yaw = -image_angle (Y-flip, CAM_ROT=0).
        Fallback: random CoM candidate when GR-ConvNet returns fewer than needed.
        """
        from tango_robot.pointcloud import compute_intrinsics

        IK_PE_THRESHOLD = float(os.environ.get("IK_PE_THRESH", "0.005"))
        rng = np.random.default_rng(self.seed)

        rgb          = obs['image']          # (H, W, 3) uint8
        depth        = obs['depth']          # (H, W) float32 metric metres
        seg          = obs.get('seg')
        cam_to_world = obs['cam_to_world']   # (4, 4) camera → world

        img_h, img_w = depth.shape
        K    = compute_intrinsics(img_w, img_h, 55.0)  # FOVY = 55°
        fx   = float(K[0, 0])   # ≈ 215.6
        px   = float(K[0, 2])   # principal point x = 112.0
        py   = float(K[1, 2])   # principal point y = 112.0

        for obj_id in self.env.obj_ids:
            try:
                com = self.env.get_obj_com_pos(obj_id)
            except Exception:
                com = self.env.get_obj_pos(obj_id)

            gx, gy = float(com[0]), float(com[1])
            gz     = float(com[2]) + GRASP_Z_TABLE_MARGIN
            H      = self._compute_obj_H(obs, obj_id) if obs is not None else 0.05

            # IK pre-check identical to random path
            com_target = np.array([gx, gy, gz])
            pe_iks  = self.env.compute_ik_reachability(
                [com_target] * self.n_grasp_attempts)
            pe      = pe_iks[0]
            reachable = [i for i, e in enumerate(pe_iks) if e <= IK_PE_THRESHOLD]
            n_needed  = len(reachable) if reachable else min(3, self.n_grasp_attempts)

            # Masked RGB + depth for GR-ConvNet
            mask = (seg == obj_id) if seg is not None else np.ones(
                (img_h, img_w), dtype=bool)
            mask_rgb = np.where(np.stack([mask] * 3, axis=-1), rgb,
                                np.full_like(rgb, 0xff))
            mask_d   = np.where(mask, depth, float(depth.max()))

            grasps_2d, _ = self.grasp_generator.predict(
                mask_rgb, mask_d.copy(), n_grasps=n_needed)

            grasps = []
            for g in grasps_2d:
                row = int(np.clip(g.center[0], 0, img_h - 1))
                col = int(np.clip(g.center[1], 0, img_w - 1))

                d = float(depth[row, col])
                if d < 0.05 or d > 2.5:   # bad depth pixel → object median
                    obj_depths = depth[mask]
                    d = float(np.median(obj_depths)) if obj_depths.size else 1.115

                # Back-project: +X right, +Y up (row flip), -Z into scene
                xc = (col - px) / fx * d
                yc = -(row - py) / fx * d   # fx == fy (square pixels)
                zc = -d
                world_pt = (cam_to_world @ np.array([xc, yc, zc, 1.0]))[:3]
                x_w, y_w = float(world_pt[0]), float(world_pt[1])

                # Yaw: image angle → world yaw (Y-flip, CAM_ROT=0 → negate)
                world_yaw = float(-g.angle)

                # Width: GR-ConvNet pixel length → metres
                max_g    = float(self.grasp_generator.MAX_GRASP)    # 0.085
                pix_conv = float(self.grasp_generator.PIX_CONVERSION)  # 277
                width    = float(np.clip(
                    float(g.length) / (max_g * pix_conv) * max_g, 0.04, 0.09))

                grasps.append({
                    "position": [x_w, y_w, gz],
                    "rpy":      [np.pi, 0.0, world_yaw],
                    "width":    width,
                    "score":    float(g.quality),
                    "_metrics": {"H": H, "dz": 0.0, "dz_lift": 0.0,
                                 "need_dz": 0.0, "pe_ik": pe},
                })

            # Pad with random candidates if GR-ConvNet returned fewer than needed
            while len(grasps) < n_needed:
                grasps.append({
                    "position": [
                        gx + float(rng.uniform(-_LGGSN_SPREAD_XY, _LGGSN_SPREAD_XY)),
                        gy + float(rng.uniform(-_LGGSN_SPREAD_XY, _LGGSN_SPREAD_XY)),
                        gz,
                    ],
                    "rpy":   [np.pi, 0.0, float(rng.uniform(-np.pi / 2, np.pi / 2))],
                    "width": float(rng.uniform(0.04, 0.09)),
                    "score": 0.0,
                    "_metrics": {"H": H, "dz": 0.0, "dz_lift": 0.0,
                                 "need_dz": 0.0, "pe_ik": pe},
                })

            self.env.set_obj_grasps(obj_id, grasps, grasp_rects=[])

    def setup_grasps(self,
                     obs: Dict[str, Any],
                     visualise_grasps: bool = False):
        """
        Run inference with GR-ConvNet grasp generator on current observation
        """
        if self.backend == "mujoco":
            if os.environ.get("OWG_GRC6DOF") == "1":
                self._setup_grasps_grconvnet(obs)
            else:
                self._setup_grasps_mujoco(obs)
            return

        rgb, depth, seg = obs['image'], obs['depth'], obs['seg']

        img_size = self.grasp_generator.IMG_WIDTH
        if img_size != self.env.camera.width:
            rgb = cv2.resize(rgb, (img_size, img_size))
            depth = cv2.resize(depth, (img_size, img_size))
        for obj_id in self.env.obj_ids:
            mask = seg == obj_id
            if img_size != self.env.camera.width:
                mask = np.array(
                    Image.fromarray(mask).resize((img_size, img_size),
                                                 Image.LANCZOS))
            grasps, grasp_rects = self.grasp_generator.predict_grasp_from_mask(
                rgb, depth, mask, n_grasps=self.n_grasp_attempts, show_output=False)
            if img_size != self.env.camera.width:
                # normalize to original size
                for j, gr in enumerate(grasp_rects):
                    grasp_rects[j][0] = int(gr[0] / img_size *
                                            self.env.camera.width)
                    grasp_rects[j][1] = int(gr[1] / img_size *
                                            self.env.camera.width)
                    grasp_rects[j][4] = int(gr[4] / img_size *
                                            self.env.camera.width)
                    grasp_rects[j][3] = int(gr[3] / img_size *
                                            self.env.camera.width)
            grasp_rects = [
                Grasp2D.from_vector(
                    x=g[1],
                    y=g[0],
                    w=g[4],
                    h=g[3],
                    theta=g[2],
                    W=self.env.camera.width,
                    H=self.env.camera.width,
                    normalized=False,
                    line_offset=5,
                ) for g in grasp_rects
            ]
            self.env.set_obj_grasps(obj_id, grasps, grasp_rects)

        if visualise_grasps:
            LID = []
            for obj_id in self.env.obj_ids:
                grasps = self.env.get_obj_grasps(obj_id)
                color = np.random.rand(3).tolist()
                for g in grasps:
                    LID = self.env.draw_predicted_grasp(g,
                                                        color=color,
                                                        lineIDs=LID)

            time.sleep(1)
            self.env.remove_drawing(LID)

    def step(self, action):
        '''
        Wrapper around TANGO action predictions and implemented robot primitives.

        Args:
          action: Predicted action by TANGO 
            - `action`: Either `remove` to place blocking object in free space, or `pick` to put target in tray.
            - `input`: The object ID of object to manipulate.
        '''
        # ---- 统一处理 grasps：如果 planner 给的是 [] 或 None，就回退到默认 grasp 集合 ----
        grasp_indices = action.get('grasps')
        if not grasp_indices:   # None 或 []
            print("[WARN] Empty grasp list from planner, fallback to default indices")
            # 默认用前 n_grasp_attempts 个 grasp
            grasp_indices = list(range(self.n_grasp_attempts))
            action['grasps'] = grasp_indices

        if action['action'] == 'remove':
            success_grasp, success_target = self.env.put_obj_in_free_space(
                action['input'], grasp_indices=action['grasps'])
        elif action['action'] == 'pick':
            success_grasp, success_target = self.env.put_obj_in_tray(
                action['input'], grasp_indices=action['grasps'])

        for _ in range(30):
            self.env.step_simulation()

        if not success_grasp:
            print(f'Action failed...')
            success, done = False, False
        else:
            print(f'Done {action["action"]} {action["input"]}')
            success = True
            done = bool(success_target) and (action['input'] == action['target_id'])

        return success, done

    def run(self, initial_query: Optional[str] = None, once: bool = False):
        """
        initial_query: 如果传入，就不走 input()，直接用这条自然语言指令跑一轮
        once: True 时跑完一轮就退出（适合 stage3/4 做演示）
        """
        query_used = False

        while True:
            # 每一轮先把机械臂复位
            self.env.reset_robot()

            # 1) 优先用外部传入的 prompt
            if initial_query is not None and not query_used:
                user_input = str(initial_query).strip()
                query_used = True
                print(f'[AUTO] Using query: "{user_input}"')
            else:
                user_input = input("请输入对象描述（自然语言）：").strip()

            if not user_input:
                continue

            # 特殊命令
            if user_input == ':reset':
                self.reset(new=False)
                self.env.dummy_simulation_steps(10)
                if once:
                    self.env.close()
                    break
                continue

            elif user_input == ':new':
                self.reset(new=True)
                self.env.dummy_simulation_steps(10)
                if once:
                    self.env.close()
                    break
                continue

            elif user_input == ':all':
                # 抓取当前场景里所有 obj_ids，一次性评估成功率
                results = []

                # 先拿一次 grasps（2D/6D 都兼容）
                if self.grasp_rank_3d:
                    all_grasps = {int(k): self.env.get_obj_grasps(k) for k in self.env.obj_ids}
                else:
                    all_grasps = {int(k): self.env.get_obj_grasp_rects(k) for k in self.env.obj_ids}

                for tid in list(self.env.obj_ids):
                    tid = int(tid)

                    # 每个物体前都重置一下机器人，避免状态干扰
                    self.env.reset_robot()
                    obs = self.update()

                    action = {
                        'action': 'pick',
                        'input': tid,
                        'target_id': tid,
                        'grasps': []
                    }

                    # 如果 stage4 启用了 grasp_ranker，就对该物体的 grasp 排序
                    _n_grasps = len(all_grasps.get(tid, []))
                    _top1_score = None
                    if getattr(self.policy, 'use_grasp_ranker', False) and getattr(self.policy, 'grasp_ranker', None) is not None:
                        obj_grasps = all_grasps.get(tid, [])
                        if len(obj_grasps) > 0:
                            try:
                                order, scores = self.policy.grasp_ranker.rank(
                                    obj_grasps, query_text=str(tid), obj_type=None
                                )
                                action['grasps'] = order.tolist()
                                _top1_score = float(scores[order[0]])
                            except Exception as e:
                                print('⚠️ rank failed:', e)

                    success, done = self.step(action)
                    results.append((tid, bool(success)))
                    log_exec({
                        "stage": self._stage,
                        "path": "all_numeric",
                        "object_id": tid,
                        "n_grasps": _n_grasps,
                        "lggsn_score_top1": _top1_score,
                        "success": bool(success),
                    })

                    self.env.dummy_simulation_steps(30)

                ok = sum(int(s) for _, s in results)
                print(f"[ALL] success {ok}/{len(results)} -> {results}")

                if once:
                    self.env.close()
                    break
                continue
            attempt = 0
            while True:
                self.env.reset_robot()
                obs = self.update()

                if self.grasp_rank_3d:
                    all_grasps = {k: self.env.get_obj_grasps(k) for k in self.env.obj_ids}
                else:
                    all_grasps = {k: self.env.get_obj_grasp_rects(k) for k in self.env.obj_ids}

                print("UI env obj_ids:", getattr(self.env, "obj_ids", None))
                print("UI env obj_names:", getattr(self.env, "obj_names", None))
                
                action = self.policy.predict(
                    obs,
                    user_input,
                    all_grasps,
                    obj_names=getattr(self.env, "obj_names", None),
                    env_obj_ids=getattr(self.env, "obj_ids", None),
                )

                if action['action'] == 'fail':
                    success, done = False, False
                else:
                    success, done = self.step(action)

                log_exec({
                    "stage": self._stage,
                    "path": "language",
                    "query": user_input,
                    "object_id": action.get("input"),
                    "n_grasps": len(all_grasps.get(action.get("input"), [])),
                    "attempt": attempt,
                    "success": bool(success),
                })

                # 日志保持不变（你原来的 log 代码可以继续留着）

                if once:
                    break

                if success:
                    break

                if not success:
                    attempt += 1
                    if attempt >= self.n_action_attempts:
                        print('Action failed. No more atempts.')
                        break
                    print(f'Action failed. {attempt} attempt. Retrying..')
                    continue

                attempt = 0
                self.env.dummy_simulation_steps(30)
                continue

            # 2) once=True：跑完这一轮就退出
            if once:
                self.env.close()
                break

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="env.yaml",
        help="Path to environment config file (YAML).",
    )
    args = parser.parse_args()

    # 创建 UI 并运行主循环
    ui = RobotEnvUI(args.config)
    ui.run()


if __name__ == "__main__":
    main()

