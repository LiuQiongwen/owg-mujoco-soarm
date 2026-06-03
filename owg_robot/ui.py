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
    from owg_robot.env import *                      # PyBullet Environment (default)
except ImportError:
    pass                                             # pybullet not available (MuJoCo-only env)
from owg_robot.env_soarm import (EnvironmentSoArm,      # MuJoCo backend
                                  GRASP_MODE_PHYSICS,
                                  GRASP_MODE_DEMO_ATTACH,
                                  GRASP_Z_TABLE_MARGIN)
try:
    from owg_robot.camera import Camera
except ImportError:
    Camera = None                                    # pybullet not available
from owg_robot.objects import YcbObjects
from owg.policy import OwgPolicy
from owg.utils.config import load_config
from owg.utils.grasp import Grasp2D
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
            './owg_robot/assets/ycb_objects',
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

        # init OWG policy
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
            # self.objects = YcbObjects('./owg_robot/assets/ycb_objects',
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

    def _setup_grasps_mujoco(self):
        """MuJoCo backend: generate centroid-based grasps from actual object pose.

        GR-ConvNet's camera→robot coordinate transform is calibrated for PyBullet
        and produces wrong x/y in MuJoCo. Instead, sample grasps directly from
        the object's 3D CoM position — identical to the collect_mujoco_transitions
        approach.

        Uses get_obj_com_pos() (body world-frame CoM) rather than get_obj_pos()
        (free-joint qpos) so the jaw-midpoint IK target z equals the object CoM
        height.  YCB meshes have varying origin offsets relative to the CoM; a
        fixed GRASP_Z_OFFSET over qpos_z caused the jaw to sit above short objects
        (e.g. Banana: 4 cm tall, offset was 6 cm) with no contact on close.
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

            candidates = [
                (float(rng.uniform(-np.pi / 2, np.pi / 2)),
                 float(rng.uniform(0.04, 0.09)))
                for _ in range(self.n_grasp_attempts)
            ]

            target = np.array([gx, gy, gz])
            pe_iks = self.env.compute_ik_reachability([target] * len(candidates))
            pe = pe_iks[0]  # same target for all candidates within episode

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

            grasps = []
            for i in use_idx:
                yaw, opening = candidates[i]
                grasps.append({
                    "position": [gx, gy, gz],
                    "rpy":      [np.pi, 0.0, yaw],
                    "width":    opening,
                    "score":    0.0,
                    "_metrics": {"H": 0.05, "pe_ik": pe},
                })

            self.env.set_obj_grasps(obj_id, grasps, grasp_rects=[])

    def setup_grasps(self,
                     obs: Dict[str, Any],
                     visualise_grasps: bool = False):
        """
        Run inference with GR-ConvNet grasp generator on current observation
        """
        if self.backend == "mujoco":
            self._setup_grasps_mujoco()
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
        Wrapper around OWG action predictions and implemented robot primitives.

        Args:
          action: Predicted action by OWG 
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

