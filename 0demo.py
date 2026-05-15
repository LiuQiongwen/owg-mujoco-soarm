from owg_robot.ui import RobotEnvUI
from owg.utils.config import load_config
import argparse
import os, sys

# 将 grconvnet 目录加入模块搜索路径
grconvnet_path = os.path.join(os.path.dirname(__file__), "third_party/grconvnet")
if grconvnet_path not in sys.path:
    sys.path.insert(0, grconvnet_path)

# ========== ✅ 新增命令行参数 ==========
parser = argparse.ArgumentParser(description="Run OWG demo without Tkinter popup.")
parser.add_argument('--n_objects', type=int, help='Number of objects to load', default=None)
parser.add_argument('--seed', type=int, help='Random seed', default=None)
parser.add_argument('--vis', type=int, help='Enable PyBullet visualizer', default=None)
parser.add_argument('--verbose', type=int, help='Set verbosity level', default=None)
parser.add_argument('--prompt', type=str, help='Language instruction for the grasp task', default="grasp the red cup")
kwargs = vars(parser.parse_args())

# ========== ✅ 加载配置文件 ==========
cfg = load_config('./config/pyb/env.yaml')
cfg.n_objects = kwargs['n_objects'] or cfg.n_objects
cfg.seed = kwargs['seed'] or cfg.seed
cfg.policy.vis = kwargs['vis'] or cfg.policy.vis
cfg.policy.verbose = kwargs['verbose'] or cfg.policy.verbose

# ✅ 将 prompt 注入配置（给后续 ui.py 使用）
cfg.prompt = kwargs['prompt']
print("\n=== Loaded Configuration ===")
print(cfg)
print("============================\n")

# ========== ✅ 运行 UI 控制器 ==========
demo = RobotEnvUI(cfg)

# 🔹 Monkey patch: 直接注入 prompt，避免调用 Tkinter 弹窗
if hasattr(demo, "user_input") is False:
    demo.user_input = cfg.prompt

if hasattr(demo, "set_user_prompt"):
    demo.set_user_prompt(cfg.prompt)

print(f"[INFO] Using prompt: \"{cfg.prompt}\"")

demo.run()

