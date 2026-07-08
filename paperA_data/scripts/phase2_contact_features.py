import sys, os, json
sys.path.insert(0, "/lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding")
os.chdir("/lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from scipy.stats import mannwhitneyu

from tango_robot.env_soarm import EnvironmentSoArm
from grasp_6dof.grasp_sampler import rpy_to_R, local_point_density, normal_consistency, contact_width_ratio

DIAG = "/home/lina/.claude/jobs/b899ad73/tmp/phase0_diag"
data = json.load(open(f"{DIAG}/data_with_ik.json"))
raw = [json.loads(l) for l in open(f"{DIAG}/ui_grasp_exec_snapshot.jsonl")]
tray = [r for r in raw if r.get("mode") == "tray"]
for d, r in zip(data, tray):
    d["width"] = r["opening_len"]

OBJ_YCB = {"Pear": "YcbPear", "MustardBottle": "YcbMustardBottle", "CrackerBox": "YcbCrackerBox"}
ORIENTS = [5,6,7,8,9]

def spawn_and_get_pc(seed, ycb_name, obj_name):
    np.random.seed(seed)
    env = EnvironmentSoArm(vis=False)
    env.remove_all_obj()
    env.load_isolated_obj(ycb_name, obj_name, False, False, pos=[0.0, -0.30, env.OBJECT_INIT_HEIGHT])
    env.dummy_simulation_steps(100)
    obs = env.get_obs(pointcloud=True)
    pc = obs["points"].reshape(-1, 3)
    env.close()
    return pc

for obj, ycb_name in OBJ_YCB.items():
    for oseed in ORIENTS:
        pc = spawn_and_get_pc(oseed, ycb_name, obj)
        cell = [d for d in data if d["object"] == obj and d["orient_seed"] == oseed]
        for d in cell:
            R = rpy_to_R(np.pi, 0.0, d["yaw"])
            xyz = np.array([d["x"], d["y"], d["z"]])
            w = d["width"]
            d["local_point_density"] = local_point_density(xyz, R, w, pc)
            d["normal_consistency"]  = normal_consistency(xyz, R, w, pc)
            d["contact_width_ratio"] = contact_width_ratio(xyz, R, w, pc)
        print(f"{obj} orient={oseed}: computed contact features for {len(cell)} trials, pc size={len(pc)}")

with open(f"{DIAG}/data_with_contact_feats.json", "w") as f:
    json.dump(data, f)
print("saved data_with_contact_feats.json")

print()
print("=" * 80)
print("Discriminative power: succ vs fail, per object, for each non-pose feature")
print("=" * 80)
OBJECTS = ["Pear", "MustardBottle", "CrackerBox"]
FEATURES = ["local_point_density", "normal_consistency", "contact_width_ratio", "ik_pe_mm"]
for obj in OBJECTS:
    obj_data = [d for d in data if d["object"] == obj]
    succ = [d for d in obj_data if d["success"]]
    fail = [d for d in obj_data if not d["success"]]
    print(f"--- {obj} (n_succ={len(succ)}, n_fail={len(fail)}) ---")
    for feat in FEATURES:
        s = np.array([d[feat] for d in succ])
        f_ = np.array([d[feat] for d in fail])
        try:
            u, p = mannwhitneyu(s, f_, alternative='two-sided')
        except ValueError:
            p = float('nan')
        print(f"    {feat:22s}: succ mean={s.mean():.5f} std={s.std():.5f} | "
              f"fail mean={f_.mean():.5f} std={f_.std():.5f} | p={p:.4f}")
    print()
