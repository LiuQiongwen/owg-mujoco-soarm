import sys, os, json
sys.path.insert(0, "/lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding")
os.chdir("/lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import torch.nn as nn

T_DIM, POSE_DIM, VIS_DIM = 3, 6, 256
IN_DIM = T_DIM + POSE_DIM + VIS_DIM

class SinusoidalTimeEmbed(nn.Module):
    def forward(self, t):
        t = t.unsqueeze(-1)
        return torch.cat([t, torch.sin(np.pi * t), torch.cos(np.pi * t)], dim=-1)

class VelocityNet(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbed()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, POSE_DIM),
        )
    def forward(self, t, xt, cond):
        te = self.t_embed(t)
        return self.net(torch.cat([te, xt, cond], dim=-1))

def sample_poses_fixed(model, cond, n, steps, seed, device):
    model.eval()
    cond = cond.unsqueeze(0).expand(n, -1).to(device) if cond.dim() == 1 else cond.to(device)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    x = torch.randn(n, POSE_DIM, device=device, generator=gen)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = torch.full((n,), i * dt, device=device)
        with torch.no_grad():
            x = x + model(t_val, x, cond) * dt
    return x.cpu().numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = json.load(open("grasp_6dof/models/cfm_allobj_ot_stats.json"))
model = VelocityNet(hidden=512)
model.load_state_dict(torch.load("grasp_6dof/models/cfm_allobj_ot.pt", map_location="cpu", weights_only=False))
model.eval(); model = model.to(device)
pmean = np.array(stats["pose_mean"], dtype=np.float32)
pstd = np.array(stats["pose_std"], dtype=np.float32)
cond_pear = torch.tensor(stats["mean_vis_per_obj"]["pear"], dtype=torch.float32, device=device)

# Spawn positions for Pear at each orientation (reuse the exact spawn recipe)
from tango_robot.env_soarm import EnvironmentSoArm
def spawn_and_get_com(seed, obj_ycb_name, obj_name):
    np.random.seed(seed)
    env = EnvironmentSoArm(vis=False)
    env.remove_all_obj()
    spawn_pos = [0.0, -0.30, env.OBJECT_INIT_HEIGHT]
    env.load_isolated_obj(obj_ycb_name, obj_name, False, False, pos=spawn_pos)
    env.dummy_simulation_steps(100)
    com = env.get_obj_com_pos(env.obj_ids[0])
    env.close()
    return float(com[0]), float(com[1]), float(com[2])

from tango_robot.headless_ik import HeadlessIKSolver
solver = HeadlessIKSolver()

ORIENT_SEEDS = [5,6,7,8,9]
BASES_N10 = [1,11,21,31,41,51,61,71,81,91]
GRASP_Z_MARGIN = 0.020  # GRASP_Z_TABLE_MARGIN from env_soarm.py

results = []
for oseed in ORIENT_SEEDS:
    gx, gy, gcom_z = spawn_and_get_com(oseed, "YcbPear", "Pear")
    gz = gcom_z + GRASP_Z_MARGIN
    for base in BASES_N10:
        ensemble = []
        for i in range(10):
            pn = sample_poses_fixed(model, cond_pear, n=1, steps=20, seed=base+i, device=device)
            pose = (pn * pstd + pmean)[0]
            ensemble.append(pose)
        ensemble = np.array(ensemble)
        pe_iks = []
        for p in ensemble:
            tx = float(p[0]) - float(pmean[0]) + gx
            ty = float(p[1]) - float(pmean[1]) + gy
            target = np.array([tx, ty, gz])
            _, pe, _ = solver.solve_ik_jaw_pos_only(target, silent=True)
            pe_iks.append(pe)
        winner_idx = int(np.argmin(pe_iks))
        results.append({
            "orient_seed": oseed, "ensemble_base": base,
            "pe_iks": pe_iks, "winner_idx": winner_idx,
            "winner_pe_ik_mm": pe_iks[winner_idx]*1000,
            "median_pe_ik_mm": float(np.median(pe_iks))*1000,
            "max_pe_ik_mm": float(np.max(pe_iks))*1000,
        })
        print(f"orient={oseed} base={base}: winner_pe={pe_iks[winner_idx]*1000:.3f}mm  "
              f"ensemble_pe_range=[{min(pe_iks)*1000:.2f},{max(pe_iks)*1000:.2f}]mm")

with open("/home/lina/.claude/jobs/b899ad73/tmp/phase1_v2/pear_ensemble_reconstruction.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved")
