"""
Extends the phase0 (IK reachability) + phase2 (contact geometry features)
diagnostic pipeline to 3 more objects: TomatoSoupCan, PowerDrill, Scissors.
Mirrors phase0_full_diagnostic.py + phase2_contact_features.py exactly (same
IK solver, same contact-feature functions), applied to the new trial data in
phase0_diag_extended/ instead of the original 3-object phase0_diag/ data.

Input: phase0_diag_extended/trials_new3.jsonl (150 trials, produced by
run_phase0_extended.sh) + ui_grasp_exec_snapshot_new3.jsonl (paired 1:1 by
mode=="tray" records, same pairing assumption as phase0_full_diagnostic.py).

Output: phase0_diag_extended/data_with_contact_feats_new3.json -- same schema
as phase0_diag/data_with_contact_feats.json, so run_contact_features_stats.py
can be trivially extended to load both and cover all 6 objects.
"""
import sys, os, json
sys.path.insert(0, "/lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding")
os.chdir("/lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

DIAG = "/lena/projects/OWG-main/paperA_data/phase0_diag_extended"

# ── pair trials with logged poses (same logic as phase0_full_diagnostic.py) ──
trials = [json.loads(l) for l in open(f"{DIAG}/trials_new3.jsonl")]
raw = [json.loads(l) for l in open(f"{DIAG}/ui_grasp_exec_snapshot_new3.jsonl")]
tray_recs = [r for r in raw if r.get("mode") == "tray"]

assert len(trials) == len(tray_recs), f"{len(trials)} trials vs {len(tray_recs)} tray records — order mismatch!"

data = []
for t, r in zip(trials, tray_recs):
    data.append({
        "object": t["object"], "orient_seed": t["orient_seed"], "gen_seed": t["gen_seed"],
        "success": t["success"] == "true",
        "x": r["x"], "y": r["y"], "z": r["z"], "yaw": r["yaw"],
        "width": r["opening_len"],
    })

print(f"paired {len(data)} records")
from collections import defaultdict
check = defaultdict(list)
for d in data:
    check[(d["object"], d["orient_seed"])].append(d["success"])
for obj in ["TomatoSoupCan", "PowerDrill", "Scissors"]:
    print(obj, [sum(check[(obj, o)]) for o in [5, 6, 7, 8, 9]])
print()

# ── IK reachability (phase0) ────────────────────────────────────────────────
from tango_robot.headless_ik import HeadlessIKSolver
solver = HeadlessIKSolver()
for d in data:
    target = np.array([d["x"], d["y"], d["z"]])
    ok, pe, _ = solver.solve_ik_jaw_pos_only(target, silent=True)
    d["ik_ok"] = bool(ok)
    d["ik_pe_mm"] = float(pe) * 1000

# ── contact geometry features (phase2) ──────────────────────────────────────
from tango_robot.env_soarm import EnvironmentSoArm
from grasp_6dof.grasp_sampler import rpy_to_R, local_point_density, normal_consistency, contact_width_ratio

OBJ_YCB = {"TomatoSoupCan": "YcbTomatoSoupCan", "PowerDrill": "YcbPowerDrill", "Scissors": "YcbScissors"}
ORIENTS = [5, 6, 7, 8, 9]


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
            d["normal_consistency"] = normal_consistency(xyz, R, w, pc)
            d["contact_width_ratio"] = contact_width_ratio(xyz, R, w, pc)
        print(f"{obj} orient={oseed}: computed contact features for {len(cell)} trials, pc size={len(pc)}")

with open(f"{DIAG}/data_with_contact_feats_new3.json", "w") as f:
    json.dump(data, f)
print("saved data_with_contact_feats_new3.json")
