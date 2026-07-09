"""
Post-hoc, single-feature follow-up for PowerDrill (2026-07-09).

PowerDrill is the only object with a clean null across all 3 contact
features in contact_features_bonferroni_bh_5obj_clean.csv (all p_raw>=0.17).
grasp_6dof/grasp_sampler.py already implements lateral_score(R, major_axis)
(gripper-axis vs. object-major-axis alignment) but it was never tested here.
Unlike elongation_ratio (rotation-invariant, ~constant per object -- not a
valid within-object feature), lateral_score depends on the trial's yaw via R,
so it has real within-object variance and is a valid candidate.

This is a SINGLE exploratory test, deliberately NOT folded into the existing
3-feature x 5-object = 15-test Bonferroni/BH family in
contact_features_bonferroni_bh_5obj_clean.csv -- reported separately so the
already-published 15-test family's p-values/significance flags don't need
re-deriving.

No new grasp trials: reuses PowerDrill's already-logged 50 trials (x/y/z/yaw/
width/success) from phase0_diag_extended/data_with_contact_feats_new3.json.
Only re-does the 5 lightweight per-orient_seed object spawns (to get the
point cloud for compute_pca_axes) -- same cost class as the original contact
features, not a new-sim-trials cost.

Output: formal_results/powerdrill_lateral_score_posthoc.csv
"""
import sys, os, json
sys.path.insert(0, "/lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding")
os.chdir("/lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from scipy.stats import mannwhitneyu

BASE = "/lena/projects/OWG-main/paperA_data"
DIAG = f"{BASE}/phase0_diag_extended"

data = json.load(open(f"{DIAG}/data_with_contact_feats_new3.json"))
pd_trials = [d for d in data if d["object"] == "PowerDrill"]
assert len(pd_trials) == 50, f"expected 50 PowerDrill trials, got {len(pd_trials)}"

from tango_robot.env_soarm import EnvironmentSoArm
from grasp_6dof.grasp_sampler import rpy_to_R, compute_pca_axes, lateral_score

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


for oseed in ORIENTS:
    pc = spawn_and_get_pc(oseed, "YcbPowerDrill", "PowerDrill")
    major_axis, _minor_axis, elongation = compute_pca_axes(pc)
    cell = [d for d in pd_trials if d["orient_seed"] == oseed]
    for d in cell:
        R = rpy_to_R(np.pi, 0.0, d["yaw"])
        d["lateral_score"] = lateral_score(R, major_axis)
        d["elongation_ratio_static"] = elongation  # reported for context only, NOT tested (no within-object variance)
    print(f"PowerDrill orient={oseed}: lateral_score computed for {len(cell)} trials, "
          f"elongation_ratio={elongation:.3f}, pc size={len(pc)}")

succ = np.array([d["lateral_score"] for d in pd_trials if d["success"]])
fail = np.array([d["lateral_score"] for d in pd_trials if not d["success"]])
u, p = mannwhitneyu(succ, fail, alternative="two-sided")
rank_biserial = 1 - 2 * u / (len(succ) * len(fail))

row = {
    "object": "PowerDrill",
    "feature": "lateral_score",
    "n_succ": len(succ),
    "n_fail": len(fail),
    "succ_mean": round(float(succ.mean()), 5),
    "succ_std": round(float(succ.std()), 5),
    "fail_mean": round(float(fail.mean()), 5),
    "fail_std": round(float(fail.std()), 5),
    "mannwhitney_U": u,
    "p_raw": p,
    "rank_biserial_effect_size": round(rank_biserial, 4),
    "note": ("post-hoc single exploratory test, NOT part of the pre-specified "
             "3-feature x 5-object=15-test Bonferroni/BH family in "
             "contact_features_bonferroni_bh_5obj_clean.csv; no correction applied"),
}

import csv
out_path = f"{BASE}/formal_results/powerdrill_lateral_score_posthoc.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    w.writeheader()
    w.writerow(row)
print(f"\nMann-Whitney U={u}, p={p:.5f}, rank-biserial={rank_biserial:.4f}")
print(f"wrote {out_path}")

# Also persist the per-trial lateral_score data for reproducibility/audit.
with open(f"{DIAG}/data_with_lateral_score_powerdrill.json", "w") as f:
    json.dump(pd_trials, f)
print(f"wrote {DIAG}/data_with_lateral_score_powerdrill.json")
