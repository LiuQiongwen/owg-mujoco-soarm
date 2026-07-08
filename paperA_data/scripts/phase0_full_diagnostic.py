import sys, os, json
sys.path.insert(0, "/lena/projects/OWG-main/.claude/worktrees/headless-ik-w1")
os.chdir("/lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding")
import numpy as np

DIAG = "/home/lina/.claude/jobs/b899ad73/tmp/phase0_diag"

# ── load and pair trials with logged poses ──────────────────────────────────
trials = [json.loads(l) for l in open(f"{DIAG}/trials.jsonl")]
raw = [json.loads(l) for l in open(f"{DIAG}/ui_grasp_exec_snapshot.jsonl")]
tray_recs = [r for r in raw if r.get("mode") == "tray"]

assert len(trials) == len(tray_recs), f"{len(trials)} trials vs {len(tray_recs)} tray records — order mismatch!"

data = []
for t, r in zip(trials, tray_recs):
    data.append({
        "object": t["object"], "orient_seed": t["orient_seed"], "gen_seed": t["gen_seed"],
        "success": t["success"] == "true",
        "x": r["x"], "y": r["y"], "z": r["z"], "yaw": r["yaw"],
    })

print(f"paired {len(data)} records")

# sanity cross-check: recompute per-cell success counts from paired data, compare to earlier
from collections import defaultdict
check = defaultdict(list)
for d in data:
    check[(d["object"], d["orient_seed"])].append(d["success"])
for obj in ["Pear", "MustardBottle", "CrackerBox"]:
    print(obj, [sum(check[(obj,o)]) for o in [5,6,7,8,9]])
print()

# ── Q1: IK reachability of each candidate ───────────────────────────────────
from tango_robot.headless_ik import HeadlessIKSolver
solver = HeadlessIKSolver()

for d in data:
    target = np.array([d["x"], d["y"], d["z"]])
    ok, pe, _ = solver.solve_ik_jaw_pos_only(target, silent=True)
    d["ik_ok"] = bool(ok)
    d["ik_pe_mm"] = float(pe) * 1000

with open(f"{DIAG}/data_with_ik.json", "w") as f:
    json.dump(data, f)
print("saved data_with_ik.json")

print()
print("=== Q1: among FAILED trials, IK-unreachable vs IK-reachable-but-failed ===")
for obj in ["Pear", "MustardBottle", "CrackerBox"]:
    obj_data = [d for d in data if d["object"] == obj]
    failed = [d for d in obj_data if not d["success"]]
    succeeded = [d for d in obj_data if d["success"]]
    fail_unreachable = sum(1 for d in failed if not d["ik_ok"])
    fail_reachable = sum(1 for d in failed if d["ik_ok"])
    succ_unreachable = sum(1 for d in succeeded if not d["ik_ok"])
    succ_reachable = sum(1 for d in succeeded if d["ik_ok"])
    print(f"{obj}: n_failed={len(failed)} (IK-unreachable={fail_unreachable}, IK-reachable-but-failed={fail_reachable})  "
          f"n_succeeded={len(succeeded)} (IK-unreachable={succ_unreachable} <- should be near 0, IK-reachable={succ_reachable})")
