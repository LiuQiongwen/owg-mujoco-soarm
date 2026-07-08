import json
import numpy as np
from scipy.stats import mannwhitneyu

DIAG = "/home/lina/.claude/jobs/b899ad73/tmp/phase0_diag"
data = json.load(open(f"{DIAG}/data_with_ik.json"))

# pair in width (opening_len) from the raw snapshot, same 150-record order
raw = [json.loads(l) for l in open(f"{DIAG}/ui_grasp_exec_snapshot.jsonl")]
tray = [r for r in raw if r.get("mode") == "tray"]
assert len(tray) == len(data)
for d, r in zip(data, tray):
    d["width"] = r["opening_len"]

OBJECTS = ["Pear", "CrackerBox"]
ORIENTS = [5,6,7,8,9]

print("=" * 75)
print("(a) Does 'distance from consensus' survive within the IK-reachable subset?")
print("=" * 75)
for obj in OBJECTS:
    print(f"--- {obj} ---")
    all_d_succ, all_d_fail = [], []
    for o in ORIENTS:
        cell = [d for d in data if d["object"] == obj and d["orient_seed"] == o and d["ik_ok"]]
        if len(cell) < 5:
            continue
        n_succ = sum(d["success"] for d in cell)
        if n_succ == 0 or n_succ == len(cell):
            continue
        pts = np.array([[d["x"], d["y"], d["yaw"]*0.05] for d in cell])
        median = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - median, axis=1)
        succ_mask = np.array([d["success"] for d in cell])
        d_succ, d_fail = dists[succ_mask], dists[~succ_mask]
        all_d_succ.extend(d_succ.tolist()); all_d_fail.extend(d_fail.tolist())
        print(f"    orient={o} (ik_ok subset, n={len(cell)}, succ={n_succ}): "
              f"succ_dist={d_succ.mean():.4f}  fail_dist={d_fail.mean():.4f}")
    if all_d_succ and all_d_fail:
        u, p = mannwhitneyu(all_d_succ, all_d_fail, alternative='two-sided')
        print(f"  POOLED (ik_ok only): succ mean={np.mean(all_d_succ):.4f} (n={len(all_d_succ)}) "
              f"vs fail mean={np.mean(all_d_fail):.4f} (n={len(all_d_fail)})  Mann-Whitney p={p:.4f}")
    print()

print("=" * 75)
print("(b) IK error margin (pe_ik, even within reachable) and width: succ vs fail")
print("=" * 75)
for obj in OBJECTS:
    obj_data = [d for d in data if d["object"] == obj]
    succ = [d for d in obj_data if d["success"]]
    fail = [d for d in obj_data if not d["success"]]
    print(f"--- {obj} ---")
    for field, label in [("ik_pe_mm", "IK error (mm, lower=more margin)"), ("width", "gripper width (m)")]:
        s = np.array([d[field] for d in succ])
        f = np.array([d[field] for d in fail])
        u, p = mannwhitneyu(s, f, alternative='two-sided')
        print(f"    {label}: succ mean={s.mean():.4f} std={s.std():.4f} | "
              f"fail mean={f.mean():.4f} std={f.std():.4f} | Mann-Whitney p={p:.4f}")
    # within only ik_ok candidates, does pe_ik margin still separate succ/fail?
    ok_data = [d for d in obj_data if d["ik_ok"]]
    s_ok = np.array([d["ik_pe_mm"] for d in ok_data if d["success"]])
    f_ok = np.array([d["ik_pe_mm"] for d in ok_data if not d["success"]])
    if len(s_ok) > 1 and len(f_ok) > 1:
        u, p = mannwhitneyu(s_ok, f_ok, alternative='two-sided')
        print(f"    WITHIN ik_ok only -- pe_ik margin: succ mean={s_ok.mean():.4f} fail mean={f_ok.mean():.4f} p={p:.4f}")
    print()
