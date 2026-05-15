import json, csv
from datetime import datetime
from pathlib import Path

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def parse_time(t):
    return datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

ROOT = Path(".")
nl_path = ROOT / "logs/ui_nl_exec.jsonl"
grasp_path = ROOT / "logs/ui_grasp_exec.jsonl"
out_path = ROOT / "logs/ui_joint_episodes.csv"

nl_rows = load_jsonl(nl_path)
grasp_rows = load_jsonl(grasp_path)

for r in nl_rows:
    r["_dt"] = parse_time(r["time"])
for r in grasp_rows:
    r["_dt"] = parse_time(r["time"])

nl_rows.sort(key=lambda r: r["_dt"])
grasp_rows.sort(key=lambda r: r["_dt"])

joint_rows = []
used_grasp_idx = set()

for i, nl in enumerate(nl_rows):
    t0 = nl["_dt"]
    t1 = nl_rows[i+1]["_dt"] if i+1 < len(nl_rows) else None

    act = nl.get("action", {}) or {}
    a_type = act.get("action")
    a_input = act.get("input")
    a_target = act.get("target_id")
    grasps_idx = act.get("grasps", [])

    # 在 t0 ~ t1 区间内，找最近的一条 grasp 日志（模式 tray）
    cand = []
    for j, g in enumerate(grasp_rows):
        if j in used_grasp_idx:
            continue
        gt = g["_dt"]
        if gt < t0:
            continue
        if t1 is not None and gt >= t1:
            continue
        cand.append((gt, j, g))
    cand.sort(key=lambda x: x[0])

    if cand:
        _, j, g = cand[0]
        used_grasp_idx.add(j)
        row = {
            "time_nl": nl["time"],
            "query": nl.get("query", ""),
            "hl_action": a_type,
            "input_id": a_input,
            "target_id": a_target,
            "grasps_idx": ";".join(str(x) for x in grasps_idx),
            "success_nl": int(bool(nl.get("success"))),

            "time_grasp": g["time"],
            "mode": g.get("mode", ""),
            "obj_id": g.get("obj_id", ""),
            "obj_height": g.get("obj_height", ""),
            "x": g.get("x", ""),
            "y": g.get("y", ""),
            "z": g.get("z", ""),
            "yaw": g.get("yaw", ""),
            "opening_len": g.get("opening_len", ""),
            "success_grasp": int(bool(g.get("success_grasp"))),
            "success_target": int(bool(g.get("success_target"))),
        }
    else:
        # 没找到匹配的 grasp，就只记 NL 部分
        row = {
            "time_nl": nl["time"],
            "query": nl.get("query", ""),
            "hl_action": a_type,
            "input_id": a_input,
            "target_id": a_target,
            "grasps_idx": ";".join(str(x) for x in grasps_idx),
            "success_nl": int(bool(nl.get("success"))),

            "time_grasp": "",
            "mode": "",
            "obj_id": "",
            "obj_height": "",
            "x": "",
            "y": "",
            "z": "",
            "yaw": "",
            "opening_len": "",
            "success_grasp": "",
            "success_target": "",
        }

    joint_rows.append(row)

fields = list(joint_rows[0].keys()) if joint_rows else []
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(joint_rows)

print("Wrote joint episodes ->", out_path, "N =", len(joint_rows))


