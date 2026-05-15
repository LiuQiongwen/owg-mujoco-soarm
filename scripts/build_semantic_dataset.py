import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

joint_path = LOG_DIR / "ui_joint_episodes.csv"
out_path   = LOG_DIR / "semantic_episodes_lggsn.csv"

print("[INFO] Load joint episodes from:", joint_path)

rows = []
with open(joint_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print("[INFO] Total joint rows:", len(rows))

def str2bool(v):
    # ui_joint_episodes 里是 'True'/'False' 字符串
    return str(v).strip().lower() == "true"

clean_rows = []

for r in rows:
    # 1) 语言理解成功
    if not str2bool(r.get("success_nl", "")):
        continue

    # 2) 抓取执行成功（抓取 + 放置都成功）
    if not str2bool(r.get("success_grasp", "")):
        continue
    if not str2bool(r.get("success_target", "")):
        continue

    # 3) 必须有 grasp 日志（有些 remove 失败的没有位姿信息）
    if not r.get("time_grasp"):
        continue

    # 4) 取出我们关心的字段
    try:
        time_nl   = r.get("time_nl", "")
        query     = r.get("query", "")
        mode      = r.get("mode", "")          # tray / free 等
        obj_id    = int(r.get("obj_id", -1))   # PyBullet 里的 object id
        target_id = r.get("target_id", "")     # HL planner 选的 target_id

        x   = float(r.get("x", "nan"))
        y   = float(r.get("y", "nan"))
        z   = float(r.get("z", "nan"))
        yaw = float(r.get("yaw", "nan"))
        w   = float(r.get("opening_len", "nan"))
        h   = float(r.get("obj_height", "nan"))

        # 目前 env 是 4-DoF，所以 roll/pitch 先置 0，后面接 6-DoF 时再替换
        roll  = 0.0
        pitch = 0.0

    except ValueError:
        # 有坏行就跳过
        continue

    clean_rows.append({
        "time": time_nl,
        "query": query,
        "mode": mode,
        "obj_id": obj_id,
        "target_id": target_id,

        # 统一成 LG-GSN 以后要吃的位姿字段
        "x": x,
        "y": y,
        "z": z,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "width": w,
        "obj_height": h,

        # 这里全是成功 episode，就不再重复 success 标记了
    })

print("[INFO] Clean successful NL+grasp episodes:", len(clean_rows))

# 写出为 CSV
fields = [
    "time", "query", "mode", "obj_id", "target_id",
    "x", "y", "z", "roll", "pitch", "yaw", "width", "obj_height"
]

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(clean_rows)

print("[OK] Saved semantic dataset ->", out_path)
