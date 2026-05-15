import csv
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

src_path = LOG_DIR / "semantic_episodes_lggsn.csv"
out_path = LOG_DIR / "lggsn_pairs.csv"

print("[INFO] Load semantic episodes from:", src_path)

rows = []
with open(src_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print("[INFO] semantic episodes N =", len(rows))
if len(rows) < 2:
    print("[WARN] Not enough episodes to build negatives.")
    exit(0)

# 所有 query 列表
all_queries = [r["query"] for r in rows]

pairs = []

for r in rows:
    # 正样本：query 与自身 grasp
    pos = {
        "query": r["query"],
        "x": r["x"],
        "y": r["y"],
        "z": r["z"],
        "roll": r["roll"],
        "pitch": r["pitch"],
        "yaw": r["yaw"],
        "width": r["width"],
        "obj_height": r["obj_height"],
        "label": 1,
    }
    pairs.append(pos)

    # 负样本：随机换一个“不同”的 query（简单负样本构造）
    neg_query = r["query"]
    tries = 0
    while neg_query == r["query"] and tries < 10:
        neg_query = random.choice(all_queries)
        tries += 1

    neg = {
        "query": neg_query,
        "x": r["x"],
        "y": r["y"],
        "z": r["z"],
        "roll": r["roll"],
        "pitch": r["pitch"],
        "yaw": r["yaw"],
        "width": r["width"],
        "obj_height": r["obj_height"],
        "label": 0,
    }
    pairs.append(neg)

print("[INFO] built pairs N =", len(pairs))

fields = [
    "query", "x", "y", "z",
    "roll", "pitch", "yaw",
    "width", "obj_height",
    "label",
]

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(pairs)

print("[OK] Saved LG-GSN training pairs ->", out_path)
