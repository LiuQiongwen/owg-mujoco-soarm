#!/usr/bin/env python
import pandas as pd
from pathlib import Path

# 脚本在 scripts/ 目录下，ROOT 指向项目根目录
ROOT = Path(__file__).resolve().parent.parent

joint_path = ROOT / "logs/ui_joint_episodes.csv"
out_path = ROOT / "lggsn_dataset.csv"

print("Loading episodes from:", joint_path)
joint = pd.read_csv(joint_path)

print("Total episodes:", len(joint))

# 1) 过滤掉没有几何信息的行（失败时 x,y,z,yaw 等经常是空/null）
geom_mask = joint["x"].notna() & (joint["x"] != "")
joint = joint[geom_mask].copy()
print("Episodes with geometry:", len(joint))

# 2) 转成数值类型
for col in ["x", "y", "z", "yaw", "opening_len",
            "obj_height", "success_nl", "success_grasp", "success_target"]:
    if col in joint.columns:
        joint[col] = pd.to_numeric(joint[col], errors="coerce")

# 3) 定义一个 “成功标记”：
#    success = 自然语言成功 且 实际物体成功放到 tray
joint["success"] = (
    (joint["success_nl"] == 1) &
    (joint["success_target"] == 1)
).astype(int)

# 4) 现在先简单一点：label 就等于 success
#    - label = 1: 这个 grasp + 这个 query 的整体任务成功
#    - label = 0: 要么 NL 失败，要么执行失败（但仍有几何信息）
joint["label"] = joint["success"]

# 5) 没有 roll/pitch，用 0 填
joint["roll"] = 0.0
joint["pitch"] = 0.0

# 6) 把 opening_len 重命名成 width，字段对齐 LG-GSN 设计
joint.rename(columns={
    "opening_len": "width"
}, inplace=True)

# 7) 加上场景 / 来源两个标记字段，方便以后 multi-dataset 混合
joint["scene"] = "owg_tray"
joint["source"] = "ui_log_auto_v1"

# 8) 按你之前约定的字段顺序导出
cols = [
    "query",
    "x", "y", "z",
    "roll", "pitch", "yaw",
    "width", "obj_height",
    "success", "label",
    "scene", "source",
]

joint_out = joint[cols].copy()
joint_out.to_csv(out_path, index=False)

print("Wrote:", out_path, "N =", len(joint_out))
print("Label distribution:")
print(joint_out["label"].value_counts())

