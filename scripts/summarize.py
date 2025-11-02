#!/usr/bin/env python
# summarize.py — 从 summary.csv 读取结果，汇总/排序打印 Top-N，并给出每个(yaw, voxel, zm)的均值
import csv, sys, statistics
from collections import defaultdict

csv_path = sys.argv[1] if len(sys.argv) > 1 else "grasp_6dof/out/summary.csv"
rows=[]
with open(csv_path, newline="") as f:
    r=csv.DictReader(f)
    for t in r:
        try:
            succ=float(t.get("success_rate","") or 0.0)
        except:
            succ=0.0
        rows.append({
            "time":t.get("time",""),
            "obj":t.get("obj",""),
            "cube_scale":t.get("cube_scale",""),
            "topk":t.get("topk",""),
            "seed":t.get("seed",""),
            "descent_step":t.get("descent_step",""),
            "descend_clear":t.get("descend_clear",""),
            "success_rate":succ,
            # 从日志命名解析参数（若你需要可更严谨地从文件名/命令记录）
        })

# 打印 Top 10
rows_sorted=sorted(rows, key=lambda x: x["success_rate"], reverse=True)
print("Top 10 (by success_rate):")
for r in rows_sorted[:10]:
    print(f'{r["time"]}  obj={r["obj"]} scale={r["cube_scale"]} '
          f'topk={r["topk"]} seed={r["seed"]}  succ={r["success_rate"]:.3f}')

# 如果你按照 run_grid.sh 的命名规则，validated 文件名可用于聚合参数
# 这里示例：从 summary.csv 同目录下的 validated 文件名再做一次统计（可选）
print("\n[hint] 若需要更细粒度分组( yaw/voxel/zm )，可从数据集文件名解析后再聚合。")

