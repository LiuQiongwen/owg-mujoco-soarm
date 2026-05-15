#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch generator for 6-DoF grasp dataset.

For each object:
1) Call grasp_6dof/grasp_generator_6dof.py to sample candidate grasps.
2) Call grasp_6dof/validate_grasps_panda.py to label success in PyBullet.
3) Convert validated grasps into LG-GSN friendly JSON/CSV:

Fields:
    x, y, z, roll, pitch, yaw, width, score, success
"""

import json
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # OWG-main 根目录

# ======= 在这里配置要处理的物体 =======
OBJECTS = [
    dict(
        name="cylinder_0p08",
        mesh="grasp_6dof/assets/cylinder_0p08.ply",   # 如果路径不同，这里改一下
        urdf="grasp_6dof/assets/cylinder.urdf",
        table_z=-0.004,
        n=2048,
        topk=512,
        topk_val=64,
        default_width=0.08,   # 0.08m gripper opening for training
    ),
    dict(
        name="sphere_small_0p08",
        mesh="grasp_6dof/assets/sphere_small_0p08.ply",
        urdf="grasp_6dof/assets/sphere_small.urdf",
        table_z=-0.004,
        n=2048,
        topk=512,
        topk_val=64,
        default_width=0.08,
    ),
    # 以后加更多物体，只要 append 新的 dict 即可
]


def run_cmd(cmd, cwd=None):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def convert_validated_to_lggsn(in_path: Path, out_json: Path, out_csv: Path,
                               default_width: float = 0.08):
    """
    把 *_grasps_validated.json 转成 LG-GSN 训练友好的 JSON + CSV
    统一字段:
        x,y,z,roll,pitch,yaw,width,score,success
    """
    with in_path.open() as f:
        data = json.load(f)

    if isinstance(data, dict):
        grasps = data.get("grasps", [])
    elif isinstance(data, list):
        grasps = data
    else:
        raise RuntimeError(f"Unknown json structure in {in_path}")

    rows = []
    for g in grasps:
        pos = g.get("position") or g.get("pos") or [None, None, None]
        rpy = g.get("rpy") or g.get("euler") or [None, None, None]

        width = g.get("width", default_width)
        score = g.get("score", 1.0)
        succ = bool(g.get("success", True))

        row = dict(
            x=pos[0],
            y=pos[1],
            z=pos[2],
            roll=rpy[0],
            pitch=rpy[1],
            yaw=rpy[2],
            width=width,
            score=score,
            success=int(succ),   # 0/1 更方便训练
        )
        rows.append(row)

    # 写 JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(rows, f, indent=2)
    # 写 CSV
    fieldnames = ["x", "y", "z", "roll", "pitch", "yaw", "width", "score", "success"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] {in_path.name} -> {out_json.name}, {out_csv.name} (N={len(rows)})")


def main():
    for cfg in OBJECTS:
        name = cfg["name"]
        print("\n" + "=" * 80)
        print(f"[OBJECT] {name}")
        print("=" * 80)

        mesh = ROOT / cfg["mesh"]
        urdf = ROOT / cfg["urdf"]
        dataset_dir = ROOT / "grasp_6dof" / "dataset"
        out_dir = ROOT / "grasp_6dof" / "out"

        dataset_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_json = dataset_dir / f"{name}_grasps.json"
        validated_json = out_dir / f"{name}_grasps_validated.json"

        # 1) 采样 6-DoF grasps
        cmd_gen = [
            "python", "grasp_6dof/grasp_generator_6dof.py",
            "--obj", str(mesh),
            "--out", str(raw_json),
            "--n", str(cfg.get("n", 2048)),
            "--topk", str(cfg.get("topk", 512)),
            "--table_z", str(cfg.get("table_z", -0.004)),
        ]
        run_cmd(cmd_gen, cwd=ROOT)

        # 2) PyBullet 中做 Panda 验证，打 success label
        cmd_val = [
            "python", "grasp_6dof/validate_grasps_panda.py",
            "--obj", str(urdf),
            "--grasps", str(raw_json),
            "--out", str(validated_json),
            "--vis", "0",
            "--topk", str(cfg.get("topk_val", 64)),
            "--seed", "0",
            "--descent-step", "0.0006",
            "--descend-clear", "0.02",
            "--vel-close", "1.0",
            "--pos-close", "900",
            "--squeeze", "1.0",
        ]
        run_cmd(cmd_val, cwd=ROOT)

        # 3) 转成 LG-GSN 统一格式 (JSON + CSV)
        lggsn_json = out_dir / f"{name}_lggsn.json"
        lggsn_csv = out_dir / f"{name}_lggsn.csv"
        convert_validated_to_lggsn(
            validated_json, lggsn_json, lggsn_csv,
            default_width=cfg.get("default_width", 0.08),
        )

    print("\nAll objects processed. 6-DoF dataset ready for LG-GSN.")


if __name__ == "__main__":
    main()
