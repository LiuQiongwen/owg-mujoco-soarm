#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
semantic_pc_from_masks.py

离线脚本：读取 grounding 日志 .npz（image + depth + masks + K [+ pose_cam] [+ query]），
把选中的 2D 语义 mask 回投成 3D 语义点云，保存到 semantic_pc/ 目录。

用法示例：
    cd ~/OWG-main
    python tools/semantic_pc_from_masks.py \
        --log-dir logs/grounding_examples \
        --out-dir semantic_pc \
        --min-depth 0.1 \
        --max-depth 2.0 \
        --stride 2
"""

import os
import argparse
import glob
import numpy as np


def backproject_mask_to_points(mask, depth, K, pose_cam=None,
                               min_depth=0.0, max_depth=np.inf, stride=1):
    """
    将 2D mask + depth 回投成 3D 点云。
    mask: (H, W) bool
    depth: (H, W) float32, 以米为单位
    K: (3, 3) 相机内参矩阵
    pose_cam: (4, 4) 可选，世界坐标系下的相机位姿，
              如果提供，则把点从相机坐标系变换到世界坐标系
    min_depth, max_depth: 深度范围过滤
    stride: 下采样步长，减小点数（例如 stride=2）
    """
    H, W = depth.shape
    assert mask.shape == depth.shape, "mask 和 depth 尺寸不一致"

    # 取出 mask 范围内的像素
    ys, xs = np.nonzero(mask)
    if stride > 1:
        ys = ys[::stride]
        xs = xs[::stride]

    d = depth[ys, xs]
    # 深度过滤
    valid = (d > min_depth) & (d < max_depth)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    d = d[valid].astype(np.float32)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 像素坐标 → 相机坐标系
    # X_cam = ( (u-cx)*z/fx, (v-cy)*z/fy, z )
    X = (xs - cx) * d / fx
    Y = (ys - cy) * d / fy
    Z = d

    pts_cam = np.stack([X, Y, Z], axis=1)  # (N, 3)

    if pose_cam is None:
        return pts_cam.astype(np.float32)

    # 扩展成齐次坐标，并用 pose_cam (4x4) 变换到世界坐标
    R = pose_cam[:3, :3]
    t = pose_cam[:3, 3]
    pts_world = (R @ pts_cam.T + t.reshape(3, 1)).T  # (N, 3)
    return pts_world.astype(np.float32)


def process_file(path, out_dir, args):
    """
    处理单个 grounding .npz 文件，生成选中对象的 3D 点云并保存。
    """
    data = np.load(path, allow_pickle=True)
    base = os.path.basename(path).replace(".npz", "")

    image = data["image"]             # HxWx3
    depth = data["depth"]             # HxW
    masks = data["masks"]             # NxHxW
    labels = data["labels"]           # (N,)
    selected_ids = data["selected_ids"]  # (M,)

    # ✅ 1) 如果有真实 K，就用真实的；否则构造一个默认 K
    if "K" in data.files:
        K = data["K"]
    else:
        H, W = depth.shape
        # 假设水平 FOV = 60°，构造 pinhole intrinsics
        fov_deg = 60.0
        f = W / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))  # fx = fy = f
        cx, cy = W / 2.0, H / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], dtype=np.float32)
        print(f"[WARN] {path} has no K, using default intrinsics with fov={fov_deg}°")

    pose_cam = data["pose_cam"] if "pose_cam" in data.files else None
    query = str(data["query"]) if "query" in data.files else ""

    H, W, _ = image.shape
    N = masks.shape[0]

    print(f"[INFO] Processing {base}: image={H}x{W}, N_masks={N}, selected_ids={selected_ids}")

    # 为了简单：只处理 selected_ids 中每个 id 对应的 mask
    pcs = []
    metas = []

    for sid in selected_ids:
        # 找到 label == sid 的那一个 mask（有可能没有，做个防御）
        idxs = np.where(labels == sid)[0]
        if len(idxs) == 0:
            print(f"  ⚠️ selected id {sid} not found in labels {labels}")
            continue
        idx = idxs[0]
        mask = masks[idx].astype(bool)

        pts = backproject_mask_to_points(
            mask,
            depth,
            K,
            pose_cam=pose_cam,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            stride=args.stride,
        )
        if pts.shape[0] == 0:
            print(f"  ⚠️ No valid points for selected id {sid}")
            continue

        pcs.append(pts)
        metas.append({
            "sid": int(sid),
            "label_index": int(idx),
        })

    if not pcs:
        print(f"[WARN] No point cloud generated for {base}, skip.")
        return

    # 拼成一个大的点云，或也可以分开保存
    pc_all = np.concatenate(pcs, axis=0)  # (M_total, 3)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, base + "_semantic_pc.npz")
    np.savez_compressed(
        out_path,
        pc=pc_all,          # 所有选中目标的 3D 点
        query=query,
        metas=np.array(metas, dtype=object),
    )
    print(f"[OK] Saved semantic point cloud → {out_path}, N_pts={pc_all.shape[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="logs/grounding_examples",
                        help="grounding .npz 日志所在目录")
    parser.add_argument("--out-dir", type=str, default="semantic_pc",
                        help="输出 3D 语义点云保存目录")
    parser.add_argument("--min-depth", type=float, default=0.05,
                        help="最小有效深度（米）")
    parser.add_argument("--max-depth", type=float, default=2.0,
                        help="最大有效深度（米）")
    parser.add_argument("--stride", type=int, default=1,
                        help="像素下采样步长（>1 可减少点数）")

    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.log_dir, "*.npz")))
    if not paths:
        print(f"[WARN] No .npz found in {args.log_dir}")
        return

    print(f"[INFO] Found {len(paths)} grounding logs in {args.log_dir}")
    for p in paths:
        try:
            process_file(p, args.out_dir, args)
        except Exception as e:
            print(f"[ERROR] Failed to process {p}: {e}")


if __name__ == "__main__":
    main()

