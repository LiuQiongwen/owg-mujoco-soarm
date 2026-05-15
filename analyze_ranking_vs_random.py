#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ranking vs Random for geometric grasps on YCB objects.

For each validated grasps file data/grasps_val/*_geom_val.json:
- Load the list of grasps (each has 'success' and usually 'score').
- Compute SR@k when grasps are sorted by 'score' (geom ranking).
- Compute SR@k when grasps are in random order, averaged over many shuffles.
Outputs:
- Prints per-object SR@k (geom vs random) for k in {1,3,5}.
- Prints average improvement across objects.
- Saves a figure 'results/ranking_vs_random.png' with mean SR@k curves.
"""

import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# top-k list
K_LIST = [1, 3, 5, 10, 20, 32, 64]
N_SHUFFLES = 50  # how many random permutations per object


def load_grasps(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a list")
    return data


def sr_at_k(success_array, k):
    """success_array: 1D numpy array of 0/1"""
    n = len(success_array)
    kk = min(k, n)
    if kk <= 0:
        return 0.0
    return float(success_array[:kk].mean())


def compute_geom_and_random_sr(grasps, k_list, n_shuffles=50):
    """
    grasps: list of dicts with 'success' and optionally 'score'.
    Returns:
        sr_geom: dict{k: value}
        sr_rand_mean: dict{k: value}
        sr_rand_std: dict{k: value}
    """
    n = len(grasps)
    if n == 0:
        zero = {k: 0.0 for k in k_list}
        return zero, zero, zero

    # 1) Geom ranking: sort by 'score' descending if exists, otherwise keep as is
    if "score" in grasps[0]:
        sorted_grasps = sorted(grasps, key=lambda g: g.get("score", 0.0), reverse=True)
    else:
        sorted_grasps = list(grasps)

    succ_geom = np.array([1.0 if g.get("success") else 0.0 for g in sorted_grasps], dtype=float)
    sr_geom = {k: sr_at_k(succ_geom, k) for k in k_list}

    # 2) Random ranking: shuffle multiple times, average SR@k
    succ_orig = np.array([1.0 if g.get("success") else 0.0 for g in grasps], dtype=float)
    sr_rand_all = {k: [] for k in k_list}

    idx = np.arange(n)
    for _ in range(n_shuffles):
        np.random.shuffle(idx)
        shuffled = succ_orig[idx]
        for k in k_list:
            sr_rand_all[k].append(sr_at_k(shuffled, k))

    sr_rand_mean = {k: float(np.mean(sr_rand_all[k])) for k in k_list}
    sr_rand_std = {k: float(np.std(sr_rand_all[k])) for k in k_list}
    return sr_geom, sr_rand_mean, sr_rand_std


def main():
    val_files = sorted(glob.glob("data/grasps_val/*_geom_val.json"))
    if not val_files:
        print("[ERROR] No files matched data/grasps_val/*_geom_val.json")
        return

    rows = []
    print(f"[INFO] Found {len(val_files)} validated grasp files.")

    for f in val_files:
        obj_id = Path(f).stem.replace("_geom_val", "")
        grasps = load_grasps(f)
        sr_geom, sr_rand_mean, sr_rand_std = compute_geom_and_random_sr(
            grasps, K_LIST, n_shuffles=N_SHUFFLES
        )

        row = {"obj_id": obj_id}
        for k in K_LIST:
            row[f"geom_SR@{k}"] = sr_geom[k]
            row[f"rand_SR@{k}"] = sr_rand_mean[k]
            row[f"rand_std@{k}"] = sr_rand_std[k]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("obj_id").reset_index(drop=True)

    os.makedirs("results", exist_ok=True)
    out_csv = "results/ycb_ranking_vs_random.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved per-object SR@k (geom vs random) to {out_csv}\n")

    # 打印每个物体在 k=1,3,5 的对比
    print("Per-object SR@k (geom vs random), k in {1,3,5}:")
    display_cols = ["obj_id"]
    for k in [1, 3, 5]:
        display_cols += [f"geom_SR@{k}", f"rand_SR@{k}"]
    print(df[display_cols].to_string(index=False))

    # 计算跨物体平均
    mean_row = {"obj_id": "MEAN"}
    for k in K_LIST:
        mean_row[f"geom_SR@{k}"] = df[f"geom_SR@{k}"].mean()
        mean_row[f"rand_SR@{k}"] = df[f"rand_SR@{k}"].mean()
    df_mean = pd.DataFrame([mean_row])

    print("\nMean SR@k across objects:")
    cols_mean = ["obj_id"]
    for k in [1, 3, 5, 10, 20, 32, 64]:
        cols_mean += [f"geom_SR@{k}", f"rand_SR@{k}"]
    print(df_mean[cols_mean].to_string(index=False))

    # 画一张平均 SR@k 曲线图（所有物体平均）
    ks = K_LIST
    geom_mean = [mean_row[f"geom_SR@{k}"] for k in ks]
    rand_mean = [mean_row[f"rand_SR@{k}"] for k in ks]

    plt.figure()
    plt.plot(ks, geom_mean, marker="o", label="Geom ranking")
    plt.plot(ks, rand_mean, marker="s", linestyle="--", label="Random ranking")
    plt.xlabel("k")
    plt.ylabel("SR@k")
    plt.title("Average SR@k across YCB objects (geom vs random ranking)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out_fig = "results/ranking_vs_random.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    print(f"\n[INFO] Saved ranking vs random figure to {out_fig}")


if __name__ == "__main__":
    main()

