"""
Validate the flat_frac predictability framework on YCB meshes.

Loads each object's canonical .obj mesh, samples a surface point cloud,
computes flat_frac (fraction of surface points with z < 0.001 m) and
sigma_H (std of z), then measures Spearman correlation with our
experimental net improvements.

Usage:
    conda run -n owg-mujoco python scripts/validate_flat_frac.py
"""

import os
import numpy as np
import open3d as o3d
from scipy.stats import spearmanr

ASSETS = "tango_robot/assets/ycb_objects"

OBJECTS = {
    "Banana":        ("YcbBanana",        "textured_simple_reoriented.obj"),
    "CrackerBox":    ("YcbCrackerBox",     "textured_simple_reoriented.obj"),
    "MustardBottle": ("YcbMustardBottle",  "textured_simple_reoriented.obj"),
    "PowerDrill":    ("YcbPowerDrill",     "textured_simple_reoriented.obj"),
    "Scissors":      ("YcbScissors",       "textured_simple_reoriented.obj"),
    "TomatoSoupCan": ("YcbTomatoSoupCan",  "textured_simple_reoriented.obj"),
    "Pear":          ("YcbPear",           "textured_reoriented.obj"),
}

# Experimental net improvements from Table 1 (MuJoCo/SO-ARM101, 25 seeds)
NET = {
    "Banana":        0,
    "CrackerBox":   -2,
    "MustardBottle":-1,
    "PowerDrill":   +4,
    "Scissors":     -4,
    "TomatoSoupCan":-1,
    "Pear":         -1,
}

N_SAMPLE = 50_000
H_THRESH = 0.001  # metres — same threshold used in simulation


def compute_stats(mesh_path: str):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=N_SAMPLE)
    pts = np.asarray(pcd.points)

    # Normalise so object base sits at z=0
    z = pts[:, 2]
    z = z - z.min()

    flat_frac = float(np.mean(z < H_THRESH))
    sigma_H   = float(np.std(z))
    z_max     = float(z.max())
    return flat_frac, sigma_H, z_max


def main():
    rows = []
    print(f"\n{'Object':<16} {'flat_frac':>10} {'sigma_H':>10} {'z_max':>8} {'net':>5}")
    print("-" * 55)
    for name, (folder, obj_file) in OBJECTS.items():
        path = os.path.join(ASSETS, folder, obj_file)
        if not os.path.exists(path):
            print(f"  {name:<14}  FILE NOT FOUND: {path}")
            continue
        ff, sh, zm = compute_stats(path)
        net = NET[name]
        rows.append((name, ff, sh, zm, net))
        sign = "+" if net >= 0 else ""
        print(f"  {name:<14} {ff:10.4f} {sh:10.4f} {zm:8.4f} {sign}{net:4d}")

    print("-" * 55)

    flat_fracs = [r[1] for r in rows]
    nets       = [r[4] for r in rows]

    rho_ff, p_ff = spearmanr(flat_fracs, nets)
    sigma_Hs = [r[2] for r in rows]
    rho_sh, p_sh = spearmanr(sigma_Hs, nets)

    print(f"\nSpearman(flat_frac, net) : rho = {rho_ff:+.3f}  p = {p_ff:.4f}")
    print(f"Spearman(sigma_H,   net) : rho = {rho_sh:+.3f}  p = {p_sh:.4f}")

    print("\n--- Rank comparison ---")
    exp_rank  = sorted(rows, key=lambda r: r[4], reverse=True)
    pred_rank = sorted(rows, key=lambda r: r[1])          # low flat_frac → good
    print(f"{'Experimental (net ↓)':30}  {'Predicted by flat_frac (↑ flat = worse)':}")
    for e, p in zip(exp_rank, pred_rank):
        match = "✓" if e[0] == p[0] else " "
        print(f"  {e[0]:<18} net={e[4]:+2d}    {p[0]:<18} ff={p[1]:.4f}  {match}")

    print()
    if rho_ff < -0.7 and p_ff < 0.1:
        print("✅  flat_frac predicts reranking direction from mesh geometry alone.")
    elif rho_ff < -0.5:
        print("~  Weak negative correlation — trend holds but not strongly.")
    else:
        print("✗  flat_frac from canonical mesh does not predict our experimental order.")


if __name__ == "__main__":
    main()
