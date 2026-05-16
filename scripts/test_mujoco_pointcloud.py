#!/usr/bin/env python3
"""Validation script for owg_robot.pointcloud — MuJoCo depth → world point cloud.

Spawns a red box and a blue cylinder on a table inside a minimal MuJoCo scene
(no arm, no YCB assets needed).  Renders depth and segmentation from the same
overhead camera used in env_soarm.py, then validates the new pointcloud module.

Outputs
───────
  results/pointcloud_debug.npz   — arrays: rgb, depth, seg, pc_all,
                                   pc_box, pc_cylinder, K, cam_to_world
  results/pointcloud_debug.png   — 3-panel: RGB | depth | segmentation

Usage
─────
  conda run -n owg-mujoco python scripts/test_mujoco_pointcloud.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

from owg_robot.pointcloud import (
    compute_intrinsics,
    depth_to_camera_points,
    transform_points,
    mujoco_depth_to_world_points,
)

# ── scene constants matching env_soarm.py ────────────────────────────────────
TABLE_Z  = 0.785
FOVY     = 55.0
IMG_SIZE = 224
CAM_X, CAM_Y, CAM_Z = 0.05, -0.52, 1.90
CAM_EULER = "0 0 0"   # same as env_soarm CAM_EULER; default MuJoCo cam looks -Z

# Logical object IDs (chosen to match env_soarm convention: 1-based)
BOX_ID = 1
CYL_ID = 2

# Box: 6cm cube resting on table → CoM at TABLE_Z + 0.03
BOX_POS = f"0.05 -0.30 {TABLE_Z + 0.03:.4f}"
# Cylinder: radius=2.5cm, half-height=4cm → CoM at TABLE_Z + 0.04
CYL_POS = f"-0.08 -0.36 {TABLE_Z + 0.04:.4f}"

SCENE_XML = f"""
<mujoco model="pc_test">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <asset>
    <texture name="grid"   type="2d" builtin="checker"
             rgb1=".8 .7 .6" rgb2=".7 .6 .5" width="64" height="64"/>
    <material name="table_mat" texture="grid" texrepeat="4 4"/>
    <material name="box_mat"   rgba="0.82 0.18 0.18 1"/>
    <material name="cyl_mat"   rgba="0.18 0.45 0.82 1"/>
  </asset>

  <worldbody>
    <!-- floor (prevents objects from falling through) -->
    <geom type="plane" size="3 3 0.1" pos="0 0 0"
          contype="1" conaffinity="1"/>

    <!-- table surface matching env_soarm TABLE_TOP_Z=0.785 -->
    <geom name="table_top" type="box" size="0.45 0.45 0.02"
          pos="0 -0.45 {TABLE_Z - 0.02:.4f}"
          material="table_mat" contype="1" conaffinity="1"/>

    <!-- box object (logical id = {BOX_ID}) -->
    <body name="box_body" pos="{BOX_POS}">
      <freejoint name="box_joint"/>
      <geom name="box_geom" type="box" size="0.03 0.03 0.03"
            material="box_mat" contype="1" conaffinity="1" mass="0.15"/>
    </body>

    <!-- cylinder object (logical id = {CYL_ID}) -->
    <body name="cyl_body" pos="{CYL_POS}">
      <freejoint name="cyl_joint"/>
      <geom name="cyl_geom" type="cylinder" size="0.025 0.04"
            material="cyl_mat" contype="1" conaffinity="1" mass="0.10"/>
    </body>

    <!-- overhead camera — identical parameters to env_soarm.py -->
    <camera name="overhead"
            pos="{CAM_X} {CAM_Y} {CAM_Z}"
            euler="{CAM_EULER}"
            fovy="{FOVY}"/>
  </worldbody>
</mujoco>
"""


def build_cam_to_world() -> np.ndarray:
    """Replicate env_soarm._get_transform_matrix(cx, cy, cz, rot=0).

    With CAM_ROT = 0 this reduces to a pure translation by camera position.
    The camera-frame coordinate flip (+X right, +Y up, -Z into scene) is
    already handled inside depth_to_camera_points.
    """
    rot = 0.0   # CAM_ROT in env_soarm.py
    T = np.array([[np.cos(rot), -np.sin(rot), 0.0, CAM_X],
                  [np.sin(rot),  np.cos(rot), 0.0, CAM_Y],
                  [0.0,          0.0,         1.0, CAM_Z],
                  [0.0,          0.0,         0.0, 1.0  ]], dtype=np.float64)
    return T


def settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int = 400) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)


def render(model: mujoco.MjModel,
           data: mujoco.MjData,
           size: int = IMG_SIZE) -> tuple:
    """Return (rgb, depth_metric, seg_logical) arrays."""
    renderer = mujoco.Renderer(model, height=size, width=size)

    renderer.update_scene(data, camera="overhead")
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="overhead")
    depth = renderer.render().copy().astype(np.float32)
    renderer.disable_depth_rendering()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="overhead")
    seg_raw = renderer.render()
    renderer.disable_segmentation_rendering()
    renderer.close()

    # seg_raw[:,:,0] = geom_id,  seg_raw[:,:,1] = body_id
    geom_ids = seg_raw[:, :, 0].astype(np.int32)
    seg = np.zeros_like(geom_ids)
    try:
        seg[geom_ids == model.geom("box_geom").id] = BOX_ID
        seg[geom_ids == model.geom("cyl_geom").id] = CYL_ID
    except Exception as e:
        print(f"[WARN] segmentation remap error: {e}")

    return rgb, depth, seg


def print_stats(name: str, pts: np.ndarray) -> None:
    n = len(pts)
    if n == 0:
        print(f"  {name}: 0 points")
        return
    print(f"  {name}: {n:6d} points  "
          f"x=[{pts[:,0].min():.4f}, {pts[:,0].max():.4f}]  "
          f"y=[{pts[:,1].min():.4f}, {pts[:,1].max():.4f}]  "
          f"z=[{pts[:,2].min():.4f}, {pts[:,2].max():.4f}]  "
          f"centroid=({pts[:,0].mean():.4f}, {pts[:,1].mean():.4f}, {pts[:,2].mean():.4f})")


def save_png(rgb: np.ndarray, depth: np.ndarray,
             seg: np.ndarray, path: str,
             pc_box: np.ndarray, pc_cyl: np.ndarray) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(rgb)
        axes[0].set_title("RGB (overhead)")
        axes[0].axis("off")

        im = axes[1].imshow(depth, cmap="viridis_r")
        axes[1].set_title("Depth — metric metres")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(seg, cmap="tab10", vmin=0, vmax=9)
        axes[2].set_title(f"Seg  0=bg  {BOX_ID}=box  {CYL_ID}=cyl")
        axes[2].axis("off")

        plt.suptitle(
            f"MuJoCo point cloud validation | "
            f"box N={len(pc_box)}  cyl N={len(pc_cyl)}", y=1.01)
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved PNG: {path}")
    except ImportError:
        print("  matplotlib not available; skipping PNG output")


def main() -> int:
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── build model & settle ─────────────────────────────────────────────────
    print("Building MuJoCo model …")
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data  = mujoco.MjData(model)
    settle(model, data, steps=400)

    box_body_z = data.xpos[model.body("box_body").id, 2]
    cyl_body_z = data.xpos[model.body("cyl_body").id, 2]
    print(f"  After settling: box z={box_body_z:.4f}  cyl z={cyl_body_z:.4f}  "
          f"(table top z={TABLE_Z})")

    # ── render ───────────────────────────────────────────────────────────────
    print("Rendering depth + segmentation …")
    rgb, depth, seg = render(model, data)
    print(f"  depth: min={depth.min():.4f}  max={depth.max():.4f}  "
          f"mean={depth.mean():.4f}  shape={depth.shape}")
    unique_ids = np.unique(seg).tolist()
    print(f"  seg unique ids: {unique_ids}")
    box_pix = int((seg == BOX_ID).sum())
    cyl_pix = int((seg == CYL_ID).sum())
    print(f"  box pixels={box_pix}  cyl pixels={cyl_pix}")

    # ── intrinsics ───────────────────────────────────────────────────────────
    K = compute_intrinsics(IMG_SIZE, IMG_SIZE, FOVY)
    cam_to_world = build_cam_to_world()
    print(f"\nIntrinsics K (fovy={FOVY}°  {IMG_SIZE}×{IMG_SIZE}):\n{K.round(3)}")
    print(f"cam_to_world:\n{cam_to_world.round(4)}")

    # ── point clouds ─────────────────────────────────────────────────────────
    print("\nGenerating point clouds …")
    pc_all  = mujoco_depth_to_world_points(
        depth, seg=None, cam_to_world=cam_to_world, fov_deg=FOVY)
    pc_bg   = mujoco_depth_to_world_points(
        depth, seg, cam_to_world, FOVY, target_ids=None)  # all non-bg
    pc_box  = mujoco_depth_to_world_points(
        depth, seg, cam_to_world, FOVY, target_ids=[BOX_ID])
    pc_cyl  = mujoco_depth_to_world_points(
        depth, seg, cam_to_world, FOVY, target_ids=[CYL_ID])
    pc_both = mujoco_depth_to_world_points(
        depth, seg, cam_to_world, FOVY, target_ids=[BOX_ID, CYL_ID])

    print("\nPoint cloud statistics (world / robot-base frame):")
    print_stats("all pixels  ", pc_all)
    print_stats("non-bg (seg>0)", pc_bg)
    print_stats(f"box  (id={BOX_ID}) ", pc_box)
    print_stats(f"cyl  (id={CYL_ID}) ", pc_cyl)
    print_stats("box+cyl     ", pc_both)

    # ── validate Z above table ────────────────────────────────────────────────
    print("\nValidation checks:")
    checks: list[tuple[str, bool]] = []

    def check(label: str, cond: bool) -> None:
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {label}")
        checks.append((label, cond))

    check(f"depth > 0 everywhere",
          bool(depth.min() > 0))
    check(f"N all pixels = {IMG_SIZE**2}",
          len(pc_all) == IMG_SIZE * IMG_SIZE)
    check(f"box pixels detected",
          box_pix > 0)
    check(f"cylinder pixels detected",
          cyl_pix > 0)

    if len(pc_box):
        bz = float(pc_box[:, 2].mean())
        check(f"box centroid Z={bz:.4f} in [{TABLE_Z:.3f}, {TABLE_Z+0.15:.3f}]",
              TABLE_Z <= bz <= TABLE_Z + 0.15)

    if len(pc_cyl):
        cz = float(pc_cyl[:, 2].mean())
        check(f"cyl centroid Z={cz:.4f} in [{TABLE_Z:.3f}, {TABLE_Z+0.15:.3f}]",
              TABLE_Z <= cz <= TABLE_Z + 0.15)

    check("pc_all has 3 columns",
          pc_all.ndim == 2 and pc_all.shape[1] == 3)
    check("pc_all dtype is float32",
          pc_all.dtype == np.float32)

    # ── compare with env_soarm._depth_to_pointcloud ──────────────────────────
    # Inline the existing implementation to verify identical output.
    h, w = depth.shape
    fy2 = (h / 2) / np.tan(np.deg2rad(FOVY / 2))
    fx2 = fy2
    ys2, xs2 = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    z2   = depth
    xc2  = (xs2 - w / 2) / fx2 * z2
    yc2  = (ys2 - h / 2) / fy2 * z2
    pc_ref = np.stack([xc2, -yc2, -z2], axis=-1).reshape(-1, 3).astype(np.float32)
    pc_hom = np.vstack([pc_ref.T, np.ones((1, len(pc_ref)))])
    pc_ref_world = (cam_to_world @ pc_hom)[:3].T.astype(np.float32)

    max_err = float(np.abs(pc_all - pc_ref_world).max())
    check(f"matches env_soarm._depth_to_pointcloud (max err={max_err:.2e})",
          max_err < 1e-4)

    # ── save debug files ──────────────────────────────────────────────────────
    print("\nSaving debug files …")
    npz_path = os.path.join(results_dir, "pointcloud_debug.npz")
    np.savez_compressed(npz_path,
                        rgb=rgb,
                        depth=depth,
                        seg=seg,
                        pc_all=pc_all,
                        pc_box=pc_box,
                        pc_cylinder=pc_cyl,
                        pc_both_objects=pc_both,
                        K=K,
                        cam_to_world=cam_to_world)
    print(f"  Saved NPZ: {npz_path}")

    save_png(rgb, depth, seg,
             os.path.join(results_dir, "pointcloud_debug.png"),
             pc_box, pc_cyl)

    # ── summary ───────────────────────────────────────────────────────────────
    n_pass = sum(1 for _, ok in checks if ok)
    n_fail = sum(1 for _, ok in checks if not ok)
    print(f"\n{'─'*55}")
    print(f"Result: {n_pass}/{len(checks)} checks passed"
          + (f"  ({n_fail} FAILED)" if n_fail else "  — all OK"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
