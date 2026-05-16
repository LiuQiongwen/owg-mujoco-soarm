"""Simulator-independent point cloud generation for OWG.

Replaces the scattered PyBullet-specific depth conversion in
owg_robot/camera.py (Camera.get_pointcloud) and owg_robot/env.py
with a single tested module.

Key difference from the legacy camera.py implementation
────────────────────────────────────────────────────────
  - PyBullet depth is a z-buffer value in [0, 1]; it must be linearised
    to metric depth before back-projection.
  - MuJoCo enable_depth_rendering() already returns linear metric depth
    in metres along the optical axis.  No z-buffer conversion is needed.

Camera coordinate convention used throughout
────────────────────────────────────────────
  +X  right (image columns, left → right)
  +Y  up    (image rows, top → bottom flipped to up)
  -Z  into  the scene (depth is positive, so camera-frame Z is negative)

This matches the convention in env_soarm._depth_to_pointcloud, which this
module supersedes without changing the numerical output.

Public API
──────────
  compute_intrinsics(width, height, fov_deg)          → (3, 3) K
  depth_to_camera_points(depth, intrinsics, mask)     → (N, 3) camera frame
  transform_points(points, T)                         → (N, 3) transformed
  mujoco_depth_to_world_points(depth, seg, ...)       → (N, 3) world frame
  pybullet_depth_to_world_points(zbuffer, seg, ...)   → (N, 3) world frame
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional


# ── core primitives ───────────────────────────────────────────────────────────

def compute_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Build a 3×3 pinhole camera intrinsics matrix.

    Uses the vertical FOV (FOVY) convention that both MuJoCo and PyBullet
    expose.  Square pixels are assumed (fx == fy).

    Args:
        width   : image width in pixels.
        height  : image height in pixels.
        fov_deg : vertical field of view in degrees.

    Returns:
        K : (3, 3) float64  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    fy = (height / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    fx = fy          # square pixels
    cx = width  / 2.0
    cy = height / 2.0
    return np.array([[fx,  0.0, cx],
                     [0.0, fy,  cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def depth_to_camera_points(depth: np.ndarray,
                            intrinsics: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Back-project a metric depth image to 3-D points in camera frame.

    Works identically for MuJoCo (pass depth directly) and PyBullet (pass
    the linearised depth from pybullet_depth_to_world_points).

    Coordinate convention: +X right, +Y up, -Z into scene.
    Image rows increase downward, so this function negates Y and Z to match
    the robotics camera convention used by OWG.

    Args:
        depth     : (H, W) float32, metric depth in metres.
        intrinsics: (3, 3) camera intrinsics K.
        mask      : optional (H, W) bool array.  When given only the True
                    pixels are returned; otherwise all H×W pixels.

    Returns:
        points : (N, 3) float32 in camera frame.
                 N = H*W when mask is None, else count of True pixels.
    """
    h, w = depth.shape
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32),
                         indexing='ij')
    z  = depth.astype(np.float32)
    xc = (xs - cx) / fx * z     # +X right
    yc = (ys - cy) / fy * z     # image-Y down → flip to +Y up below

    # Convention: +X right, +Y up (flip image Y), -Z into scene
    cam_pts = np.stack([xc, -yc, -z], axis=-1)

    if mask is not None:
        return cam_pts[mask].astype(np.float32)
    return cam_pts.reshape(-1, 3).astype(np.float32)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous transform to an (N, 3) point array.

    Args:
        points : (N, 3) float32.
        T      : (4, 4) float64 homogeneous transform (R | t).

    Returns:
        (N, 3) float32 in the target frame.
    """
    if len(points) == 0:
        return points
    pts_h = np.hstack([points,
                       np.ones((len(points), 1), dtype=points.dtype)])
    return (T @ pts_h.T)[:3].T.astype(np.float32)


# ── simulator-specific wrappers ───────────────────────────────────────────────

def mujoco_depth_to_world_points(
        depth: np.ndarray,
        seg: Optional[np.ndarray],
        cam_to_world: np.ndarray,
        fov_deg: float,
        target_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """MuJoCo pipeline: metric depth → world-frame point cloud.

    MuJoCo's Renderer.enable_depth_rendering() returns linear metric depth
    in metres along the optical axis; no z-buffer conversion is required.

    This replicates env_soarm.EnvironmentSoArm._depth_to_pointcloud, factored
    out so it can be tested and reused independently.

    Args:
        depth        : (H, W) float32, metric depth in metres (from MuJoCo).
        seg          : (H, W) int, segmentation map (pixel value = logical
                       obj_id, 0 = background).  Pass None to keep all pixels.
        cam_to_world : (4, 4) camera-to-world (or camera-to-robot-base) matrix,
                       e.g. env.cam_to_robot_base from env_soarm.py.
        fov_deg      : vertical field of view in degrees (FOVY constant).
        target_ids   : if given, restrict output to pixels whose seg value is
                       in this list.  seg must not be None in this case.

    Returns:
        points : (N, 3) float32 in world/robot-base frame.
    """
    h, w = depth.shape
    K = compute_intrinsics(w, h, fov_deg)

    if target_ids is not None and seg is not None:
        mask: Optional[np.ndarray] = np.isin(seg, target_ids)
    elif seg is not None:
        mask = seg > 0
    else:
        mask = None

    cam_pts = depth_to_camera_points(depth, K, mask=mask)
    return transform_points(cam_pts, cam_to_world)


def pybullet_depth_to_world_points(
        zbuffer: np.ndarray,
        seg: Optional[np.ndarray],
        cam_to_world: np.ndarray,
        fov_deg: float,
        near: float,
        far: float,
        target_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """PyBullet pipeline: z-buffer depth → world-frame point cloud.

    PyBullet returns depth as a normalised z-buffer value in [0, 1].
    This function linearises it to metric depth using the OpenGL perspective
    inverse formula, then delegates to the same camera-frame back-projection
    as the MuJoCo path.

    Matches the existing behaviour of owg_robot/camera.py::Camera.get_pointcloud
    followed by owg_robot/env.py's pc[:,1] = -pc[:,1]; pc[:,2] = -pc[:,2]
    coordinate flip.

    Args:
        zbuffer      : (H, W) z-buffer values in [0, 1] from pybullet.getCameraImage.
        seg          : (H, W) segmentation map, or None.
        cam_to_world : (4, 4) camera-to-world matrix (env.cam_to_robot_base).
        fov_deg      : vertical field of view in degrees.
        near         : near clipping plane distance in metres.
        far          : far clipping plane distance in metres.
        target_ids   : restrict to specific obj_ids when given.

    Returns:
        points : (N, 3) float32 in world frame.
    """
    z = np.asarray(zbuffer, dtype=np.float32)
    # OpenGL perspective linearisation: z-buffer → metric depth
    depth = (2.0 * near * far) / (far + near - (2.0 * z - 1.0) * (far - near))

    h, w = depth.shape
    K = compute_intrinsics(w, h, fov_deg)

    if target_ids is not None and seg is not None:
        mask: Optional[np.ndarray] = np.isin(seg, target_ids)
    elif seg is not None:
        mask = np.asarray(seg) > 0
    else:
        mask = None

    cam_pts = depth_to_camera_points(depth, K, mask=mask)
    return transform_points(cam_pts, cam_to_world)


# ── convenience: object-only point cloud from an obs dict ────────────────────

def obs_to_object_points(obs: dict,
                         obj_ids: Optional[List[int]] = None,
                         backend: str = "mujoco",
                         fov_deg: float = 55.0,
                         cam_to_world: Optional[np.ndarray] = None,
                         near: float = 0.01,
                         far: float = 10.0) -> np.ndarray:
    """Extract world-frame points for specific objects from a get_obs() dict.

    Compatible with both MuJoCo and PyBullet observation dicts:
        obs['depth'] : (H, W) depth image
        obs['seg']   : (H, W) segmentation map (obj_ids)

    If obs already contains 'points' (the pre-computed full point cloud from
    env.get_obs), this function will still reproject from the raw depth so
    that object-only masking is applied correctly.

    Args:
        obs          : dict returned by env.get_obs() or env_soarm.get_obs().
        obj_ids      : logical object ids to extract.  None = all non-background.
        backend      : "mujoco" or "pybullet".
        fov_deg      : vertical FOV (must match the camera used for the obs).
        cam_to_world : (4, 4) transform.  If None, tries obs.get('cam_to_world')
                       then falls back to a pure identity (camera frame output).
        near / far   : clipping planes, only used for pybullet backend.

    Returns:
        points : (N, 3) float32 in world/robot-base frame.
    """
    depth = obs['depth']
    seg   = obs.get('seg') or obs.get('segmentation')

    if cam_to_world is None:
        cam_to_world = obs.get('cam_to_world', np.eye(4))

    if backend == "mujoco":
        return mujoco_depth_to_world_points(
            depth, seg, cam_to_world, fov_deg, target_ids=obj_ids)
    elif backend == "pybullet":
        return pybullet_depth_to_world_points(
            depth, seg, cam_to_world, fov_deg, near, far,
            target_ids=obj_ids)
    else:
        raise ValueError(f"Unknown backend: {backend!r}.  Use 'mujoco' or 'pybullet'.")
