"""
Transition logger for world-model data collection.

Stores each grasp attempt as one compressed .npz file plus a running meta.json.

Feature vector layout (FEATURE_DIM = 22):
  [0:6]   grasp_pose      (x, y, z, yaw, opening_len, obj_height)
  [6:9]   obj_pos_before  (x, y, z)
  [9:13]  obj_quat_before (w, x, y, z)
  [13:22] pc_stats        (cx, cy, cz, sx, sy, sz, min_z, max_z, n_pts_norm)

Label vector layout (LABEL_DIM = 3):
  [0] success   (0/1)
  [1] dz        (float, metres)
  [2] fell_off  (0/1)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRANSITIONS_DIR = Path("data/transitions")
FEATURE_DIM = 22   # grasp(6) + pos(3) + quat(4) + pc_stats(9)
LABEL_DIM   = 3    # success + dz + fell_off


# ── Transition dataclass ──────────────────────────────────────────────────────

@dataclass
class Transition:
    episode_id:        int
    obj_name:          str
    obj_id:            int
    yaw_mode:          str

    # Pre-grasp observation
    obj_pos_before:    np.ndarray    # (3,)
    obj_quat_before:   np.ndarray    # (4,) [w,x,y,z]
    pc_stats_before:   np.ndarray    # (9,)
    depth_mean_before: float

    # Action
    grasp_pose:        np.ndarray    # (6,) [x,y,z,yaw,opening_len,obj_height]

    # Post-grasp observation
    obj_pos_after:     np.ndarray    # (3,)
    obj_quat_after:    np.ndarray    # (4,)

    # Derived labels
    success:           bool
    dz:                float         # obj_pos_after[2] - obj_pos_before[2]
    fell_off:          bool          # object left workspace
    pose_delta:        np.ndarray    # (6,) [dx,dy,dz, dqx,dqy,dqz]

    timestamp: float = field(default_factory=time.time)


# ── Shared obs utilities ──────────────────────────────────────────────────────

def compute_pc_stats(obs: dict, obj_id: int) -> np.ndarray:
    """
    9-dim pointcloud statistics for one object segment.

    Layout: centroid(3) + std(3) + min_z(1) + max_z(1) + n_pts_norm(1)
    Returns zeros if the segment is absent or too small.
    """
    seg    = obs.get("seg")
    points = obs.get("points")
    zero   = np.zeros(9, dtype=np.float32)

    if seg is None or points is None:
        return zero

    flat_seg = seg.ravel()
    flat_pts = points.reshape(-1, 3) if points.ndim == 3 else points
    n_min    = min(len(flat_seg), len(flat_pts))
    flat_seg = flat_seg[:n_min]
    flat_pts = flat_pts[:n_min]

    mask = flat_seg == obj_id
    if mask.sum() < 5:
        return zero

    obj_pts  = flat_pts[mask]
    centroid = obj_pts.mean(axis=0)
    std      = obj_pts.std(axis=0) + 1e-6
    min_z    = float(obj_pts[:, 2].min())
    max_z    = float(obj_pts[:, 2].max())
    n_norm   = float(min(mask.sum() / 1000.0, 1.0))

    return np.concatenate([centroid, std, [min_z, max_z, n_norm]]).astype(np.float32)


def compute_pose_delta(pos_before: np.ndarray, quat_before: np.ndarray,
                       pos_after:  np.ndarray, quat_after:  np.ndarray) -> np.ndarray:
    """6-dim pose change: [dx, dy, dz, dqx, dqy, dqz] (quaternion w dropped)."""
    d_pos  = (pos_after  - pos_before).astype(np.float32)
    d_quat = (quat_after - quat_before).astype(np.float32)
    return np.concatenate([d_pos, d_quat[:3]])


def build_feature(grasp_pose:  np.ndarray,
                  obj_pos:     np.ndarray,
                  obj_quat:    np.ndarray,
                  pc_stats:    np.ndarray) -> np.ndarray:
    """
    Assemble the 22-dim feature vector used by the MLP.

    All inputs are already in robot-base frame (same coordinate system as env).
    """
    feat = np.concatenate([
        np.asarray(grasp_pose, dtype=np.float32).ravel()[:6],
        np.asarray(obj_pos,    dtype=np.float32).ravel()[:3],
        np.asarray(obj_quat,   dtype=np.float32).ravel()[:4],
        np.asarray(pc_stats,   dtype=np.float32).ravel()[:9],
    ])
    if len(feat) != FEATURE_DIM:
        raise ValueError(f"build_feature: expected {FEATURE_DIM} dims, got {len(feat)}")
    return feat


# ── TransitionLogger ──────────────────────────────────────────────────────────

class TransitionLogger:
    """
    Append-only episode logger.

    Layout on disk::

        data/transitions/
            meta.json          — list of episode metadata dicts
            ep_00000.npz       — arrays for episode 0
            ep_00001.npz       — arrays for episode 1
            ...
    """

    def __init__(self, out_dir: Path = TRANSITIONS_DIR):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.out_dir / "meta.json"
        self._meta: list = self._load_meta()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_meta(self) -> list:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text())
        return []

    def _flush(self):
        self.meta_path.write_text(json.dumps(self._meta, indent=2))

    # ── public API ────────────────────────────────────────────────────────────

    def log(self, t: Transition) -> int:
        """Write one transition to disk. Returns episode id."""
        fname = f"ep_{t.episode_id:05d}.npz"
        np.savez_compressed(
            self.out_dir / fname,
            obj_pos_before  = t.obj_pos_before.astype(np.float32),
            obj_quat_before = t.obj_quat_before.astype(np.float32),
            pc_stats_before = t.pc_stats_before.astype(np.float32),
            grasp_pose      = t.grasp_pose.astype(np.float32),
            obj_pos_after   = t.obj_pos_after.astype(np.float32),
            obj_quat_after  = t.obj_quat_after.astype(np.float32),
            pose_delta      = t.pose_delta.astype(np.float32),
        )
        self._meta.append({
            "ep_id":             t.episode_id,
            "obj_name":          t.obj_name,
            "yaw_mode":          t.yaw_mode,
            "success":           bool(t.success),
            "dz":                float(t.dz),
            "fell_off":          bool(t.fell_off),
            "depth_mean_before": float(t.depth_mean_before),
            "timestamp":         t.timestamp,
            "file":              fname,
        })
        self._flush()
        return t.episode_id

    def load_dataset(self) -> tuple:
        """
        Load all logged transitions into numpy arrays.

        Returns
        -------
        X    : (N, FEATURE_DIM)  float32
        y    : (N, LABEL_DIM)    float32   columns = [success, dz, fell_off]
        meta : list[dict]
        """
        X_rows, y_rows, meta_rows = [], [], []
        for ep in self._meta:
            path = self.out_dir / ep["file"]
            if not path.exists():
                continue
            d = np.load(path)
            feat = build_feature(
                d["grasp_pose"],
                d["obj_pos_before"],
                d["obj_quat_before"],
                d["pc_stats_before"],
            )
            label = np.array([
                float(ep["success"]),
                float(ep["dz"]),
                float(ep["fell_off"]),
            ], dtype=np.float32)
            X_rows.append(feat)
            y_rows.append(label)
            meta_rows.append(ep)

        if not X_rows:
            return (np.zeros((0, FEATURE_DIM), dtype=np.float32),
                    np.zeros((0, LABEL_DIM),   dtype=np.float32),
                    [])
        return np.array(X_rows), np.array(y_rows), meta_rows

    @property
    def n_episodes(self) -> int:
        return len(self._meta)

    def summary(self) -> dict:
        if not self._meta:
            return {"n": 0}
        s  = [ep["success"]  for ep in self._meta]
        ff = [ep["fell_off"] for ep in self._meta]
        dz = [ep["dz"]       for ep in self._meta]
        return {
            "n":            len(self._meta),
            "success_rate": round(float(np.mean(s)),  3),
            "fell_off_rate":round(float(np.mean(ff)), 3),
            "dz_mean":      round(float(np.mean(dz)), 4),
            "dz_std":       round(float(np.std(dz)),  4),
        }
