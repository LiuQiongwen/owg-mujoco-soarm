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


def compute_pc_stats_local(obs: dict, obj_id: int, center_xyz: np.ndarray,
                            radius: float = 0.04) -> np.ndarray:
    """
    Candidate-level counterpart to compute_pc_stats(): same 9-dim layout
    (centroid(3) + std(3) + min_z(1) + max_z(1) + n_pts_norm(1)), but computed
    over only the object-masked points within `radius` of `center_xyz` -- the
    specific candidate's own gripper position -- instead of the whole object.

    Fixes a real defect: compute_pc_stats() is called once per scene, before
    any candidate is sampled, so every candidate in a scene previously
    received the identical scene-level vector (see feature() in
    world_model/train_counterfactual_critic.py). This function gives each
    candidate its own local geometry instead.

    radius=0.04 was chosen empirically, not arbitrarily: swept 0.02-0.10m
    against real MuJoCo point clouds (tango_robot/env_soarm.py's own
    finger_length=0.04 was the physical anchor). 0.02 (roughly the jaw span)
    is too tight -- some real candidates get zero points and degenerate to
    the zero-fallback below. By 0.10 crops start reabsorbing most of the
    object's points (over 900 of ~1600 in the swept scene) and candidates'
    local centroids start converging back toward each other -- i.e. drifting
    back toward the whole-object average this function exists to avoid. 0.04
    sits in the middle of the range that gave 6/6 distinct candidates without
    that convergence, and matches a real, already-named physical constant
    (finger_length) rather than a round number picked for convenience.

    All inputs (`obs`, `obj_id`) are PRE_EXECUTION-admissible in exactly the
    same sense compute_pc_stats()'s are: `obs` is captured once per scene
    before any candidate executes, and `center_xyz` is the candidate's own
    generator-time pose, not anything derived from its outcome. Returns
    zeros if too few points fall in the local crop (same convention as
    compute_pc_stats() for a missing/tiny segment).
    """
    seg = obs.get("seg")
    points = obs.get("points")
    zero = np.zeros(9, dtype=np.float32)
    if seg is None or points is None:
        return zero

    flat_seg = seg.ravel()
    flat_pts = points.reshape(-1, 3) if points.ndim == 3 else points
    n_min = min(len(flat_seg), len(flat_pts))
    flat_seg = flat_seg[:n_min]
    flat_pts = flat_pts[:n_min]

    obj_mask = flat_seg == obj_id
    if obj_mask.sum() < 5:
        return zero
    obj_pts = flat_pts[obj_mask]

    center = np.asarray(center_xyz, dtype=np.float32).reshape(1, 3)
    dist = np.linalg.norm(obj_pts - center, axis=1)
    local_mask = dist <= radius
    if local_mask.sum() < 5:
        return zero
    local_pts = obj_pts[local_mask]

    centroid = local_pts.mean(axis=0)
    std = local_pts.std(axis=0) + 1e-6
    min_z = float(local_pts[:, 2].min())
    max_z = float(local_pts[:, 2].max())
    n_norm = float(min(local_mask.sum() / 1000.0, 1.0))
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

    KNOWN DEFECT, NOT FIXED HERE (found 2026-08-05, documented not patched): every caller
    (world_model/rerank_grasps.py's rerank()/score_grasps(), consumed by
    scripts/risk_gated_vla_phase1_eval.py's "world_critic" method) passes the SAME
    scene-level `pc_stats` for every candidate in a pool -- the identical bug fixed
    in world_model/train_counterfactual_critic.py::feature() (see
    data.transition_logger.compute_pc_stats_local() and that fix's commit history),
    just in this older feature assembly function instead.

    Deliberately NOT fixed here: this function backs `world_model/mlp_predictor.pkl`,
    the exact "predecessor"/"stale checkpoint" the risk-gated VLA paper's own
    diagnosis (Section 3.4/4.1) already found chance-level (AUROC=0.4996) for two
    OTHER, independent reasons (seed-coupling and success-criterion defects in the
    harness that trained/evaluated it). This defect is consistent with, not a
    contradiction of, that finding -- a model trained on features with zero
    candidate-specific point-cloud signal would plausibly fail regardless of the
    other two fixes. Patching this function would mean retraining a pipeline the
    project's own paper already correctly diagnosed and moved away from in favor of
    the object-relative counterfactual critic; there is no result that depends on
    doing so. Left as a documented fact for whoever next touches this code, not a
    silent gap.
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
