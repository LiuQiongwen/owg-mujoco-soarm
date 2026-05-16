"""EpisodeWriter — saves episodes in two formats.

Format A (current, backward-compatible):
    data/transitions/
        meta.json
        ep_00000.npz  ← same layout as data/transition_logger.py

Format B (LeRobot-style):
    data/lerobot/
        episodes/
            ep_00000/
                steps.npz       ← per-step obs/action arrays
                frames/
                    step_0000_rgb.png   ← only when record_frames=True
                    step_0000_depth.npy
        meta.jsonl
        dataset_info.json

Both formats are written simultaneously when enabled.  Downstream training code
can read either; the legacy TransitionLogger is still authoritative for the MLP
world model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from datasets.episode import Episode


_TRANSITIONS_DIR = Path("data/transitions")
_LEROBOT_DIR     = Path("data/lerobot")


class EpisodeWriter:
    """
    Writes one Episode to disk.

    Parameters
    ----------
    out_dir_legacy : Path | str
        Root for format A (npz + meta.json).  Defaults to data/transitions/.
    out_dir_lerobot : Path | str | None
        Root for format B.  Pass None to skip LeRobot output.
    record_frames : bool
        When True, save per-step RGB/depth images inside format B episodes.
    """

    def __init__(
        self,
        out_dir_legacy:   Path | str = _TRANSITIONS_DIR,
        out_dir_lerobot:  Optional[Path | str] = _LEROBOT_DIR,
        record_frames:    bool = False,
    ):
        self._legacy      = Path(out_dir_legacy)
        self._lerobot     = Path(out_dir_lerobot) if out_dir_lerobot else None
        self._rec_frames  = record_frames

        self._legacy.mkdir(parents=True, exist_ok=True)
        if self._lerobot:
            (self._lerobot / "episodes").mkdir(parents=True, exist_ok=True)
            (self._lerobot / "meta.jsonl").touch(exist_ok=True)

        self._meta_path = self._legacy / "meta.json"
        self._meta: list = self._load_meta()

    # ── public ────────────────────────────────────────────────────────────────

    def write(self, ep: Episode) -> None:
        self._write_legacy(ep)
        if self._lerobot:
            self._write_lerobot(ep)

    # ── format A ──────────────────────────────────────────────────────────────

    def _write_legacy(self, ep: Episode) -> None:
        """Write episode in data/transition_logger.py-compatible .npz format."""
        if ep.obj_pos_before is None:
            return   # incomplete episode — skip

        fname = f"ep_{ep.episode_id:05d}.npz"
        arrays: dict = {}

        def _save(key, val):
            if val is not None:
                arrays[key] = np.asarray(val, dtype=np.float32)

        _save("obj_pos_before",  ep.obj_pos_before)
        _save("obj_quat_before", ep.obj_quat_before)
        _save("pc_stats_before", ep.pc_stats_before)
        _save("obj_pos_after",   ep.obj_pos_after)
        _save("obj_quat_after",  ep.obj_quat_after)

        if ep.grasp_action is not None:
            arrays["grasp_pose"] = ep.grasp_action.as_vector()

        if ep.obj_pos_before is not None and ep.obj_pos_after is not None:
            d_pos  = (np.asarray(ep.obj_pos_after)  - np.asarray(ep.obj_pos_before)).astype(np.float32)
            d_quat = np.zeros(4, dtype=np.float32)
            if ep.obj_quat_before is not None and ep.obj_quat_after is not None:
                d_quat = (np.asarray(ep.obj_quat_after) - np.asarray(ep.obj_quat_before)).astype(np.float32)
            arrays["pose_delta"] = np.concatenate([d_pos, d_quat[:3]])

        np.savez_compressed(self._legacy / fname, **arrays)

        entry = {
            "ep_id":             ep.episode_id,
            "obj_name":          ep.obj_name,
            "yaw_mode":          ep.yaw_mode,
            "execution_mode":    ep.execution_mode,
            "success":           bool(ep.success) if ep.success is not None else None,
            "dz":                float(ep.dz)      if ep.dz      is not None else None,
            "fell_off":          bool(ep.fell_off) if ep.fell_off is not None else None,
            "depth_mean_before": float(ep.depth_mean_before) if ep.depth_mean_before is not None else None,
            "timestamp":         ep.timestamp,
            "file":              fname,
        }
        self._meta.append(entry)
        self._meta_path.write_text(json.dumps(self._meta, indent=2))

    # ── format B ──────────────────────────────────────────────────────────────

    def _write_lerobot(self, ep: Episode) -> None:
        """Write episode in LeRobot-compatible directory format."""
        ep_dir = self._lerobot / "episodes" / f"ep_{ep.episode_id:05d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # per-step arrays
        if ep.steps:
            obs_vecs   = [s.obs_vector() for s in ep.steps]
            act_vecs   = [s.action.as_vector() if s.action else np.zeros(6, np.float32)
                          for s in ep.steps]
            timestamps = [s.timestamp for s in ep.steps]
            np.savez_compressed(
                ep_dir / "steps.npz",
                obs       = np.array(obs_vecs,  dtype=np.float32),
                action    = np.array(act_vecs,  dtype=np.float32),
                timestamp = np.array(timestamps, dtype=np.float64),
            )

            if self._rec_frames:
                frames_dir = ep_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                for s in ep.steps:
                    if s.rgb is not None:
                        np.save(frames_dir / f"step_{s.step_idx:04d}_rgb.npy", s.rgb)
                    if s.depth is not None:
                        np.save(frames_dir / f"step_{s.step_idx:04d}_depth.npy", s.depth)

        # episode summary → meta.jsonl
        with open(self._lerobot / "meta.jsonl", "a") as f:
            f.write(json.dumps(ep.summary()) + "\n")

        # dataset_info.json (create / update)
        info_path = self._lerobot / "dataset_info.json"
        info: dict = {}
        if info_path.exists():
            info = json.loads(info_path.read_text())
        info.update({
            "robot":          "soarm101",
            "camera":         "overhead",
            "obs_dim":        10,    # joint(5) + gripper(2) + eef(3)
            "action_dim":     6,     # eef_pos(3) + yaw + opening + obj_height
            "n_episodes":     info.get("n_episodes", 0) + 1,
            "updated_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        info_path.write_text(json.dumps(info, indent=2))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_meta(self) -> list:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text())
        return []
