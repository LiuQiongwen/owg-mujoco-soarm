"""EpisodeReader — loads episodes from either storage format.

Reads both:
  - Format A: data/transitions/meta.json + ep_*.npz  (legacy)
  - Format B: data/lerobot/meta.jsonl + episodes/ep_*/steps.npz  (LeRobot-style)

Returns Episode objects for downstream use, or raw numpy arrays for training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from datasets.episode import Episode, EpisodeStep, GraspAction, GripperState, JointState


_TRANSITIONS_DIR = Path("data/transitions")
_LEROBOT_DIR     = Path("data/lerobot")

# Feature / label dimensions (match data/transition_logger.py)
FEATURE_DIM = 22
LABEL_DIM   = 3


class EpisodeReader:
    """
    Load episode data from disk.

    Parameters
    ----------
    legacy_dir : Path | str
        Root of format A storage.  Pass None to skip.
    lerobot_dir : Path | str | None
        Root of format B storage.  Pass None to skip.
    execution_mode_filter : str | None
        When set, only return episodes whose execution_mode matches.
        Use "physics" to exclude demo_attach episodes from training.
    """

    def __init__(
        self,
        legacy_dir:             Optional[Path | str] = _TRANSITIONS_DIR,
        lerobot_dir:            Optional[Path | str] = _LEROBOT_DIR,
        execution_mode_filter:  Optional[str]        = None,
    ):
        self._legacy     = Path(legacy_dir)  if legacy_dir  else None
        self._lerobot    = Path(lerobot_dir) if lerobot_dir else None
        self._mode_filter = execution_mode_filter
        self._legacy_meta: list = self._load_legacy_meta()

    # ── public API ────────────────────────────────────────────────────────────

    def n_episodes(self) -> int:
        return len(self._legacy_meta)

    def iter_legacy(self) -> Iterator[dict]:
        """Iterate over legacy meta entries (dicts from meta.json)."""
        for entry in self._legacy_meta:
            if self._mode_filter and entry.get("execution_mode") != self._mode_filter:
                continue
            yield entry

    def load_arrays(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load all legacy episodes into (X, y, meta) numpy arrays.

        X shape: (N, FEATURE_DIM)  — same layout as data/transition_logger.py
        y shape: (N, LABEL_DIM)    — [success, dz, fell_off]

        Only episodes that have all required arrays are included.
        """
        from data.transition_logger import build_feature

        X_rows, y_rows, meta_rows = [], [], []
        for entry in self.iter_legacy():
            path = self._legacy / entry["file"]
            if not path.exists():
                continue
            d = np.load(path)
            required = {"grasp_pose", "obj_pos_before", "obj_quat_before", "pc_stats_before"}
            if not required.issubset(d.files):
                continue

            feat = build_feature(
                d["grasp_pose"],
                d["obj_pos_before"],
                d["obj_quat_before"],
                d["pc_stats_before"],
            )
            label = np.array([
                float(entry.get("success",  0) or 0),
                float(entry.get("dz",       0) or 0),
                float(entry.get("fell_off", 0) or 0),
            ], dtype=np.float32)
            X_rows.append(feat)
            y_rows.append(label)
            meta_rows.append(entry)

        if not X_rows:
            return (
                np.zeros((0, FEATURE_DIM), dtype=np.float32),
                np.zeros((0, LABEL_DIM),   dtype=np.float32),
                [],
            )
        return np.array(X_rows), np.array(y_rows), meta_rows

    def iter_lerobot_steps(self) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Iterate over LeRobot-format step arrays.

        Yields (episode_id, obs, actions) where:
          obs:     (T, 10)  float32
          actions: (T, 6)   float32
        """
        if self._lerobot is None or not self._lerobot.exists():
            return
        meta_path = self._lerobot / "meta.jsonl"
        if not meta_path.exists():
            return

        for line in meta_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            ep_id = entry["episode_id"]
            if self._mode_filter and entry.get("execution_mode") != self._mode_filter:
                continue
            steps_path = self._lerobot / "episodes" / f"ep_{ep_id:05d}" / "steps.npz"
            if not steps_path.exists():
                continue
            d = np.load(steps_path)
            yield ep_id, d["obs"], d["action"]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_legacy_meta(self) -> list:
        if self._legacy is None:
            return []
        path = self._legacy / "meta.json"
        if path.exists():
            return json.loads(path.read_text())
        return []
