# -*- coding: utf-8 -*-
"""
Transition logger for 6-DoF grasp experiments.
Records before_state, action, after_state, success, reward to JSONL.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TransitionLogger:
    """
    Append-only JSONL logger.  Each line is one grasp transition.

    Schema:
    {
      "timestamp":    float,
      "episode":      int,
      "step":         int,
      "before_state": {"object_pos": [x,y,z], "object_quat": [w,x,y,z],
                       "arm_qpos": [...]},
      "action":       {"position": [x,y,z], "rpy": [r,p,y],
                       "width": float, "score": float},
      "after_state":  {"object_pos": [x,y,z], "object_quat": [w,x,y,z],
                       "arm_qpos": [...]},
      "success":      bool,
      "reward":       float,
      "info":         {}
    }
    """

    def __init__(self, log_path: str, append: bool = True):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._fh = self._path.open(mode)
        self._episode = 0
        self._step    = 0

    # ── API ──────────────────────────────────────────────────────────────────

    def new_episode(self):
        self._episode += 1
        self._step = 0

    def log(self, before_state: Dict[str, Any], action: Dict[str, Any],
            after_state: Dict[str, Any], success: bool, reward: float,
            info: Optional[Dict] = None):
        record = {
            "timestamp":    time.time(),
            "episode":      self._episode,
            "step":         self._step,
            "before_state": _to_serializable(before_state),
            "action":       _to_serializable(action),
            "after_state":  _to_serializable(after_state),
            "success":      bool(success),
            "reward":       float(reward),
            "info":         _to_serializable(info or {}),
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._step += 1

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── convenience ──────────────────────────────────────────────────────────

    @staticmethod
    def load(log_path: str):
        """Load all transitions as a list of dicts."""
        records = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def success_rate(log_path: str) -> float:
        """Compute overall grasp success rate from a log file."""
        records = TransitionLogger.load(log_path)
        if not records:
            return 0.0
        return sum(r["success"] for r in records) / len(records)


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def make_before_state(env, robot) -> Dict[str, Any]:
    """Helper: snapshot before a grasp attempt."""
    pose = env.get_object_pose()
    return {
        "object_pos":  pose["position"],
        "object_quat": pose["quaternion"],
        "arm_qpos":    robot.get_joint_positions(),
    }


def make_after_state(env, robot) -> Dict[str, Any]:
    """Helper: snapshot after a grasp attempt."""
    return make_before_state(env, robot)
