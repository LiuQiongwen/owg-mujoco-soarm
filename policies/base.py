"""PolicyBase — minimal interface every grasp policy must implement.

Concrete implementations live in the same package:
  policies/lggsn_policy.py   — LGGSN pairwise BPR ranker (current method)
  policies/random_policy.py  — random grasp baseline

A policy receives the current observation dict from env.get_obs() and
returns a GraspAction.  It may also receive a text prompt for
language-conditioned policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from datasets.episode import GraspAction


class PolicyBase(ABC):
    """Interface for grasp-selection policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logging (e.g. 'lggsn', 'random')."""
        ...

    @abstractmethod
    def select_grasp(
        self,
        obs:       Dict[str, Any],
        obj_id:    int,
        obj_name:  str,
        candidates: List[np.ndarray],
        prompt:    Optional[str] = None,
    ) -> GraspAction:
        """
        Select the best grasp from a list of candidates.

        Parameters
        ----------
        obs : dict
            Output of env.get_obs() — contains 'image', 'depth', 'seg', 'points'.
        obj_id : int
            MuJoCo / PyBullet logical object ID.
        obj_name : str
            Human-readable object name (e.g. "Banana").
        candidates : list of ndarray (6,)
            Raw grasp candidate vectors [x,y,z,yaw,opening_len,obj_height].
        prompt : str | None
            Natural-language instruction (e.g. "pick up the banana").

        Returns
        -------
        GraspAction
            The selected grasp.
        """
        ...

    def reset(self) -> None:
        """Called at the start of each episode.  Override if stateful."""
        pass
