"""Grasp ranking methods for the benchmark.

Each method receives a list of sampled grasp candidates and returns a ranked
index array (best first).  Methods are stateless with respect to the episode;
they may hold a loaded model checkpoint.

Available methods
-----------------
random        — uniform shuffle (seed-controlled baseline)
geometry      — rank by XY distance to object centroid (no model required)
lggsn         — LGGSN pairwise BPR reranker (requires trained checkpoint)
world_model   — binary MLP success predictor (requires trained checkpoint)
hybrid        — lggsn score gated by world_model confidence

Adding a new method: subclass MethodBase, register in _REGISTRY at the bottom.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


# ── base class ────────────────────────────────────────────────────────────────

class MethodBase(ABC):
    """Abstract grasp ranking method."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def rank(
        self,
        candidates: np.ndarray,    # (N, 6) [x,y,z,yaw,opening,obj_height]
        obj_pos:    np.ndarray,    # (3,)
        obs:        dict,
        obj_id:     int,
        rng:        np.random.Generator,
    ) -> np.ndarray:               # (N,) int indices, best first
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ── random ────────────────────────────────────────────────────────────────────

class RandomMethod(MethodBase):
    """Uniform random ordering — the strictest baseline."""

    @property
    def name(self) -> str:
        return "random"

    def rank(self, candidates, obj_pos, obs, obj_id, rng) -> np.ndarray:
        idx = np.arange(len(candidates))
        rng.shuffle(idx)
        return idx


# ── geometry ──────────────────────────────────────────────────────────────────

class GeometryMethod(MethodBase):
    """Rank by XY distance to object centroid (ascending) — no learned model.

    Grasps closest to the object centre are tried first.  This is the natural
    geometry-only baseline for top-down centroid-sampled grasps.
    """

    @property
    def name(self) -> str:
        return "geometry"

    def rank(self, candidates, obj_pos, obs, obj_id, rng) -> np.ndarray:
        centroid_xy = obj_pos[:2]
        dists = np.linalg.norm(candidates[:, :2] - centroid_xy, axis=1)
        return np.argsort(dists)   # ascending: closest first


# ── LGGSN ─────────────────────────────────────────────────────────────────────

class LggsnMethod(MethodBase):
    """LGGSN pairwise BPR reranker.

    Loads the LggsnGraspRanker from owg_robot/grasp_ranker_lggsn.py.
    Falls back to geometry ranking if the checkpoint is unavailable.

    Parameters
    ----------
    ckpt_path : str | None
        Path to .pt checkpoint.  None → auto-discover from env vars /
        default locations (mirrors GraspRanker behaviour).
    fallback_on_error : bool
        If True, silently fall back to geometry when the model fails.
    """

    def __init__(self, ckpt_path: Optional[str] = None,
                 fallback_on_error: bool = True):
        self._ckpt_path      = ckpt_path
        self._fallback       = fallback_on_error
        self._ranker         = None
        self._geometry       = GeometryMethod()
        self._load_error: Optional[str] = None
        self._init_ranker()

    def _init_ranker(self):
        try:
            from owg_robot.grasp_ranker_lggsn import LggsnGraspRanker
            self._ranker = LggsnGraspRanker(ckpt_path=self._ckpt_path)
        except Exception as e:
            self._load_error = str(e)
            if not self._fallback:
                raise

    @property
    def name(self) -> str:
        return "lggsn"

    def rank(self, candidates, obj_pos, obs, obj_id, rng) -> np.ndarray:
        if self._ranker is None:
            return self._geometry.rank(candidates, obj_pos, obs, obj_id, rng)

        try:
            order, _scores = self._ranker.rank(candidates.tolist())
            return np.asarray(order, dtype=int)
        except Exception:
            if self._fallback:
                return self._geometry.rank(candidates, obj_pos, obs, obj_id, rng)
            raise

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error


# ── world model ───────────────────────────────────────────────────────────────

class WorldModelMethod(MethodBase):
    """Binary MLP success predictor trained on transition data.

    Feature vector layout (22-dim) follows data/transition_logger.py:
      [0:6]   grasp_pose      (x, y, z, yaw, opening, obj_height)
      [6:9]   obj_pos_before  (x, y, z)
      [9:13]  obj_quat_before (w, x, y, z)
      [13:22] pc_stats        (cx, cy, cz, sx, sy, sz, min_z, max_z, n_pts_norm)

    Parameters
    ----------
    ckpt_path : str
        Path to the trained MLP checkpoint (.pt).
        Expected format: {'model_state': ..., 'input_dim': 22}
    """

    def __init__(self, ckpt_path: str = "data/world_model_ckpt.pt",
                 fallback_on_error: bool = True):
        self._ckpt_path = ckpt_path
        self._fallback  = fallback_on_error
        self._model     = None
        self._geometry  = GeometryMethod()
        self._load_error: Optional[str] = None
        self._init_model()

    def _init_model(self):
        import os
        if not os.path.isfile(self._ckpt_path):
            self._load_error = (
                f"world_model checkpoint not found: {self._ckpt_path}\n"
                f"Train it first: python scripts/train_world_model.py"
            )
            if not self._fallback:
                raise FileNotFoundError(self._load_error)
            return
        try:
            import torch
            from benchmark._wm_mlp import WorldModelMLP
            ck = torch.load(self._ckpt_path, map_location="cpu", weights_only=True)
            dim = ck.get("input_dim", 22)
            self._model = WorldModelMLP(input_dim=dim)
            self._model.load_state_dict(ck["model_state"])
            self._model.eval()
        except Exception as e:
            self._load_error = str(e)
            if not self._fallback:
                raise

    @property
    def name(self) -> str:
        return "world_model"

    def rank(self, candidates, obj_pos, obs, obj_id, rng) -> np.ndarray:
        if self._model is None:
            return self._geometry.rank(candidates, obj_pos, obs, obj_id, rng)
        try:
            return self._rank_with_model(candidates, obs, obj_id)
        except Exception:
            if self._fallback:
                return self._geometry.rank(candidates, obj_pos, obs, obj_id, rng)
            raise

    def _rank_with_model(self, candidates, obs, obj_id) -> np.ndarray:
        import torch
        from data.transition_logger import build_feature, compute_pc_stats

        obj_pos  = obs.get("object_pose", {})
        obj_info = obj_pos.get(obj_id, {}) if isinstance(obj_pos, dict) else {}
        pos      = np.zeros(3, dtype=np.float32)
        quat     = np.array([1, 0, 0, 0], dtype=np.float32)
        pc_stats = compute_pc_stats(obs, obj_id)

        feats = []
        for g in candidates:
            feat = build_feature(
                np.asarray(g[:6], dtype=np.float32),
                pos, quat, pc_stats,
            )
            feats.append(feat)

        X = torch.tensor(np.array(feats), dtype=torch.float32)
        with torch.no_grad():
            scores = self._model(X).squeeze(-1).numpy()

        return np.argsort(-scores)   # descending: highest predicted success first

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error


# ── hybrid ────────────────────────────────────────────────────────────────────

class HybridMethod(MethodBase):
    """LGGSN ranking gated by world-model agreement.

    Strategy: use LGGSN ranking, but if the world model score of the LGGSN
    top-1 candidate is below `wm_gate_threshold`, fall back to geometry order.

    This captures the intuition: trust LGGSN when the world model also
    believes the scene is graspable; otherwise use the simpler geometric rank.
    """

    def __init__(
        self,
        lggsn_ckpt:         Optional[str] = None,
        wm_ckpt:            str = "data/world_model_ckpt.pt",
        wm_gate_threshold:  float = 0.30,
        fallback_on_error:  bool = True,
    ):
        self._lggsn    = LggsnMethod(ckpt_path=lggsn_ckpt,
                                     fallback_on_error=fallback_on_error)
        self._wm       = WorldModelMethod(ckpt_path=wm_ckpt,
                                          fallback_on_error=fallback_on_error)
        self._geometry = GeometryMethod()
        self._gate     = wm_gate_threshold

    @property
    def name(self) -> str:
        return "hybrid"

    def rank(self, candidates, obj_pos, obs, obj_id, rng) -> np.ndarray:
        lggsn_order = self._lggsn.rank(candidates, obj_pos, obs, obj_id, rng)
        if self._wm._model is None:
            return lggsn_order

        try:
            import torch
            from data.transition_logger import build_feature, compute_pc_stats
            top_g   = candidates[lggsn_order[0]]
            pc      = compute_pc_stats(obs, obj_id)
            feat    = build_feature(top_g[:6], np.zeros(3), np.array([1,0,0,0]), pc)
            X       = torch.tensor(feat[None], dtype=torch.float32)
            with torch.no_grad():
                wm_score = float(self._wm._model(X).squeeze())
        except Exception:
            return lggsn_order

        if wm_score < self._gate:
            return self._geometry.rank(candidates, obj_pos, obs, obj_id, rng)
        return lggsn_order


# ── MLP definition (used by WorldModelMethod) ─────────────────────────────────
# Lives here as a private submodule to avoid circular imports.

class _WorldModelMLP:
    pass   # imported from benchmark._wm_mlp


# ── registry ──────────────────────────────────────────────────────────────────

_REGISTRY = {
    "random":      RandomMethod,
    "geometry":    GeometryMethod,
    "lggsn":       LggsnMethod,
    "world_model": WorldModelMethod,
    "hybrid":      HybridMethod,
}


def build_method(name: str, **kwargs) -> MethodBase:
    """Instantiate a ranking method by name.

    Extra kwargs are forwarded to the method constructor.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown method {name!r}.  "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def available_methods() -> List[str]:
    return sorted(_REGISTRY)
