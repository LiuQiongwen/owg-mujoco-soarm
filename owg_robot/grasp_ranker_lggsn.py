import os
import numpy as np
import torch

from lggsn_model import LGGSN, GC_LGGSN

_USE_DIST = os.environ.get("FEAT_DIST", "1") == "1"   # module-level default, overridden per instance
_USE_ZREL = os.environ.get("FEAT_ZREL", "1") == "1"
_FLAT_GATE = float(os.environ.get("FLAT_GATE_H_STD", "0.005"))
_RERANK_WHITELIST = set(
    s.strip() for s in os.environ.get("RERANK_WHITELIST", "").split(",") if s.strip()
)

# Canonical 14-dim feature list (both dist_to_centroid and z_rel present).
# Per-instance _feature_cols may be shorter depending on the checkpoint.
FEATURE_COLS_BASE = [
    "x", "y", "z",
    "roll", "pitch", "yaw",
    "width", "score",
    "dz", "dz_lift", "need_dz", "H",
]
FEATURE_COLS_FULL = FEATURE_COLS_BASE + ["dist_to_centroid", "z_rel"]

# Module-level alias kept for backward-compat imports
FEATURE_COLS = FEATURE_COLS_FULL if (_USE_DIST and _USE_ZREL) else (
    FEATURE_COLS_BASE + (["dist_to_centroid"] if _USE_DIST else []) + (["z_rel"] if _USE_ZREL else [])
)


class LggsnGraspRanker:
    """
    用几何 LGGSN 模型对一批 3D grasps 打分并排序。

    既兼容:
      - JSON/dict 形式的 grasp: {"position":[x,y,z], "rpy":[r,p,y], "width":w, "score":s, ...}
      - 也兼容 GR-ConvNet 在线生成时可能返回的 tuple / list 形式:
          (pos, rpy, width, score, ...)
          或 ( {dict_grasp}, ...extra )
    """

    def __init__(
        self,
        model_path: str = "grasp_6dof/models/lggsn_geom_only_live.pt",
        device: str = "cuda",
        lggsn_input_dim: int | None = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._gc_mode = os.environ.get("LGGSN_GC_MODE", "0") == "1"

        # ── Step 1: probe checkpoint to get actual input dim ──────────────────
        raw_state = torch.load(model_path, map_location="cpu", weights_only=False)
        ckpt_dim = raw_state["mlp.0.weight"].shape[1]

        # ── Step 2: validate against explicit config (if given) ───────────────
        if lggsn_input_dim is not None and lggsn_input_dim != ckpt_dim:
            raise ValueError(
                f"lggsn_input_dim={lggsn_input_dim} conflicts with checkpoint "
                f"mlp.0.weight input dim={ckpt_dim} in {model_path}"
            )

        # ── Step 3: derive per-instance feature flags from checkpoint dim ─────
        if ckpt_dim == 12:
            self._use_dist = False
            self._use_zrel = False
            self._feature_cols = FEATURE_COLS_BASE[:]
            print(f"[LggsnGraspRanker] legacy 12-dim checkpoint — "
                  f"dist_to_centroid and z_rel disabled for this instance")
        elif ckpt_dim == 14:
            self._use_dist = True
            self._use_zrel = True
            self._feature_cols = FEATURE_COLS_FULL[:]
        else:
            raise ValueError(
                f"Unsupported checkpoint input dim {ckpt_dim} in {model_path}; "
                f"expected 12 or 14"
            )

        print(f"[LggsnGraspRanker] input_dim={ckpt_dim}  "
              f"features={self._feature_cols}")

        # ── Step 4: build model with the right geom_dim ───────────────────────
        if self._gc_mode:
            self.model = GC_LGGSN(
                n_queries=1,
                geom_dim=ckpt_dim,
                query_dim=0,
                hidden_dim=40,
                context_dim=3,
            )
            print(f"[LggsnGraspRanker] GC-LGGSN mode | loading: {model_path}")
        else:
            self.model = LGGSN(
                n_queries=1,
                geom_dim=ckpt_dim,
                query_dim=0,
                hidden_dim=40,
            )
            print(f"[LggsnGraspRanker] loading checkpoint: {model_path}")

        self.model.load_state_dict(raw_state)
        self.model.to(self.device)
        self.model.eval()

        self.use_3d_prompt = True

    # --------- 输入格式适配 ---------

    def _unwrap_grasp(self, g):
        """
        把多种可能的 grasp 表示统一整理成 dict 形式:
          {"position":[x,y,z], "rpy":[r,p,y], "width":w, "score":s, ...}
        """
        # 1) 已经是 dict 的情况（离线库 / validated grasps）
        if isinstance(g, dict):
            return g

        # 2) tuple / list / np.ndarray
        if isinstance(g, (tuple, list, np.ndarray)):
            # 2.1) 形如 (dict_grasp, extra_info...)
            if len(g) > 0 and isinstance(g[0], dict):
                return g[0]

            # 2.2) 形如 (pos, rpy, width, score, ...)
            #      其中 pos/rpy 通常是长度为 3 的 list/ndarray
            if len(g) >= 2 and isinstance(g[0], (tuple, list, np.ndarray)):
                pos = list(g[0])
                rpy = list(g[1]) if isinstance(g[1], (tuple, list, np.ndarray)) else [np.pi, 0.0, 0.0]

                pos = [float(p) for p in pos[:3]] + [0.0] * (3 - len(pos))
                rpy = [float(r) for r in rpy[:3]] + [0.0] * (3 - len(rpy))

                width = 0.04
                score = 0.0
                # 从后面的标量里顺序猜 width / score
                for v in g[2:]:
                    if isinstance(v, (int, float, np.floating)):
                        if width == 0.04:
                            width = float(v)
                        elif score == 0.0:
                            score = float(v)

                return {
                    "position": pos,
                    "rpy": rpy,
                    "width": width,
                    "score": score,
                }

            # 2.3-env) 6- or 7-scalar env tuple from GR-ConvNet:
            #   6-tuple: (x, y, z, yaw, opening_len, obj_height)          [legacy]
            #   7-tuple: (x, y, z, yaw, opening_len, obj_height, quality) [current]
            #   roll=π and pitch=0 are fixed for top-down grasps.
            if len(g) in (6, 7):
                vals = [float(v) for v in g]
                x_, y_, z_, yaw_, opening_len_, obj_height_ = vals[:6]
                quality_ = vals[6] if len(vals) == 7 else 0.0
                return {
                    "position": [x_, y_, z_],
                    "rpy":      [np.pi, 0.0, yaw_],
                    "width":    opening_len_,
                    "score":    quality_,
                    "_metrics": {"H": obj_height_},
                }

            # 2.3) 完全扁平的一串数字，按 [x,y,z,roll,pitch,yaw,width,score,...] 解释
            flat = [float(v) for v in g]
            while len(flat) < 8:
                flat.append(0.0)
            pos = flat[0:3]
            rpy = flat[3:6]
            width = flat[6]
            score = flat[7]
            return {
                "position": pos,
                "rpy": rpy,
                "width": width,
                "score": score,
            }

        # 3) 其他未知类型：返回一个默认占位，避免直接崩溃
        return {
            "position": [0.0, 0.0, 0.0],
            "rpy": [np.pi, 0.0, 0.0],
            "width": 0.04,
            "score": 0.0,
        }

    # --------- 特征构造 ---------

    def _featurize_one(self, g):
        """
        从单个 grasp（任意支持的格式）抽取几何 + 质量特征。
        """
        g = self._unwrap_grasp(g)

        pos = g.get("position") or g.get("pos") or [0.0, 0.0, 0.0]
        rpy = g.get("rpy") or [np.pi, 0.0, 0.0]

        pos = [float(p) for p in pos[:3]] + [0.0] * (3 - len(pos))
        rpy = [float(r) for r in rpy[:3]] + [0.0] * (3 - len(rpy))

        width = float(g.get("width", 0.04))
        score = float(g.get("score", 0.0))
        dz = float(g.get("dz", 0.0))

        m = g.get("_metrics", {}) or {}
        dz_lift = float(m.get("dz_lift", dz))
        need_dz = float(m.get("need_dz", 0.0))
        H = float(m.get("H", 0.08))

        x, y, z = pos
        roll, pitch, yaw = rpy

        return [
            x, y, z,
            roll, pitch, yaw,
            width, score,
            dz, dz_lift, need_dz, H,
        ]

    def _featurize(self, grasps):
        feats = [self._featurize_one(g) for g in grasps]
        arr   = np.asarray(feats, dtype=np.float32)   # [N, 12]
        extra = []

        if self._use_dist or self._use_zrel:
            xy   = arr[:, :2]
            z    = arr[:, 2]
            cent = xy.mean(axis=0)
            if self._use_dist:
                dists = np.linalg.norm(xy - cent, axis=1, keepdims=True)       # [N,1]
                extra.append(dists)
            if self._use_zrel:
                z_min, z_max = z.min(), z.max()
                z_rel = ((z - z_min) / (z_max - z_min + 1e-8)).reshape(-1, 1) # [N,1]
                extra.append(z_rel)

        if extra:
            return np.concatenate([arr] + extra, axis=1)
        return arr

    # --------- 排序接口 ---------

    def rank(self, grasps, query_text: str | None = None, obj_type: str | None = None,
             verbose: bool = False):
        """
        输入:
          grasps: list[dict 或 tuple]
          verbose: if True, print per-candidate feature matrix and raw logits for diagnostics

        输出:
          order: np.array[int]，按质量从高到低的索引顺序
          scores: np.array[float]，每个 grasp 的 [0,1] 质量分
        """
        if self.model is None:
            scores = np.array([float(self._unwrap_grasp(g).get("score", 0.0)) for g in grasps], dtype=float)
            order = np.argsort(-scores)
            return order, scores
        if len(grasps) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # Object-conditional gate: fall back to Stage-3 order for unlisted objects
        if _RERANK_WHITELIST and query_text not in _RERANK_WHITELIST:
            identity = np.arange(len(grasps))
            return identity, np.full(len(grasps), 0.5, dtype=float)

        X = self._featurize(grasps)

        # Flat-object gate: if H_std < threshold, keep original (Stage-3) order
        H_col = 11  # index of H in FEATURE_COLS
        if _FLAT_GATE > 0 and X.shape[0] > 1 and float(np.std(X[:, H_col])) < _FLAT_GATE:
            identity = np.arange(len(grasps))
            return identity, np.full(len(grasps), 0.5, dtype=float)

        geom = torch.from_numpy(X).to(self.device)
        q_id = torch.zeros(len(grasps), dtype=torch.long, device=self.device)

        with torch.no_grad():
            if self._gc_mode:
                # Compute episode context z = [flat_frac, sigma_H, sigma_yaw]
                H_vals   = X[:, 11]
                yaw_vals = X[:, 5]
                flat_frac = float(np.mean(H_vals < 0.001))
                sigma_H   = float(np.std(H_vals))
                sigma_yaw = float(np.std(yaw_vals))
                ctx_vec   = np.array([[flat_frac, sigma_H, sigma_yaw]] * len(grasps),
                                     dtype=np.float32)
                ctx = torch.from_numpy(ctx_vec).to(self.device)
                logit = self.model(geom, q_id, ctx)   # GC_LGGSN forward
            else:
                logit = self.model(geom, q_id)        # standard LGGSN forward
            score = torch.sigmoid(logit).cpu().numpy()

        if verbose:
            logit_np = logit.cpu().numpy()
            print("[LGGSN diag] per-candidate features + scores:")
            header = f"  {'idx':>3}  " + "  ".join(f"{c:>8}" for c in self._feature_cols) + \
                     f"  {'logit':>7}  {'score':>6}  {'spread_from_c0':>14}"
            print(header)
            for i in range(len(grasps)):
                feat_str = "  ".join(f"{v:>8.4f}" for v in X[i])
                delta = score[i] - score[0]
                print(f"  {i:>3}  {feat_str}  {logit_np[i]:>7.4f}  {score[i]:>6.4f}  {delta:>+14.4f}")
            print(f"  score spread (max-min): {score.max() - score.min():.6f}")
            zero_cols = [self._feature_cols[j] for j in range(len(self._feature_cols))
                         if np.all(X[:, j] == X[0, j])]
            print(f"  constant features (no within-episode variance): {zero_cols}")

        order = np.argsort(-score)                # 从大到小排序
        return order, score

