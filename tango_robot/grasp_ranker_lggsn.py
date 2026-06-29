import os
import numpy as np
import torch

from lggsn_model import LGGSN, GC_LGGSN, LGGSNVision
from grasp_6dof.grasp_sampler import (
    rpy_to_R, local_point_density, normal_consistency, contact_width_ratio,
)

_USE_DIST = os.environ.get("FEAT_DIST", "1") == "1"   # module-level default, overridden per instance
_USE_ZREL = os.environ.get("FEAT_ZREL", "1") == "1"
_EPISODE_WHITEN = os.environ.get("LGGSN_WHITEN", "0") == "1"   # per-episode z-score whitening (off by default; needs retrain to take effect)
_OBJ_FRAME_NORM = os.environ.get("OBJ_FRAME_NORM", "0") == "1"  # subtract object centroid from (x,y); needs v10+ checkpoint
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
FEATURE_COLS_EXT  = FEATURE_COLS_FULL + [
    "local_point_density", "normal_consistency", "contact_width_ratio",
]
FEATURE_COLS_EXT2 = FEATURE_COLS_EXT + ["pe_ik"]

# Module-level alias kept for backward-compat imports
FEATURE_COLS = FEATURE_COLS_FULL if (_USE_DIST and _USE_ZREL) else (
    FEATURE_COLS_BASE + (["dist_to_centroid"] if _USE_DIST else []) + (["z_rel"] if _USE_ZREL else [])
)

# Query class map used by v5+ checkpoints (must match train_lggsn_v5.py OBJECT_CLASSES).
# Maps prompt/object-name variations → integer class ID.
_OBJECT_CLASS_MAP: dict[str, int] = {}
for _aliases, _cid in [
    (["banana"],                               0),
    (["can", "tomatosoupcan", "tomato"],       1),
    (["cracker", "crackerbox"],                2),
    (["cylinder", "mediumclamp", "clamp"],     3),
    (["drill", "powerdrill"],                  4),
    (["mustard", "mustardbottle"],             5),
    (["pear"],                                 6),
]:
    for _a in _aliases:
        _OBJECT_CLASS_MAP[_a] = _cid


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
        mlp_in_dim = raw_state["mlp.0.weight"].shape[1]

        # Detect v5+ query embedding: presence of query_emb.weight in state dict.
        _has_query_emb = "query_emb.weight" in raw_state
        if _has_query_emb:
            _emb_shape   = raw_state["query_emb.weight"].shape  # (n_queries, query_dim)
            _query_dim   = int(_emb_shape[1])
            _n_queries   = int(_emb_shape[0])
            ckpt_dim     = mlp_in_dim - _query_dim
        else:
            _query_dim   = 0
            _n_queries   = 1
            ckpt_dim     = mlp_in_dim

        self._use_query_emb = _has_query_emb
        self._query_dim     = _query_dim
        self._n_queries     = _n_queries

        # ── Step 2: validate against explicit config (if given) ───────────────
        if lggsn_input_dim is not None and lggsn_input_dim != ckpt_dim:
            raise ValueError(
                f"lggsn_input_dim={lggsn_input_dim} conflicts with checkpoint "
                f"geom_dim={ckpt_dim} (mlp_in={mlp_in_dim}) in {model_path}"
            )

        # ── Step 3: derive per-instance feature flags from geom_dim ──────────
        # LGGSNVision checkpoints have ckpt_dim = geom_dim + vis_dim (e.g. 18+256=274).
        _VIS_DIM = 256
        _KNOWN_GEOM = {12, 14, 17, 18}
        if ckpt_dim in _KNOWN_GEOM:
            self._vis_dim = 0
            _actual_geom  = ckpt_dim
        elif (ckpt_dim - _VIS_DIM) in _KNOWN_GEOM:
            self._vis_dim = _VIS_DIM
            _actual_geom  = ckpt_dim - _VIS_DIM
            ckpt_dim      = _actual_geom   # reuse existing branch below
        else:
            raise ValueError(
                f"Unsupported checkpoint geom_dim={ckpt_dim} (mlp_in={mlp_in_dim}) "
                f"in {model_path}; expected 12, 14, 17, 18 or 18+256=274"
            )

        if ckpt_dim == 12:
            self._use_dist = False
            self._use_zrel = False
            self._use_pc_feats = False
            self._use_pe_ik = False
            self._feature_cols = FEATURE_COLS_BASE[:]
            print(f"[LggsnGraspRanker] legacy 12-dim checkpoint — "
                  f"dist_to_centroid and z_rel disabled for this instance")
        elif ckpt_dim == 14:
            self._use_dist = True
            self._use_zrel = True
            self._use_pc_feats = False
            self._use_pe_ik = False
            self._feature_cols = FEATURE_COLS_FULL[:]
        elif ckpt_dim == 17:
            self._use_dist = True
            self._use_zrel = True
            self._use_pc_feats = True
            self._use_pe_ik = False
            self._feature_cols = FEATURE_COLS_EXT[:]
        elif ckpt_dim == 18:
            self._use_dist = True
            self._use_zrel = True
            self._use_pc_feats = True
            self._use_pe_ik = True
            self._feature_cols = FEATURE_COLS_EXT2[:]

        vis_tag = f" + vis({self._vis_dim})" if self._vis_dim else ""
        q_tag   = f" + query_emb({_n_queries}×{_query_dim})" if _has_query_emb else ""
        print(f"[LggsnGraspRanker] geom_dim={ckpt_dim}{vis_tag}{q_tag}  "
              f"features={self._feature_cols}")

        # ── Step 4: build model with the right dims ───────────────────────────
        if self._gc_mode:
            self.model = GC_LGGSN(
                n_queries=_n_queries,
                geom_dim=ckpt_dim,
                query_dim=_query_dim,
                hidden_dim=40,
                context_dim=3,
            )
            print(f"[LggsnGraspRanker] GC-LGGSN mode | loading: {model_path}")
        elif self._vis_dim > 0:
            self.model = LGGSNVision(
                n_queries=_n_queries,
                geom_dim=ckpt_dim,
                vis_dim=self._vis_dim,
                query_dim=_query_dim,
                hidden_dim=64,
            )
            print(f"[LggsnGraspRanker] LGGSNVision mode | loading: {model_path}")
        else:
            self.model = LGGSN(
                n_queries=_n_queries,
                geom_dim=ckpt_dim,
                query_dim=_query_dim,
                hidden_dim=40,
                dropout=0.1,
            )
            print(f"[LggsnGraspRanker] loading checkpoint: {model_path}")

        self.model.load_state_dict(raw_state)
        self.model.to(self.device)
        self.model.eval()

        # ── Step 5: load SAM segmentor for visual features (LGGSNVision only) ─
        self._segmentor = None
        if self._vis_dim > 0:
            try:
                from tango.markers.segmentor import SegmentAnythingMarkGenerator
                _sam_dev   = os.environ.get("LGGSN_SAM_DEVICE",
                                            "cuda" if torch.cuda.is_available() else "cpu")
                _sam_model = os.environ.get("LGGSN_SAM_MODEL", "facebook/sam-vit-base")
                print(f"[LggsnGraspRanker] Loading SAM ({_sam_model}) on {_sam_dev} ...")
                self._segmentor = SegmentAnythingMarkGenerator(
                    device=_sam_dev, model_name=_sam_model
                )
                print(f"[LggsnGraspRanker] SAM ready.")
            except Exception as e:
                print(f"[LggsnGraspRanker] SAM unavailable ({e}); vis will be zeros.")

        self.use_3d_prompt = True
        self._episode_whiten = _EPISODE_WHITEN

    # --------- Episode-level feature whitening ---------

    def _whiten(self, X: np.ndarray) -> np.ndarray:
        """Per-episode z-score whitening: zero-mean, unit-std across candidates.

        Constant features (std=0 within this scene) become 0 — their influence
        is removed. Features with true within-episode variance are amplified.
        No-op when N<=1 (nothing to compare against).
        """
        if X.shape[0] <= 1:
            return X
        mu  = X.mean(axis=0, keepdims=True)
        sig = X.std(axis=0, keepdims=True)
        return (X - mu) / (sig + 1e-6)

    # --------- SAM visual feature extraction (per-candidate) ---------

    def _get_per_candidate_vis_feats(
        self,
        grasps: list,
        obs: dict | None,
    ) -> np.ndarray:
        """Return [N, vis_dim] per-candidate SAM features.

        Each candidate's grasp position (x,y,z) is projected into the image,
        and a 3×3 region of the SAM [1,256,64,64] feature map centred on that
        pixel is mean-pooled into a 256-dim vector.

        Coordinate convention (matches tango_robot/pointcloud.py):
          camera frame: +X right, +Y up, -Z into scene
          depth  = -z_cam  (positive for visible points)
          u (col) = fx * x_cam / depth + cx
          v (row) = -fy * y_cam / depth + cy
          feature cell: fu = u * 64 / img_w,  fv = v * 64 / img_h
        """
        N = len(grasps)
        zeros = np.zeros((N, self._vis_dim), dtype=np.float32)
        if self._segmentor is None or obs is None:
            return zeros

        image        = obs.get("image")
        K            = obs.get("K")
        cam_to_world = obs.get("cam_to_world")
        if image is None or K is None or cam_to_world is None:
            return zeros

        # ── encode image once (populate SAM cache) ────────────────────────────
        try:
            from PIL import Image as PILImage
            self._segmentor._cached_embeddings = None       # invalidate old scene
            pil_img = PILImage.fromarray(image)
            self._segmentor.get_roi_feature((0, 0, 1, 1), image=pil_img)
        except Exception as e:
            print(f"[LggsnGraspRanker] SAM encode failed: {e}")
            return zeros

        feat_map = self._segmentor._cached_embeddings       # [1, 256, 64, 64]
        if feat_map is None:
            return zeros

        _, C, H_f, W_f = feat_map.shape                    # 256, 64, 64
        H_img, W_img   = image.shape[:2]                   # 224, 224

        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        world_to_cam = np.linalg.inv(cam_to_world)         # 4×4

        result = np.zeros((N, C), dtype=np.float32)
        for i, g_raw in enumerate(grasps):
            g   = self._unwrap_grasp(g_raw)
            pos = g.get("position", [0.0, 0.0, 0.0])
            xyz_w = np.array([float(pos[0]), float(pos[1]), float(pos[2]), 1.0])

            xyz_c = world_to_cam @ xyz_w                   # camera frame
            depth = float(-xyz_c[2])
            if depth <= 1e-4:
                continue

            u = fx * float(xyz_c[0]) / depth + cx         # column
            v = -fy * float(xyz_c[1]) / depth + cy        # row

            fu = u * W_f / W_img
            fv = v * H_f / H_img

            fu_i = int(round(fu))
            fv_i = int(round(fv))

            u0 = max(0, fu_i - 1); u1 = min(W_f, fu_i + 2)
            v0 = max(0, fv_i - 1); v1 = min(H_f, fv_i + 2)

            if u0 >= u1 or v0 >= v1:
                continue

            roi = feat_map[0, :, v0:v1, u0:u1]            # [256, h, w]
            result[i] = roi.mean(dim=(-2, -1)).cpu().numpy()

        return result

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

    def _featurize(self, grasps, episode_pc=None):
        feats = [self._featurize_one(g) for g in grasps]
        arr   = np.asarray(feats, dtype=np.float32)   # [N, 12]

        if _OBJ_FRAME_NORM:
            # Shift XY to centroid of the grasp candidate set.  Must match the
            # training-time centering (v10 trains on mean of episode's ~5 candidates),
            # so we use arr[:,:2].mean() here — NOT the point-cloud centroid, which
            # would introduce a systematic train/inference mismatch.
            cent_xy = arr[:, :2].mean(axis=0)
            arr = arr.copy()
            arr[:, 0] -= cent_xy[0]
            arr[:, 1] -= cent_xy[1]

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

        if self._use_pc_feats:
            if episode_pc is not None:
                pc_rows = []
                for i, g_raw in enumerate(grasps):
                    g     = self._unwrap_grasp(g_raw)
                    pos   = arr[i, :3]
                    rpy   = [float(r) for r in (g.get("rpy") or [np.pi, 0., 0.])[:3]]
                    R     = rpy_to_R(*rpy)
                    width = float(arr[i, 6])
                    pc_rows.append([
                        local_point_density(pos, R, width, episode_pc),
                        normal_consistency(pos, R, width, episode_pc),
                        contact_width_ratio(pos, R, width, episode_pc),
                    ])
                extra.append(np.array(pc_rows, dtype=np.float32))
            else:
                # no point cloud available: zero-fill so feature dim stays consistent
                extra.append(np.zeros((len(grasps), 3), dtype=np.float32))

        if self._use_pe_ik:
            pe_vals = []
            for g_raw in grasps:
                g_ = self._unwrap_grasp(g_raw)
                pe_vals.append(float((g_.get("_metrics") or {}).get("pe_ik", 0.0)))
            extra.append(np.array(pe_vals, dtype=np.float32).reshape(-1, 1))

        if extra:
            return np.concatenate([arr] + extra, axis=1)
        return arr

    # --------- 排序接口 ---------

    def rank(self, grasps, query_text: str | None = None, obj_type: str | None = None,
             verbose: bool = False, episode_pc: np.ndarray | None = None,
             obs: dict | None = None, target_obj_id: int | None = None):
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

        X = self._featurize(grasps, episode_pc=episode_pc)
        if self._episode_whiten:
            X = self._whiten(X)

        geom = torch.from_numpy(X).to(self.device)

        if self._use_query_emb and query_text is not None:
            _key = query_text.lower().replace(" ", "").replace("ycb", "")
            if _key in _OBJECT_CLASS_MAP:
                _class_id = _OBJECT_CLASS_MAP[_key]
                q_id = torch.full((len(grasps),), _class_id, dtype=torch.long, device=self.device)
                _q_emb = None  # use standard embedding lookup below
            else:
                # Unknown object: use mean of all learned embeddings as a generic prior.
                _q_emb = self.model.query_emb.weight.mean(dim=0, keepdim=True)  # [1, query_dim]
                _q_emb = _q_emb.expand(len(grasps), -1)                         # [N, query_dim]
                q_id = None
        else:
            q_id = torch.zeros(len(grasps), dtype=torch.long, device=self.device)
            _q_emb = None

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
            elif self._vis_dim > 0:
                # LGGSNVision: per-candidate SAM features via grasp position projection.
                vis_np = self._get_per_candidate_vis_feats(grasps, obs)    # (N, 256)
                vis_t  = torch.from_numpy(vis_np).to(self.device)
                logit  = self.model(geom, vis_t, q_id)  # LGGSNVision forward
            elif _q_emb is not None:
                # Unknown object fallback: bypass embedding lookup, use mean emb directly.
                x = torch.cat([geom, _q_emb.to(self.device)], dim=-1)
                logit = self.model.mlp(x).squeeze(-1)
            else:
                logit = self.model(geom, q_id)        # standard LGGSN forward
            score = torch.sigmoid(logit).cpu().numpy()

        if verbose:
            logit_np = logit.cpu().numpy()
            whiten_tag = " [whitened]" if self._episode_whiten else ""
            print(f"[LGGSN diag] per-candidate features + scores{whiten_tag}:")
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
            if zero_cols:
                print(f"  constant features (no within-episode variance): {zero_cols}")

        order = np.argsort(-score)                # 从大到小排序
        return order, score

    def rank_mc(
        self,
        grasps,
        query_text: str | None = None,
        T: int = 20,
        episode_pc: np.ndarray | None = None,
        obs: dict | None = None,
        target_obj_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Like rank() but uses MC Dropout. Returns (order, mean_score, std_score).
        Falls back to rank() for GC-LGGSN and LGGSNVision (vis path has no MC)."""
        if self._gc_mode or self._vis_dim > 0 or len(grasps) == 0:
            order, scores = self.rank(grasps, query_text=query_text,
                                      episode_pc=episode_pc,
                                      obs=obs, target_obj_id=target_obj_id)
            return order, scores, np.zeros_like(scores)

        X = self._featurize(grasps, episode_pc=episode_pc)
        if self._episode_whiten:
            X = self._whiten(X)
        geom = torch.from_numpy(X).to(self.device)
        if self._use_query_emb and query_text is not None:
            _key = query_text.lower().replace(" ", "").replace("ycb", "")
            _cid = _OBJECT_CLASS_MAP.get(_key)
            if _cid is not None:
                q_id = torch.full((len(grasps),), _cid, dtype=torch.long, device=self.device)
            else:
                q_id = torch.zeros(len(grasps), dtype=torch.long, device=self.device)
        else:
            q_id = torch.zeros(len(grasps), dtype=torch.long, device=self.device)

        mean_s, std_s = self.model.predict_with_uncertainty(geom, q_id, T=T)
        mean_np = mean_s.cpu().numpy()
        std_np  = std_s.cpu().numpy()
        order   = np.argsort(-mean_np)
        return order, mean_np, std_np

