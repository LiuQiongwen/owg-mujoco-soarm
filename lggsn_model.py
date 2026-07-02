# lggsn_model.py
# 简化版 LGGSN：只用几何特征做抓取好坏预测

import torch
import torch.nn as nn


class LGGSN(nn.Module):
    """
    Lightweight Grasp Geometry Scoring Network.

    目前只用几何 + 质量特征 (geom) 来预测抓取质量 label。
    预留了 query embedding 接口，方便以后接语言/类信息。

    Args:
        n_queries: 查询 id 的总数（现在用不到，先留接口）
        geom_dim:  几何特征维度（= feature_cols 的长度）
        query_dim: query embedding 维度（现在我们设成 0，相当于不用）
        hidden_dim: MLP 隐藏层维度
    """
    def __init__(
        self,
        n_queries: int,
        geom_dim: int = 12,
        query_dim: int = 0,
        hidden_dim: int = 40,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_query = query_dim > 0

        if self.use_query:
            self.query_emb = nn.Embedding(n_queries, query_dim)
        else:
            self.query_emb = None

        in_dim = geom_dim + (query_dim if self.use_query else 0)

        # Keep mlp keys stable (mlp.0, mlp.2) regardless of dropout so old
        # checkpoints load without key-index shifts.
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Dropout has no parameters → absent from state_dict → no key conflicts.
        self._drop: nn.Dropout | None = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(self, geom: torch.Tensor, query_id: torch.Tensor):
        """
        geom: [B, geom_dim]
        query_id: [B]，如果 use_query=False 会被忽略
        """
        x = geom
        if self.use_query:
            q = self.query_emb(query_id)  # [B, query_dim]
            x = torch.cat([geom, q], dim=-1)

        # mlp[0]=Linear, mlp[1]=ReLU, mlp[2]=Linear — inject dropout between them.
        h = self.mlp[1](self.mlp[0](x))
        if self._drop is not None:
            h = self._drop(h)
        logit = self.mlp[2](h).squeeze(-1)  # [B]
        return logit

    def predict_with_uncertainty(
        self,
        geom: torch.Tensor,
        query_id: torch.Tensor,
        T: int = 20,
        p: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout: T stochastic forward passes, returns (mean_score, std_score)."""
        drop = self._drop if self._drop is not None else nn.Dropout(p=p)
        drop.train()  # activate dropout regardless of parent module's mode
        stacked = []
        with torch.no_grad():
            for _ in range(T):
                x = geom
                if self.use_query:
                    q = self.query_emb(query_id)
                    x = torch.cat([geom, q], dim=-1)
                h = self.mlp[1](self.mlp[0](x))
                h = drop(h)
                logit = self.mlp[2](h).squeeze(-1)
                stacked.append(torch.sigmoid(logit))
        scores = torch.stack(stacked, dim=0)  # [T, N]
        return scores.mean(dim=0), scores.std(dim=0)


class GatingNetwork(nn.Module):
    """
    Geometry-conditioned feature gate.

    Takes a 3-dim episode context vector z = [flat_frac, sigma_H, sigma_yaw]
    and outputs a soft feature mask in (0,1)^geom_dim via a small MLP.

    Parameters: 3*16 + 16 + 16*geom_dim + geom_dim  (≈ 302 for geom_dim=14)
    """
    def __init__(self, context_dim: int = 3, feat_dim: int = 14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., context_dim]  ->  [..., feat_dim]  values in (0,1)"""
        return self.net(z)


class GC_LGGSN(nn.Module):
    """
    Geometry-Conditioned LGGSN (GC-LGGSN).

    Prepends a GatingNetwork that soft-masks per-candidate features based on
    three episode-level geometric statistics before passing gated features to
    the existing LGGSN scorer.  The LGGSN scorer is unchanged.

    Args:
        n_queries, geom_dim, query_dim, hidden_dim — same as LGGSN
        context_dim — dimensionality of the episode context vector (default 3)
    """
    def __init__(
        self,
        n_queries: int,
        geom_dim: int = 14,
        query_dim: int = 0,
        hidden_dim: int = 40,
        context_dim: int = 3,
    ):
        super().__init__()
        self.gate   = GatingNetwork(context_dim=context_dim, feat_dim=geom_dim)
        self.scorer = LGGSN(n_queries=n_queries, geom_dim=geom_dim,
                            query_dim=query_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        geom:     torch.Tensor,   # [B, geom_dim]
        query_id: torch.Tensor,   # [B]
        context:  torch.Tensor,   # [B, context_dim]
    ) -> torch.Tensor:            # [B]
        gate        = self.gate(context)          # [B, geom_dim]
        gated_geom  = gate * geom                 # element-wise mask
        return self.scorer(gated_geom, query_id)


class LGGSNVision(nn.Module):
    """
    LGGSN + SAM visual feature (simple concatenation).

    Two inputs are concatenated before the scoring MLP:
      - geom    [B, geom_dim]  — 18 geometric/physics features
      - vis     [B, vis_dim]   — 256-dim SAM ROI embedding (episode-level)
    A learned query embedding is also concatenated (same design as LGGSN).

    MLP input dim = geom_dim + vis_dim + query_dim  (278 with defaults).
    hidden_dim is set larger than LGGSN's 40 because the input is much wider.

    State-dict key names (query_emb.weight, mlp.0/2.weight/bias) are
    intentionally kept identical to LGGSN so grasp_ranker_lggsn.py can
    auto-detect the architecture from checkpoint shape.
    """

    def __init__(
        self,
        n_queries:  int,
        geom_dim:   int   = 18,
        vis_dim:    int   = 256,
        query_dim:  int   = 4,
        hidden_dim: int   = 64,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.geom_dim  = geom_dim
        self.vis_dim   = vis_dim
        self.use_query = query_dim > 0

        if self.use_query:
            self.query_emb = nn.Embedding(n_queries, query_dim)
        else:
            self.query_emb = None

        in_dim = geom_dim + vis_dim + (query_dim if self.use_query else 0)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self._drop: nn.Dropout | None = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(
        self,
        geom:     torch.Tensor,  # [B, geom_dim]
        vis:      torch.Tensor,  # [B, vis_dim]
        query_id: torch.Tensor,  # [B]
    ) -> torch.Tensor:           # [B]
        parts = [geom, vis]
        if self.use_query:
            parts.append(self.query_emb(query_id))  # [B, query_dim]
        x = torch.cat(parts, dim=-1)                # [B, in_dim]

        h = self.mlp[1](self.mlp[0](x))
        if self._drop is not None:
            h = self._drop(h)
        return self.mlp[2](h).squeeze(-1)           # [B]

    def predict_with_uncertainty(
        self,
        geom:     torch.Tensor,
        vis:      torch.Tensor,
        query_id: torch.Tensor,
        T:        int   = 20,
        p:        float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout uncertainty estimate. Returns (mean_score, std_score)."""
        drop = self._drop if self._drop is not None else nn.Dropout(p=p)
        drop.train()
        stacked = []
        with torch.no_grad():
            for _ in range(T):
                parts = [geom, vis]
                if self.use_query:
                    parts.append(self.query_emb(query_id))
                x = torch.cat(parts, dim=-1)
                h = self.mlp[1](self.mlp[0](x))
                h = drop(h)
                stacked.append(torch.sigmoid(self.mlp[2](h).squeeze(-1)))
        scores = torch.stack(stacked, dim=0)        # [T, N]
        return scores.mean(dim=0), scores.std(dim=0)

