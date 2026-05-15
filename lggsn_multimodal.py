"""
lggsn_multimodal.py

Multimodal LG-GSN:
- Geometry-only grasp features (from existing LGGSN)
- Semantic 3D point cloud features (Stage 3 semantic_pc)
- Text features (query embedding, placeholder for now)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPointEncoder(nn.Module):
    """
    Encode semantic point cloud P ∈ R^{N×3} into a global feature g_sem ∈ R^{D}.

    Very simple PointNet-style encoder:
        - per-point MLP
        - global max-pooling
        - linear projection
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dims: Tuple[int, int, int] = (64, 128, 256),
        out_dim: int = 256,
    ):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(last, out_dim)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts: (B, N, 3) point cloud

        Returns:
            feat: (B, out_dim)
        """
        # (B, N, C)
        x = self.mlp(pts)
        # global max-pooling over points -> (B, C)
        x = x.max(dim=1).values
        # projection -> (B, out_dim)
        x = self.proj(x)
        return x


class TextEncoderPlaceholder(nn.Module):
    """
    Placeholder text encoder.

    In a real system you would:
      - either pre-compute CLIP / OpenAI embeddings and feed them here
      - or replace this with a frozen language model encoder

    For now we just take a precomputed embedding of dim=in_dim and
    project it to dim=out_dim.
    """

    def __init__(self, in_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_emb: (B, in_dim) precomputed text embedding

        Returns:
            feat: (B, out_dim)
        """
        return self.proj(text_emb)


class LggsnMultimodal(nn.Module):
    """
    Multimodal LG-GSN head.

    It takes three kinds of features:
      - Geometric grasp feature g_geom ∈ R^{Dg}
      - Semantic point cloud feature g_sem ∈ R^{Ds}
      - Text feature g_text ∈ R^{Dt}

    and predicts a scalar success score y_hat ∈ (0,1) via a small MLP.
    """

    def __init__(
        self,
        dim_geom: int = 40,
        dim_sem: int = 256,
        dim_text: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        in_dim = dim_geom + dim_sem + dim_text
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        geom_feat: torch.Tensor,
        sem_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            geom_feat: (B, Dg) grasp geometry feature
            sem_feat:  (B, Ds) semantic point cloud feature
            text_feat: (B, Dt) text feature

        Returns:
            prob: (B,) predicted success probability in [0,1]
        """
        x = torch.cat([geom_feat, sem_feat, text_feat], dim=-1)
        logit = self.mlp(x).squeeze(-1)  # (B,)
        prob = torch.sigmoid(logit)
        return prob

