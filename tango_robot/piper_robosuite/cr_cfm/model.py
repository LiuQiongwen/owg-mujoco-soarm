"""CR-CFM Stage A model: a deliberately tiny 1D-conv flow-matching network
with FiLM conditioning on (flow time step, drift estimate) -- target VRAM
well under the 2GB budget, meant to run on the 3060 (or even CPU) for
architecture/loss validation before Stage B swaps in DinoV2+DiT on the
A100 window. See piper_robosuite/README.md's CR-CFM entry for the staged
plan this belongs to.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation -- gamma/beta from the combined
    (time, condition) embedding, applied per-channel to the conv feature map."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H); cond: (B, cond_dim)
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return x * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


class CRFlowNet(nn.Module):
    """Predicts the flow-matching velocity field v_theta(x_t, t, cond) for
    an action chunk x_t: (B, horizon, action_dim). 3-layer 1D conv core
    (matches the design doc's "Conv1D 深度仅为3层"), FiLM-conditioned on a
    combined time+drift embedding at every layer.
    """

    def __init__(self, action_dim: int = 6, horizon: int = 16, hidden: int = 64,
                 time_emb_dim: int = 32, cond_in_dim: int = 6, cond_emb_dim: int = 32):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_in_dim, cond_emb_dim), nn.SiLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
        )
        combined_dim = time_emb_dim + cond_emb_dim
        self.combined_proj = nn.Sequential(nn.Linear(combined_dim, hidden), nn.SiLU())

        self.in_proj = nn.Conv1d(action_dim, hidden, kernel_size=3, padding=1)
        self.film1 = FiLM(hidden, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.film2 = FiLM(hidden, hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.out_proj = nn.Conv1d(hidden, action_dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x_t: (B, horizon, action_dim) -> (B, action_dim, horizon) for Conv1d
        x = x_t.transpose(1, 2)
        te = self.time_embed(t)
        ce = self.cond_encoder(cond)
        c = self.combined_proj(torch.cat([te, ce], dim=-1))

        h = self.act(self.in_proj(x))
        h = self.film1(h, c)
        h = self.act(self.conv1(h))
        h = self.film2(h, c)
        h = self.act(self.conv2(h))
        out = self.out_proj(h)
        return out.transpose(1, 2)  # (B, horizon, action_dim)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
