"""WorldModelMLP — simple binary MLP for grasp success prediction.

Trained on 22-dim feature vectors from data/transition_logger.py.
This module is kept separate so that importing benchmark.methods does
not require torch unless the world_model method is actually used.

To train:
    python scripts/train_world_model.py
"""

import torch
import torch.nn as nn


class WorldModelMLP(nn.Module):
    def __init__(self, input_dim: int = 22, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
