from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ReliabilityConfig:
    in_dim: int = 5
    hidden: int = 32
    dropout: float = 0.1
    tau_g: float = 1.0


class ReliabilityPredictor(nn.Module):
    def __init__(self, cfg: ReliabilityConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u: (B, T, F) unlabeled statistics (should be detached from backbone)
        Returns:
            q: (B, T) in (0,1)
            w_pred: (B, T) softmax over ticks
        """
        if u.ndim != 3:
            raise ValueError(f"u must be (B,T,F), got {tuple(u.shape)}")
        b, t, f = u.shape
        if f != self.cfg.in_dim:
            raise ValueError(f"u last dim {f} != cfg.in_dim {self.cfg.in_dim}")

        q = torch.sigmoid(self.mlp(u).squeeze(-1))
        w = F.softmax(q / float(self.cfg.tau_g), dim=1)
        return q, w

