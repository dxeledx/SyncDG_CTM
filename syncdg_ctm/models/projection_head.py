from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ProjectionHeadConfig:
    in_dim: int
    hidden: int = 256
    out_dim: int = 128


class ProjectionHead(nn.Module):
    def __init__(self, cfg: ProjectionHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

