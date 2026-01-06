from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DomainDiscriminatorConfig:
    in_dim: int
    n_domains: int
    hidden: int = 256
    dropout: float = 0.2


class DomainDiscriminator(nn.Module):
    def __init__(self, cfg: DomainDiscriminatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

