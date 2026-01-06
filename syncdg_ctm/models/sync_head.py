from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SyncHeadConfig:
    d_model: int = 128
    d_out: int = 32
    num_classes: int = 4
    gamma: float = 0.9
    welford_shrinkage: float = 0.1
    eps: float = 1e-5
    eps_h: float = 1e-6


class SyncProjectionHead(nn.Module):
    """
    Online Welford -> low-rank projection energy -> length-normalized log-space features (x_tau)
    """

    def __init__(self, cfg: SyncHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.d_model <= 0 or cfg.d_out <= 0:
            raise ValueError("d_model/d_out must be > 0")
        if not (0.0 <= cfg.welford_shrinkage <= 1.0):
            raise ValueError("welford_shrinkage must be in [0,1]")
        if not (0.0 <= cfg.gamma <= 1.0):
            raise ValueError("gamma must be in [0,1]")

        self.proj = nn.Parameter(torch.randn(cfg.d_model, cfg.d_out) * 0.02)
        self.ln = nn.LayerNorm(cfg.d_out)
        self.classifier = nn.Linear(cfg.d_out, cfg.num_classes)

    def forward(self, z_ticks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_ticks: (B, T, D)
        Returns:
            x_ticks: (B, T, D_out)  (log-space, NO LN)
            logits_ticks: (B, T, C)
        """
        if z_ticks.ndim != 3:
            raise ValueError(f"z_ticks must be (B,T,D), got {tuple(z_ticks.shape)}")
        b, t, d = z_ticks.shape
        if d != self.cfg.d_model:
            raise ValueError(f"z_ticks last dim {d} != cfg.d_model {self.cfg.d_model}")

        m = z_ticks.new_zeros((b, d))
        m2 = z_ticks.new_zeros((b, d))

        h = z_ticks.new_zeros((b, self.cfg.d_out))
        s = z_ticks.new_zeros(())  # scalar

        x_list = []
        logits_list = []

        alpha = float(self.cfg.welford_shrinkage)
        gamma = float(self.cfg.gamma)
        eps = float(self.cfg.eps)
        eps_h = float(self.cfg.eps_h)

        for tau in range(1, t + 1):
            z = z_ticks[:, tau - 1, :]  # (B,D)
            delta = z - m
            m = m + delta / float(tau)
            delta2 = z - m
            m2 = m2 + delta * delta2

            denom = float(max(tau - 1, 1))
            v = m2 / denom
            v_bar = v.mean(dim=1, keepdim=True)
            v_tilde = (1.0 - alpha) * v + alpha * v_bar
            z_tilde = (z - m) / torch.sqrt(v_tilde + eps)

            r = z_tilde @ self.proj  # (B,D_out)
            h = gamma * h + r * r
            s = gamma * s + 1.0
            h_bar = h / s
            x = torch.log(h_bar + eps_h)

            x_list.append(x)
            logits_list.append(self.classifier(self.ln(x)))

        x_ticks = torch.stack(x_list, dim=1)
        logits_ticks = torch.stack(logits_list, dim=1)
        return x_ticks, logits_ticks

