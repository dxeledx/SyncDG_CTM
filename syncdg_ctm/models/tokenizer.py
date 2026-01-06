from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TokenizerConfig:
    n_channels: int = 22
    d_token: int = 64
    n_tokens: int = 16
    temporal_kernels: tuple[int, ...] = (63, 125, 250)
    dropout: float = 0.1
    temporal_filters: int = 8


class MultiScaleTokenizer(nn.Module):
    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if cfg.d_token <= 0:
            raise ValueError("d_token must be > 0")
        if cfg.n_tokens <= 0:
            raise ValueError("n_tokens must be > 0")
        if len(cfg.temporal_kernels) == 0:
            raise ValueError("temporal_kernels must be non-empty")

        branches = []
        for k in cfg.temporal_kernels:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=cfg.temporal_filters,
                        kernel_size=(1, int(k)),
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm2d(cfg.temporal_filters),
                    nn.ELU(),
                )
            )
        self.temporal_branches = nn.ModuleList(branches)

        in_ch = cfg.temporal_filters * len(cfg.temporal_kernels)
        self.spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=(cfg.n_channels, 1),
                groups=in_ch,
                bias=False,
            ),
            nn.BatchNorm2d(in_ch),
            nn.ELU(),
        )

        self.mix = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=cfg.d_token, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(cfg.d_token),
            nn.ELU(),
            nn.Dropout(cfg.dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, cfg.n_tokens))
        self.ln = nn.LayerNorm(cfg.d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            tokens: (B, L, d_token) with L = n_tokens
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,C,T), got {tuple(x.shape)}")

        x2 = x.unsqueeze(1)  # (B,1,C,T)
        feats = [branch(x2) for branch in self.temporal_branches]  # each (B,F,C,T)
        h = torch.cat(feats, dim=1)  # (B,F*Ms,C,T)
        h = self.spatial(h)  # (B,F*Ms,1,T)
        h = self.mix(h)  # (B,d_token,1,T)
        h = self.pool(h)  # (B,d_token,1,L)
        tokens = h.squeeze(2).transpose(1, 2).contiguous()  # (B,L,d_token)
        return self.ln(tokens)

