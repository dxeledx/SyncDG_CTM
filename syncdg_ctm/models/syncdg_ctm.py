from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from syncdg_ctm.models.ctm_core import CTMCore, CTMCoreConfig
from syncdg_ctm.models.reliability import ReliabilityConfig, ReliabilityPredictor
from syncdg_ctm.models.sync_head import SyncHeadConfig, SyncProjectionHead
from syncdg_ctm.models.tokenizer import MultiScaleTokenizer, TokenizerConfig


@dataclass(frozen=True)
class SyncDGCTMConfig:
    num_classes: int = 4
    tokenizer: TokenizerConfig = TokenizerConfig()
    ctm: CTMCoreConfig = CTMCoreConfig()
    sync_head: SyncHeadConfig = SyncHeadConfig()
    reliability: ReliabilityConfig = ReliabilityConfig()


class SyncDGCTM(nn.Module):
    def __init__(self, cfg: SyncDGCTMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        tok_cfg = cfg.tokenizer
        ctm_cfg = cfg.ctm
        head_cfg = cfg.sync_head
        rel_cfg = cfg.reliability

        if cfg.num_classes != head_cfg.num_classes:
            head_cfg = SyncHeadConfig(
                d_model=head_cfg.d_model,
                d_out=head_cfg.d_out,
                num_classes=cfg.num_classes,
                gamma=head_cfg.gamma,
                welford_shrinkage=head_cfg.welford_shrinkage,
                eps=head_cfg.eps,
                eps_h=head_cfg.eps_h,
            )
            self.cfg = SyncDGCTMConfig(
                num_classes=cfg.num_classes,
                tokenizer=tok_cfg,
                ctm=ctm_cfg,
                sync_head=head_cfg,
                reliability=rel_cfg,
            )

        self.tokenizer = MultiScaleTokenizer(tok_cfg)
        self.ctm = CTMCore(ctm_cfg)
        self.head = SyncProjectionHead(head_cfg)
        self.g = ReliabilityPredictor(rel_cfg)

    @staticmethod
    def _entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return -(p * torch.log(p + eps)).sum(dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, T)
        Returns:
            dict with:
              z_ticks: (B,T,D)
              x_ticks: (B,T,D_out)  (log-space, no LN)
              logits_ticks: (B,T,C)
              p_ticks: (B,T,C)
              u_ticks: (B,T,5)
              q_ticks: (B,T)
              w_pred: (B,T)
        """
        tokens = self.tokenizer(x)
        z_ticks = self.ctm(tokens)
        x_ticks, logits_ticks = self.head(z_ticks)

        p_ticks = F.softmax(logits_ticks, dim=-1)
        entropy = self._entropy(p_ticks)  # (B,T)
        top2 = p_ticks.topk(2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]

        # KL(p_tau || p_{tau-1}) with KL_0 = 0
        log_p = torch.log(p_ticks + 1e-8)
        kl = p_ticks[:, 1:, :] * (log_p[:, 1:, :] - log_p[:, :-1, :])
        kl = kl.sum(dim=-1)
        kl = torch.cat([torch.zeros_like(kl[:, :1]), kl], dim=1)

        x_l1 = x_ticks.abs().sum(dim=-1)
        delta_h = torch.cat([torch.zeros_like(entropy[:, :1]), entropy[:, :-1] - entropy[:, 1:]], dim=1)

        u = torch.stack([entropy, margin, kl, x_l1, delta_h], dim=-1)  # (B,T,5)

        # g only reads detached (label-free) statistics.
        q, w_pred = self.g(u.detach())

        return {
            "z_ticks": z_ticks,
            "x_ticks": x_ticks,
            "logits_ticks": logits_ticks,
            "p_ticks": p_ticks,
            "u_ticks": u,
            "q_ticks": q,
            "w_pred": w_pred,
        }

