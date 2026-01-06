from __future__ import annotations

import torch
import torch.nn.functional as F


def tick_diversity_loss(z_ticks: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """
    Anti-collapse: encourage decorrelation between ticks.

    Args:
        z_ticks: (B, T, D)
    """
    if z_ticks.ndim != 3:
        raise ValueError(f"z_ticks must be (B,T,D), got {tuple(z_ticks.shape)}")

    b, t, d = z_ticks.shape
    u = F.layer_norm(z_ticks, (d,), eps=eps)
    g = (u @ u.transpose(1, 2)) / float(d)  # (B,T,T)
    offdiag = g - torch.diag_embed(torch.diagonal(g, dim1=1, dim2=2))
    loss = (offdiag**2).mean()
    if torch.isnan(loss):
        return z_ticks.new_tensor(0.0)
    return loss

