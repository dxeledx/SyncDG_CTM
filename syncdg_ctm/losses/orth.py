from __future__ import annotations

import torch


def orth_loss(p: torch.Tensor) -> torch.Tensor:
    """
    ||P^T P - I||_F^2

    Args:
        p: (D, D_o)
    """
    if p.ndim != 2:
        raise ValueError(f"p must be 2D, got {tuple(p.shape)}")
    d_out = p.shape[1]
    gram = p.T @ p
    eye = torch.eye(d_out, device=p.device, dtype=p.dtype)
    return ((gram - eye) ** 2).mean()

