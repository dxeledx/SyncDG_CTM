from __future__ import annotations

import torch
import torch.nn.functional as F

from syncdg_ctm.losses.grl import grad_reverse


def cdan_onehot_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    domain: torch.Tensor,
    discriminator,
    *,
    n_classes: int,
    grl_lambda: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Conditional Domain Adversarial Network (CDAN) using true one-hot labels.

    Args:
        x: (B, F) feature
        y: (B,) class index
        domain: (B,) domain index
        discriminator: module mapping (B, F*n_classes) -> (B, n_domains)
        sample_weight: (B,) optional weight (detached)
    """
    if x.ndim != 2:
        raise ValueError(f"x must be (B,F), got {tuple(x.shape)}")
    if y.ndim != 1 or domain.ndim != 1:
        raise ValueError("y/domain must be 1D")
    if not (len(x) == len(y) == len(domain)):
        raise ValueError("x/y/domain must have same batch size")

    y_onehot = F.one_hot(y, num_classes=n_classes).to(dtype=x.dtype)
    g = torch.bmm(x.unsqueeze(2), y_onehot.unsqueeze(1))  # (B,F,C)
    g = g.view(x.shape[0], -1)
    g = grad_reverse(g, grl_lambda)

    dom_logits = discriminator(g)
    loss = F.cross_entropy(dom_logits, domain, reduction="none")
    if sample_weight is not None:
        loss = loss * sample_weight.detach()
    return loss.mean()

