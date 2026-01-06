from __future__ import annotations

import torch


def coral_loss(
    x: torch.Tensor,
    domain: torch.Tensor,
    *,
    n_domains: int,
    eps: float = 1e-6,
    ridge: float = 1e-3,
) -> torch.Tensor:
    """
    CORAL loss: align covariance matrices across domains.

    Args:
        x: (B, F) features
        domain: (B,) int in [0, n_domains-1]
    """
    if x.ndim != 2:
        raise ValueError(f"x must be (B,F), got {tuple(x.shape)}")
    if domain.ndim != 1:
        raise ValueError(f"domain must be (B,), got {tuple(domain.shape)}")

    b, feat_dim = x.shape
    covs = []
    for d in range(n_domains):
        xd = x[domain == d]
        if xd.numel() == 0:
            continue
        xd = xd - xd.mean(dim=0, keepdim=True)
        denom = max(int(xd.shape[0]) - 1, 1)
        cd = (xd.T @ xd) / float(denom)
        cd = cd + ridge * torch.eye(feat_dim, device=x.device, dtype=x.dtype)
        covs.append(cd)

    if len(covs) <= 1:
        return x.new_tensor(0.0)

    covs_stacked = torch.stack(covs, dim=0)  # (D,F,F)
    c_bar = covs_stacked.mean(dim=0, keepdim=True)
    loss = ((covs_stacked - c_bar) ** 2).mean()
    if torch.isnan(loss):
        return x.new_tensor(0.0)
    return loss

