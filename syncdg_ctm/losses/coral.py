from __future__ import annotations

from dataclasses import dataclass

import torch


def coral_loss(
    x: torch.Tensor,
    domain: torch.Tensor,
    *,
    n_domains: int,
    ridge: float = 1e-3,
) -> torch.Tensor:
    """
    CORAL loss: align covariance matrices across domains (per-batch version).

    Args:
        x: (B, F) features
        domain: (B,) int in [0, n_domains-1]
    """
    if x.ndim != 2:
        raise ValueError(f"x must be (B,F), got {tuple(x.shape)}")
    if domain.ndim != 1:
        raise ValueError(f"domain must be (B,), got {tuple(domain.shape)}")

    feat_dim = x.shape[1]
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


@dataclass
class CovarianceEMAMemory:
    """
    EMA memory for per-domain covariance matrices.

    Used to stabilize CORAL under small per-domain batch sizes.
    """

    n_domains: int
    feat_dim: int
    momentum: float = 0.01  # zeta in design
    ridge: float = 1e-3  # lambda in design
    shrinkage: float = 0.0  # optional towards scaled identity

    def __post_init__(self) -> None:
        if self.n_domains <= 0 or self.feat_dim <= 0:
            raise ValueError("Invalid n_domains/feat_dim.")
        if not (0.0 < self.momentum <= 1.0):
            raise ValueError("momentum must be in (0,1].")
        if self.ridge < 0.0:
            raise ValueError("ridge must be >= 0.")
        if not (0.0 <= self.shrinkage <= 1.0):
            raise ValueError("shrinkage must be in [0,1].")

        self.cov = torch.zeros(self.n_domains, self.feat_dim, self.feat_dim)
        self.initialized = torch.zeros(self.n_domains, dtype=torch.bool)

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> "CovarianceEMAMemory":
        self.cov = self.cov.to(device=device, dtype=dtype)
        self.initialized = self.initialized.to(device=device)
        return self

    @torch.no_grad()
    def update(self, batch_cov: torch.Tensor, valid: torch.Tensor) -> None:
        """
        Args:
            batch_cov: (n_domains, F, F) (detached)
            valid: (n_domains,) boolean mask
        """
        if batch_cov.shape != self.cov.shape:
            raise ValueError(f"batch_cov must be {tuple(self.cov.shape)}, got {tuple(batch_cov.shape)}")
        if valid.shape != self.initialized.shape:
            raise ValueError(f"valid must be {tuple(self.initialized.shape)}, got {tuple(valid.shape)}")

        m = float(self.momentum)
        init_mask = (~self.initialized) & valid
        if init_mask.any():
            self.cov[init_mask] = batch_cov[init_mask]
            self.initialized[init_mask] = True

        upd_mask = self.initialized & valid
        if upd_mask.any():
            self.cov[upd_mask] = (1.0 - m) * self.cov[upd_mask] + m * batch_cov[upd_mask]


def _shrink_cov(cov: torch.Tensor, shrinkage: float) -> torch.Tensor:
    if shrinkage <= 0.0:
        return cov
    f = cov.shape[-1]
    trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    mu = trace / float(f)
    eye = torch.eye(f, device=cov.device, dtype=cov.dtype)
    return (1.0 - shrinkage) * cov + shrinkage * mu.unsqueeze(-1) * eye


def coral_ema_loss(
    x: torch.Tensor,
    domain: torch.Tensor,
    *,
    memory: CovarianceEMAMemory,
) -> torch.Tensor:
    """
    CORAL with EMA covariance memory.

    Gradient flows through current batch covariances; EMA memory provides a stable mean covariance target.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be (B,F), got {tuple(x.shape)}")
    if domain.ndim != 1:
        raise ValueError(f"domain must be (B,), got {tuple(domain.shape)}")
    if x.shape[1] != memory.feat_dim:
        raise ValueError("Feature dim mismatch with covariance memory.")

    feat_dim = x.shape[1]
    batch_cov = x.new_zeros((memory.n_domains, feat_dim, feat_dim), dtype=torch.float32)
    valid = torch.zeros((memory.n_domains,), device=x.device, dtype=torch.bool)

    x_f = x.float()
    for d in range(memory.n_domains):
        mask = domain == d
        if not mask.any():
            continue
        xd = x_f[mask]
        xd = xd - xd.mean(dim=0, keepdim=True)
        denom = max(int(xd.shape[0]) - 1, 1)
        cov = (xd.T @ xd) / float(denom)
        cov = _shrink_cov(cov, float(memory.shrinkage))
        cov = cov + float(memory.ridge) * torch.eye(feat_dim, device=x.device, dtype=torch.float32)
        batch_cov[d] = cov
        valid[d] = True

    if valid.sum() <= 1:
        if valid.any():
            memory.update(batch_cov.detach(), valid.detach())
        return x.new_tensor(0.0)

    if memory.initialized.any():
        c_bar = memory.cov[memory.initialized].mean(dim=0).float().detach()
    else:
        c_bar = batch_cov[valid].mean(dim=0).float().detach()

    diffs = batch_cov[valid] - c_bar.unsqueeze(0)
    loss = (diffs**2).mean()
    if torch.isnan(loss):
        loss = x.new_tensor(0.0)

    memory.update(batch_cov.detach(), valid.detach())
    return loss
