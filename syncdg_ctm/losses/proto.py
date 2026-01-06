from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PrototypeEMAMemory:
    n_domains: int
    n_classes: int
    feat_dim: int
    momentum: float = 0.1

    def __post_init__(self) -> None:
        if self.n_domains <= 0 or self.n_classes <= 1 or self.feat_dim <= 0:
            raise ValueError("Invalid n_domains/n_classes/feat_dim.")
        if not (0.0 < self.momentum <= 1.0):
            raise ValueError("momentum must be in (0,1].")

        self.mu = torch.zeros(self.n_domains, self.n_classes, self.feat_dim)
        self.initialized = torch.zeros(self.n_domains, self.n_classes, dtype=torch.bool)

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> "PrototypeEMAMemory":
        self.mu = self.mu.to(device=device, dtype=dtype)
        self.initialized = self.initialized.to(device=device)
        return self

    @torch.no_grad()
    def update(self, batch_mu: torch.Tensor, valid: torch.Tensor) -> None:
        """
        Update EMA prototypes.

        Args:
            batch_mu: (n_domains, n_classes, feat_dim)
            valid: (n_domains, n_classes) boolean mask, True where batch_mu is meaningful
        """
        if batch_mu.shape != self.mu.shape:
            raise ValueError(f"batch_mu must be {tuple(self.mu.shape)}, got {tuple(batch_mu.shape)}")
        if valid.shape != self.initialized.shape:
            raise ValueError(f"valid must be {tuple(self.initialized.shape)}, got {tuple(valid.shape)}")

        momentum = float(self.momentum)

        init_mask = (~self.initialized) & valid
        if init_mask.any():
            self.mu[init_mask] = batch_mu[init_mask]
            self.initialized[init_mask] = True

        upd_mask = self.initialized & valid
        if upd_mask.any():
            self.mu[upd_mask] = (1.0 - momentum) * self.mu[upd_mask] + momentum * batch_mu[upd_mask]


def proto_alignment_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    domain: torch.Tensor,
    *,
    memory: PrototypeEMAMemory,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prototype alignment (class-conditional, multi-source).

    Returns:
        loss: scalar
        batch_mu: (n_domains, n_classes, feat_dim) differentiable
    """
    if x.ndim != 2:
        raise ValueError(f"x must be (B,F), got {tuple(x.shape)}")
    if y.ndim != 1 or domain.ndim != 1:
        raise ValueError("y/domain must be (B,)")
    if not (len(x) == len(y) == len(domain)):
        raise ValueError("x/y/domain must have same batch size")

    device = x.device
    feat_dim = x.shape[1]
    n_domains = memory.n_domains
    n_classes = memory.n_classes
    if feat_dim != memory.feat_dim:
        raise ValueError("Feature dim mismatch with memory.")

    batch_mu = torch.zeros(n_domains, n_classes, feat_dim, device=device, dtype=x.dtype)
    counts = torch.zeros(n_domains, n_classes, device=device, dtype=torch.long)

    for d in range(n_domains):
        for c in range(n_classes):
            mask = (domain == d) & (y == c)
            if mask.any():
                batch_mu[d, c] = x[mask].mean(dim=0)
                counts[d, c] = int(mask.sum())

    valid = counts > 0

    # Update EMA with detached batch prototypes (no leakage into gradients).
    memory.update(batch_mu.detach(), valid)

    # Global prototypes from EMA over initialized domains only (constant w.r.t. current batch).
    init = memory.initialized  # (D,C)
    global_mu = torch.zeros(1, n_classes, feat_dim, device=device, dtype=x.dtype)
    for c in range(n_classes):
        mask = init[:, c]
        if mask.any():
            global_mu[0, c] = memory.mu[mask, c].mean(dim=0)
        elif valid[:, c].any():
            global_mu[0, c] = batch_mu[valid[:, c], c].mean(dim=0)
        else:
            global_mu[0, c] = 0.0

    if valid.sum() == 0:
        return x.new_tensor(0.0), batch_mu

    diffs = batch_mu - global_mu.expand_as(batch_mu)
    loss = (diffs[valid] ** 2).mean()
    return loss, batch_mu
