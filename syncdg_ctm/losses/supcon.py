from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Supervised contrastive loss (Khosla et al., 2020).

    Args:
        features: (B, D) float, will be l2-normalized.
        labels: (B,) int.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be (B,D), got {tuple(features.shape)}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be (B,) and match features batch size")

    feats = F.normalize(features, p=2, dim=1)
    logits = (feats @ feats.T) / float(temperature)

    # Mask out self-contrast.
    b = feats.shape[0]
    self_mask = torch.eye(b, device=feats.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, float("-inf"))

    labels = labels.contiguous().view(-1, 1)
    pos_mask = labels.eq(labels.T) & (~self_mask)

    # log_prob_i,j = log softmax over j
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_log_prob = log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1)
    pos_count = pos_mask.sum(dim=1).clamp_min(1)

    loss = -(pos_log_prob / pos_count).mean()
    if torch.isnan(loss):
        return features.new_tensor(0.0)
    return loss

