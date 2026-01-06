from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def compute_channel_stats(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute channel-wise mean/std over trials and time.

    Args:
        x: ndarray of shape (N, C, T)
    Returns:
        mean: (C,)
        std: (C,)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (N,C,T), got {x.shape}")
    mean = x.mean(axis=(0, 2))
    std = x.std(axis=(0, 2))
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass(frozen=True)
class ChannelZScore:
    mean: np.ndarray  # (C,)
    std: np.ndarray  # (C,)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (C,T), got {x.shape}")
        return (x - self.mean[:, None]) / self.std[:, None]

