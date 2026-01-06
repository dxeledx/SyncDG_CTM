from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EEGTrialDataset(Dataset):
    x: np.ndarray  # (N, C, T)
    y: np.ndarray  # (N,)
    domain: np.ndarray  # (N,)
    transform: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.x.ndim != 3:
            raise ValueError(f"x must have shape (N,C,T), got {self.x.shape}")
        if self.y.ndim != 1 or self.domain.ndim != 1:
            raise ValueError(f"y/domain must be 1D, got y={self.y.shape} domain={self.domain.shape}")
        if not (len(self.x) == len(self.y) == len(self.domain)):
            raise ValueError("x/y/domain must have same length")

        self.x = self.x.astype(np.float32, copy=False)
        self.y = self.y.astype(np.int64, copy=False)
        self.domain = self.domain.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.x[idx]
        if self.transform is not None:
            x = self.transform(x)
        return {
            "x": torch.from_numpy(x).float(),  # (C,T)
            "y": torch.tensor(int(self.y[idx]), dtype=torch.long),
            "domain": torch.tensor(int(self.domain[idx]), dtype=torch.long),
        }

