from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class BalancedDomainClassBatchSampler(Sampler[list[int]]):
    """
    Per batch: sample P domains, and for each domain sample K trials per class.

    Batch size = P * n_classes * K
    """

    def __init__(
        self,
        labels: np.ndarray,
        domains: np.ndarray,
        *,
        n_classes: int,
        domains_per_batch: int,
        samples_per_class: int,
        steps_per_epoch: int | None = None,
        seed: int = 0,
    ) -> None:
        if labels.ndim != 1 or domains.ndim != 1:
            raise ValueError("labels/domains must be 1D arrays")
        if len(labels) != len(domains):
            raise ValueError("labels and domains must have same length")
        if n_classes <= 1:
            raise ValueError("n_classes must be > 1")
        if domains_per_batch <= 0 or samples_per_class <= 0:
            raise ValueError("domains_per_batch and samples_per_class must be > 0")

        self.labels = labels.astype(np.int64, copy=False)
        self.domains = domains.astype(np.int64, copy=False)
        self.n_classes = int(n_classes)
        self.domains_per_batch = int(domains_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0

        unique_domains = np.unique(self.domains)
        self.unique_domains = unique_domains.astype(np.int64, copy=False)

        self.domain_class_to_indices: dict[tuple[int, int], np.ndarray] = {}
        for d in self.unique_domains:
            for c in range(self.n_classes):
                idx = np.flatnonzero((self.domains == d) & (self.labels == c))
                if len(idx) == 0:
                    raise ValueError(f"No samples for domain={int(d)} class={c}.")
                self.domain_class_to_indices[(int(d), int(c))] = idx

        if steps_per_epoch is None:
            batch_size = self.domains_per_batch * self.n_classes * self.samples_per_class
            steps_per_epoch = max(1, len(self.labels) // batch_size)
        self.steps_per_epoch = int(steps_per_epoch)

        if self.domains_per_batch > len(self.unique_domains):
            raise ValueError(
                f"domains_per_batch={self.domains_per_batch} exceeds available domains={len(self.unique_domains)}"
            )

    def __len__(self) -> int:  # type: ignore[override]
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:  # type: ignore[override]
        rng = np.random.default_rng(self.seed + self.epoch)

        for _ in range(self.steps_per_epoch):
            batch_indices: list[int] = []
            chosen_domains = rng.choice(self.unique_domains, size=self.domains_per_batch, replace=False)
            for d in chosen_domains:
                d_int = int(d)
                for c in range(self.n_classes):
                    pool = self.domain_class_to_indices[(d_int, c)]
                    chosen = rng.choice(pool, size=self.samples_per_class, replace=len(pool) < self.samples_per_class)
                    batch_indices.extend(int(i) for i in chosen.tolist())
            rng.shuffle(batch_indices)
            yield batch_indices

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
