from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss


@dataclass(frozen=True)
class Metrics:
    acc: float
    kappa: float
    macro_f1: float
    nll: float


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 15) -> float:
    """
    ECE for multiclass using max probability as confidence.
    """
    y_true = y_true.astype(int)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        ece += float(mask.mean()) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = y_prob.argmax(axis=1)
    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "nll": float(log_loss(y_true, y_prob, labels=list(range(y_prob.shape[1])))),
        "ece": float(expected_calibration_error(y_true, y_prob)),
    }
    return metrics

