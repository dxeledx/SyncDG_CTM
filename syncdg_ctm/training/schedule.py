from __future__ import annotations

import math


def dann_ramp(progress: float, *, gamma: float = 10.0) -> float:
    """
    Classic DANN schedule: 2/(1+exp(-gamma*p)) - 1, p in [0,1]
    """
    p = max(0.0, min(1.0, float(progress)))
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)

