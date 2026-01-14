from __future__ import annotations

import numpy as np


def softmin(x: np.ndarray, tau: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    m = float(np.min(x))
    # stable softmin via shift by min
    return float(-np.log(np.sum(np.exp(-tau * (x - m)))) / tau + m)


def agg_e1(x: np.ndarray, tau: float, w_soft: float = 0.5, w_mean: float = 0.5) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return w_soft * softmin(x, tau) + w_mean * float(np.mean(x))
