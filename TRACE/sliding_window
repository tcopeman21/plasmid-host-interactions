from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch

from trace.dna import clean_dna, onehot_encode


@dataclass(frozen=True)
class SlidingWindowConfig:
    window: int = 192
    stride: int = 20
    batch_size: int = 512


@torch.no_grad()
def window_scores(
    model: torch.nn.Module,
    seq: str,
    cfg: SlidingWindowConfig,
    device: torch.device,
) -> np.ndarray:
    """
    Returns scalar score per window (mean over positions of risk map).
    """
    seq = clean_dna(seq)
    if len(seq) < cfg.window:
        return np.empty((0,), dtype=np.float32)

    windows = [seq[i:i + cfg.window] for i in range(0, len(seq) - cfg.window + 1, cfg.stride)]
    if not windows:
        return np.empty((0,), dtype=np.float32)

    X = np.stack([onehot_encode(w) for w in windows]).astype(np.float32)  # (Nw, L, 4)
    Xt = torch.from_numpy(X)

    model.eval()
    scores: list[np.ndarray] = []
    for i in range(0, Xt.shape[0], cfg.batch_size):
        xb = Xt[i:i + cfg.batch_size].to(device)
        risk = model(xb)  # (B, L)
        scores.append(risk.mean(dim=1).detach().cpu().numpy())

    return np.concatenate(scores).astype(np.float32)
