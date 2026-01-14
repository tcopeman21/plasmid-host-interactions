from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AugmentConfig:
    shift_max: int = 3
    mut_rate: float = 0.02


def load_pkl_sequences(pkl_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Expects dict with keys: 'sequences' (N, 194, 4) float32 and 'expressions' (N,) float32.
    Crops first+last bp -> (N, 192, 4).
    """
    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)

    X_full = np.asarray(blob["sequences"], dtype=np.float32)
    y = np.asarray(blob["expressions"], dtype=np.float32).reshape(-1)

    if X_full.ndim != 3 or X_full.shape[-1] != 4:
        raise ValueError(f"Expected sequences shape (N, L, 4). Got {X_full.shape}.")

    if X_full.shape[1] < 192:
        raise ValueError(f"Sequence length too short: {X_full.shape[1]}")

    X = X_full[:, 1:-1, :]  # 194 -> 192
    return X, y


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if len(X) != len(y):
            raise ValueError("X and y length mismatch.")
        self.X = torch.from_numpy(X)  # (N, L, 4)
        self.y = torch.from_numpy(y)  # (N,)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def rc_onehot(x: torch.Tensor) -> torch.Tensor:
    """
    Reverse-complement of one-hot encoded DNA.
    x: (B, L, 4) where channels are A,C,G,T in that order.
    """
    idx = torch.tensor([3, 2, 1, 0], device=x.device)
    return torch.flip(x, dims=[1]).index_select(2, idx)


def augment_batch(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    """
    x: (B, L, 4)
    - random circular shift along L
    - random base mutations at cfg.mut_rate
    """
    x_aug = x.clone()
    B, L, _ = x_aug.shape

    if cfg.shift_max > 0:
        shifts = torch.randint(-cfg.shift_max, cfg.shift_max + 1, (B,), device=x.device)
        for i in range(B):
            x_aug[i] = torch.roll(x_aug[i], shifts=int(shifts[i].item()), dims=0)

    if cfg.mut_rate > 0:
        mask = torch.rand((B, L), device=x.device) < cfg.mut_rate
        new_bases = F.one_hot(
            torch.randint(0, 4, (B, L), device=x.device),
            num_classes=4
        ).float()
        x_aug = torch.where(mask.unsqueeze(-1), new_bases, x_aug)

    return x_aug
