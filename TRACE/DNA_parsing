from __future__ import annotations

import numpy as np


def clean_dna(s: str) -> str:
    s = (s or "").upper().replace("U", "T")
    return "".join(ch for ch in s if ch in "ACGTN")


def onehot_encode(seq: str) -> np.ndarray:
    seq = clean_dna(seq)
    x = np.zeros((len(seq), 4), dtype=np.float32)
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, ch in enumerate(seq):
        j = m.get(ch, None)
        if j is not None:
            x[i, j] = 1.0
    return x
