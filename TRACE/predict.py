from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from models.unet_sequence_model.model import GCNUNet
from trace.aggregate import agg_e1
from trace.sliding_window import SlidingWindowConfig, window_scores


@dataclass(frozen=True)
class TraceConfig:
    window: int = 192
    stride: int = 20
    batch_size: int = 512
    tau_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    e1_w_soft: float = 0.5
    e1_w_mean: float = 0.5


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    m = GCNUNet().to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    m.eval()
    return m


def ensemble_predict(
    ckpts: Sequence[str],
    seqs: Sequence[str],
    trace_cfg: TraceConfig,
    device: torch.device,
    tau: float,
) -> np.ndarray:
    sw_cfg = SlidingWindowConfig(window=trace_cfg.window, stride=trace_cfg.stride, batch_size=trace_cfg.batch_size)
    preds = []

    for ckpt in ckpts:
        model = load_model(ckpt, device=device)
        one_model = []
        for s in seqs:
            ws = window_scores(model, s, cfg=sw_cfg, device=device)
            one_model.append(agg_e1(ws, tau=tau, w_soft=trace_cfg.e1_w_soft, w_mean=trace_cfg.e1_w_mean))
        preds.append(np.asarray(one_model, dtype=np.float32))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.mean(np.stack(preds, axis=0), axis=0)


def tune_tau_on_val(
    ckpts: Sequence[str],
    val_seqs: Sequence[str],
    val_y: np.ndarray,
    trace_cfg: TraceConfig,
    device: torch.device,
) -> float:
    best_tau = trace_cfg.tau_grid[0]
    best_rho = -1e9

    mask = np.isfinite(val_y)
    if mask.sum() < 3:
        raise ValueError("Not enough finite validation labels to tune tau.")

    for tau in trace_cfg.tau_grid:
        pv = ensemble_predict(ckpts, val_seqs, trace_cfg, device, tau=tau)
        rho = spearmanr(pv[mask], val_y[mask]).correlation
        if rho > best_rho:
            best_rho = rho
            best_tau = tau

    return float(best_tau)


def fit_val_transform(pred_val: np.ndarray, y_val: np.ndarray) -> tuple[np.ndarray, callable]:
    """
    Quantile-map then linear-fit on VAL. Returns (mapped_val, transform_fn_for_test).
    """
    mask = np.isfinite(y_val)
    pred_val = np.asarray(pred_val)[mask]
    y_val = np.asarray(y_val)[mask]

    q = np.linspace(0, 1, 200)
    p_q = np.quantile(pred_val, q)
    y_q = np.quantile(y_val, q)

    pred_qm = np.interp(pred_val, p_q, y_q)
    lr = LinearRegression().fit(pred_qm.reshape(-1, 1), y_val)

    def transform(pred_raw: np.ndarray) -> np.ndarray:
        pred_raw = np.asarray(pred_raw)
        pred_qm_test = np.interp(pred_raw, p_q, y_q)
        return (lr.predict(pred_qm_test.reshape(-1, 1))).astype(np.float32)

    return pred_qm.astype(np.float32), transform
