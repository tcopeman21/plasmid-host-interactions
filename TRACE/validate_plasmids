from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from trace.dna import clean_dna
from trace.predict import TraceConfig, ensemble_predict, fit_val_transform, tune_tau_on_val


SEQ_KEYS = ["Position 1 Sequence", "Position 2 Sequence", "Position 3 Sequence", "Position 4 Sequence"]


def build_sequence(row: dict) -> str:
    return "".join(clean_dna(row.get(k, "")) for k in SEQ_KEYS)


def combo_key(row: dict) -> tuple[str, str, str, str]:
    return tuple(clean_dna(row.get(k, "")) for k in SEQ_KEYS)


def split_by_combo(rows: list[dict], y: np.ndarray, test_ratio: float, val_frac: float, rare_threshold: int, seed: int):
    rng = np.random.RandomState(seed)

    combo_to_idx = defaultdict(list)
    for i, r in enumerate(rows):
        combo_to_idx[combo_key(r)].append(i)

    rare = [c for c, idxs in combo_to_idx.items() if len(idxs) <= rare_threshold]
    common = [c for c in combo_to_idx.keys() if c not in set(rare)]

    test = []
    for c in rare:
        test.extend(combo_to_idx[c])

    target_test = int(len(rows) * test_ratio)
    remain = [i for c in common for i in combo_to_idx[c]]
    rng.shuffle(remain)
    need = target_test - len(test)
    if need > 0:
        test.extend(remain[:need])
        train_val = remain[need:]
    else:
        train_val = remain
        test = test[:target_test]

    rng.shuffle(train_val)
    val_n = int(len(train_val) * val_frac)
    val = train_val[:val_n]
    train = train_val[val_n:]
    return np.array(train), np.array(val), np.array(test)


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(pred)
    if m.sum() < 3:
        return {"n": int(m.sum())}
    return {
        "n": int(m.sum()),
        "pearson_r": float(pearsonr(pred[m], y[m])[0]),
        "spearman_rho": float(spearmanr(pred[m], y[m])[0]),
    }


def run(args):
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.lfc_col])

    rows = df.to_dict(orient="records")
    seqs = [build_sequence(r) for r in rows]
    y = df[args.lfc_col].astype(np.float32).values

    train_idx, val_idx, test_idx = split_by_combo(
        rows, y, test_ratio=args.test_ratio, val_frac=args.val_frac,
        rare_threshold=args.rare_threshold, seed=args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trace_cfg = TraceConfig(window=args.window, stride=args.stride, batch_size=args.batch_size)

    # Tune tau on VAL for baseline and finetuned separately (no leakage)
    tau_base = tune_tau_on_val(args.base_ckpts, [seqs[i] for i in val_idx], y[val_idx], trace_cfg, device)
    tau_ft = tune_tau_on_val(args.ft_ckpts, [seqs[i] for i in val_idx], y[val_idx], trace_cfg, device)

    # Baseline raw preds
    pred_val_base = ensemble_predict(args.base_ckpts, [seqs[i] for i in val_idx], trace_cfg, device, tau=tau_base)
    _, transform_base = fit_val_transform(pred_val_base, y[val_idx])
    pred_test_base_raw = ensemble_predict(args.base_ckpts, [seqs[i] for i in test_idx], trace_cfg, device, tau=tau_base)
    pred_test_base = transform_base(pred_test_base_raw)

    # Finetuned raw preds
    pred_val_ft = ensemble_predict(args.ft_ckpts, [seqs[i] for i in val_idx], trace_cfg, device, tau=tau_ft)
    _, transform_ft = fit_val_transform(pred_val_ft, y[val_idx])
    pred_test_ft_raw = ensemble_predict(args.ft_ckpts, [seqs[i] for i in test_idx], trace_cfg, device, tau=tau_ft)
    pred_test_ft = transform_ft(pred_test_ft_raw)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "trace_cfg": asdict(trace_cfg),
        "tau_baseline": float(tau_base),
        "tau_finetuned": float(tau_ft),
        "baseline_test_metrics": metrics(pred_test_base, y[test_idx]),
        "finetuned_test_metrics": metrics(pred_test_ft, y[test_idx]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    with open(os.path.join(out_dir, "trace_validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    export = pd.DataFrame({
        "row_index": test_idx,
        "true_LFC": y[test_idx],
        "baseline_pred": pred_test_base,
        "finetuned_pred": pred_test_ft,
        "baseline_pred_raw": pred_test_base_raw,
        "finetuned_pred_raw": pred_test_ft_raw,
    })
    export.to_csv(os.path.join(out_dir, "trace_test_predictions.csv"), index=False)

    print(json.dumps(summary, indent=2))
    print(f"Saved to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--lfc-col", default="Average")
    p.add_argument("--out", required=True)

    p.add_argument("--base-ckpts", nargs="+", required=True)
    p.add_argument("--ft-ckpts", nargs="+", required=True)

    p.add_argument("--window", type=int, default=192)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)

    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--rare-threshold", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
