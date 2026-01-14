from __future__ import annotations

import argparse
import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from models.unet_sequence_model.data import load_pkl_sequences
from models.unet_sequence_model.model import GCNUNet


@torch.no_grad()
def predict_indices(model: torch.nn.Module, X: torch.Tensor, idx: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        bidx = idx[i:i + batch]
        xb = X[bidx].to(device)
        risk = model(xb)
        out.append(risk.mean(dim=1).detach().cpu().numpy())
    return np.concatenate(out)


def run(pkl_path: str, ckpt_dir: str, out_dir: str, k_folds: int, seed: int, batch: int, max_points: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    X_np, y = load_pkl_sequences(pkl_path)
    X = torch.from_numpy(X_np)
    N = len(y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = np.full(N, np.nan, dtype=np.float32)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for fold, (_, te_idx) in enumerate(kf.split(range(N)), start=1):
        ckpt = os.path.join(ckpt_dir, f"fold{fold}_unet.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        model = GCNUNet().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        te_idx = np.asarray(te_idx)
        preds[te_idx] = predict_indices(model, X, te_idx, device=device, batch=batch)

    mask = np.isfinite(preds)
    y_true = y[mask]
    y_pred = preds[mask]

    r_p = pearsonr(y_pred, y_true)[0]
    r_s = spearmanr(y_pred, y_true)[0]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    tox_thresh = np.percentile(y_true, 25)
    y_tox = y_true <= tox_thresh
    auc = roc_auc_score(y_tox, -y_pred)

    metrics = {
        "pearson_r": float(r_p),
        "spearman_rho": float(r_s),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "toxicity_auc_bottom25": float(auc),
        "n": int(mask.sum()),
    }

    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot
    if len(y_pred) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_pred), size=max_points, replace=False)
        x_plot = y_pred[idx]
        y_plot = y_true[idx]
    else:
        x_plot, y_plot = y_pred, y_true

    plt.figure(figsize=(5.2, 5.2), dpi=220)
    plt.scatter(x_plot, y_plot, s=9, alpha=0.35)
    plt.xlabel("Predicted LFC")
    plt.ylabel("Measured LFC")
    plt.title(f"UNet CV hold-out\nPearson r={r_p:.3f}, Spearman ρ={r_s:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "predict_vs_test.png"), bbox_inches="tight")
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--max-points", type=int, default=5000)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.pkl, a.ckpt_dir, a.out, a.k, a.seed, a.batch, a.max_points)
