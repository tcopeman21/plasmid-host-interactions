"""
Evaluate a saved RF pipeline on a dataset CSV and write figures/metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


SLOTS = ["Position 1", "Position 2", "Position 3", "Position 4"]


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(5, 4), dpi=300)
    plt.scatter(y_pred, y_true, alpha=0.6, s=18)
    plt.xlabel("Predicted LFC")
    plt.ylabel("Measured LFC")
    plt.title("Predicted vs measured")
    plt.text(0.05, 0.95, f"R² = {r2:.3f}", transform=plt.gca().transAxes, va="top")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

    # Q-Q
    stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q plot")

    # Residuals vs predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=14)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs predicted")

    # Histogram
    axes[2].hist(residuals, bins=25, alpha=0.8)
    axes[2].set_title("Residual histogram")
    axes[2].set_xlabel("Residual")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True, help="rf_pipeline.joblib")
    ap.add_argument("--target", default="Average")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    X = df[SLOTS].copy()
    y = df[args.target].astype(float).values

    model = joblib.load(args.model)
    y_pred = model.predict(X)

    metrics = {
        "mse": float(mean_squared_error(y, y_pred)),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
        "n": int(len(y)),
    }
    (out_dir / "metrics_full_dataset.json").write_text(json.dumps(metrics, indent=2))

    plot_pred_vs_true(y, y_pred, out_dir / "pred_vs_true.png")
    plot_residual_diagnostics(y, y_pred, out_dir / "residual_diagnostics.png")

    # Save predictions table
    pd.DataFrame({"y_true": y, "y_pred": y_pred}).to_csv(out_dir / "predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
