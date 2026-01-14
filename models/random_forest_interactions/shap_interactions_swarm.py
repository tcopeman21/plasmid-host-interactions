#!/usr/bin/env python3
"""
Plot present-only SHAP swarm plots for:
  1) main effects (one-hot features)
  2) true 2-way interaction features (between different slots)

Consumes:
  - rf_pipeline.joblib
  - assembly CSV (to rebuild X)
  - main_shap_per_feature.csv (SHAP values per poly feature)

This script is plotting-only: it does not recompute SHAP.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


SLOTS = ["Position 1", "Position 2", "Position 3", "Position 4"]


def make_unique_labels(labels: list[str]) -> list[str]:
    counts = Counter()
    out = []
    for lab in labels:
        counts[lab] += 1
        out.append(f"{lab} ({counts[lab]})" if counts[lab] > 1 else lab)
    return out


def plot_present_only_swarm(
    shap_vals: np.ndarray,
    X_bin: np.ndarray,
    labels: list[str],
    out_path: Path,
    xlabel: str,
    title: str,
    figsize=(12, 14),
    left_adjust=0.35,
) -> None:
    labels = make_unique_labels(labels)

    all_shap = []
    all_feat = []
    for i, lab in enumerate(labels):
        mask = X_bin[:, i] == 1
        vals = shap_vals[mask, i]
        if vals.size == 0:
            # keep the category present for ordering (one NaN)
            all_shap.append(np.nan)
            all_feat.append(lab)
        else:
            all_shap.extend(vals.tolist())
            all_feat.extend([lab] * len(vals))

    dfp = pd.DataFrame({"SHAP value": all_shap, "Feature": all_feat})
    dfp["Feature"] = pd.Categorical(dfp["Feature"], categories=labels, ordered=True)

    plt.figure(figsize=figsize, dpi=300)
    sns.swarmplot(
        data=dfp,
        y="Feature",
        x="SHAP value",
        order=labels,
        size=5,
        alpha=0.6,
        orient="h",
    )

    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.title(title)
    plt.grid(axis="x", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.subplots_adjust(left=left_adjust)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--assembly-csv", required=True)
    ap.add_argument("--shap-feature-csv", required=True, help="main_shap_per_feature.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=20, help="Top N interaction terms by mean |SHAP|")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(args.pipeline)
    ct = pipe.named_steps["columntransformer"]
    poly = pipe.named_steps["polynomialfeatures"]

    # Build feature names
    ohe = ct.named_transformers_["onehot"]
    feat_ohe = ohe.get_feature_names_out(SLOTS)
    feat_poly = poly.get_feature_names_out(feat_ohe)

    # Load SHAP values
    shap_df = pd.read_csv(args.shap_feature_csv)
    shap_vals = shap_df.drop(columns=["Sample"], errors="ignore").values
    if shap_vals.shape[1] != len(feat_poly):
        raise ValueError(
            f"SHAP feature count mismatch: shap has {shap_vals.shape[1]} columns, "
            f"but poly features are {len(feat_poly)}."
        )

    # Rebuild X_poly (binary presence matrix for swarm filtering)
    asm = pd.read_csv(args.assembly_csv)
    X_raw = asm[SLOTS]
    X_ohe = ct.transform(X_raw)
    if hasattr(X_ohe, "toarray"):
        X_ohe = X_ohe.toarray()
    X_poly = poly.transform(X_ohe)

    # Identify main effects (exactly the OHE features)
    main_idx = [i for i, f in enumerate(feat_poly) if f in set(feat_ohe)]
    main_names = [feat_poly[i] for i in main_idx]

    # Identify true 2-way interactions: "A B" where A and B are different slots
    interaction_pattern = re.compile(r"^(Position \d+_[^\s]+) (Position \d+_[^\s]+)$")
    inter_idx = []
    inter_names = []
    for i, f in enumerate(feat_poly):
        m = interaction_pattern.match(f)
        if not m:
            continue
        a, b = m.groups()
        if a.split("_")[0] != b.split("_")[0]:
            inter_idx.append(i)
            inter_names.append(f)

    if not inter_idx:
        raise RuntimeError("No interaction features detected. Check feature naming / pipeline.")

    # Pick top-N interactions by mean absolute SHAP
    mean_abs = np.abs(shap_vals[:, inter_idx]).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][: args.top_n]
    top_inter_idx = [inter_idx[i] for i in order]
    top_inter_names = [inter_names[i] for i in order]

    # Plot swarms
    plot_present_only_swarm(
        shap_vals[:, main_idx],
        np.asarray(X_poly)[:, main_idx],
        labels=main_names,
        out_path=out_dir / "swarm_main_effects.png",
        xlabel="SHAP value",
        title="Main effects (parts)",
        figsize=(12, 16),
        left_adjust=0.40,
    )

    plot_present_only_swarm(
        shap_vals[:, top_inter_idx],
        np.asarray(X_poly)[:, top_inter_idx],
        labels=top_inter_names,
        out_path=out_dir / "swarm_top_interactions.png",
        xlabel="SHAP value",
        title=f"Top {args.top_n} two-part interactions",
        figsize=(12, 16),
        left_adjust=0.40,
    )

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
