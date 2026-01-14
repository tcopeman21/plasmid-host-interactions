"""
Pairwise ranking ablation for plasmid fitness mechanisms.

What it does
- Loads a per-plasmid dataset with LFC and burden features.
- Optionally derives TF-binding category totals (Global/Stress/Local/Other) from per-TF columns.
- Trains a simple pairwise logistic ranking model on random pairs.
- Evaluates Spearman correlation between ranking score and LFC.
- Repeats across seeds for stability; reports mean ± SD Spearman.
- Saves results table + publication-ready figures.

Expected input columns (minimum)
- Average_LFC_DH5a
- Total_TX_Sites
- Total_TFBS_Sites (required if you want TFBS totals-only ablations or to compute "Other" category)
- DH5a_Average_Concatemer_Percentage
Optional
- Plasmid (identifier; used only for bookkeeping)
- Per-TF binding count columns (numeric), e.g. "IHF_xxx", "Fis_xxx", ... used for TF categories

Usage example
python run_pairwise_ranking_ablation.py \
  --data merged_plasmid_TFBS_LFC_dataset_with_totals.csv \
  --outdir outputs/pairwise_ablation \
  --mode tf_categories \
  --n_pairs 60000 \
  --repeats 20 \
  --base_seed 42 \
  --make_scatter_grid
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr


# -----------------------------
# Regulatory category sets
# -----------------------------
GLOBAL_TFS = {
    "IHF", "Fis", "H-NS", "CRP", "FNR", "LRP",
    "rpoD15", "rpoD16", "rpoD17", "rpoD18", "rpoD19",
    "rpoE", "rpoH2", "rpoH3", "rpoN", "rpoS17", "rpoS18"
}
STRESS_TFS = {"SoxS", "OxyR", "MarA", "LexA", "CpxR"}
LOCAL_TFS = {"AraC", "ArgR", "GalR", "TrpR", "NagC", "MalT", "PurR",
             "PhoB", "OmpR", "CynR", "PdhR", "TyrR", "GcvA", "DeoR"}


# -----------------------------
# Core helpers
# -----------------------------
def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _infer_tf_columns(df: pd.DataFrame, reserved: List[str]) -> List[str]:
    """
    Infer per-TF binding columns: numeric columns not in reserved list.
    This matches your prior logic but makes it explicit.
    """
    tf_cols = []
    for c in df.columns:
        if c in reserved:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            tf_cols.append(c)
    return tf_cols


def add_tf_category_features(
    df: pd.DataFrame,
    total_tfbs_col: str = "Total_TFBS_Sites",
    reserved: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Adds:
      - Global_TF_Binding_Sites
      - Stress_TF_Binding_Sites
      - Local_TF_Binding_Sites
      - Other_TF_Binding_Sites = Total_TFBS_Sites - (Global+Stress+Local)

    Assumes per-TF columns are named like "<TF>_..." so splitting at "_" gets TF.
    """
    if reserved is None:
        reserved = [
            "Plasmid", "Average_LFC_DH5a",
            "Total_TX_Sites", "DH5a_Average_Concatemer_Percentage",
            "sequence_length", "GC_percent", total_tfbs_col
        ]

    _require_columns(df, [total_tfbs_col])

    tf_cols = _infer_tf_columns(df, reserved=reserved)

    def cat_sum(tf_set: set) -> pd.Series:
        cols = [c for c in tf_cols if str(c).split("_")[0] in tf_set]
        if not cols:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
        return df[cols].sum(axis=1).astype(float)

    df = df.copy()
    df["Global_TF_Binding_Sites"] = cat_sum(GLOBAL_TFS)
    df["Stress_TF_Binding_Sites"] = cat_sum(STRESS_TFS)
    df["Local_TF_Binding_Sites"] = cat_sum(LOCAL_TFS)

    df["Other_TF_Binding_Sites"] = (
        df[total_tfbs_col].astype(float)
        - (df["Global_TF_Binding_Sites"] + df["Stress_TF_Binding_Sites"] + df["Local_TF_Binding_Sites"])
    )

    # Guard against negative due to inconsistent totals (can happen with naming mismatches)
    df["Other_TF_Binding_Sites"] = df["Other_TF_Binding_Sites"].clip(lower=0.0)

    return df


def pairwise_rank_spearman(
    X: np.ndarray,
    y: np.ndarray,
    n_pairs: int,
    seed: int,
    max_iter: int = 2000
) -> Tuple[float, np.ndarray]:
    """
    Trains logistic regression on pairwise differences and returns:
      - Spearman rho between scores and y
      - scores per sample
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)

    mask = y[i] != y[j]
    i, j = i[mask], j[mask]
    if len(i) < 10:
        raise ValueError("Too few non-tied pairs. Check y distribution or reduce filtering.")

    Z = X[i] - X[j]
    labels = (y[i] > y[j]).astype(int)

    model = LogisticRegression(max_iter=max_iter)
    model.fit(Z, labels)

    scores = X @ model.coef_.ravel()
    rho, _ = spearmanr(scores, y)
    return float(rho), scores


@dataclass
class AblationResult:
    model: str
    features: List[str]
    rho_mean: float
    rho_sd: float


def run_ablations(
    df: pd.DataFrame,
    ablations: Dict[str, List[str]],
    y_col: str,
    n_pairs: int,
    repeats: int,
    base_seed: int
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """
    Returns:
      - results dataframe sorted by rho_mean
      - per-model detail: rhos + last-run scores (for optional scatter grid)
    """
    y = df[y_col].values.astype(float)

    results: List[AblationResult] = []
    detail: Dict[str, Dict[str, np.ndarray]] = {}

    for name, feats in ablations.items():
        X = df[feats].values.astype(float)

        rhos = []
        scores_last = None

        for k in range(repeats):
            seed = base_seed + k
            rho, scores = pairwise_rank_spearman(X, y, n_pairs=n_pairs, seed=seed)
            rhos.append(rho)
            scores_last = scores

        rhos_arr = np.array(rhos, dtype=float)
        results.append(
            AblationResult(
                model=name,
                features=feats,
                rho_mean=float(np.nanmean(rhos_arr)),
                rho_sd=float(np.nanstd(rhos_arr, ddof=1) if repeats > 1 else 0.0),
            )
        )

        detail[name] = {
            "rhos": rhos_arr,
            "scores": scores_last if scores_last is not None else np.full_like(y, np.nan),
        }

    res_df = pd.DataFrame([{
        "Model": r.model,
        "Features": ", ".join(r.features),
        "Spearman_mean": r.rho_mean,
        "Spearman_sd": r.rho_sd,
        "Repeats": repeats,
        "Pairs_per_repeat": n_pairs
    } for r in results])

    res_df = res_df.sort_values("Spearman_mean", ascending=True).reset_index(drop=True)
    return res_df, detail


# -----------------------------
# Plotting
# -----------------------------
def save_barplot(
    res_df: pd.DataFrame,
    out_pdf: str,
    out_png: str,
    title: str,
    color: str = "#6CA6CD"
) -> None:
    """
    Horizontal bar plot with mean Spearman and error bars (SD across repeats).
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    y_labels = res_df["Model"].tolist()
    x = res_df["Spearman_mean"].values
    xerr = res_df["Spearman_sd"].values

    ax.barh(
        y=np.arange(len(y_labels)),
        width=x,
        xerr=xerr,
        color=color,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.95
    )

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)

    ax.set_xlabel("Spearman rank correlation", fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)

    # Outline box
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="-", linewidth=0.8, alpha=0.25)
    ax.grid(axis="y", linestyle="")

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_scatter_grid(
    df: pd.DataFrame,
    y_col: str,
    detail: Dict[str, Dict[str, np.ndarray]],
    out_pdf: str,
    out_png: str,
    title: str,
    color: str = "#6CA6CD"
) -> None:
    """
    Grid of scatter plots: ranking score vs LFC for each ablation (last repeat).
    """
    y = df[y_col].values.astype(float)

    labels = list(detail.keys())
    n_models = len(labels)
    n_cols = 2
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        dpi=300,
        squeeze=False
    )
    axes = axes.flatten()

    for ax, name in zip(axes, labels):
        scores = detail[name]["scores"]
        rho, pval = spearmanr(scores, y)

        p_display = "p < 0.001" if (pval is not None and pval < 1e-3) else f"p = {pval:.3f}"

        ax.set_axisbelow(True)
        ax.grid(alpha=0.25, linestyle="-", linewidth=0.7)

        ax.scatter(scores, y, s=10, alpha=0.6, color=color)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Pairwise ranking score", fontsize=10)
        ax.set_ylabel("Average log2 fold change (LFC)", fontsize=10)

        ax.text(
            0.05, 0.95,
            f"ρ = {rho:.2f}\n{p_display}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="black",
                alpha=0.75,
            ),
        )

    # Delete unused axes
    for k in range(n_models, len(axes)):
        fig.delaxes(axes[k])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Ablation definitions
# -----------------------------
def build_ablations(mode: str) -> Dict[str, List[str]]:
    """
    mode:
      - "tf_categories": uses derived Global/Stress/Local/Other TF binding site features
      - "tf_total": uses Total_TFBS_Sites only
    """
    TX = ["Total_TX_Sites"]
    CONCAT = ["DH5a_Average_Concatemer_Percentage"]

    if mode == "tf_categories":
        TF_CAT = [
            "Global_TF_Binding_Sites",
            "Stress_TF_Binding_Sites",
            "Local_TF_Binding_Sites",
            "Other_TF_Binding_Sites",
        ]

        return {
            "Transcription only": TX,
            "Global TF binding only": ["Global_TF_Binding_Sites"],
            "Local TF binding only": ["Local_TF_Binding_Sites"],
            "Stress TF binding only": ["Stress_TF_Binding_Sites"],
            "Other TF binding only": ["Other_TF_Binding_Sites"],
            "All TF binding categories": TF_CAT,
            "Concatemer percentage only": CONCAT,
            "Transcription + Global TF binding": TX + ["Global_TF_Binding_Sites"],
            "Transcription + TF binding categories": TX + TF_CAT,
            "Transcription + Concatemer percentage": TX + CONCAT,
            "TF binding categories + Concatemer percentage": TF_CAT + CONCAT,
            "All mechanisms": TX + TF_CAT + CONCAT,
        }

    if mode == "tf_total":
        TFBS = ["Total_TFBS_Sites"]
        return {
            "Transcription only": TX,
            "TFBS count only": TFBS,
            "Concatemer percentage only": CONCAT,
            "Transcription + TFBS": TX + TFBS,
            "Transcription + Concatemer": TX + CONCAT,
            "TFBS + Concatemer": TFBS + CONCAT,
            "All mechanisms": TX + TFBS + CONCAT,
        }

    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Input CSV path")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--mode", choices=["tf_categories", "tf_total"], default="tf_categories")
    p.add_argument("--y_col", default="Average_LFC_DH5a")
    p.add_argument("--n_pairs", type=int, default=60000)
    p.add_argument("--repeats", type=int, default=20, help="Number of repeats (different seeds)")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--make_scatter_grid", action="store_true", help="Save score-vs-LFC scatter grid")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)

    # Validate common required columns
    _require_columns(df, [args.y_col, "Total_TX_Sites", "DH5a_Average_Concatemer_Percentage"])

    # Mode-specific preparation
    if args.mode == "tf_categories":
        # Need totals for "Other" category at least
        _require_columns(df, ["Total_TFBS_Sites"])
        df = add_tf_category_features(df, total_tfbs_col="Total_TFBS_Sites")
    else:
        _require_columns(df, ["Total_TFBS_Sites"])

    ablations = build_ablations(args.mode)

    # Validate all feature columns exist
    needed_feats = sorted({f for feats in ablations.values() for f in feats})
    _require_columns(df, needed_feats)

    res_df, detail = run_ablations(
        df=df,
        ablations=ablations,
        y_col=args.y_col,
        n_pairs=args.n_pairs,
        repeats=args.repeats,
        base_seed=args.base_seed
    )

    # Save table
    out_csv = os.path.join(args.outdir, f"pairwise_ablation_results_{args.mode}.csv")
    res_df.to_csv(out_csv, index=False)

    # Figures
    title = "Pairwise ranking ablation of plasmid burden mechanisms"
    out_bar_pdf = os.path.join(args.outdir, f"pairwise_ablation_bar_{args.mode}.pdf")
    out_bar_png = os.path.join(args.outdir, f"pairwise_ablation_bar_{args.mode}.png")
    save_barplot(res_df, out_pdf=out_bar_pdf, out_png=out_bar_png, title=title)

    if args.make_scatter_grid:
        out_sc_pdf = os.path.join(args.outdir, f"pairwise_ablation_scatter_{args.mode}.pdf")
        out_sc_png = os.path.join(args.outdir, f"pairwise_ablation_scatter_{args.mode}.png")
        save_scatter_grid(
            df=df,
            y_col=args.y_col,
            detail=detail,
            out_pdf=out_sc_pdf,
            out_png=out_sc_png,
            title="Pairwise ranking ablation: score vs fitness"
        )

    # Minimal console output (reviewer-safe)
    print("Wrote:", out_csv)
    print("Wrote:", out_bar_pdf)
    if args.make_scatter_grid:
        print("Wrote:", out_sc_pdf)


if __name__ == "__main__":
    main()
