#!/usr/bin/env python3
"""
mechanism_analysis.py

Combines three mechanism metrics and plot against plasmid fitness (LFC):
  1) Spurious transcription load (tx_total)
  2) Concatemerisation (%)
  3) TFBS burden (total TFBS sites per plasmid)

Also computes Spearman correlations and optionally a TF-class effect plot.

Typical use:
  python mechanism_analysis.py \
    --lfc Plasmid_LFC_Summary_with_Sequences.csv \
    --tx reads_out.tsv \
    --concat Concatemer_Percentage_Summary.csv \
    --tfbs FIMO_presence_matrix.csv \
    --outdir results/mechanisms \
    --make-tf-class
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# TF class mapping ---------------------

GLOBAL_TFS = {
    "arcA", "crp", "fis", "fnr", "fur",
    "hns", "ihf", "lrp", "ompR", "narL", "narP"
}

STRESS_TFS = {
    "oxyR", "soxS", "rpoE", "rpoH2", "rpoH3",
    "rpoN", "rpoS17", "rpoS18", "marR"
}

SIGMA70_TFS = {"rpoD15", "rpoD16", "rpoD17", "rpoD18", "rpoD19"}

TWO_COMPONENT_TFS = {"cpxR", "phoB", "phoB3", "torR", "ntrC", "narL", "narP", "ompR"}

METABOLIC_TFS = {
    "araC", "argR", "argR2", "cysB", "cytR", "deoR",
    "fadR", "fruR", "galR", "gcvA", "glpR", "iclR",
    "ilvY", "lacI", "malT", "metJ", "metJ3", "metR",
    "modE", "nagC", "pdhR", "purR", "rhaS", "trpR",
    "tyrR", "cynR"
}

TF_CLASS_ORDER = [
    "Global",
    "Stress/AltSigma",
    "Sigma70-like",
    "Metabolic",
    "Two-component",
    "Pathway-specific",
]


def classify_tf(tf: str) -> str:
    if tf in GLOBAL_TFS:
        return "Global"
    if tf in STRESS_TFS:
        return "Stress/AltSigma"
    if tf in SIGMA70_TFS:
        return "Sigma70-like"
    if tf in METABOLIC_TFS:
        return "Metabolic"
    if tf in TWO_COMPONENT_TFS:
        return "Two-component"
    return "Pathway-specific"

# IO + validation helpers-------------------

def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    # TSV for tx; otherwise csv
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}\nAvailable: {list(df.columns)}")


def _normalise_plasmid(df: pd.DataFrame, plasmid_col: str) -> pd.DataFrame:
    """Strip whitespace; enforce string key for robust joins."""
    out = df.copy()
    out[plasmid_col] = out[plasmid_col].astype(str).str.strip()
    return out

def compute_tfbs_total(tfbs_matrix: pd.DataFrame, plasmid_col: str = "plasmid") -> pd.DataFrame:
    """
    From a presence matrix (plasmid + TF columns), compute:
      - tfbs_total
      - n_unique_tfs
    """
    _require_cols(tfbs_matrix, [plasmid_col], "TFBS matrix")
    tf_cols = [c for c in tfbs_matrix.columns if c != plasmid_col]
    if not tf_cols:
        raise ValueError("TFBS matrix has no TF columns (only plasmid column?).")

    df = tfbs_matrix.copy()
    df[tf_cols] = df[tf_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    out = df[[plasmid_col]].copy()
    out["tfbs_total"] = df[tf_cols].sum(axis=1)
    out["n_unique_tfs"] = (df[tf_cols] > 0).sum(axis=1)
    return out

# Plot style helpers-------------------

def add_lfc_bands(ax: plt.Axes) -> None:
    ax.set_ylim(-4.25, 1.0)
    ax.axhspan(-4.25, -3.0, facecolor="#c7c7c7", alpha=0.9, zorder=0)
    ax.axhspan(-3.0, -2.0, facecolor="#d8d8d8", alpha=0.8, zorder=0)
    ax.axhspan(-2.0, -0.5, facecolor="#e8e8e8", alpha=0.7, zorder=0)
    ax.axhspan(-0.5, 1.0, facecolor="#ffffff", alpha=1.0, zorder=0)


def style_axes(ax: plt.Axes, grid_linestyle: str = "--") -> None:
    ax.grid(True, alpha=0.4, linewidth=0.5, linestyle=grid_linestyle)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.tick_params(direction="out", length=4, width=0.8)


def add_stats_box(ax: plt.Axes, rho: float, p: float) -> None:
    txt = f"Spearman ρ = {rho:.2f}\np = {p:.2g}"
    ax.text(
        0.96, 0.96,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.4),
    )

# Main plots --------------------

def plot_mechanism_panel(
    df_tx: pd.DataFrame,
    df_concat: pd.DataFrame,
    df_tfbs: pd.DataFrame,
    lfc_mean_col: str,
    lfc_std_col: str,
    out_png: Path,
    out_pdf: Path,
) -> pd.DataFrame:
    """
    Make the 1×3 panel and return Spearman stats table.
    """
    marker_color = "#4A90E2"
    marker_edge = "#1f3b4d"

    # Spearman stats
    rho_tx, p_tx = spearmanr(df_tx["tx_total"], df_tx[lfc_mean_col])
    rho_concat, p_concat = spearmanr(df_concat["concat_mean"], df_concat[lfc_mean_col])
    rho_tfbs, p_tfbs = spearmanr(df_tfbs["tfbs_total"], df_tfbs[lfc_mean_col])

    stats = pd.DataFrame([
        {"metric": "tx_total", "spearman_rho": rho_tx, "p_value": p_tx, "n": len(df_tx)},
        {"metric": "concat_mean", "spearman_rho": rho_concat, "p_value": p_concat, "n": len(df_concat)},
        {"metric": "tfbs_total", "spearman_rho": rho_tfbs, "p_value": p_tfbs, "n": len(df_tfbs)},
    ])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300, sharey=True)

    # Panel 1: TX
    ax = axes[0]
    add_lfc_bands(ax)
    ax.errorbar(
        df_tx["tx_total"],
        df_tx[lfc_mean_col],
        yerr=df_tx[lfc_std_col],
        fmt="o",
        markersize=6,
        markerfacecolor=marker_color,
        markeredgecolor=marker_edge,
        markeredgewidth=0.7,
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        alpha=0.9,
        linestyle="none",
        zorder=2,
    )
    ax.set_xlabel("Total transcriptional read count", fontsize=12)
    ax.set_ylabel("Log$_2$ Fold Change (LFC)", fontsize=12)
    ax.set_title("LFC vs Spurious Transcription", fontsize=14)
    style_axes(ax)
    add_stats_box(ax, rho_tx, p_tx)

    # Panel 2: concat
    ax = axes[1]
    add_lfc_bands(ax)
    ax.errorbar(
        df_concat["concat_mean"],
        df_concat[lfc_mean_col],
        xerr=df_concat["concat_std"],
        yerr=df_concat[lfc_std_col],
        fmt="o",
        markersize=6,
        markerfacecolor=marker_color,
        markeredgecolor=marker_edge,
        markeredgewidth=0.7,
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        alpha=0.9,
        linestyle="none",
        zorder=2,
    )
    ax.set_xlabel("Concatemer percentage (%)", fontsize=12)
    ax.set_title("LFC vs Concatemer Percentage", fontsize=14)
    style_axes(ax)
    add_stats_box(ax, rho_concat, p_concat)

    # Panel 3: TFBS
    ax = axes[2]
    add_lfc_bands(ax)
    ax.errorbar(
        df_tfbs["tfbs_total"],
        df_tfbs[lfc_mean_col],
        yerr=df_tfbs[lfc_std_col],
        fmt="o",
        markersize=6,
        markerfacecolor=marker_color,
        markeredgecolor=marker_edge,
        markeredgewidth=0.7,
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        alpha=0.9,
        linestyle="none",
        zorder=2,
    )
    ax.set_xlabel("Total TFBS sites per plasmid", fontsize=12)
    ax.set_title("LFC vs TFBS homology", fontsize=14)
    style_axes(ax)
    add_stats_box(ax, rho_tfbs, p_tfbs)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return stats


def plot_tf_class_effect(
    tfbs_matrix: pd.DataFrame,
    lfc: pd.DataFrame,
    plasmid_col: str,
    lfc_mean_col: str,
    out_png: Path,
    out_pdf: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expand plasmid–TF pairs where TF count > 0, assign TF classes, and plot box+points per class.
    Returns:
      - long_df
      - summary_stats
    """
    _require_cols(tfbs_matrix, [plasmid_col], "TFBS matrix")
    _require_cols(lfc, [plasmid_col, lfc_mean_col], "LFC table")

    df = tfbs_matrix.merge(lfc[[plasmid_col, lfc_mean_col]], on=plasmid_col, how="inner").copy()
    tf_cols = [c for c in df.columns if c not in {plasmid_col, lfc_mean_col, "tfbs_total", "n_unique_tfs"}]
    if not tf_cols:
        raise ValueError("No TF columns found for TF-class analysis (matrix looks totals-only).")

    long = df.melt(
        id_vars=[plasmid_col, lfc_mean_col],
        value_vars=tf_cols,
        var_name="TF",
        value_name="count",
    )
    long["count"] = pd.to_numeric(long["count"], errors="coerce").fillna(0)
    long = long[long["count"] > 0].copy()
    long["TF_class"] = long["TF"].apply(classify_tf)

    summary = (
        long.groupby("TF_class")[lfc_mean_col]
            .agg(count="count", mean="mean", std="std", median="median")
            .reindex(TF_CLASS_ORDER)
            .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    add_lfc_bands(ax)

    data = [long.loc[long["TF_class"] == c, lfc_mean_col].values for c in TF_CLASS_ORDER]
    positions = np.arange(len(TF_CLASS_ORDER))

    ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#4A90E2", alpha=0.35, edgecolor="black"),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    rng = np.random.default_rng(0)  # deterministic jitter
    for i, c in enumerate(TF_CLASS_ORDER):
        y = long.loc[long["TF_class"] == c, lfc_mean_col].values
        x = rng.normal(loc=i, scale=0.07, size=len(y))
        ax.scatter(x, y, s=12, alpha=0.55, color="black", linewidths=0, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(TF_CLASS_ORDER, rotation=45, ha="right")
    ax.set_xlabel("TF binding site category", fontsize=12)
    ax.set_ylabel("Log$_2$ Fold Change (LFC)", fontsize=12)
    ax.set_title("Impact of Transcription Factor Class on Plasmid Fitness", fontsize=14)
    style_axes(ax, grid_linestyle="-")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return long, summary

# CLI ------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mechanism proxy analysis + plotting.")
    p.add_argument("--lfc", required=True, type=Path, help="LFC CSV path.")
    p.add_argument("--tx", required=True, type=Path, help="TX table path (TSV/CSV).")
    p.add_argument("--concat", required=True, type=Path, help="Concatemer summary CSV path.")
    p.add_argument("--tfbs", required=True, type=Path, help="TFBS presence matrix CSV path.")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory.")

    p.add_argument("--lfc-mean-col", default="Average_LFC_DH5a", help="Mean LFC column.")
    p.add_argument("--lfc-std-col", default="LFC_StdDev_DH5a", help="LFC std column.")

    p.add_argument("--tx-left-col", default="tx_left", help="TX left reads column.")
    p.add_argument("--tx-right-col", default="tx_right", help="TX right reads column.")
    p.add_argument("--tx-total-col", default=None, help="If provided, use this instead of left+right.")

    p.add_argument("--concat-mean-col", default="Average_End_Concatemer_%", help="Concatemer mean column.")
    p.add_argument("--concat-std-col", default="Std_End_Concatemer_%", help="Concatemer std column.")

    p.add_argument("--plasmid-col-lfc", default="plasmid", help="Plasmid column name in LFC table.")
    p.add_argument("--plasmid-col-tx", default="plasmid", help="Plasmid column name in TX table.")
    p.add_argument("--plasmid-col-concat", default="plasmid", help="Plasmid column name in concat table.")
    p.add_argument("--plasmid-col-tfbs", default="plasmid", help="Plasmid column name in TFBS matrix.")

    p.add_argument("--make-tf-class", action="store_true", help="Also generate TF-class plot + summary.")
    p.add_argument("--prefix", type=str, default="mechanisms", help="Prefix for output filenames.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    lfc = _read_table(args.lfc)
    tx = _read_table(args.tx)
    concat = _read_table(args.concat)
    tfbs = _read_table(args.tfbs)

    # Reviewer-friendly explicit renaming (common in your files)
    if "Plasmid" in lfc.columns and args.plasmid_col_lfc == "plasmid":
        lfc = lfc.rename(columns={"Plasmid": "plasmid"})
    if "Plasmid" in tfbs.columns and args.plasmid_col_tfbs == "plasmid":
        tfbs = tfbs.rename(columns={"Plasmid": "plasmid"})
    if "region" in concat.columns and args.plasmid_col_concat == "plasmid":
        concat = concat.rename(columns={"region": "plasmid"})

    lfc = _normalise_plasmid(lfc, args.plasmid_col_lfc)
    tx = _normalise_plasmid(tx, args.plasmid_col_tx)
    concat = _normalise_plasmid(concat, args.plasmid_col_concat)
    tfbs = _normalise_plasmid(tfbs, args.plasmid_col_tfbs)

    _require_cols(lfc, [args.plasmid_col_lfc, args.lfc_mean_col, args.lfc_std_col], "LFC table")
    _require_cols(concat, [args.plasmid_col_concat, args.concat_mean_col, args.concat_std_col], "Concat table")

    # TX total
    if args.tx_total_col is not None:
        _require_cols(tx, [args.plasmid_col_tx, args.tx_total_col], "TX table")
        tx_total = tx[[args.plasmid_col_tx, args.tx_total_col]].copy()
        tx_total = tx_total.rename(columns={args.tx_total_col: "tx_total"})
    else:
        _require_cols(tx, [args.plasmid_col_tx, args.tx_left_col, args.tx_right_col], "TX table")
        tx_total = tx[[args.plasmid_col_tx, args.tx_left_col, args.tx_right_col]].copy()
        tx_total[args.tx_left_col] = pd.to_numeric(tx_total[args.tx_left_col], errors="coerce").fillna(0)
        tx_total[args.tx_right_col] = pd.to_numeric(tx_total[args.tx_right_col], errors="coerce").fillna(0)
        tx_total["tx_total"] = tx_total[args.tx_left_col] + tx_total[args.tx_right_col]
        tx_total = tx_total[[args.plasmid_col_tx, "tx_total"]]

    tx_total = tx_total.rename(columns={args.plasmid_col_tx: "plasmid"})
    lfc_keyed = lfc.rename(columns={args.plasmid_col_lfc: "plasmid"}).copy()

    # Concat tidy
    concat_tidy = concat[[args.plasmid_col_concat, args.concat_mean_col, args.concat_std_col]].copy()
    concat_tidy = concat_tidy.rename(columns={
        args.plasmid_col_concat: "plasmid",
        args.concat_mean_col: "concat_mean",
        args.concat_std_col: "concat_std",
    })

    # TFBS total tidy
    tfbs_total = compute_tfbs_total(tfbs, plasmid_col=args.plasmid_col_tfbs).rename(
        columns={args.plasmid_col_tfbs: "plasmid"}
    )

    df_tx = lfc_keyed.merge(tx_total, on="plasmid", how="inner")
    df_concat = lfc_keyed.merge(concat_tidy, on="plasmid", how="inner")
    df_tfbs = lfc_keyed.merge(tfbs_total, on="plasmid", how="inner")

    fig_png = args.outdir / f"{args.prefix}_figure_mechanisms_1x3.png"
    fig_pdf = args.outdir / f"{args.prefix}_figure_mechanisms_1x3.pdf"
    stats = plot_mechanism_panel(
        df_tx=df_tx,
        df_concat=df_concat,
        df_tfbs=df_tfbs,
        lfc_mean_col=args.lfc_mean_col,
        lfc_std_col=args.lfc_std_col,
        out_png=fig_png,
        out_pdf=fig_pdf,
    )

    stats_path = args.outdir / f"{args.prefix}_spearman_stats.csv"
    stats.to_csv(stats_path, index=False)

    print(f"[OK] Wrote: {fig_png}")
    print(f"[OK] Wrote: {fig_pdf}")
    print(f"[OK] Wrote: {stats_path}")

    if args.make_tf_class:
        # Ensure plasmid column is literally "plasmid" for class plot
        tfbs_matrix = tfbs.copy()
        if args.plasmid_col_tfbs != "plasmid":
            tfbs_matrix = tfbs_matrix.rename(columns={args.plasmid_col_tfbs: "plasmid"})
        lfc_for_class = lfc_keyed[["plasmid", args.lfc_mean_col]].copy()

        tfclass_png = args.outdir / f"{args.prefix}_figure_tf_class.png"
        tfclass_pdf = args.outdir / f"{args.prefix}_figure_tf_class.pdf"
        long_df, summary = plot_tf_class_effect(
            tfbs_matrix=tfbs_matrix,
            lfc=lfc_for_class,
            plasmid_col="plasmid",
            lfc_mean_col=args.lfc_mean_col,
            out_png=tfclass_png,
            out_pdf=tfclass_pdf,
        )

        long_path = args.outdir / f"{args.prefix}_tf_class_long_table.csv"
        summary_path = args.outdir / f"{args.prefix}_tf_class_summary_stats.csv"
        long_df.to_csv(long_path, index=False)
        summary.to_csv(summary_path, index=False)

        print(f"[OK] Wrote: {tfclass_png}")
        print(f"[OK] Wrote: {tfclass_pdf}")
        print(f"[OK] Wrote: {long_path}")
        print(f"[OK] Wrote: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
