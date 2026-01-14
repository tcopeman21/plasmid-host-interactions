"""
compute_tfbs_presence.py

Convert MEME-suite FIMO output (TSV) into a plasmid-by-TF presence/count matrix,
plus summary tables.

Typical FIMO TSV columns include:
  motif_id, motif_alt_id, sequence_name, start, stop, strand, score, p-value, q-value, matched_sequence

This script:
  - groups hits by (plasmid, TF) and counts sites
  - produces a wide CSV: plasmid rows, TF columns (counts), plus a tfbs_total
  - produces summary tables per plasmid and per TF

Example:
  python compute_tfbs_presence.py \
    --fimo /path/to/fimo.tsv \
    --outdir /path/to/out_tfbs \
    --plasmid-col sequence_name \
    --tf-col motif_alt_id \
    --qvalue 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_FIMO_COLS = {
    "motif_id": "motif_id",
    "motif_alt_id": "motif_alt_id",
    "sequence_name": "sequence_name",
    "p-value": "p-value",
    "q-value": "q-value",
}


def _read_fimo(fimo_path: Path) -> pd.DataFrame:
    # FIMO TSV is typically tab-separated with a header.
    # Comment lines may start with '#'.
    df = pd.read_csv(fimo_path, sep="\t", comment="#", dtype=str)
    if df.empty:
        raise ValueError(f"FIMO file is empty after parsing: {fimo_path}")
    return df


def _ensure_cols(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "FIMO TSV missing required columns: "
            + ", ".join(missing)
            + f"\nFound columns: {list(df.columns)}"
        )


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_presence_matrix(
    fimo_df: pd.DataFrame,
    plasmid_col: str,
    tf_col: str,
    qvalue_threshold: float | None = None,
    pvalue_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      presence_matrix: plasmid x TF counts + tfbs_total
      per_plasmid_summary
      per_tf_summary
    """
    _ensure_cols(fimo_df, [plasmid_col, tf_col])

    # Convert p/q columns to numeric if present (optional thresholds)
    fimo_df = _coerce_numeric(fimo_df, ["p-value", "q-value"])

    df = fimo_df.copy()

    if qvalue_threshold is not None:
        if "q-value" not in df.columns:
            raise KeyError("Requested --qvalue filter but FIMO TSV has no 'q-value' column.")
        df = df[df["q-value"].notna() & (df["q-value"] <= qvalue_threshold)].copy()

    if pvalue_threshold is not None:
        if "p-value" not in df.columns:
            raise KeyError("Requested --pvalue filter but FIMO TSV has no 'p-value' column.")
        df = df[df["p-value"].notna() & (df["p-value"] <= pvalue_threshold)].copy()

    # Normalise identifiers (reviewer friendliness)
    df[plasmid_col] = df[plasmid_col].astype(str).str.strip()
    df[tf_col] = df[tf_col].astype(str).str.strip()

    # Count sites per (plasmid, TF)
    counts = (
        df.groupby([plasmid_col, tf_col])
          .size()
          .reset_index(name="site_count")
    )

    # Wide matrix: plasmid rows, TF columns
    presence = counts.pivot_table(
        index=plasmid_col,
        columns=tf_col,
        values="site_count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    presence = presence.rename(columns={plasmid_col: "plasmid"})
    # Total sites per plasmid
    tf_cols = [c for c in presence.columns if c != "plasmid"]
    presence["tfbs_total"] = presence[tf_cols].sum(axis=1)
    presence["n_unique_tfs"] = (presence[tf_cols] > 0).sum(axis=1)

    # Per-plasmid summary
    per_plasmid = presence[["plasmid", "tfbs_total", "n_unique_tfs"]].sort_values(
        ["tfbs_total", "n_unique_tfs"], ascending=False
    )

    # Per-TF summary
    per_tf = (
        counts.groupby(tf_col)
              .agg(total_sites=("site_count", "sum"),
                   n_plasmids=(plasmid_col, "nunique"))
              .reset_index()
              .rename(columns={tf_col: "TF"})
              .sort_values(["total_sites", "n_plasmids"], ascending=False)
    )

    # Ensure deterministic column order: plasmid then TF columns sorted then totals
    sorted_tf_cols = sorted([c for c in presence.columns if c not in {"plasmid", "tfbs_total", "n_unique_tfs"}])
    presence = presence[["plasmid"] + sorted_tf_cols + ["tfbs_total", "n_unique_tfs"]]

    return presence, per_plasmid, per_tf


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TFBS presence/count matrix from FIMO TSV.")
    p.add_argument("--fimo", required=True, type=Path, help="Path to FIMO TSV output.")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    p.add_argument("--plasmid-col", default=DEFAULT_FIMO_COLS["sequence_name"],
                   help="Column name in FIMO TSV that contains plasmid/sequence IDs.")
    p.add_argument("--tf-col", default=DEFAULT_FIMO_COLS["motif_alt_id"],
                   help="Column name in FIMO TSV that contains TF name/label.")
    p.add_argument("--qvalue", type=float, default=None, help="Optional q-value threshold (<=).")
    p.add_argument("--pvalue", type=float, default=None, help="Optional p-value threshold (<=).")
    p.add_argument("--prefix", type=str, default="TFBS",
                   help="Prefix for output files.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    fimo_df = _read_fimo(args.fimo)

    presence, per_plasmid, per_tf = build_presence_matrix(
        fimo_df=fimo_df,
        plasmid_col=args.plasmid_col,
        tf_col=args.tf_col,
        qvalue_threshold=args.qvalue,
        pvalue_threshold=args.pvalue,
    )

    presence_path = args.outdir / f"{args.prefix}_presence_matrix.csv"
    per_plasmid_path = args.outdir / f"{args.prefix}_summary_per_plasmid.csv"
    per_tf_path = args.outdir / f"{args.prefix}_summary_per_tf.csv"

    presence.to_csv(presence_path, index=False)
    per_plasmid.to_csv(per_plasmid_path, index=False)
    per_tf.to_csv(per_tf_path, index=False)

    print(f"[OK] Wrote: {presence_path}")
    print(f"[OK] Wrote: {per_plasmid_path}")
    print(f"[OK] Wrote: {per_tf_path}")
    print(f"[INFO] Plasmids: {presence.shape[0]}")
    print(f"[INFO] TF columns: {presence.shape[1] - 3} (excluding plasmid + totals)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
