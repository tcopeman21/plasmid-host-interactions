"""
Compute concatemer metrics per reference region from BAMs.

Definition:
  concatemer if read_length >= 2 * adjusted_ref_len
  adjusted_ref_len = ref_len + backbone_bp

Outputs (CSVs):
  - concatemer_per_region.csv (Start/End columns + summary columns)
  - fold_increase_per_region.csv (Start/End columns + summary columns)
  - total_reads_per_region.csv
  - concatemer_reads_per_region.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pysam
from Bio import SeqIO


def load_reference_lengths(fasta_path: str, backbone_bp: int) -> pd.DataFrame:
    rows = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        rows.append({"region": rec.id, "ref_len": len(rec.seq)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No records found in FASTA: {fasta_path}")
    df["adjusted_ref_len"] = df["ref_len"] + int(backbone_bp)
    return df


def bam_read_lengths(bam_path: str) -> pd.DataFrame:
    """
    One row per aligned read: region, sequence_length
    """
    records = []
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for aln in bam:
            if aln.is_unmapped:
                continue
            if aln.query_sequence is None or aln.reference_name is None:
                continue
            records.append({"region": aln.reference_name, "sequence_length": len(aln.query_sequence)})
    return pd.DataFrame(records)


def compute_concatemer_tables(
    sample_to_bam: Dict[str, str],
    ref_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      total_reads_df: region x samples
      concat_reads_df: region x samples
      concat_pct_df: region x samples
      fold_inc_df: region x samples (mean fold increase among concatemers)
    """
    total_reads = {}
    concat_reads = {}
    concat_pct = {}
    fold_inc = {}

    ref_small = ref_df[["region", "adjusted_ref_len"]].copy()

    for sample, bam_path in sample_to_bam.items():
        reads = bam_read_lengths(bam_path)
        if reads.empty:
            total_reads[sample] = pd.Series(dtype=int)
            concat_reads[sample] = pd.Series(dtype=int)
            concat_pct[sample] = pd.Series(dtype=float)
            fold_inc[sample] = pd.Series(dtype=float)
            continue

        merged = reads.merge(ref_small, on="region", how="inner")
        if merged.empty:
            total_reads[sample] = pd.Series(dtype=int)
            concat_reads[sample] = pd.Series(dtype=int)
            concat_pct[sample] = pd.Series(dtype=float)
            fold_inc[sample] = pd.Series(dtype=float)
            continue

        merged["is_concatemer"] = merged["sequence_length"] >= (2 * merged["adjusted_ref_len"])

        total = merged.groupby("region").size().astype(int)
        conc = merged.groupby("region")["is_concatemer"].sum().astype(int)

        pct = (conc / total * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        conc_only = merged[merged["is_concatemer"]].copy()
        if not conc_only.empty:
            conc_only["fold_increase"] = conc_only["sequence_length"] / conc_only["adjusted_ref_len"]
            fold = conc_only.groupby("region")["fold_increase"].mean()
        else:
            fold = pd.Series(dtype=float)

        total_reads[sample] = total
        concat_reads[sample] = conc
        concat_pct[sample] = pct
        fold_inc[sample] = fold

    total_reads_df = pd.DataFrame(total_reads).fillna(0).astype(int)
    concat_reads_df = pd.DataFrame(concat_reads).fillna(0).astype(int)
    concat_pct_df = pd.DataFrame(concat_pct).fillna(0.0)
    fold_inc_df = pd.DataFrame(fold_inc).fillna(0.0)

    # enforce sample order
    cols = list(sample_to_bam.keys())
    return (
        total_reads_df.reindex(columns=cols, fill_value=0),
        concat_reads_df.reindex(columns=cols, fill_value=0),
        concat_pct_df.reindex(columns=cols, fill_value=0.0),
        fold_inc_df.reindex(columns=cols, fill_value=0.0),
    )


def add_end_stats_and_deltas(
    df: pd.DataFrame,
    start_col: str,
    end_cols: List[str],
    prefix: str,
) -> pd.DataFrame:
    """
    Adds:
      - {prefix}_End_Mean
      - {prefix}_End_Std
      - {prefix}_Delta_EndMean_minus_Start
    """
    out = df.copy()

    for c in [start_col] + end_cols:
        if c not in out.columns:
            raise ValueError(f"Missing column '{c}' in table.")

    end_mat = out[end_cols]
    out[f"{prefix}_End_Mean"] = end_mat.mean(axis=1)
    out[f"{prefix}_End_Std"] = end_mat.std(axis=1, ddof=1)
    out[f"{prefix}_Delta_EndMean_minus_Start"] = out[f"{prefix}_End_Mean"] - out[start_col]

    return out


def parse_sample_map(sample_map: List[str]) -> Dict[str, str]:
    m = {}
    for item in sample_map:
        if "=" not in item:
            raise ValueError(f"Invalid --sample '{item}'. Use SampleName=/path/to.bam")
        k, v = item.split("=", 1)
        m[k] = v
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="append", required=True, help="SampleName=/path/to.bam (repeatable).")
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--backbone_bp", type=int, default=0, help="Backbone bp to add to ref length (default 0).")
    ap.add_argument("--start", default="Start")
    ap.add_argument("--ends", default="End_1,End_2,End_3")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    sample_to_bam = parse_sample_map(args.sample)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_cols = [x.strip() for x in args.ends.split(",") if x.strip()]

    ref_df = load_reference_lengths(args.reference_fasta, args.backbone_bp)

    total_df, conc_df, pct_df, fold_df = compute_concatemer_tables(sample_to_bam, ref_df)

    pct_summary = add_end_stats_and_deltas(pct_df, args.start, end_cols, prefix="ConcatPct")
    fold_summary = add_end_stats_and_deltas(fold_df, args.start, end_cols, prefix="FoldInc")

    (out_dir / "total_reads_per_region.csv").write_text("")  # ensures folder exists even if saving fails
    total_df.to_csv(out_dir / "total_reads_per_region.csv")
    conc_df.to_csv(out_dir / "concatemer_reads_per_region.csv")
    pct_summary.to_csv(out_dir / "concatemer_per_region.csv")
    fold_summary.to_csv(out_dir / "fold_increase_per_region.csv")

    print(f"Wrote: {out_dir / 'total_reads_per_region.csv'}")
    print(f"Wrote: {out_dir / 'concatemer_reads_per_region.csv'}")
    print(f"Wrote: {out_dir / 'concatemer_per_region.csv'}")
    print(f"Wrote: {out_dir / 'fold_increase_per_region.csv'}")
    print(f"Regions: {len(pct_df)}")


if __name__ == "__main__":
    main()
