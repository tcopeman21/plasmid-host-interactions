"""
Compute RPM and LFC from aligned BAM files.

Inputs:
  - One or more BAM files mapped to the same reference set (reference names = plasmids/contigs).
  - A sample map specifying which BAM corresponds to Start / End replicates.

Outputs:
  - rpm.csv: RPM per reference per sample
  - lfc.csv: LFC_End_i, Average_LFC, LFC_StdDev per reference

Filtering:
  - coverage (%) = query_alignment_length / reference_length * 100
  - accuracy (%) = cigar_matches / query_alignment_length * 100
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pysam


def iter_filtered_alignments(
    bam_path: str,
    min_coverage: float,
    min_accuracy: float,
) -> List[str]:
    """
    Returns a list of reference names for reads passing filters.
    One entry per passing read (used for counting).
    """
    refs = []

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam:
            if read.is_unmapped:
                continue

            ref_name = bam.get_reference_name(read.reference_id)
            if ref_name is None:
                continue

            aln_len = read.query_alignment_length
            if aln_len <= 0:
                continue

            ref_len = bam.get_reference_length(ref_name)
            if ref_len <= 0:
                continue

            # CIGAR stats: [matches, ins, del, ...] per operation category.
            # read.get_cigar_stats()[0][0] is "M" count in many cases; for minimap2,
            # this is typically acceptable as a match proxy.
            cigar_stats = read.get_cigar_stats()
            matches = cigar_stats[0][0] if cigar_stats and cigar_stats[0] else 0

            accuracy = (matches / aln_len) * 100.0
            coverage = (aln_len / ref_len) * 100.0

            if coverage >= min_coverage and accuracy >= min_accuracy:
                refs.append(ref_name)

    return refs


def compute_rpm(
    sample_to_bam: Dict[str, str],
    min_coverage: float,
    min_accuracy: float,
) -> pd.DataFrame:
    """
    Returns RPM dataframe indexed by reference, columns = samples.
    """
    rpm_tables = []

    for sample, bam_path in sample_to_bam.items():
        refs = iter_filtered_alignments(bam_path, min_coverage, min_accuracy)
        if len(refs) == 0:
            # keep empty column
            rpm_tables.append(pd.DataFrame({"reference": [], sample: []}))
            continue

        counts = pd.Series(refs).value_counts()
        total = float(counts.sum())
        rpm = (counts * 1e6) / total

        rpm_tables.append(
            pd.DataFrame({"reference": rpm.index.astype(str), sample: rpm.values})
        )

    # outer-join all columns on reference
    out = rpm_tables[0]
    for df in rpm_tables[1:]:
        out = out.merge(df, on="reference", how="outer")

    out = out.fillna(0.0).set_index("reference")
    # stable column order as provided
    out = out.loc[:, list(sample_to_bam.keys())]
    return out


def compute_lfc(
    rpm_df: pd.DataFrame,
    start_col: str,
    end_cols: List[str],
    pseudocount: float,
) -> pd.DataFrame:
    """
    Compute LFC for each end replicate and summary stats.
    """
    if start_col not in rpm_df.columns:
        raise ValueError(f"Start column '{start_col}' not found in RPM columns: {rpm_df.columns.tolist()}")

    missing = [c for c in end_cols if c not in rpm_df.columns]
    if missing:
        raise ValueError(f"Missing end columns in RPM: {missing}")

    out = rpm_df.copy()

    for c in end_cols:
        out[f"LFC_{c}"] = np.log2((out[c] + pseudocount) / (out[start_col] + pseudocount))

    lfc_cols = [f"LFC_{c}" for c in end_cols]
    out["Average_LFC"] = out[lfc_cols].mean(axis=1)
    out["LFC_StdDev"] = out[lfc_cols].std(axis=1, ddof=1)

    # keep only relevant columns (optional; comment out if you want to keep RPM too)
    keep = [start_col] + end_cols + lfc_cols + ["Average_LFC", "LFC_StdDev"]
    return out[keep]


def parse_sample_map(sample_map: List[str]) -> Dict[str, str]:
    """
    Parse repeated --sample SampleName=/path/to.bam entries into a dict.
    """
    m = {}
    for item in sample_map:
        if "=" not in item:
            raise ValueError(f"Invalid --sample '{item}'. Use SampleName=/path/to.bam")
        k, v = item.split("=", 1)
        m[k] = v
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample",
        action="append",
        required=True,
        help="SampleName=/path/to.bam. Provide multiple times.",
    )
    ap.add_argument("--start", default="Start", help="Name of the start sample column (default: Start).")
    ap.add_argument(
        "--ends",
        default="End_1,End_2,End_3",
        help="Comma-separated list of end sample names (must match --sample keys).",
    )
    ap.add_argument("--min_coverage", type=float, default=90.0)
    ap.add_argument("--min_accuracy", type=float, default=90.0)
    ap.add_argument("--pseudocount", type=float, default=1.0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    sample_to_bam = parse_sample_map(args.sample)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_cols = [x.strip() for x in args.ends.split(",") if x.strip()]

    rpm_df = compute_rpm(
        sample_to_bam=sample_to_bam,
        min_coverage=args.min_coverage,
        min_accuracy=args.min_accuracy,
    )
    rpm_path = out_dir / "rpm.csv"
    rpm_df.to_csv(rpm_path)

    lfc_df = compute_lfc(
        rpm_df=rpm_df,
        start_col=args.start,
        end_cols=end_cols,
        pseudocount=args.pseudocount,
    )
    lfc_path = out_dir / "lfc.csv"
    lfc_df.to_csv(lfc_path)

    # Minimal run summary to stdout
    print(f"Wrote: {rpm_path}")
    print(f"Wrote: {lfc_path}")
    print(f"References: {len(rpm_df)}")


if __name__ == "__main__":
    main()
