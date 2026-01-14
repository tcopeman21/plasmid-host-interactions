"""
Merge assembly designs with:
  - per-slot SHAP attributions (from main_shap_per_slot.csv)
  - solo part LFC table (per part)
Apply name corrections, report missing mappings, and export:
  - merged row-level table
  - per-part summary stats (mean SHAP across contexts vs solo LFC)

This script does NOT recompute SHAP; it consumes the exported CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SLOTS = ["Position 1", "Position 2", "Position 3", "Position 4"]


DEFAULT_NAME_MAP = {
    # slot-3 (MTK4a -> MTK3a) examples from your notebook
    "MTK4a_001": "MTK3a_001",
    "MTK4a_005": "MTK3a_005",
    "MTK4a_007": "MTK3a_007",
    "MTK4a_008": "MTK3a_004",
    # slot-4 (MTK4b -> MTK4)
    "MTK4b_001": "MTK4_001",
    "MTK4b_002": "MTK4_002",
    "MTK4b_003": "MTK4_003",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembly-csv", required=True)
    ap.add_argument("--solo-csv", required=True)
    ap.add_argument("--solo-id-col", default="Plasmid")
    ap.add_argument("--solo-lfc-col", default="Average_LFC_DH5a")
    ap.add_argument("--solo-sd-col", default="LFC_StdDev_DH5a")
    ap.add_argument("--shap-slot-csv", required=True, help="main_shap_per_slot.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name-map-csv", default=None, help="Optional CSV with columns: bad, good")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # name mapping
    name_map = dict(DEFAULT_NAME_MAP)
    if args.name_map_csv:
        nm = pd.read_csv(args.name_map_csv)
        if not {"bad", "good"}.issubset(nm.columns):
            raise ValueError("name-map-csv must have columns: bad, good")
        name_map.update(dict(zip(nm["bad"].astype(str), nm["good"].astype(str))))

    asm = pd.read_csv(args.assembly_csv)
    solo = pd.read_csv(args.solo_csv)
    shap_slot = pd.read_csv(args.shap_slot_csv)

    # Clean + apply name map on assembly slots
    for col in SLOTS:
        asm[col] = asm[col].astype(str).str.strip().replace(name_map)

    # Prepare solo stats
    solo = solo.copy()
    solo["PartID"] = solo[args.solo_id_col].astype(str).str.strip().replace(name_map)

    solo_stats = solo[["PartID", args.solo_lfc_col, args.solo_sd_col]].drop_duplicates()
    solo_stats = solo_stats.rename(
        columns={
            args.solo_lfc_col: "Solo_LFC",
            args.solo_sd_col: "Solo_LFC_SD",
        }
    )
    # If you always have n=3 replicates in solo table, keep it explicit; otherwise drop.
    solo_stats["Solo_LFC_SEM"] = solo_stats["Solo_LFC_SD"] / np.sqrt(3)

    # Attach per-slot shap columns to assembly rows (assumes same row order)
    if len(shap_slot) != len(asm):
        raise ValueError(
            f"Row mismatch: assembly has {len(asm)} rows, shap_slot has {len(shap_slot)} rows. "
            "These must correspond 1:1 in the same order."
        )

    df = pd.concat(
        [asm.reset_index(drop=True), shap_slot.drop(columns=["Sample"], errors="ignore").reset_index(drop=True)],
        axis=1,
    )

    # Merge solo stats for each slot
    missing_report = {}
    for p in range(1, 5):
        slot_col = f"Position {p}"
        df = df.merge(
            solo_stats.rename(
                columns={
                    "PartID": slot_col,
                    "Solo_LFC": f"Solo_LFC_slot{p}",
                    "Solo_LFC_SD": f"Solo_LFC_slot{p}_sd",
                    "Solo_LFC_SEM": f"Solo_LFC_slot{p}_sem",
                }
            ),
            on=slot_col,
            how="left",
        )

        miss = df.loc[df[f"Solo_LFC_slot{p}"].isna(), slot_col].dropna().unique().tolist()
        missing_report[p] = sorted(map(str, miss))

    # Report missing mappings
    for p, miss in missing_report.items():
        if not miss:
            print(f"✅ Slot {p}: all parts matched to solo table")
        else:
            print(f"⚠️ Slot {p}: {len(miss)} parts missing in solo table")

    # Save merged table
    merged_path = out_dir / "assembly_with_solo_and_shap.csv"
    df.to_csv(merged_path, index=False)
    print(f"✅ Wrote merged table: {merged_path}")

    # Part-level summary stats per slot
    summary_rows = []
    for p in range(1, 5):
        part_col = f"Position {p}"
        shap_col = f"MainSHAP_slot{p}"
        solo_col = f"Solo_LFC_slot{p}"
        solo_sd_col = f"Solo_LFC_slot{p}_sd"
        solo_sem_col = f"Solo_LFC_slot{p}_sem"

        grp = (
            df.groupby(part_col)
            .agg(
                mean_solo=(solo_col, "first"),
                sd_solo=(solo_sd_col, "first"),
                sem_solo=(solo_sem_col, "first"),
                mean_shap=(shap_col, "mean"),
                sd_shap=(shap_col, lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
                sem_shap=(shap_col, lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
                n_appear=(shap_col, "size"),
            )
            .reset_index()
            .rename(columns={part_col: "PartID"})
        )
        grp["Slot"] = f"S{p}"
        summary_rows.append(grp)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_path = out_dir / "Part_SoloLFC_SHAP_stats.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Wrote part-level summary: {summary_path}")


if __name__ == "__main__":
    main()
