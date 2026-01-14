"""
Compute TreeSHAP values for a trained RF pipeline and export:
  1) per-feature SHAP values
  2) per-slot aggregated SHAP values (sum of features containing that slot)

Assumes the pipeline structure:
  columntransformer -> polynomialfeatures -> randomforestregressor
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from tqdm import tqdm


SLOTS = ["Position 1", "Position 2", "Position 3", "Position 4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True, help="Path to rf_pipeline.joblib")
    ap.add_argument("--assembly-csv", required=True, help="Assembly dataset CSV")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--batch-size", type=int, default=256, help="SHAP batch size")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(args.pipeline)

    # enforce expected steps
    ct = pipe.named_steps.get("columntransformer", None)
    poly = pipe.named_steps.get("polynomialfeatures", None)
    rf = pipe.named_steps.get("randomforestregressor", None)
    if ct is None or poly is None or rf is None:
        raise ValueError(
            "Pipeline must contain named steps: "
            "'columntransformer', 'polynomialfeatures', 'randomforestregressor'."
        )

    # Load data
    asm = pd.read_csv(args.assembly_csv)
    missing = [c for c in SLOTS if c not in asm.columns]
    if missing:
        raise ValueError(f"Missing required slot columns: {missing}")
    X_raw = asm[SLOTS]

    # Build model input matrix (X_poly)
    X_ohe = ct.transform(X_raw)
    if hasattr(X_ohe, "toarray"):
        X_ohe = X_ohe.toarray()
    X_ohe = X_ohe.astype(np.float64)
    X_poly = poly.transform(X_ohe).astype(np.float64)

    # Build feature names
    ohe = ct.named_transformers_["onehot"]
    feat_ohe = ohe.get_feature_names_out(SLOTS)
    feat_poly = poly.get_feature_names_out(feat_ohe)

    # SHAP
    explainer = shap.TreeExplainer(rf)

    n = X_poly.shape[0]
    all_shap = []
    for start in tqdm(range(0, n, args.batch_size), desc="Computing SHAP"):
        end = min(start + args.batch_size, n)
        Xb = X_poly[start:end]
        # check_additivity=False avoids warnings for ensembles
        sb = explainer.shap_values(Xb, check_additivity=False)
        all_shap.append(sb)

    shap_values = np.vstack(all_shap)  # (n_samples, n_features)

    # Map each polynomial feature -> which slot(s) appear in its name
    # Example feature name includes strings like "Position 1_*", etc.
    pat = re.compile(r"Position (\d+)_")
    slot_map = {
        idx: {int(m) for m in pat.findall(name)}
        for idx, name in enumerate(feat_poly)
    }

    # Aggregate SHAP per slot (sum SHAP of all features containing that slot)
    main_shap_slot = {p: np.zeros(n, dtype=float) for p in [1, 2, 3, 4]}
    for feat_idx in range(len(feat_poly)):
        vals = shap_values[:, feat_idx]
        for p in slot_map[feat_idx]:
            main_shap_slot[p] += vals

    # Save
    pd.DataFrame(shap_values, columns=feat_poly).assign(Sample=np.arange(n)).to_csv(
        out_dir / "main_shap_per_feature.csv", index=False
    )

    pd.DataFrame(
        {"Sample": np.arange(n), **{f"MainSHAP_slot{p}": main_shap_slot[p] for p in [1, 2, 3, 4]}}
    ).to_csv(out_dir / "main_shap_per_slot.csv", index=False)

    # Also save feature name files for convenience
    pd.Series(feat_poly, name="feature").to_csv(out_dir / "feature_names_poly.csv", index=False)
    pd.Series(feat_ohe, name="feature").to_csv(out_dir / "feature_names_ohe.csv", index=False)

    print(f"✅ Wrote SHAP outputs to: {out_dir}")


if __name__ == "__main__":
    main()
