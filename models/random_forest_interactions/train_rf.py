"""
Train a RandomForestRegressor on 4-part plasmid designs using one-hot encoding and
explicit pairwise interaction features.

Pipeline:
    OneHotEncoder (positions) -> PolynomialFeatures(interaction_only=True) -> RandomForestRegressor

Outputs:
    - joblib pipeline
    - train/test metrics JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


SLOTS = ["Position 1", "Position 2", "Position 3", "Position 4"]


def build_pipeline(
    n_estimators: int,
    random_state: int,
    n_jobs: int,
    max_depth: int | None,
    min_samples_leaf: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("onehot", OneHotEncoder(handle_unknown="ignore"), SLOTS)],
        remainder="drop",
        sparse_threshold=0.3,  # keep sparse when possible
    )

    # PolynomialFeatures will expand to (main + pairwise interactions).
    # NOTE: PolynomialFeatures may densify; check memory if you have many categories.
    poly = PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False,
    )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )

    return Pipeline(
        steps=[
            ("columntransformer", preprocessor),
            ("polynomialfeatures", poly),
            ("randomforestregressor", rf),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Assembly dataset CSV")
    ap.add_argument("--target", default="Average", help="Target column name")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=-1)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV.")

    missing = [c for c in SLOTS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required slot columns: {missing}")

    X = df[SLOTS].copy()
    y = df[args.target].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    model = build_pipeline(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "test_mse": mse,
        "test_r2": r2,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "slots": SLOTS,
        "target": args.target,
        "seed": args.seed,
        "rf_params": model.named_steps["randomforestregressor"].get_params(),
    }

    # Save
    joblib.dump(model, out_dir / "rf_pipeline.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
