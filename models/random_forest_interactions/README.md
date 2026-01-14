# Random forest with explicit pairwise interactions

This folder contains a RandomForestRegressor trained on 4-part plasmid designs.
Inputs are categorical part identities for Positions 1–4. Features include:
- one-hot encoded main effects
- explicit pairwise interaction terms generated via `PolynomialFeatures(interaction_only=True)`

## Files

- `train_rf.py`  
  Trains the pipeline and saves `rf_pipeline.joblib`.

- `evaluate_rf.py`  
  Loads a saved pipeline and reproduces metrics and figures.

- `shap_main_effects.py`  
  Computes TreeSHAP attributions per feature and exports CSVs.

- `shap_replot_and_merge.py`  
  Applies part-name corrections, merges solo-LFC tables, reports missing part mappings,
  and exports merged analysis tables.

- `shap_interactions_swarm.py`  
  Identifies true 2-way interaction features and produces present-only swarm plots
  for main effects and top interactions.
