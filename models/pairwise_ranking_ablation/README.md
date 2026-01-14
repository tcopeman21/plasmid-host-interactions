# Pairwise ranking ablation

This folder contains code to reproduce the pairwise-ranking ablation analyses used to evaluate
burden mechanisms (transcriptional activity, TF-binding burden, concatemerisation).

## Input data
A CSV with (at minimum):
- `Average_LFC_DH5a`
- `Total_TX_Sites`
- `Total_TFBS_Sites`
- `DH5a_Average_Concatemer_Percentage`

If using TF-category mode, the CSV should also contain per-TF binding count columns (numeric),
named like `<TF>_...` so that the TF name can be inferred from the prefix before `_`.

## Run
TF category ablation:
```bash
python run_pairwise_ranking_ablation.py \
  --data path/to/merged_plasmid_TFBS_LFC_dataset_with_totals.csv \
  --outdir outputs/tf_categories \
  --mode tf_categories \
  --n_pairs 60000 \
  --repeats 20 \
  --base_seed 42
