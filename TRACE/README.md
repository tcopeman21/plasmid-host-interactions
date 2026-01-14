# TRACE

This folder contains the TRACE pipeline, which applies a UNet trained on short DNA inserts
to full-length plasmids using sliding-window inference.

## Files

### Helper modules (not run directly)
- `dna_parsing.py` – DNA cleaning and one-hot encoding
- `sliding_window.py` – sliding-window inference on long sequences
- `aggregate.py` – window aggregation functions (softmin, mean, E1)
- `predict.py` – ensemble prediction, τ tuning, and calibration utilities

### Executable scripts
- `finetune_plasmids.py` – optional fine-tuning of UNet models on plasmid data
- `validate_plasmids.py` – evaluates TRACE on held-out plasmids and generates figures/CSVs
