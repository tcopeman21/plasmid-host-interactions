# UNet sequence model

This folder contains the core UNet-based sequence model used to predict plasmid fitness from short DNA inserts.
The model is trained exclusively on fixed-length synthetic sequences and is later reused by the TRACE
pipeline for inference on full-length plasmids.

## Scope

This code defines:
- the UNet architecture
- data loading and augmentation
- cross-validated training
- hold-out evaluation on short sequences

Application of the trained model to full plasmids is handled separately in `trace/`.

---

## Model overview

The model is a 1D UNet operating on one-hot encoded DNA sequences.
It produces a position-wise risk / fitness map which is reduced to a scalar prediction by averaging
across sequence length.

Key design features:
- circular shift augmentation
- random base mutation augmentation
- reverse-complement consistency regularisation
- auxiliary ranking loss between high- and low-fitness sequences
- Huber regression loss for robustness

---

## Files

### Training and evaluation scripts

- `train_unet_cv.py`  
  Cross-validated training of the UNet on short DNA inserts.
  Performs K-fold training, early stopping, and checkpointing.

- `evaluate_unet_cv.py`  
  Aggregates cross-validated hold-out predictions and reports final metrics.
  Generates the predict–vs–test plots used in the manuscript.

### Model definition

- `model.py`  
  Definition of the UNet architecture and supporting encoder/decoder blocks.

### Utilities

- `data.py`  
  Dataset loading, one-hot encoding, and preprocessing utilities.

- `augment.py`  
  Sequence-level augmentation functions (shifts, mutations, reverse complement).
