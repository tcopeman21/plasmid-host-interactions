# Plasmid host–interaction analysis code

This repository contains the analysis, metric extraction, and modelling code used in the manuscript, "Quantitative profiling of millions of nucleotides reveals sequence-encoded interactions that govern plasmid propagation".

## What’s in here

The repo is organised around four main components:

1. **Metric extraction (`Pipeline/metrics/`)**  
   Scripts that compute quantitative per-plasmid features.
2. **Mechanistic analysis (`Pipeline/analysis/`)**  
   Scripts that combine metrics with fitness and generate publication figures.
3. **Predictive models (`models/`)**  
   Random forest, pairwise/ranking, and unet models.
4. **TRACE (`TRACE/`)**  
   Sliding-window application of sequence-trained models to full plasmids + aggregation + finetuning + validation.

---

## Repository structure

```text
Pipeline/
  metrics/
  analysis/

models/
  random_forest/
  pairwise_ranking/
  unet_sequence_model/

TRACE/
  dna_parsing/
  sliding_window/
  predict/
  validate_plasmids/
  aggregate/

data/
  README.md
