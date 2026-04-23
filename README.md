# Plasmid host–interaction analysis code

This repository contains the analysis, metric extraction, and modelling code used in the manuscript, "Quantitative profiling of millions of nucleotides reveals sequence-encoded interactions that govern plasmid propagation".

Source data is available on Zenodo at https://doi.org/10.5281/zenodo.19698537

---

## What's in here

The repo is organised around four main components:

1. **Metric extraction (`Pipeline/metrics/`)**  
   Scripts that compute LFC, concatemer percentage, and TFBS burden from raw sequencing data.

2. **Mechanistic analysis (`Pipeline/mechanisms/`)**  
   Combines metric outputs with plasmid fitness to generate publication figures and Spearman correlations.

3. **Predictive models (`models/`)**  
   Random forest part-identity model, pairwise ranking ablation model, and U-Net sequence model.

4. **TRACE (`TRACE/`)**  
   Sliding-window application of the trained U-Net to full-length plasmid sequences, including aggregation, fine-tuning, and validation.

---

## Repository structure

```text
Pipeline/
  metrics/
    calculate_rpm_and_lfc.py
    calculate_concatemer_metrics.py
    calculate_tfbs.py
  mechanisms/
    mechanism_analysis.py

models/
  random_forest_interactions/
    train_rf.py
    evaluate_rf.py
    shap_main_effects.py
    shap_interactions_swarm.py
    shap_merge_solo_and_summarise.py
  pairwise_ranking_ablation/
    run_pairwise_ranking_ablation.py
  unet_sequence_model/
    model.py
    data.py
    train_cv.py
    eval_cv.py

TRACE/
  DNA_parsing.py
  sliding_window.py
  aggregate.py
  predict.py
  finetune_plasmids.py
  validate_plasmids.py

data/
  README.md
```

---

## System requirements

**Software dependencies**

- Python 3.11
- torch >= 2.0
- numpy >= 2.3.4
- scipy >= 1.11
- scikit-learn >= 1.3
- pandas >= 2.0
- matplotlib >= 3.10.7
- biopython >= 1.86
- shap >= 0.43
- pysam >= 0.22

Install all dependencies via pip:

```bash
pip install torch numpy scipy scikit-learn pandas matplotlib biopython shap pysam
```

**Operating systems**

Tested on Google Colaboratory (Linux-based environment). Compatible with any system running Python 3.11 with CUDA-enabled PyTorch.

**Non-standard hardware**

None required. All code runs on standard CPU. NVIDIA GPU (CUDA-compatible) will accelerate U-Net model training but is not required.

---

## Installation guide

1. Clone the repository:

```bash
git clone https://github.com/tcopeman21/plasmid-host-interactions.git
cd plasmid-host-interactions
```

2. Install dependencies:

```bash
pip install torch numpy scipy scikit-learn pandas matplotlib biopython shap pysam
```

Typical install time: approximately 5–10 minutes on a standard desktop computer.

---

## Demo

All data required to run these scripts is available on Zenodo at https://doi.org/10.5281/zenodo.19698537

Download and unzip the source data. The file structure is:

```text
Source Data/
  Single Part Assay/
    Plasmid_LFC_Summary_with_Sequences.csv
    Concatemer Metrics/
      New Concat AnalysisDH5a_Concatemer_Percentage_Summary.csv
    Predicted and Measured Transcription Metrics/
      reads_out.tsv
    TFBS Metrics/
      FIMO_presence_matrix (1).csv
  4-Part Assembly Assay (rf training data)/
    GGS2 normalised full data set with sequences.csv
  Pairwise Ranking Ablation Study/
    Plasmid_LFC_TSS_TFBS_Concat_Data.csv
  UNet Training Data/
    Deg_Seq_LFC_UNet_Training_Data.pkl
```

---

## How to run on your data

### 1. Metric extraction

**Compute LFC from aligned BAM files:**

```bash
python Pipeline/metrics/calculate_rpm_and_lfc.py \
  --bam <start_rep1.bam> <start_rep2.bam> <end_rep1.bam> <end_rep2.bam> \
  --sample-map <sample_map.csv> \
  --outdir <output_directory>
```

Outputs: `rpm.csv`, `lfc.csv`

**Compute concatemer metrics from BAM files:**

```bash
python Pipeline/metrics/calculate_concatemer_metrics.py \
  --bam <bam_files> \
  --fasta <reference.fasta> \
  --backbone-bp <backbone_length> \
  --outdir <output_directory>
```

Outputs: `concatemer_per_region.csv`, `fold_increase_per_region.csv`

**Compute TFBS burden from FIMO output:**

```bash
python Pipeline/metrics/calculate_tfbs.py \
  --fimo <fimo.tsv> \
  --outdir <output_directory> \
  --qvalue 0.05
```

Outputs: `tfbs_presence_matrix.csv`

---

### 2. Mechanistic analysis

Uses: `Plasmid_LFC_Summary_with_Sequences.csv`, `reads_out.tsv`, `New Concat AnalysisDH5a_Concatemer_Percentage_Summary.csv`, `FIMO_presence_matrix (1).csv`

```bash
python Pipeline/mechanisms/mechanism_analysis.py \
  --lfc "Plasmid_LFC_Summary_with_Sequences.csv" \
  --tx "reads_out.tsv" \
  --concat "New Concat AnalysisDH5a_Concatemer_Percentage_Summary.csv" \
  --tfbs "FIMO_presence_matrix (1).csv" \
  --outdir results/mechanisms \
  --make-tf-class
```

Reproduces Figures 1–2.

---

### 3. Random forest part-identity model

Uses: `GGS2 normalised full data set with sequences.csv`

**Train:**

```bash
python models/random_forest_interactions/train_rf.py \
  --data "GGS2 normalised full data set with sequences.csv" \
  --outdir <output_directory>
```

**Evaluate and compute SHAP values:**

```bash
python models/random_forest_interactions/evaluate_rf.py \
  --model <trained_pipeline.joblib> \
  --data "GGS2 normalised full data set with sequences.csv"

python models/random_forest_interactions/shap_main_effects.py \
  --model <trained_pipeline.joblib> \
  --data "GGS2 normalised full data set with sequences.csv"
```

Reproduces Figure 3.

---

### 4. Pairwise ranking ablation

Uses: `Plasmid_LFC_TSS_TFBS_Concat_Data.csv`

```bash
python models/pairwise_ranking_ablation/run_pairwise_ranking_ablation.py \
  --data "Plasmid_LFC_TSS_TFBS_Concat_Data.csv" \
  --outdir outputs/pairwise_ablation \
  --mode tf_categories \
  --n_pairs 60000 \
  --repeats 20
```

Reproduces Figure 2E.

---

### 5. U-Net sequence model

Uses: `Deg_Seq_LFC_UNet_Training_Data.pkl`

**Train with 5-fold cross-validation:**

```bash
python models/unet_sequence_model/train_cv.py \
  --pkl "Deg_Seq_LFC_UNet_Training_Data.pkl" \
  --out <output_directory> \
  --k 5 \
  --epochs 25
```

**Evaluate:**

```bash
python models/unet_sequence_model/eval_cv.py \
  --pkl "Deg_Seq_LFC_UNet_Training_Data.pkl" \
  --model-dir <output_directory>
```

---

### 6. TRACE — sliding-window inference on full-length plasmids

Uses: `Plasmid_LFC_Summary_with_Sequences.csv`

```bash
python TRACE/validate_plasmids.py \
  --data "Plasmid_LFC_Summary_with_Sequences.csv" \
  --ckpts <path_to_model_checkpoints> \
  --out <output_directory>
```

Expected output: a CSV containing predicted TRACE scores per plasmid alongside measured LFC values and Spearman correlation between predicted and measured fitness.

Expected run time: approximately 5–10 minutes for 192 sequences on a standard CPU.

---

## Reproduction instructions

To reproduce all quantitative results in the manuscript:

1. Download source data from Zenodo: https://doi.org/10.5281/zenodo.19698537
2. Run `Pipeline/metrics/` scripts on raw BAM files to regenerate derived metrics, or use the pre-computed files from Zenodo directly
3. Run `Pipeline/mechanisms/mechanism_analysis.py` to reproduce Figures 1–2
4. Run `models/random_forest_interactions/train_rf.py` to reproduce Figure 3
5. Run `models/pairwise_ranking_ablation/run_pairwise_ranking_ablation.py` to reproduce Figure 2E
6. Run `models/unet_sequence_model/train_cv.py` to train the U-Net model
7. Run `TRACE/validate_plasmids.py` to reproduce Figure 4 TRACE predictions

---

## Licence

CC BY-NC 4.0 — Creative Commons Attribution-NonCommercial 4.0 International. See `LICENSE` file for details.
