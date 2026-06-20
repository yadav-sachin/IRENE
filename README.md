<img src="media/KDD_logo.png" height="80" align="right"/>

# IRENE: Extreme Meta-Classification for Large-Scale Zero-Shot Retrieval

[![KDD 2024](https://img.shields.io/badge/Venue-KDD%202024-blue)](https://dl.acm.org/doi/10.1145/3637528.3671843)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](http://manikvarma.org/pubs/yadav24.pdf)

Official PyTorch implementation of **"Extreme Meta-Classification for Large-Scale Zero-Shot Retrieval"** (KDD 2024).

> **Authors:** Sachin Yadav, Deepak Saini, Anirudh Buvanesh, Bhawna Paliwal, Kunal Dahiya, Siddarth Asokan, Yashoteja Prabhu, Jian Jiao, Manik Varma
> Microsoft Research, India | Microsoft, USA | IIT Delhi, India

---

## Overview

**Problem:** Large-scale retrieval settings face a fundamental challenge when novel (zero-shot) items arrive continuously — existing Extreme Classification (XC) classifiers cannot be trained for unseen items due to data and latency constraints, while Siamese encoders lack the representational capacity of discriminative classifiers.

**EMMETT** (Extreme Meta-classification for METa-classifiers Training) is the general algorithmic framework introduced in this paper to bridge this gap. EMMETT efficiently trains meta-classifiers for novel items by leveraging the classifiers already learned for observed items.

**IRENE** is the practical instantiation of EMMETT, comprising two components:

- **Classifier Selector (S):** Given a novel item's embedding, retrieves its *K* nearest neighbor labels from the seen training set using an ANNS index over label embeddings.
- **Meta-Classifier Generator (G):** A lightweight Transformer encoder that takes the novel label's embedding and the classifiers of its *K* selected neighbors as input, and outputs a meta-classifier that can be used directly for retrieval.

<p align="center">
  <img src="media/KDD_logo.png" height="60"/>
</p>

IRENE is **modular** — it can be plugged on top of any dense retriever (NGAME, ANCE, DPR, MACLR) without any fine-tuning of the base encoder, and adds only **~0.4 ms** of per-item overhead at inference.

On average across datasets and base encoders, IRENE improves P@1 by **+10.1%** and R@10 by **+11.9%** in the zero-shot setting, and P@1 by **+15.5%** and R@10 by **+11.5%** in the generalized zero-shot setting.

---

## System Requirements

| Resource | Requirement |
|----------|-------------|
| GPU | NVIDIA A100 / V100 (≥ 32 GB VRAM for 1.3M-scale datasets) |
| RAM | ≥ 64 GB |
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.12 |
| CUDA | ≥ 11.3 |

---

## Installation

```bash
conda create -n irene python=3.8 -y
conda activate irene

pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

---

## Datasets

We evaluate IRENE on four public benchmarks from the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html) and an internal sponsored-search dataset. Zero-shot splits were created by randomly partitioning 10% of labels as novel items.

| Dataset | Application | Feature Type | Train Queries | Seen Labels | Novel Labels |
|---------|-------------|-------------|---------------|-------------|--------------|
| LF-AOL-270K-10 | Query Completion | Short-Text | 3,689,542 | 245,543 | 27,282 |
| LF-WikiHierarchy-550K-10 | Taxonomy Completion | Short-Text | 1,587,567 | 494,733 | 54,970 |
| LF-AmazonTitles-1.3M-10 | Product Recommendation | Short-Text | 2,225,354 | 1,174,739 | 130,526 |
| LF-Wikipedia-500K-10 | Category Annotation | Long-Text | 1,781,890 | 450,963 | 50,107 |

---

## Data Preparation

Training IRENE requires precomputed embeddings and classifiers from a base extreme classifier (e.g., [NGAME](https://github.com/nilesh2797/NGAME)). Organize the files as follows:

```
Datasets/
└── <Dataset>/                              # e.g., LF-AmazonTitles-1.3M_10
    ├── Y.trn.npz                           # train relevance matrix  [N_trn × L_seen]  (scipy sparse)
    ├── Y.tst_zero.npz                      # zero-shot test relevance [N_tst × L_zero]  (scipy sparse)
    ├── Y.tst_full.npz                      # generalized zero-shot   [N_tst × (L_seen + L_zero)]  (scipy sparse)
    ├── filter_labels_test_zero.txt         # (query, label) pairs to filter in zero-shot eval
    └── filter_labels_test_full.txt         # (query, label) pairs to filter in generalized eval

Dataset_Assets/
└── <Dataset>/
    └── <Base Retriever>/                   # e.g., NGAME
        ├── trn_X_unnorm.npy                # train doc embeddings        [N_trn × D]
        ├── tst_X_zero_unnorm.npy           # zero-shot test embeddings   [N_tst × D]
        ├── tst_X_full_unnorm.npy           # full test embeddings        [N_tst × D]
        ├── Y_trn_unnorm.npy                # seen label embeddings       [L_seen × D]
        ├── Y_zero_unnorm.npy               # zero-shot label embeddings  [L_zero × D]
        ├── Y_full_unnorm.npy               # full label embeddings       [(L_seen + L_zero) × D]
        ├── Y_trn_classifiers_unnorm.npy    # seen label classifiers      [L_seen × D]
        ├── Y_trn_neighbor_indices.npy      # k-NN label indices          [L_seen × K_max]
        └── Y_trn_neighbor_scores.npy       # k-NN label scores           [L_seen × K_max]
```

---

## Training

### Step 1 — Train Base XC Classifier

Train a base encoder with one-vs-all BCE loss (e.g., using [NGAME](https://github.com/nilesh2797/NGAME) or [Renée](https://github.com/nilesh2797/Renee)) on the **seen** labels to obtain per-label embeddings and classifiers. Then compute k-NN neighbor indices and scores in label embedding space using an ANNS library (e.g., [DiskANN](https://github.com/microsoft/DiskANN)).

Place all outputs under `Dataset_Assets/<Dataset>/<Base Retriever>/` as shown above.

### Step 2 — Train IRENE (Meta-Classifier Generator)

IRENE's meta-classifier generator **G** is a single Transformer encoder layer that combines:
- The **embedding** of the novel label (from the base encoder)
- The **classifiers** of its *K* nearest neighbor seen labels (retrieved by **S**)
- **Score embeddings** encoding the similarity of each neighbor
- **Positional embeddings** distinguishing the target label from its neighbors

Training is fully driven by YAML config files. See `configs/<Dataset>/meta_clf_gen.yaml` for all hyperparameters and `configs/base_config.yaml` for shared defaults.

```bash
dataset="LF-WikiHierarchy-550K_10"

python train.py configs/${dataset}/meta_clf_gen.yaml
```

**Override any parameter at the command line:**
```bash
python train.py configs/${dataset}/meta_clf_gen.yaml \
    --num_neighbors 3 \
    --num_layers 1 \
    --device cuda:0
```

**Multi-GPU balanced clustering** (set `cls_devices` to a comma-separated list of GPU IDs):
```bash
python train.py configs/${dataset}/meta_clf_gen.yaml --cls_devices "0,1,2,3"
```

**Disable W&B logging:**
```bash
WANDB_MODE=disabled python train.py configs/${dataset}/meta_clf_gen.yaml
```

### Training Details (from Appendix C)

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `num_neighbors` (K) | 3 | Neighbor classifiers fed to G |
| `num_layers` (D) | 1 | Transformer encoder depth in G |
| `num_heads` | 4 | Attention heads in G |
| `dim` | 768 | Embedding dimension |
| `dropout` | 0.1 | Dropout rate |
| `batch_size` | 2048 | Training batch size |
| `lr` | 2e-4 | Peak learning rate (linear warmup, 500 steps) |
| `num_epochs` | 40 | Total epochs |
| `margin` | 0.3 / 0.1 / 0.2 | BCE margin (AOL & Amazon / WikiHierarchy / Wikipedia) |
| `cl_start_ep` | 10 | Epoch to begin curriculum clustering |
| `cl_update` | 5 | Cluster refresh interval (epochs) |

Training IRENE on LF-AmazonTitles-1.3M-10 atop NGAME takes ~**6 hours** on a single V100 GPU (vs. 83 hours for NGAME itself).

### Dataset-Specific Commands

```bash
# LF-AOL-270K_10
python train.py configs/LF-AOL-270K_10/meta_clf_gen.yaml

# LF-WikiHierarchy-550K_10
python train.py configs/LF-WikiHierarchy-550K_10/meta_clf_gen.yaml

# LF-AmazonTitles-1.3M_10
python train.py configs/LF-AmazonTitles-1.3M_10/meta_clf_gen.yaml

# LF-Wikipedia-500K_10
python train.py configs/LF-Wikipedia-500K_10/meta_clf_gen.yaml
```

---

## Results

Metrics: **P@k** (Precision@k), **R@k** (Recall@k). Best in each encoder pair shown in **bold** in the paper.

### Zero-Shot Retrieval (Novel Labels Only)

| Model | LF-AOL-270K | LF-WikiHierarchy-550K | LF-AmazonTitles-1.3M | LF-Wikipedia-500K |
|-------|:-----------:|:---------------------:|:--------------------:|:-----------------:|
| | P@1 / R@10 | P@1 / R@10 | P@1 / R@10 | P@1 / R@10 |
| NGAME | 30.90 / 54.20 | 46.01 / 58.66 | 30.42 / 36.44 | 46.96 / 65.27 |
| **NGAME + IRENE** | **36.47 / 59.57** | **69.29 / 80.40** | **31.56 / 38.83** | 44.91 / 67.79 |
| ANCE | 33.43 / 67.84 | 43.06 / 56.28 | 22.38 / 30.72 | 30.67 / 58.91 |
| **ANCE + IRENE** | **36.84 / 67.82** | **66.54 / 82.10** | **22.75 / 32.72** | **41.59 / 71.59** |
| DPR | 30.38 / 53.82 | 44.84 / 59.29 | 31.10 / 40.98 | 42.90 / 71.20 |
| **DPR + IRENE** | **36.80 / 60.22** | **69.65 / 80.01** | 30.49 / 40.31 | 42.19 / 70.50 |
| MACLR | 11.31 / 18.24 | 30.37 / 35.47 | 21.93 / 28.59 | 39.56 / 68.53 |
| **MACLR + IRENE** | **34.32 / 61.29** | **69.45 / 81.43** | 21.56 / 28.77 | **44.64 / 73.05** |
| SemSup-XC | 26.27 / 36.31 | 57.45 / 46.81 | 11.28 / 11.68 | 46.60 / 57.08 |
| DEXA | 21.68 / 41.85 | 54.83 / 66.89 | 28.83 / 35.19 | 42.76 / 67.37 |

### Generalized Zero-Shot Retrieval (Seen + Novel Labels)

| Model | LF-AOL-270K | LF-WikiHierarchy-550K | LF-AmazonTitles-1.3M | LF-Wikipedia-500K |
|-------|:-----------:|:---------------------:|:--------------------:|:-----------------:|
| | P@1 / R@10 | P@1 / R@10 | P@1 / R@10 | P@1 / R@10 |
| NGAME | 20.16 / 38.27 | 66.19 / 27.08 | 45.14 / 30.25 | 81.86 / 69.58 |
| **NGAME + IRENE** | **35.11 / 52.30** | **91.33 / 40.09** | **47.77 / 31.49** | 78.99 / 69.27 |
| ANCE | 22.63 / 49.72 | 68.76 / 25.89 | 27.65 / 17.31 | 42.91 / 43.39 |
| **ANCE + IRENE** | **30.84 / 51.75** | **90.72 / 39.74** | **36.78 / 22.34** | **71.39 / 63.46** |
| DPR | 19.71 / 37.99 | 65.19 / 26.73 | 38.18 / 26.93 | 51.54 / 61.84 |
| **DPR + IRENE** | **35.07 / 52.57** | **89.52 / 39.84** | **43.08 / 29.25** | **70.39 / 66.71** |
| MACLR | 9.26 / 7.52 | 59.44 / 14.31 | 27.50 / 15.99 | 46.59 / 46.62 |
| **MACLR + IRENE** | **30.40 / 44.71** | **88.81 / 38.37** | **31.49 / 18.85** | **70.52 / 62.82** |

> Full results including R@3, R@5, R@30, R@100, P@3, P@5 for all models are in Tables 14–15 of the supplementary document.

### Inference Efficiency (LF-AmazonTitles-1.3M-10, single V100)

| Method | Rep. Time (ms/item) | Retrieval Time (ms/query) |
|--------|:-------------------:|:-------------------------:|
| NGAME | 0.08 | 0.43 |
| DEXA | 0.48 | 0.43 |
| **NGAME + IRENE** | **0.54** | **0.43** |
| SemSup-XC | N/A | 151.51 |

IRENE adds only **+0.4 ms** per item and is **~350× faster** than SemSup-XC at inference.

---

## Repository Structure

```
IRENE/
├── train.py                    # Main training script
├── nets.py                     # MetaClfGen: Transformer meta-classifier generator (G)
├── datasets.py                 # Dataset, collate, and batch-sampling utilities
├── aol_ablations.sh            # Example training script for LF-AOL-270K_10
├── utils/
│   ├── helper_utils.py         # YAML config loading with dependency resolution
│   ├── eval_utils.py           # Evaluation: ANNS / OvA prediction, XC metrics
│   └── cluster_gpu.py          # Multi-GPU balanced k-means clustering for curriculum
└── configs/
    ├── base_config.yaml                          # Shared defaults for all datasets
    ├── LF-AOL-270K_10/
    │   ├── dataset.yaml                          # Dataset-specific fields
    │   └── meta_clf_gen.yaml                     # Full training config (inherits base_config)
    ├── LF-WikiHierarchy-550K_10/
    ├── LF-AmazonTitles-1.3M_10/
    └── LF-Wikipedia-500K_10/
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yadav2024extrememetaclassification,
  author    = {Yadav, Sachin and Saini, Deepak and Buvanesh, Anirudh and Paliwal, Bhawna
               and Dahiya, Kunal and Asokan, Siddarth and Prabhu, Yashoteja
               and Jiao, Jian and Varma, Manik},
  title     = {Extreme Meta-Classification for Large-Scale Zero-Shot Retrieval},
  booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  series    = {KDD '24},
  year      = {2024},
  doi       = {10.1145/3637528.3672046}
}
```
