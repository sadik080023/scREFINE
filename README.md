# scREFINE

scREFINE is a semi-supervised single-cell RNA-seq clustering pipeline that refines scGPT-derived cell embeddings using limited labeled cells and structure-aware neighborhood regularization.

## Overview

scREFINE performs:

1. data loading and preprocessing from `.h5ad` files;
2. stratified labeled, unlabeled, validation, and test splitting;
3. inductive scGPT embedding extraction;
4. semi-supervised refinement learning;
5. SSD-based structure-aware neighborhood regularization;
6. clustering evaluation using ARI, NMI, AMI, and ACC.

## Project structure

```text
scREFINE/
├── data/
├── result/
├── scgpt/
├── scGPT_pretrained/
├── README.md
├── requirements.txt
├── run_scREFINE.py
├── model.py
├── ssd_clustering.py
├── utils_splits.py
└── extract_scgpt_embeddings.py
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## scGPT files

Place the scGPT source code in:

```text
scgpt/
```

Place the pretrained scGPT files in:

```text
scGPT_pretrained/
├── best_model.pt
└── vocab.json
```

## Input data

Place input `.h5ad` files in the `data/` folder.

The AnnData object should contain a cell-type label column. Supported label column names include:

```text
cell_type
celltype
CellType
label
labels
```

Example:

```text
data/Brain.h5ad
```

## Usage

Run scREFINE with:

```bash
python3 run_scREFINE.py \
  --data data/Brain.h5ad \
  --output-dir result/Brain_5pct_w01 \
  --labeled-ratio 0.05 \
  --seed 42 \
  --epochs 100 \
  --w-structure 0.1
```

## Arguments

| Argument | Description |
|---|---|
| `--data` | Path to the input `.h5ad` dataset. |
| `--output-dir` | Directory where results will be saved. |
| `--labeled-ratio` | Fraction of labeled cells used during semi-supervised training. |
| `--seed` | Random seed for reproducibility. |
| `--epochs` | Maximum number of training epochs. |
| `--w-structure` | Weight of the SSD structure-aware loss. |

## Output

Results are saved in the specified output directory, for example:

```text
result/Brain_5pct_w01/
```

The pipeline reports validation and test performance using:

```text
ARI
NMI
AMI
ACC
```

It also reports runtime information for preprocessing, embedding extraction, training, and evaluation.

## Example output

```text
TEST SET RESULTS:
   ARI:  0.8441
   NMI:  0.8553
   AMI:  0.8461
   ACC:  0.8705
```

## Main files

| File | Description |
|---|---|
| `run_scREFINE.py` | Main script for running the full scREFINE pipeline. |
| `model.py` | Refinement head, semi-supervised loss, and training module. |
| `ssd_clustering.py` | SSD structural layer for neighborhood graph construction and stability estimation. |
| `utils_splits.py` | Stratified split creation, partial-label generation, and leakage checking. |
| `extract_scgpt_embeddings.py` | Inductive scGPT embedding extraction with train-only PCA and scaling. |

## Citation

If you use this code, please cite the associated scREFINE manuscript.