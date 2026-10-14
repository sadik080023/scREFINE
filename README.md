# scREFINE

scREFINE is a label-efficient semi-supervised single-cell RNA-seq clustering framework that refines frozen scGPT-derived cell embeddings using only 5% labeled cells, combining supervised refinement with stability-aware soft structural regularization over local neighborhoods.

Companion code for: *"scREFINE: Label-Efficient Semi-Supervised Clustering of scRNA-seq Data via Foundation-Model Embedding Refinement"* (Computational Biology and Chemistry).

## Overview

scREFINE performs:

1. data loading and preprocessing from `.h5ad` files;
2. stratified splitting into labeled (exactly 5%), unlabeled, validation, and test partitions (exact indices in `splits/`);
3. inductive scGPT embedding extraction (HVG selection, PCA, and scaling fitted on training cells only — no test information leakage);
4. semi-supervised refinement learning (triplet, prototype, pairwise, supervised contrastive, and cross-entropy losses on labeled cells only);
5. stability-aware soft structural regularization over training-cell neighborhoods (no hard pseudo-labels; test cells never enter the graph);
6. K-means clustering on test cells and evaluation with ARI, NMI, AMI, and ACC (n_init = 10, mean ± SD over repeated fits).

## Project structure

```text
scREFINE/
├── analysis/                  # K-sensitivity/estimation, raw baseline, marker dotplots, UMAP figures
├── splits/                    # stratified split indices (.npz) for 5 datasets × 4 seeds
├── run_configs/               # launcher scripts for every experiment in the paper
├── baseline_patches/          # patches for scSemiAAE / scAGCL / scCCTR (original + scGPT-backbone)
├── results/
│   └── results_log_v2.csv     # master log of all runs
├── data/                      # input .h5ad files (links in the paper; not included)
├── scgpt/ , scGPT_pretrained/ # scGPT source + checkpoint (download instructions above)
├── run_scREFINE.py            # main pipeline script
├── model.py
├── utils_splits.py
├── extract_scgpt_embeddings.py
├── ssd_clustering.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## scGPT files

Download the scGPT pretrained checkpoint from the scGPT repository and place:

```text
scGPT_pretrained/
├── best_model.pt
└── vocab.json
```

## Input data

Place the `.h5ad` files in the `data/` folder. The AnnData object must contain a cell-type label column (e.g., `cell_type`, `celltype`, `label`). The five benchmark datasets are publicly available (links in the paper's Data availability section):

```text
data/Brain.h5ad
data/Fat.h5ad
data/Retina.h5ad
data/Breast_carcinoma.h5ad
data/Quadriceps_tendon.h5ad
```

## Usage (main 5%-label setting)

```bash
python3 scREFINE/run_scREFINE.py \
  --data data/Brain.h5ad \
  --output-dir result/Brain_s42 \
  --labeled-ratio 0.05 --seed 42 --split-seed 42 \
  --epochs 100 --w-structure 0.1
```

## Key arguments

| Argument | Description |
|---|---|
| `--data` | Path to the input `.h5ad` dataset. |
| `--output-dir` | Directory where results are saved. |
| `--labeled-ratio` | Fraction of **all** cells used as labeled (0.05 = 5%). |
| `--seed` | Model/init seed. |
| `--split-seed` | Data-split seed (labeled-cell selection); defaults to `--seed`. |
| `--epochs` | Maximum training epochs (early stopping: patience 10 on validation CE). |
| `--w-structure` | Weight of the soft structural loss (0.1 in the main experiments). |
| `--hard-pl` | Hard pseudo-label structural variant (ablation only). |
| `--no-graph` | Refinement-only ablation (no structure graph). |

## Output

Each run writes `results.json` (metrics as mean ± SD over 5 K-means fits), refined embeddings (`fused_embeddings.npy`), and appends to `results/results_log_v2.csv`. Example (Brain, seed 42):

```text
TEST (mean over K-means repeats):
   ARI: 0.8314 ± 0.0077
   NMI: 0.8367 ± 0.0037
   AMI: 0.8259 ± 0.0040
   ACC: 0.8530 ± 0.0065
```

## Reproducing the paper

All experiment families (main table, loss ablation, hard-PL, label selection, low-label, weight sensitivity, label ratio, baselines) are launched by the scripts in `run_configs/`; see `run_configs/README.md` for the mapping to paper tables/figures. All baselines were evaluated on exactly the cells in `splits/`.

## Note on checkpoints

Training is deterministic given the published seeds and takes minutes per run on CPU; all reported results are regenerated exactly by the provided configs.

## Main files

| File | Description |
|---|---|
| `run_scREFINE.py` | Main pipeline script. |
| `model.py` | Refinement head, semi-supervised losses, structure-aware trainer. |
| `ssd_clustering.py` | Neighborhood graph construction and edge-stability estimation. |
| `utils_splits.py` | Stratified splits, partial labels, leakage checks. |
| `extract_scgpt_embeddings.py` | Inductive scGPT embedding extraction (train-fitted PCA/scaler). |

## Citation

If you use this code, please cite the associated scREFINE manuscript.
