"""
export_splits.py — export scREFINE v2 split indices so that BASELINE methods
(scSemiAAE / scAGCL / scCCTR) train on EXACTLY the same labeled/unlabeled/val/test cells.
Creates splits/<Dataset>_s<seed>.npz for each dataset and each seed in {17,42,109,172}.
"""
import os
import numpy as np
import scanpy as sc
import sys
sys.path.insert(0, ".")
from utils_splits import create_stratified_splits

SEEDS = [17, 42, 109, 172]
DATA = {
    "Brain": ("data/Brain.h5ad", 22),
    "Fat": ("data/Fat.h5ad", 34),
    "Retina": ("data/Retina.h5ad", 72),
    "Breast_carcinoma": ("data/Breast_carcinoma.h5ad", 19),
    "Quadriceps_tendon": ("data/Quadriceps_tendon.h5ad", 22),
}

os.makedirs("splits", exist_ok=True)

for name, (path, k) in DATA.items():
    adata = sc.read_h5ad(path)
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str)
        adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    found = None
    for key in ["Group","cell_type1","cell_type","celltype","cell_ontology_class","label","Label","cluster"]:
        if key in adata.obs.columns:
            found = key; break
    assert found, f"{name}: no label column"
    adata.obs["Group"] = adata.obs[found].astype(str)
    labels_str = adata.obs["Group"].astype(str).values
    unique_classes = np.unique(labels_str)
    y_full = np.array([{v: i for i, v in enumerate(unique_classes)}[v] for v in labels_str])

    uc, cc = np.unique(y_full, return_counts=True)
    rare = uc[cc < 2]
    if len(rare):
        m = ~np.isin(y_full, rare)
        adata = adata[m].copy()
        labels_str = labels_str[m]
        unique_classes = np.unique(labels_str)
        y_full = np.array([{v: i for i, v in enumerate(unique_classes)}[v] for v in labels_str])

    n_classes = len(unique_classes)
    for s in SEEDS:
        idx_lab, idx_unlab, idx_val, idx_test = create_stratified_splits(
            n_samples=len(y_full), labels=y_full,
            labeled_ratio=0.05, val_ratio=0.1, test_ratio=0.2, random_state=s)
        out = f"splits/{name}_s{s}.npz"
        np.savez(out,
                 idx_train_labeled=idx_lab, idx_train_unlabeled=idx_unlab,
                 idx_val=idx_val, idx_test=idx_test,
                 y_full=y_full, class_names=unique_classes,
                 cell_ids=adata.obs_names.values)
        print(f"saved {out}  (labeled={len(idx_lab)}, test={len(idx_test)}, classes={n_classes})")
print("DONE — splits/ folder ready for baseline re-run")