"""
make_scgpt_inputs.py — regenerate the EXACT scREFINE v2 input embeddings for each
(dataset, seed) and save as scgpt_inputs/{D}_s{s}.h5  [X=embeddings, Y=labels, n_classes attr, cell_names]
These feed the scGPT+ baseline variants (controlled common-embedding comparison).
"""
import os
import numpy as np
import scanpy as sc
import h5py
import sys
sys.path.insert(0, ".")
from utils_splits import create_stratified_splits
from extract_scgpt_embeddings import extract_scgpt_embeddings_inductive

SEEDS = [17, 42, 109, 172]
DATA = {
    "Brain": ("data/Brain.h5ad", 22),
    "Fat": ("data/Fat.h5ad", 34),
    "Retina": ("data/Retina.h5ad", 71),
    "Breast_carcinoma": ("data/Breast_carcinoma.h5ad", 19),
    "Quadriceps_tendon": ("data/Quadriceps_tendon.h5ad", 22),
}
os.makedirs("scgpt_inputs", exist_ok=True)

for name, (path, k) in DATA.items():
    for s in SEEDS:
        sp = np.load(f"splits/{name}_s{s}.npz", allow_pickle=True)
        cell_ids = list(sp["cell_ids"]); y_full = sp["y_full"]

        adata = sc.read_h5ad(path)
        if "feature_name" in adata.var.columns:
            adata.var_names = adata.var["feature_name"].astype(str); adata.var_names_make_unique()
        sc.pp.filter_cells(adata, min_genes=200); sc.pp.filter_genes(adata, min_cells=3)
        adata = adata[cell_ids, :].copy()
        assert list(adata.obs_names) == cell_ids

        idx_lab, idx_unlab, idx_val, idx_test = create_stratified_splits(
            n_samples=len(y_full), labels=y_full, labeled_ratio=0.05,
            val_ratio=0.1, test_ratio=0.2, random_state=s)
        idx_train_all = np.concatenate([idx_lab, idx_unlab])

        # HVG on train only (same as scREFINE v2)
        adata_tr = adata[idx_train_all]
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata_tr, n_top_genes=2000, flavor="seurat")
        hvg = adata_tr.var["highly_variable"].reindex(adata.var_names).fillna(False).astype(bool)
        adata_h = adata[:, hvg].copy()

        emb_tr, emb_v, emb_te = extract_scgpt_embeddings_inductive(
            adata_h, train_indices=idx_train_all, val_indices=idx_val,
            test_indices=idx_test, random_state=s)

        E = np.zeros((len(y_full), emb_tr.shape[1]), dtype=np.float32)
        E[idx_train_all] = emb_tr
        E[idx_val] = emb_v
        E[idx_test] = emb_te

        out = f"scgpt_inputs/{name}_s{s}.h5"
        with h5py.File(out, "w") as f:
            f.create_dataset("X", data=E)
            f.create_dataset("Y", data=y_full.astype(np.int64))
            f.create_dataset("cell_names", data=np.array(cell_ids, dtype="S"))
            f.attrs["n_classes"] = len(np.unique(y_full))
        print(f"saved {out}  X={E.shape}, classes={len(np.unique(y_full))}")
print("DONE -> scgpt_inputs/")