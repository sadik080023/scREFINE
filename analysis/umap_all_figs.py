"""
umap_all_figs.py — ALL UMAP figures for the revised paper.
  Fig 3 : Brain/Retina/Breast  raw vs refined (test cells, seed 42)
  Fig S1: Fat / Quadriceps     raw vs refined
  Fig S2: same common scGPT embedding colored by true type (5 datasets)
Run from ~/scREFINE/scREFINE_for_review_experiments:
  python3 umap_all_figs.py
Outputs -> umap_figs/
"""
import os
import numpy as np
import h5py
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")
from utils_splits import create_stratified_splits
from extract_scgpt_embeddings import extract_scgpt_embeddings_inductive

S = 42
DATA = {"Brain": ("data/Brain.h5ad", 22), "Fat": ("data/Fat.h5ad", 34),
        "Retina": ("data/Retina.h5ad", 71), "Breast_carcinoma": ("data/Breast_carcinoma.h5ad", 19),
        "Quadriceps_tendon": ("data/Quadriceps_tendon.h5ad", 22)}
FIG3 = ["Brain", "Retina", "Breast_carcinoma"]
SI1 = ["Fat", "Quadriceps_tendon"]
os.makedirs("umap_figs", exist_ok=True)

def umap_coords(E, seed=42):
    ad = sc.AnnData(E)
    sc.pp.neighbors(ad, n_neighbors=15, use_rep="X")
    sc.tl.umap(ad, random_state=seed)
    return ad.obsm["X_umap"]

def plot_umap(X2, y, names, title, out):
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    cats = np.unique(y)
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(cats), 20)))
    for i, c in enumerate(cats):
        m = y == c
        ax.scatter(X2[m, 0], X2[m, 1], s=3, color=cmap[i], label=names[c], alpha=0.75, linewidths=0)
    ax.legend(fontsize=4.5, markerscale=2.5, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, borderaxespad=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved", out)

for name, k in DATA.items():
    sp = np.load(f"splits/{name}_s{S}.npz", allow_pickle=True)
    cell_ids = list(sp["cell_ids"]); y_full = sp["y_full"]
    names = np.array(sp["class_names"], dtype=str)
    path, _ = DATA[name]
    adata = sc.read_h5ad(path)
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str); adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200); sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[cell_ids, :].copy()

    idx_lab, idx_unlab, idx_val, idx_test = create_stratified_splits(
        n_samples=len(y_full), labels=y_full, labeled_ratio=0.05, val_ratio=0.1,
        test_ratio=0.2, random_state=S)
    idx_tr = np.concatenate([idx_lab, idx_unlab])

    adata_tr = adata[idx_tr]
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata_tr, n_top_genes=2000, flavor="seurat")
    hvg = adata_tr.var["highly_variable"].reindex(adata.var_names).fillna(False).astype(bool)
    adata_h = adata[:, hvg].copy()

    emb_tr, emb_v, emb_te = extract_scgpt_embeddings_inductive(
        adata_h, train_indices=idx_tr, val_indices=idx_val, test_indices=idx_test, random_state=S)
    E_raw = np.zeros((len(y_full), emb_tr.shape[1]), dtype=np.float32)
    E_raw[idx_tr] = emb_tr; E_raw[idx_val] = emb_v; E_raw[idx_test] = emb_te

    Z = np.load(f"result/{name}_s{S}/fused_embeddings.npy")
    y_test = y_full[idx_test]

    # Fig S2 panel: common embedding colored by true type (all 5 datasets)
    X2 = umap_coords(E_raw[idx_test])
    plot_umap(X2, y_test, names, f"{name}: common scGPT embedding", f"umap_figs/Fig.S2_{name}.png")

    tag = {"Brain": "a_Brain", "Retina": "b_Retina", "Breast_carcinoma": "c_Breast",
           "Fat": "a_Fat", "Quadriceps_tendon": "b_Quadriceps"}[name]
    if name in FIG3 or name in SI1:
        plot_umap(umap_coords(E_raw[idx_test]), y_test, names,
                  f"{name}: raw scGPT", f"umap_figs/Fig.3{tag}_raw.png")
        plot_umap(umap_coords(Z[idx_test]), y_test, names,
                  f"{name}: scREFINE refined", f"umap_figs/Fig.3{tag}_refined.png")
print("ALL DONE -> umap_figs/")