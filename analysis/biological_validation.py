"""
biological_validation.py — Reviewer #1 Major-8
Predicted-cluster marker analysis (no retraining).

What it produces (per dataset, per seed):
  1. cluster_predictions_<ds>_s<seed>.csv  — test cells: true type, predicted cluster, matched type
  2. top_markers_<ds>_s<seed>.csv          — per cluster: size, matched true type, top-10 DE marker genes
  3. known_markers_<ds>_s<seed>.csv        — per cluster: mean expression & pct-expressing of KNOWN markers
  4. known_markers_<ds>_s<seed>.png        — dotplot figure (paper-ready)

Usage:
  python3 biological_validation.py --data data/Brain.h5ad \
      --emb result/Brain_s42/fused_embeddings.npy --split-seed 42 --n-classes 22 --dataset-name Brain
"""
import argparse, json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import sys
sys.path.insert(0, ".")
from utils_splits import create_stratified_splits

KNOWN_MARKERS = {
 "Brain": {
   "Neuron": ["RBFOX3","SNAP25","STMN2"],
   "Astrocyte": ["AQP4","GFAP","APOE"],
   "Oligodendrocyte": ["MBP","PLP1","MOG"],
   "OPC": ["PDGFRA","VCAN"],
   "Microglia": ["P2RY12","CSF1R","C3"],
   "Endothelial": ["CLDN5","FLT1","VWF"],
   "Pericyte": ["RGS5","PDGFRB"],
 },
 "Breast_carcinoma": {
   "T cell": ["CD3D","CD3E"],
   "B cell": ["CD79A","MS4A1"],
   "Macrophage": ["CD68","CD14","LYZ"],
   "NK cell": ["NKG7","GNLY"],
   "Epithelial": ["EPCAM","KRT8","KRT18","KRT19"],
   "Fibroblast": ["COL1A1","DCN","LUM"],
   "Endothelial": ["PECAM1","VWF"],
   "Mast cell": ["TPSAB1","CPA3"],
 },
}

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--emb", required=True)
ap.add_argument("--split-seed", type=int, required=True)
ap.add_argument("--n-classes", type=int, required=True)
ap.add_argument("--dataset-name", required=True)
ap.add_argument("--n-init", type=int, default=10)
ap.add_argument("--out-dir", default=None)
args = ap.parse_args()

outdir = args.out_dir or f"bioval_{args.dataset_name}_s{args.split_seed}"
import os; os.makedirs(outdir, exist_ok=True)

# ---------- rebuild labels + test split (same as k_sensitivity.py) ----------
adata = sc.read_h5ad(args.data)
if "feature_name" in adata.var.columns:
    adata.var_names = adata.var["feature_name"].astype(str); adata.var_names_make_unique()
sc.pp.filter_cells(adata, min_genes=200); sc.pp.filter_genes(adata, min_cells=3)
found=None
for key in ["Group","cell_type1","cell_type","celltype","cell_ontology_class","label","Label","cluster"]:
    if key in adata.obs.columns:
        adata.obs["Group"]=adata.obs[key].astype(str); found=key; break
labels_str = adata.obs["Group"].astype(str).values
unique_classes = np.unique(labels_str)
y_full = np.array([{v:i for i,v in enumerate(unique_classes)}[v] for v in labels_str])
uc, cc = np.unique(y_full, return_counts=True)
rare = uc[cc<2]
if len(rare):
    m=~np.isin(y_full,rare); adata=adata[m].copy(); y_full=y_full[m]; labels_str=labels_str[m]
    unique_classes=np.unique(labels_str)

_,_,_,idx_test = create_stratified_splits(n_samples=len(y_full), labels=y_full,
    labeled_ratio=0.05, val_ratio=0.1, test_ratio=0.2, random_state=args.split_seed)

# ---------- predicted clusters on test cells ----------
Z = np.load(args.emb)
X_test = Z[idx_test]; y_test = y_full[idx_test]
km = KMeans(n_clusters=args.n_classes, n_init=args.n_init, random_state=args.split_seed)
pred = km.fit_predict(X_test)

# match clusters -> true types (Hungarian, same as ACC)
from sklearn import metrics
cont = metrics.cluster.contingency_matrix(y_test, pred)
r_ind, c_ind = linear_sum_assignment(-cont)
cluster2type = {c: unique_classes[t] for t, c in zip(r_ind, c_ind)}

pd.DataFrame({"cell_index": idx_test,
              "true_type": [unique_classes[i] for i in y_test],
              "pred_cluster": pred,
              "matched_type": [cluster2type[p] for p in pred]}
            ).to_csv(f"{outdir}/cluster_predictions.csv", index=False)
acc = cont[r_ind, c_ind].sum()/cont.sum()
print(f"Cluster->type ACC on test: {acc:.4f}")

# ---------- marker gene analysis (test cells only) ----------
adata_test = adata[idx_test].copy()
adata_test.obs["true_type"] = [unique_classes[i] for i in y_test]
adata_test.obs["pred_cluster"] = [str(p) for p in pred]
sc.pp.normalize_total(adata_test, target_sum=1e4); sc.pp.log1p(adata_test)

adata_test.raw = None 
sc.tl.rank_genes_groups(adata_test, groupby="pred_cluster", method="t-test", n_genes=10)
top = sc.get.rank_genes_groups_df(adata_test, group=None)
top_rows=[]
for cl in sorted(set(pred), key=lambda x: str(x)):
    sub = top[top.group==str(cl)].head(10)
    top_rows.append({"cluster": cl, "matched_type": cluster2type[cl],
                     "n_cells": int((pred==cl).sum()),
                     "top_markers": "; ".join(sub["names"].tolist())})
pd.DataFrame(top_rows).to_csv(f"{outdir}/top_markers.csv", index=False)

# ---------- known canonical markers ----------
known = KNOWN_MARKERS.get(args.dataset_name, {})
if known:
    all_m = [m for ms in known.values() for m in ms if m in adata_test.var_names]
    missing = [m for ms in known.values() for m in ms if m not in adata_test.var_names]
    if missing: print("NOTE: markers not in data:", missing)
    X = adata_test[:, all_m].X
    X = X.toarray() if hasattr(X,"toarray") else np.asarray(X)
    df = pd.DataFrame(X, columns=all_m)
    df["cluster"] = pred
    rows=[]
    for cl in sorted(set(pred)):
        sub = df[df.cluster==cl]
        row={"cluster": cl, "matched_type": cluster2type[cl]}
        for m in all_m:
            v = sub[m].values
            row[f"{m}_mean"] = round(float(v.mean()),3)
            row[f"{m}_pct"] = round(float((v>0).mean()*100),1)
        rows.append(row)
    kdf = pd.DataFrame(rows)
    kdf.to_csv(f"{outdir}/known_markers.csv", index=False)

    # dotplot: mean expression as dot size=pct, color=mean (paper-ready)
    import matplotlib.pyplot as plt
    long=[]
    for _,row in kdf.iterrows():
        for m in all_m:
            long.append({"cluster": f"{row.cluster}:{row.matched_type[:12]}",
                         "marker": m, "mean": row[f"{m}_mean"], "pct": row[f"{m}_pct"]})
    L = pd.DataFrame(long)
    fig, ax = plt.subplots(figsize=(0.55*len(all_m)+3, 0.5*len(kdf)+2))
    sc_mat = L.pivot(index="cluster", columns="marker", values="pct")
    col_mat = L.pivot(index="cluster", columns="marker", values="mean")
    ax.scatter(*np.meshgrid(np.arange(len(all_m)), np.arange(len(kdf))), 
               s=sc_mat.values.flatten()*3, c=col_mat.values.flatten(),
               cmap="Reds", vmin=0, edgecolors="grey", linewidths=0.3)
    ax.set_xticks(range(len(all_m))); ax.set_xticklabels(all_m, rotation=90)
    ax.set_yticks(range(len(kdf))); ax.set_yticklabels(sc_mat.index)
    ax.set_title(f"{args.dataset_name} — known markers per predicted cluster (seed {args.split_seed})")
    plt.tight_layout(); plt.savefig(f"{outdir}/known_markers.png", dpi=200)
    print("saved dotplot ->", f"{outdir}/known_markers.png")

print("ALL outputs in", outdir)
