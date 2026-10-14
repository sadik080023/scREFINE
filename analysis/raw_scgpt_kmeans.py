"""
raw_scgpt_kmeans.py — Raw pretrained scGPT embeddings + KMeans (Table 3 row 2).
NO refinement training. Uses the EXACT same inductive extraction pipeline as
scREFINE (HVG on train only, PCA/Scaler fit on train only), then KMeans on test cells.
Usage: python3 raw_scgpt_kmeans.py --data data/Brain.h5ad --seed 42 --n-classes 22
"""
import argparse, json, time
import numpy as np
import scanpy as sc
import sys
sys.path.insert(0, ".")
from utils_splits import create_stratified_splits
from extract_scgpt_embeddings import extract_scgpt_embeddings_inductive
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score)
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def evaluate(true, pred):
    ari = adjusted_rand_score(true, pred)
    nmi = normalized_mutual_info_score(true, pred)
    ami = adjusted_mutual_info_score(true, pred)
    cont = metrics.cluster.contingency_matrix(true, pred)
    r, c = linear_sum_assignment(-cont)
    acc = cont[r, c].sum() / cont.sum()
    return {"ARI": ari, "NMI": nmi, "AMI": ami, "ACC": acc}

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n-classes", type=int, required=True)
ap.add_argument("--n-repeats", type=int, default=5)
ap.add_argument("--n-init", type=int, default=10)
args = ap.parse_args()

adata = sc.read_h5ad(args.data)
if "feature_name" in adata.var.columns:
    adata.var_names = adata.var["feature_name"].astype(str); adata.var_names_make_unique()
sc.pp.filter_cells(adata, min_genes=200); sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
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
    m=~np.isin(y_full,rare); adata=adata[m].copy(); labels_str=labels_str[m]
    unique_classes=np.unique(labels_str)
    y_full=np.array([{v:i for i,v in enumerate(unique_classes)}[v] for v in labels_str])

idx_lab, idx_unlab, idx_val, idx_test = create_stratified_splits(
    n_samples=len(y_full), labels=y_full, labeled_ratio=0.05, val_ratio=0.1,
    test_ratio=0.2, random_state=args.seed)
idx_train_all = np.concatenate([idx_lab, idx_unlab])

# HVG on train only (same as scREFINE v2)
adata_train = adata[idx_train_all]
sc.pp.highly_variable_genes(adata_train, n_top_genes=2000, flavor="seurat")
hvg = adata_train.var["highly_variable"].reindex(adata.var_names).fillna(False).astype(bool)
adata = adata[:, hvg].copy()

emb_train, emb_val, emb_test = extract_scgpt_embeddings_inductive(
    adata, train_indices=idx_train_all, val_indices=idx_val, test_indices=idx_test,
    random_state=args.seed)

X_test = emb_test; y_test = y_full[idx_test]
allm=[]
for r in range(args.n_repeats):
    km = KMeans(n_clusters=args.n_classes, n_init=args.n_init, random_state=args.seed+r)
    pred = km.fit_predict(X_test)
    allm.append(evaluate(y_test, pred))
mean_m = {k: round(float(np.mean([m[k] for m in allm])),4) for k in allm[0]}
std_m  = {k: round(float(np.std([m[k] for m in allm])),4) for k in allm[0]}
print("RAW scGPT + KMeans:", mean_m, "+-", std_m)
out = {"seed": args.seed, "mean": mean_m, "std": std_m}
op = f"result/{args.data.split('/')[-1].replace('.h5ad','')}_s{args.seed}/raw_scgpt_kmeans.json"
import os; os.makedirs(os.path.dirname(op), exist_ok=True)
json.dump(out, open(op,"w"), indent=2)
print("saved ->", op)