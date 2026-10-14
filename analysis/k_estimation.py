"""
k_estimation.py — oracle-K follow-up experiment (Reviewer #1 Major-2)
Estimates cluster count K from refined embeddings WITHOUT ground truth
(bootstrap stability + silhouette, method adapted from scFUSE's StabilityBasedClustering),
then evaluates ARI at the ESTIMATED K.

Usage:
  python3 k_estimation.py --data data/Brain.h5ad --emb result/Brain_s42/fused_embeddings.npy \
      --split-seed 42 --true-k 22
"""
import argparse, json, time
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
import sys
sys.path.insert(0, ".")
from utils_splits import create_stratified_splits


def stability_score(X, k, n_boot=5, random_state=42):
    n = X.shape[0]
    boot = int(0.8 * n)
    results = []
    for i in range(n_boot):
        rng = np.random.RandomState(random_state + i)
        idx = rng.choice(n, size=boot, replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state + i)
        lab = km.fit_predict(X[idx])
        results.append(dict(zip(idx.tolist(), lab.tolist())))
    scores = []
    for i in range(n_boot):
        for j in range(i + 1, n_boot):
            common = set(results[i]) & set(results[j])
            if len(common) < 10:
                continue
            li = [results[i][c] for c in common]
            lj = [results[j][c] for c in common]
            scores.append(adjusted_rand_score(li, lj))
    return float(np.mean(scores)) if scores else 0.0


ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--emb", required=True)
ap.add_argument("--split-seed", type=int, required=True)
ap.add_argument("--true-k", type=int, required=True)
ap.add_argument("--window", type=int, default=8, help="search true_k +- window")
ap.add_argument("--out", default=None)
args = ap.parse_args()

# rebuild labels + test split (identical to other analysis scripts)
adata = sc.read_h5ad(args.data)
if "feature_name" in adata.var.columns:
    adata.var_names = adata.var["feature_name"].astype(str); adata.var_names_make_unique()
sc.pp.filter_cells(adata, min_genes=200); sc.pp.filter_genes(adata, min_cells=3)
for key in ["Group","cell_type1","cell_type","celltype","cell_ontology_class","label","Label","cluster"]:
    if key in adata.obs.columns:
        adata.obs["Group"] = adata.obs[key].astype(str); break
labels_str = adata.obs["Group"].astype(str).values
unique_classes = np.unique(labels_str)
y_full = np.array([{v:i for i,v in enumerate(unique_classes)}[v] for v in labels_str])
uc, cc = np.unique(y_full, return_counts=True)
rare = uc[cc < 2]
if len(rare):
    m = ~np.isin(y_full, rare); y_full = y_full[m]

_,_,_,idx_test = create_stratified_splits(n_samples=len(y_full), labels=y_full,
    labeled_ratio=0.05, val_ratio=0.1, test_ratio=0.2, random_state=args.split_seed)

Z = np.load(args.emb)
X_test = Z[idx_test]; y_test = y_full[idx_test]

k_range = list(range(max(2, args.true_k - args.window), args.true_k + args.window + 1))
print(f"Searching K in {k_range[0]}..{k_range[-1]} (true K={args.true_k}), n_test={len(X_test)}")

stabs, sils, times = [], [], []
for k in k_range:
    t0 = time.perf_counter()
    st = stability_score(X_test, k, n_boot=5, random_state=args.split_seed)
    km = KMeans(n_clusters=k, n_init=10, random_state=args.split_seed)
    lab = km.fit_predict(X_test)
    si = silhouette_score(X_test, lab) if len(np.unique(lab)) > 1 else -1.0
    stabs.append(st); sils.append(si)
    times.append(round(time.perf_counter() - t0, 1))
    print(f"  K={k:3d}: stability={st:.3f} silhouette={si:.3f} ({times[-1]}s)")

sn = np.array(stabs); ln = np.array(sils)
sn = (sn - sn.min()) / (sn.max() - sn.min() + 1e-8) if sn.max() > sn.min() else sn
ln = (ln - ln.min()) / (ln.max() - ln.min() + 1e-8) if ln.max() > ln.min() else ln
combined = 0.6 * sn + 0.4 * ln
est_k = k_range[int(np.argmax(combined))]
print(f"Estimated K={est_k} (true K={args.true_k})")

km = KMeans(n_clusters=est_k, n_init=10, random_state=args.split_seed)
pred = km.fit_predict(X_test)
ari = adjusted_rand_score(y_test, pred); nmi = normalized_mutual_info_score(y_test, pred)

out = {"true_k": args.true_k, "estimated_k": int(est_k),
       "k_range": k_range, "stability": stabs, "silhouette": sils,
       "combined": [round(float(c),4) for c in combined],
       "ARI_at_estimated_K": round(ari,4), "NMI_at_estimated_K": round(nmi,4)}
op = args.out or args.emb.replace("fused_embeddings.npy","k_estimation.json")
json.dump(out, open(op,"w"), indent=2)
print(f"ARI at estimated K={est_k}: {ari:.4f}  (saved -> {op})")