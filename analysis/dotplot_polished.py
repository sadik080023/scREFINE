"""
dotplot_polished.py — paper-ready marker dotplot WITH colorbar + size legend.
Run: python3 dotplot_polished.py --data data/Brain.h5ad --emb result/Brain_s42/fused_embeddings.npy \
       --split-seed 42 --n-classes 22 --dataset-name Brain
(Brain.h5ad raw issue fixed: adata.raw cleared before rank_genes_groups)
"""
import argparse, os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn import metrics
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
ap.add_argument("--out", default=None)
args = ap.parse_args()
outdir = args.out or f"bioval_{args.dataset_name}_s{args.split_seed}_fig"
os.makedirs(outdir, exist_ok=True)

# rebuild labels + test split
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

Z = np.load(args.emb)
km = KMeans(n_clusters=args.n_classes, n_init=10, random_state=args.split_seed)
pred = km.fit_predict(Z[idx_test])
y_test = y_full[idx_test]

cont = metrics.cluster.contingency_matrix(y_test, pred)
r_ind, c_ind = linear_sum_assignment(-cont)
c2t = {c: unique_classes[t] for t,c in zip(r_ind,c_ind)}

adata_test = adata[idx_test].copy()
adata_test.obs["pred_cluster"] = [f"{p}: {c2t[p][:14]}" for p in pred]
sc.pp.normalize_total(adata_test, target_sum=1e4); sc.pp.log1p(adata_test)
adata_test.raw = None   # FIX: avoid raw-count usage in rank_genes_groups

known = KNOWN_MARKERS[args.dataset_name]
all_m = [m for ms in known.values() for m in ms if m in adata_test.var_names]
X = adata_test[:, all_m].X
X = X.toarray() if hasattr(X,"toarray") else np.asarray(X)
df = pd.DataFrame(X, columns=all_m); df["cluster"] = pred

rows=[]
for cl in sorted(set(pred)):
    sub = df[df.cluster==cl]; row={"cluster": f"{cl}: {c2t[cl][:14]}"}
    for m in all_m:
        v = sub[m].values
        row[m] = float(v.mean()); row[m+"_pct"] = float((v>0).mean()*100)
    rows.append(row)
kdf = pd.DataFrame(rows).set_index("cluster")
kdf.to_csv(f"{outdir}/known_markers_full.csv")

# polished dotplot
fig, ax = plt.subplots(figsize=(0.62*len(all_m)+2.2, 0.42*len(kdf)+1.6))
scat = ax.scatter(*np.meshgrid(np.arange(len(all_m)), np.arange(len(kdf))),
           s=kdf[[m+"_pct" for m in all_m]].values.flatten()*3.2,
           c=kdf[all_m].values.flatten(), cmap="RdBu_r", vmin=0, vmax=max(2.0, float(kdf[all_m].values.max())),
           edgecolors="grey", linewidths=0.3)
ax.set_xticks(range(len(all_m))); ax.set_xticklabels(all_m, rotation=90, fontsize=9)
ax.set_yticks(range(len(kdf))); ax.set_yticklabels(kdf.index, fontsize=8.5)
ax.set_title(f"{args.dataset_name}: canonical marker support per predicted cluster (test cells)", fontsize=10)
cb = fig.colorbar(scat, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("mean log-normalized expression", fontsize=8)
handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor='0.6', markersize=np.sqrt(p*3.2),
           label=f"{p}%") for p in [10, 50, 90]]
ax.legend(handles=handles, title="% cells expressing", loc="upper left",
          bbox_to_anchor=(1.18, 1.0), frameon=False, fontsize=8, title_fontsize=8)
plt.savefig(f"{outdir}/fig_marker_dotplot.png", dpi=300, bbox_inches="tight")
print("saved ->", f"{outdir}/fig_marker_dotplot.png")