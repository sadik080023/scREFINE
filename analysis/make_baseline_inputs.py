"""
make_baseline_inputs.py — create baseline input files ALIGNED with scREFINE v2 splits.
Every file preserves the exact cell order of splits/*.npz (same filter chain as export_splits.py).
Outputs to baseline_inputs/:
  {D}.h5                    -> X (raw counts), Y (int codes), cell_names   [scSemiAAE + scAGCL]
  {D}_sccctr.csv            -> genes x cells raw counts (index=gene names)  [scCCTR]
  {D}_labels.csv            -> Cell_Name, Cluster (true codes)             [scCCTR]
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import h5py

DATA = {
    "Brain": "data/Brain.h5ad",
    "Fat": "data/Fat.h5ad",
    "Retina": "data/Retina.h5ad",
    "Breast_carcinoma": "data/Breast_carcinoma.h5ad",
    "Quadriceps_tendon": "data/Quadriceps_tendon.h5ad",
}
os.makedirs("baseline_inputs", exist_ok=True)

for name, path in DATA.items():
    sp = np.load(f"splits/{name}_s17.npz", allow_pickle=True)
    cell_ids = list(sp["cell_ids"])
    y_full = sp["y_full"]

    adata = sc.read_h5ad(path)
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str)
        adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # align to npz order (same chain -> same set, but enforce order anyway)
    adata = adata[cell_ids, :].copy()
    assert list(adata.obs_names) == cell_ids, f"{name}: cell order mismatch!"

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = np.nan_to_num(X).astype(np.float32)

    # h5 for scSemiAAE / scAGCL
    with h5py.File(f"baseline_inputs/{name}.h5", "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=y_full.astype(np.int64))
        f.create_dataset("cell_names", data=np.array(cell_ids, dtype="S"))

    # CSV for scCCTR (genes x cells)
    df = pd.DataFrame(X.T, index=list(adata.var_names), columns=cell_ids)
    df.to_csv(f"baseline_inputs/{name}_sccctr.csv")
    pd.DataFrame({"Cell_Name": cell_ids, "Cluster": y_full}).to_csv(
        f"baseline_inputs/{name}_labels.csv", index=False)
    print(f"{name}: {X.shape[0]} cells x {X.shape[1]} genes, classes={len(np.unique(y_full))}")

print("DONE -> baseline_inputs/")