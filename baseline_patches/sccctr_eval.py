"""
sccctr_eval.py — test-only evaluation of scCCTR final clusters, aligned with scREFINE splits.
Run AFTER main.py finishes:
  python3 sccctr_eval.py --splits_npz <split.npz> --clusters_csv <final_clusters_save_path>
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score)
from scipy.optimize import linear_sum_assignment

ap = argparse.ArgumentParser()
ap.add_argument("--splits_npz", required=True)
ap.add_argument("--clusters_csv", required=True)
args = ap.parse_args()

sp = np.load(args.splits_npz, allow_pickle=True)
res = pd.read_csv(args.clusters_csv)
res = res.set_index("cell_name").reindex(list(sp["cell_ids"]))
assert res["Cluster"].notna().all(), "cell alignment failed"
pred = res["Cluster"].astype(int).values
true = sp["y_full"]
te = sp["idx_test"]

def acc(t, p):
    D = max(t.max(), p.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(t)):
        w[p[i], t[i]] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return w[r, c].sum() / len(t)

print("Final result: ACC= %.4f, NMI= %.4f, AMI= %.4f, ARI= %.4f" % (
    acc(true[te], pred[te]),
    normalized_mutual_info_score(true[te], pred[te]),
    adjusted_mutual_info_score(true[te], pred[te]),
    adjusted_rand_score(true[te], pred[te])))