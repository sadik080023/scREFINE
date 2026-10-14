#run_scgpt_sccctr_direct
import argparse
import os
import random
import h5py
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def load_scgpt_h5(path):
    with h5py.File(os.path.expanduser(path), "r") as f:
        X = np.asarray(f["X"][:], dtype=np.float32)
        y = np.asarray(f["Y"][:], dtype=np.int64)
        n_classes = int(f.attrs["n_classes"])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, y, n_classes


def evaluate(y, pred):
    return {
        "ARI": adjusted_rand_score(y, pred),
        "NMI": normalized_mutual_info_score(y, pred),
        "AMI": adjusted_mutual_info_score(y, pred),
        "ACC": cluster_acc(y, pred),
    }


def run_direct_consensus(data_path, out_dir, num_clusters=None, max_iterations=6, seed=42, splits_npz=None):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    X, y, n_classes = load_scgpt_h5(data_path)
    if num_clusters is None:
        num_clusters = n_classes

    # PATCH: exact scREFINE v2 splits — test-only evaluation
    idx_test = None
    if splits_npz:
        sp = np.load(splits_npz, allow_pickle=True)
        assert X.shape[0] == len(sp['y_full']), f"cell count mismatch: {X.shape[0]} vs {len(sp['y_full'])}"
        idx_test = sp['idx_test']
        print(f"PATCH: scREFINE splits loaded, test={len(idx_test)}")

    print("=" * 70)
    print("scGPT + scCCTR direct consensus clustering")
    print("data:", data_path)
    print("X:", X.shape)
    print("classes:", n_classes)
    print("num_clusters:", num_clusters)
    print("max_iterations:", max_iterations)
    print("=" * 70)

    all_clusterings = []
    previous_stable = None
    stable_clusters = None

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")

        # KMeans with different random_state per iteration to build consensus.
        km = KMeans(
            n_clusters=num_clusters,
            init="k-means++",
            n_init=20,
            random_state=seed + iteration,
        )
        clusters = km.fit_predict(X)
        all_clusterings.append(clusters)

        consensus_matrix = np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
        for c in all_clusterings:
            consensus_matrix += (c[:, None] == c[None, :]).astype(np.float32)
        consensus_matrix /= len(all_clusterings)

        distance_matrix = 1.0 - consensus_matrix

        agg = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric="precomputed",
            linkage="average",
        )
        stable_clusters = agg.fit_predict(distance_matrix)
        all_clusterings[-1] = stable_clusters

        yt = y[idx_test] if idx_test is not None else y  # PATCH
        pt = stable_clusters[idx_test] if idx_test is not None else stable_clusters
        scores = evaluate(yt, pt)
        print(
            "Stable clustering: ARI %.4f NMI %.4f AMI %.4f ACC %.4f"
            % (scores["ARI"], scores["NMI"], scores["AMI"], scores["ACC"])
        )

        if previous_stable is not None:
            ari_change = adjusted_rand_score(previous_stable, stable_clusters)
            print("ARI between current and previous stable clusters: %.4f" % ari_change)
            if ari_change > 0.95:
                print("Early stop: consensus clusters converged.")
                break

        previous_stable = stable_clusters.copy()

    yt = y[idx_test] if idx_test is not None else y  # PATCH
    pt = stable_clusters[idx_test] if idx_test is not None else stable_clusters
    scores = evaluate(yt, pt)

    pd.DataFrame({
        "cell_index": np.arange(len(y)),
        "true_label": y,
        "pred_cluster": stable_clusters,
    }).to_csv(os.path.join(out_dir, "scgpt_sccctr_direct_consensus_clusters.csv"), index=False)

    print("\nFinal result:")
    print(
        "ARI %.4f NMI %.4f AMI %.4f ACC %.4f"
        % (scores["ARI"], scores["NMI"], scores["AMI"], scores["ACC"])
    )

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_clusters", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits_npz", type=str, default=None, help="PATCH: scREFINE split file")
    args = parser.parse_args()

    run_direct_consensus(
        data_path=args.data_path,
        out_dir=args.out_dir,
        num_clusters=args.num_clusters,
        max_iterations=args.max_iterations,
        seed=args.seed,
        splits_npz=args.splits_npz,
    )


if __name__ == "__main__":
    main()