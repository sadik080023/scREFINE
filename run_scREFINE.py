"""
run_scREFINE.py (revision v2)
- HVG selected on train cells only (no test/val leakage)
- --split-seed independent of --seed
- CLI control of supervised loss weights (ablation-ready)
- KMeans: n_init=10, plus repeated fits for mean±SD
- Auto-logs every run to results_log_v2.csv
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import warnings
import argparse
import json
import time
import csv
from datetime import datetime

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from extract_scgpt_embeddings import extract_scgpt_embeddings_inductive
from model import RefinementHead, FusionTrainer
from ssd_clustering import SSDStructuralLayer
from utils_splits import create_stratified_splits, create_partial_labels, verify_no_leakage


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_clustering(true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ami = adjusted_mutual_info_score(true_labels, pred_labels)

    def cluster_accuracy(t, p):
        cont = metrics.cluster.contingency_matrix(t, p)
        row_ind, col_ind = linear_sum_assignment(-cont)
        return cont[row_ind, col_ind].sum() / np.sum(cont)

    acc = cluster_accuracy(true_labels, pred_labels)
    return {"ARI": ari, "NMI": nmi, "AMI": ami, "ACC": acc}


def load_and_preprocess_data(data_path: str):
    """QC + normalization only. HVG selection is deferred to AFTER the split
    so that it is fitted on training cells exclusively."""
    print("[1/5] Loading and preprocessing data...")

    adata = sc.read_h5ad(data_path)

    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str)
        adata.var_names_make_unique()
        print("   Using adata.var['feature_name'] as gene symbols")

    adata.var_names_make_unique()
    print(f"   Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Per-cell operations only (no cross-cell information -> no leakage)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    candidate_labels = ["Group", "cell_type1", "cell_type", "celltype",
                        "cell_ontology_class", "label", "Label", "cluster"]
    found = None
    for key in candidate_labels:
        if key in adata.obs.columns:
            found = key
            break
    if found:
        adata.obs["Group"] = adata.obs[found].astype(str)
        print(f"   Detected label column: '{found}' ({adata.obs['Group'].nunique()} classes)")
    else:
        raise ValueError("No label column found!")

    return adata


def select_hvg_train_only(adata, train_indices, n_top_genes: int = 2000):
    """Fit highly-variable-gene selection on TRAIN cells only, apply mask to all."""
    print(f"   Selecting HVGs on TRAIN cells only (n_top_genes={n_top_genes})...")
    adata_train = adata[train_indices]
    sc.pp.highly_variable_genes(adata_train, n_top_genes=n_top_genes, flavor="seurat")
    hvg_mask = adata_train.var["highly_variable"].reindex(adata.var_names).fillna(False).astype(bool)
    adata_out = adata[:, hvg_mask].copy()
    print(f"   Retained {adata_out.n_vars} highly variable genes (train-only fit)")
    return adata_out


def repeated_kmeans_eval(X, y_true, n_clusters, base_seed, n_repeats=5, n_init=10):
    """KMeans with n_init restarts, repeated with different seeds -> mean+-SD."""
    all_metrics = []
    for r in range(n_repeats):
        km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=base_seed + r)
        pred = km.fit_predict(X)
        all_metrics.append(evaluate_clustering(y_true, pred))
    mean_m = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    std_m = {k: float(np.std([m[k] for m in all_metrics])) for k in all_metrics[0]}
    return mean_m, std_m


def main():
    parser = argparse.ArgumentParser(description="scREFINE Pipeline (v2)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--labeled-ratio", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42, help="model/init seed")
    parser.add_argument("--split-seed", type=int, default=None, help="data-split seed (default: =seed)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="output_v2")
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--hard-pl", action="store_true", help="hard pseudo-label structural variant")
    parser.add_argument("--no-graph", action="store_true", help="refinement-only: no structure graph at all")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--w-structure", type=float, default=0.1)
    # supervised loss weights (ablation switches)
    parser.add_argument("--w-triplet", type=float, default=1.0)
    parser.add_argument("--w-proto", type=float, default=0.5)
    parser.add_argument("--w-pairwise", type=float, default=0.1)
    parser.add_argument("--w-supcon", type=float, default=0.5)
    parser.add_argument("--w-ce", type=float, default=1.0)
    # KMeans protocol
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--n-kmeans-repeats", type=int, default=5)
    parser.add_argument("--hvg-n-top", type=int, default=2000)
    parser.add_argument("--results-log", type=str, default="results_log_v2.csv")
    args = parser.parse_args()

    split_seed = args.split_seed if args.split_seed is not None else args.seed
    dataset_name = os.path.splitext(os.path.basename(args.data))[0]

    print(f"\n===== v2 | data={dataset_name} | seed={args.seed} | split_seed={split_seed} | "
          f"labeled={args.labeled_ratio} | w_structure={args.w_structure} "
          f"| losses(tri={args.w_triplet},proto={args.w_proto},pair={args.w_pairwise},"
          f"supcon={args.w_supcon},ce={args.w_ce}) =====")

    runtime_log = {}
    t0_total = time.perf_counter()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.makedirs(args.output_dir, exist_ok=True)

    # [1/5] Load
    t0 = time.perf_counter()
    adata = load_and_preprocess_data(args.data)
    runtime_log["load_preprocess_sec"] = round(time.perf_counter() - t0, 3)
    n_cells = adata.n_obs

    labels_str = adata.obs["Group"].astype(str).values
    unique_classes = np.unique(labels_str)
    n_classes = len(unique_classes)
    label_map = {v: i for i, v in enumerate(unique_classes)}
    y_full = np.array([label_map[v] for v in labels_str], dtype=int)
    print(f"\nDataset: {n_cells} cells, {n_classes} classes")

    # [2/5] Splits (split_seed independent)
    print("\n[2/5] Creating stratified data splits...")
    t0 = time.perf_counter()

    unique_classes_arr, class_counts = np.unique(y_full, return_counts=True)
    rare_classes = unique_classes_arr[class_counts < 2]
    if len(rare_classes) > 0:
        print(f"   Pre-filtering {len(rare_classes)} rare class(es) with < 2 samples")
        valid_mask = ~np.isin(y_full, rare_classes)
        adata = adata[valid_mask].copy()
        y_full = y_full[valid_mask]
        labels_str = labels_str[valid_mask]
        unique_classes = np.unique(labels_str)
        label_map = {v: i for i, v in enumerate(unique_classes)}
        y_full = np.array([label_map[v] for v in labels_str], dtype=int)
        n_cells = adata.n_obs
        n_classes = len(unique_classes)
        print(f"   Filtered: {n_cells} cells, {n_classes} classes")

    idx_train_labeled, idx_train_unlabeled, idx_val, idx_test = create_stratified_splits(
        n_samples=n_cells, labels=y_full,
        labeled_ratio=args.labeled_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        random_state=split_seed,
    )
    verify_no_leakage(idx_train_labeled, idx_train_unlabeled, idx_val, idx_test)
    runtime_log["split_sec"] = round(time.perf_counter() - t0, 3)

    idx_train_all = np.concatenate([idx_train_labeled, idx_train_unlabeled])

    # [2.5] HVG on TRAIN ONLY (leakage-safe)
    adata = select_hvg_train_only(adata, idx_train_all, n_top_genes=args.hvg_n_top)

    # [3/5] scGPT embeddings (inductive PCA/Scaler)
    print("\n[3/5] Extracting scGPT embeddings (INDUCTIVE)...")
    t0 = time.perf_counter()
    emb_train, emb_val, emb_test = extract_scgpt_embeddings_inductive(
        adata, train_indices=idx_train_all, val_indices=idx_val, test_indices=idx_test,
        random_state=args.seed)
    runtime_log["scgpt_embedding_sec"] = round(time.perf_counter() - t0, 3)
    print(f"   Train: {emb_train.shape} | Val: {emb_val.shape if emb_val is not None else None} | Test: {emb_test.shape}")

    # [4/5] Train
    print("\n[4/5] Training scREFINE model...")
    t0 = time.perf_counter()
    ssd_layer = SSDStructuralLayer(knn_k=args.knn_k, device="cpu")
    trainer = FusionTrainer(input_dim=emb_train.shape[1], n_classes=n_classes,
                            ssd_layer=ssd_layer, lr=args.lr, weight_decay=args.weight_decay,
                            device="cpu", seed=args.seed, hard_pl=args.hard_pl, no_graph=args.no_graph)

    trainer.criterion.w_structure = args.w_structure
    trainer.criterion.w_triplet = args.w_triplet
    trainer.criterion.w_proto = args.w_proto
    trainer.criterion.w_pairwise = args.w_pairwise
    trainer.criterion.w_supcon = args.w_supcon
    trainer.criterion.w_ce = args.w_ce
    print(f"   w_structure={trainer.criterion.w_structure} | "
          f"w(tri,proto,pair,supcon,ce)=({args.w_triplet},{args.w_proto},{args.w_pairwise},{args.w_supcon},{args.w_ce})")

    y_partial = create_partial_labels(y_full, idx_train_labeled, idx_train_unlabeled)
    y_partial_train = y_partial[idx_train_all]
    val_labels = y_full[idx_val] if len(idx_val) > 0 else None

    final_emb_train = trainer.fit_transform(
        embeddings=emb_train, labels=y_partial_train,
        val_embeddings=emb_val, val_labels=val_labels,
        epochs=args.epochs, batch_size=args.batch_size, patience=10, verbose=True)
    runtime_log["training_sec"] = round(time.perf_counter() - t0, 3)

    trainer.refinement_head.eval()
    with torch.no_grad():
        final_emb_val = trainer.refinement_head(torch.tensor(emb_val, dtype=torch.float32)).numpy() if emb_val is not None else None
        final_emb_test = trainer.refinement_head(torch.tensor(emb_test, dtype=torch.float32)).numpy()

    final_emb = np.zeros((n_cells, final_emb_train.shape[1]), dtype=np.float32)
    final_emb[idx_train_all] = final_emb_train
    if final_emb_val is not None:
        final_emb[idx_val] = final_emb_val
    final_emb[idx_test] = final_emb_test
    np.save(f"{args.output_dir}/fused_embeddings.npy", final_emb)

    # [5/5] Evaluation: repeated KMeans -> mean+-SD
    print(f"\n[5/5] Evaluation (TEST, KMeans n_init={args.n_init}, repeats={args.n_kmeans_repeats})...")
    t0 = time.perf_counter()
    metrics_test, metrics_test_std = repeated_kmeans_eval(
        final_emb[idx_test], y_full[idx_test], n_classes, args.seed,
        n_repeats=args.n_kmeans_repeats, n_init=args.n_init)
    metrics_val, metrics_val_std = (repeated_kmeans_eval(
        final_emb[idx_val], y_full[idx_val], n_classes, args.seed,
        n_repeats=args.n_kmeans_repeats, n_init=args.n_init) if len(idx_val) > 0 else (None, None))

    print("\nTEST (mean over KMeans repeats):")
    for k, v in metrics_test.items():
        print(f"   {k}: {v:.4f} ± {metrics_test_std[k]:.4f}")
    runtime_log["evaluation_sec"] = round(time.perf_counter() - t0, 3)
    runtime_log["total_runtime_min"] = round((time.perf_counter() - t0_total) / 60.0, 2)

    results = {
        "dataset": dataset_name, "seed": args.seed, "split_seed": split_seed,
        "labeled_ratio": args.labeled_ratio, "w_structure": args.w_structure,
        "loss_weights": {"triplet": args.w_triplet, "proto": args.w_proto,
                         "pairwise": args.w_pairwise, "supcon": args.w_supcon, "ce": args.w_ce},
        "n_classes": n_classes, "n_cells": n_cells,
        "test_metrics_mean": metrics_test, "test_metrics_std": metrics_test_std,
        "val_metrics_mean": metrics_val, "val_metrics_std": metrics_val_std,
        "splits": {"train_labeled": len(idx_train_labeled), "train_unlabeled": len(idx_train_unlabeled),
                   "val": len(idx_val), "test": len(idx_test)},
        "kmeans_protocol": {"n_init": args.n_init, "n_repeats": args.n_kmeans_repeats},
        "runtime": runtime_log,
    }
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Append to master log CSV
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name, "seed": args.seed, "split_seed": split_seed,
        "labeled_ratio": args.labeled_ratio, "w_structure": args.w_structure,
        "w_triplet": args.w_triplet, "w_proto": args.w_proto, "w_pairwise": args.w_pairwise,
        "w_supcon": args.w_supcon, "w_ce": args.w_ce,
        "n_train_labeled": len(idx_train_labeled), "n_train_unlabeled": len(idx_train_unlabeled),
        "n_val": len(idx_val), "n_test": len(idx_test),
        "ARI": metrics_test["ARI"], "NMI": metrics_test["NMI"],
        "AMI": metrics_test["AMI"], "ACC": metrics_test["ACC"],
        "ARI_km_std": metrics_test_std["ARI"],
        "runtime_min": runtime_log["total_runtime_min"],
        "output_dir": args.output_dir,
    }
    log_exists = os.path.exists(args.results_log)
    with open(args.results_log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not log_exists:
            w.writeheader()
        w.writerow(row)

    print("\nRUNTIME:", {k: v for k, v in runtime_log.items()})
    print(f"Complete! -> {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()