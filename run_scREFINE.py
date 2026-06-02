"""
run_scREFINE.py 
Main orchestration script with:
- Inductive preprocessing (PCA/Scaler fit on train only)
- Clean batch forward (compute full embeddings once, slice for batch)
- True end-to-end structure loss
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import warnings
import argparse
import json
import time

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

    def cluster_accuracy(true_labels_inner, pred_labels_inner):
        cont = metrics.cluster.contingency_matrix(true_labels_inner, pred_labels_inner)
        row_ind, col_ind = linear_sum_assignment(-cont)
        return cont[row_ind, col_ind].sum() / np.sum(cont)

    acc = cluster_accuracy(true_labels, pred_labels)
    return {"ARI": ari, "NMI": nmi, "AMI": ami, "ACC": acc}


def load_and_preprocess_data(data_path: str):
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

    if "highly_variable" in adata.var.columns and adata.var["highly_variable"].sum() >= 200:
        print("   Detected existing HVG flags - reusing them")
        adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
        adata = adata[:, adata.var.highly_variable].copy()

    print(f"   Retained {adata.n_vars} highly variable genes")

    candidate_labels = ["Group", "cell_type1", "cell_type", "celltype",
                       "cell_ontology_class", "label", "Label", "cluster"]
    
    found = None
    for key in candidate_labels:
        if key in adata.obs.columns:
            found = key
            break

    if found:
        adata.obs["Group"] = adata.obs[found].astype(str)
        print(f"   Detected label column: '{found}'")
        print(f"   Total classes: {adata.obs['Group'].nunique()}")
    else:
        raise ValueError("No label column found!")

    return adata


def main():
    parser = argparse.ArgumentParser(description="scREFINE Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to .h5ad dataset")
    parser.add_argument("--labeled-ratio", type=float, default=0.2, help="Fraction of labels")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="output_clean_e2e", help="Output dir")
    parser.add_argument("--knn-k", type=int, default=15, help="K for SSD neighborhood graph")
    parser.add_argument("--w-structure", type=float, default=0.3, help="Weight for structure loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    print(
        f"\n===== Running experiment | seed={args.seed} | "
        f"labeled_ratio={args.labeled_ratio} | w_structure={args.w_structure} | data={args.data} ====="
    )

    runtime_log = {}
    t0_total = time.perf_counter()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("scREFINE Pipeline")
    print("   Inductive PCA/Scaler | Semi-supervised Refinement | Structure-aware Fusion")
    print("=" * 60)

    # [1/5] Load data
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

    # [2/5] Create stratified data splits
    print("\n[2/5] Creating stratified data splits...")
    t0 = time.perf_counter()
    
    # Rare-class filtering
    unique_classes_arr, class_counts = np.unique(y_full, return_counts=True)
    rare_classes = unique_classes_arr[class_counts < 2]
    if len(rare_classes) > 0:
        print(f"   Pre-filtering {len(rare_classes)} rare class(es) with < 2 samples")
        valid_mask = ~np.isin(y_full, rare_classes)
        adata = adata[valid_mask].copy()
        y_full = y_full[valid_mask]
        n_cells = adata.n_obs
        n_classes = len(np.unique(y_full))
        labels_str = labels_str[valid_mask]
        unique_classes = np.unique(labels_str)
        label_map = {v: i for i, v in enumerate(unique_classes)}
        y_full = np.array([label_map[v] for v in labels_str], dtype=int)
        n_cells = adata.n_obs
        n_classes = len(unique_classes)
        print(f"   Label range after filtering: {y_full.min()} to {y_full.max()}")
        print(f"   Filtered: {n_cells} cells, {n_classes} classes")
    
    idx_train_labeled, idx_train_unlabeled, idx_val, idx_test = create_stratified_splits(
        n_samples=n_cells,
        labels=y_full,
        labeled_ratio=args.labeled_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )
    verify_no_leakage(idx_train_labeled, idx_train_unlabeled, idx_val, idx_test)
    runtime_log["split_sec"] = round(time.perf_counter() - t0, 3)

    # Combine train indices for inductive preprocessing
    idx_train_all = np.concatenate([idx_train_labeled, idx_train_unlabeled])

    # [3/5] Extract scGPT embeddings (INDUCTIVE)
    print("\n[3/5] Extracting scGPT embeddings (INDUCTIVE: PCA/Scaler fit on train)...")
    t0 = time.perf_counter()
    
    emb_train, emb_val, emb_test = extract_scgpt_embeddings_inductive(
        adata, 
        train_indices=idx_train_all,
        val_indices=idx_val,
        test_indices=idx_test,
        random_state=args.seed
    )
    
    runtime_log["scgpt_embedding_sec"] = round(time.perf_counter() - t0, 3)
    
    print(f"   Train embeddings: {emb_train.shape}")
    print(f"   Val embeddings: {emb_val.shape if emb_val is not None else None}")
    print(f"   Test embeddings: {emb_test.shape}")

    # [4/5] Train with structure-aware batch forward
    print("\n[4/5] Training scREFINE model...")
    t0 = time.perf_counter()
    
    # Initialize SSD Structural Layer
    ssd_layer = SSDStructuralLayer(knn_k=args.knn_k, device="cpu")
    
    # Initialize Fusion Trainer
    trainer = FusionTrainer(
        input_dim=emb_train.shape[1],
        n_classes=n_classes,
        ssd_layer=ssd_layer,
        lr=args.lr,
        device="cpu",
        seed=args.seed
    )
    
    # Set structure loss weight
    trainer.criterion.w_structure = args.w_structure
    print(f"   Structural loss weight: {trainer.criterion.w_structure}")
    
    # Prepare validation data
    val_emb = emb_val
    val_labels = y_full[idx_val] if len(idx_val) > 0 else None
    
    # Prepare training labels
    y_partial = create_partial_labels(y_full, idx_train_labeled, idx_train_unlabeled)
    y_partial_train = y_partial[idx_train_all]
    
    # Train with clean batch forward
    final_emb_train = trainer.fit_transform(
        embeddings=emb_train,
        labels=y_partial_train,
        val_embeddings=val_emb,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=10,
        verbose=True
    )
    
    runtime_log["training_sec"] = round(time.perf_counter() - t0, 3)
    
    print(f"   Final training embeddings shape: {final_emb_train.shape}")

    # Transform val and test
    print("\n   Transforming validation and test sets...")
    trainer.refinement_head.eval()
    with torch.no_grad():
        if emb_val is not None:
            val_emb_tensor = torch.tensor(emb_val, dtype=torch.float32)
            final_emb_val = trainer.refinement_head(val_emb_tensor).cpu().numpy()
        else:
            final_emb_val = None
            
        test_emb_tensor = torch.tensor(emb_test, dtype=torch.float32)
        final_emb_test = trainer.refinement_head(test_emb_tensor).cpu().numpy()

    # Build full embedding array
    final_emb = np.zeros((n_cells, final_emb_train.shape[1]), dtype=np.float32)
    final_emb[idx_train_all] = final_emb_train
    if final_emb_val is not None:
        final_emb[idx_val] = final_emb_val
    final_emb[idx_test] = final_emb_test
    
    np.save(f"{args.output_dir}/fused_embeddings.npy", final_emb)
    print(f"   Full embeddings shape: {final_emb.shape}")


    # [5/5] Evaluation
    print("\n[5/5] Evaluation (TEST SET)...")
    t0 = time.perf_counter()

    X_test = final_emb[idx_test]
    y_test_true = y_full[idx_test]
    
    kmeans = KMeans(n_clusters=n_classes, n_init="auto", random_state=args.seed)
    y_test_pred = kmeans.fit_predict(X_test)
    
    metrics_test = evaluate_clustering(y_test_true, y_test_pred)
    
    print(f"\nTEST SET RESULTS ({len(idx_test)} cells):")
    print(f"   ARI:  {metrics_test['ARI']:.4f}")
    print(f"   NMI:  {metrics_test['NMI']:.4f}")
    print(f"   AMI:  {metrics_test['AMI']:.4f}")
    print(f"   ACC:  {metrics_test['ACC']:.4f}")
    
    if len(idx_val) > 0:
        X_val = final_emb[idx_val]
        y_val_true = y_full[idx_val]
        y_val_pred = kmeans.fit_predict(X_val)
        metrics_val = evaluate_clustering(y_val_true, y_val_pred)
        
        print(f"\nVALIDATION SET RESULTS ({len(idx_val)} cells):")
        print(f"   ARI:  {metrics_val['ARI']:.4f}")
        print(f"   NMI:  {metrics_val['NMI']:.4f}")
        print(f"   AMI:  {metrics_val['AMI']:.4f}")
        print(f"   ACC:  {metrics_val['ACC']:.4f}")
    else:
        metrics_val = None

    runtime_log["evaluation_sec"] = round(time.perf_counter() - t0, 3)
    runtime_log["total_runtime_sec"] = round(time.perf_counter() - t0_total, 3)
    runtime_log["total_runtime_min"] = round(runtime_log["total_runtime_sec"] / 60.0, 2)

    results = {
        "seed": args.seed,
        "labeled_ratio": args.labeled_ratio,
        "w_structure": args.w_structure,
        "n_classes": n_classes,
        "n_cells": n_cells,
        "test_metrics": metrics_test,
        "val_metrics": metrics_val,
        "splits": {
            "train_labeled": len(idx_train_labeled),
            "train_unlabeled": len(idx_train_unlabeled),
            "val": len(idx_val),
            "test": len(idx_test),
        },
        "hyperparams": {
            "knn_k": args.knn_k,
            "w_structure": args.w_structure,
            "lr": args.lr,
            "epochs": args.epochs,
        },
        "runtime": runtime_log,
    }
    
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nRUNTIME SUMMARY:")
    for k, v in runtime_log.items():
        if isinstance(v, (int, float)):
            print(f"   {k}: {v}")
    print(f"\nComplete! Results saved to {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()