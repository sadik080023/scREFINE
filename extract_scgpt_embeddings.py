import os
import json
import torch
import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from scGPT.scgpt.model.model import TransformerModel
except ImportError:
    try:
        from scgpt.scgpt.model.model import TransformerModel
    except ImportError:
        from scgpt.model.model import TransformerModel


def extract_scgpt_embeddings_inductive(adata, train_indices, val_indices, test_indices, 
                                        pca_dim: int = 128, random_state: int = 42):
    """
    Extract cell-level embeddings using frozen scGPT gene embeddings.
    INDUCTIVE version: PCA and StandardScaler fitted ONLY on training data.
    
    Args:
        adata: Full AnnData object
        train_indices: Indices for training set (PCA/Scaler fit on this)
        val_indices: Indices for validation set
        test_indices: Indices for test set
        pca_dim: Target dimension after PCA
        random_state: Random seed for PCA
    
    Returns:
        emb_train, emb_val, emb_test: Embeddings for each split with consistent transform
    """
    print("[Embedding] Generating scGPT embeddings (INDUCTIVE: PCA/Scaler fit on train only)...")

    ckpt_path = "scGPT_pretrained/best_model.pt"
    vocab_path = "scGPT_pretrained/vocab.json"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    # ── Step 1: Load vocab ─────────────────────────────────────────────────────
    with open(vocab_path) as f:
        vocab_json = json.load(f)

    if isinstance(vocab_json, dict):
        if "itos" in vocab_json:
            tokens = vocab_json["itos"]
        elif "gene_vocab" in vocab_json:
            tokens = vocab_json["gene_vocab"]
        elif "vocab" in vocab_json:
            tokens = vocab_json["vocab"]
        else:
            tokens = list(vocab_json.keys())
    elif isinstance(vocab_json, list):
        tokens = vocab_json
    else:
        raise ValueError("Unrecognized vocab file format")

    pad_token = "<pad>"
    if pad_token not in tokens:
        tokens = [pad_token] + tokens

    vocab_dict = {tok: i for i, tok in enumerate(tokens)}
    print(f"   ✓ Vocab size: {len(tokens)} tokens")

    # ── Step 2: Load model, extract frozen gene embedding matrix ──────────────
    state = torch.load(ckpt_path, map_location="cpu")

    model = TransformerModel(
        ntoken=len(tokens),
        d_model=512,
        nhead=8,
        d_hid=512,
        nlayers=6,
        dropout=0.1,
        pad_token=pad_token,
        vocab=vocab_dict,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"   ✓ Model loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()

    with torch.no_grad():
        emb_matrix = model.encoder.embedding.weight.cpu().numpy()

    # ── Step 3: Gene overlap ───────────────────────────────────────────────────
    gene_names = list(adata.var_names)
    overlap = [g for g in gene_names if g in vocab_dict]

    if len(overlap) == 0:
        raise ValueError(
            "No overlapping genes found between dataset and vocab!\n"
            "Check gene name format — vocab uses gene symbols (e.g. 'GAPDH'), "
            "not Ensembl IDs (e.g. 'ENSG00000111640')."
        )

    overlap_pct = 100.0 * len(overlap) / len(gene_names)
    print(f"   ✓ Gene overlap: {len(overlap)}/{len(gene_names)} ({overlap_pct:.1f}%)")

    if overlap_pct < 20.0:
        print(
            f"   ⚠ WARNING: Low gene overlap ({overlap_pct:.1f}%). "
            "Embeddings may be poor. Check gene naming convention."
        )

    overlap_vocab_idx = [vocab_dict[g] for g in overlap]
    overlap_data_idx = [gene_names.index(g) for g in overlap]

    gene_embs = emb_matrix[overlap_vocab_idx].astype(np.float32)

    # ── Step 4: Build log-normalized expression matrix for ALL cells ──────────
    X_raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    X_overlap = X_raw[:, overlap_data_idx]

    cell_totals = X_overlap.sum(axis=1, keepdims=True)
    X_lognorm = np.log1p(X_overlap / (cell_totals + 1e-8) * 1e4)

    print(f"   ✓ Log-normalized expression matrix: {X_lognorm.shape}")

    # ── Step 5: Expression-weighted average of gene embeddings ────────────────
    cell_embs = X_lognorm @ gene_embs
    weight_sums = X_lognorm.sum(axis=1, keepdims=True)
    cell_embs = cell_embs / (weight_sums + 1e-8)

    norms = np.linalg.norm(cell_embs, axis=1, keepdims=True)
    cell_embs = cell_embs / (norms + 1e-8)

    print(f"   ✓ Cell embeddings (L2-norm): {cell_embs.shape}")

    # ── Step 6: Split into train/val/test ─────────────────────────────────────
    cell_embs_train = cell_embs[train_indices]
    cell_embs_val = cell_embs[val_indices] if len(val_indices) > 0 else None
    cell_embs_test = cell_embs[test_indices]

    # ── Step 7: Fit PCA ONLY on training data ─────────────────────────────────
    max_dim = min(len(train_indices) - 1, cell_embs.shape[1])
    pca_dim_use = min(pca_dim, max_dim)

    if pca_dim_use < pca_dim:
        print(f"   ⚠ pca_dim capped {pca_dim} → {pca_dim_use} (n_train limit)")

    pca = PCA(n_components=pca_dim_use, random_state=random_state)
    pca.fit(cell_embs_train)  # Fit on train only!
    
    # Transform all splits with same PCA
    cell_embs_pca_train = pca.transform(cell_embs_train)
    cell_embs_pca_val = pca.transform(cell_embs_val) if cell_embs_val is not None else None
    cell_embs_pca_test = pca.transform(cell_embs_test)
    
    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(
        f"   ✓ PCA: {cell_embs.shape[1]}D → {pca_dim_use}D "
        f"({var_explained:.1f}% variance explained) [fit on train]"
    )

    # ── Step 8: Fit StandardScaler ONLY on training data ──────────────────────
    scaler = StandardScaler()
    scaler.fit(cell_embs_pca_train)  # Fit on train only!
    
    # Transform all splits with same scaler
    cell_embs_final_train = scaler.transform(cell_embs_pca_train)
    cell_embs_final_val = scaler.transform(cell_embs_pca_val) if cell_embs_pca_val is not None else None
    cell_embs_final_test = scaler.transform(cell_embs_pca_test)

    print(f"   ✓ Final embeddings: train={cell_embs_final_train.shape}, "
          f"val={cell_embs_final_val.shape if cell_embs_final_val is not None else None}, "
          f"test={cell_embs_final_test.shape} (StandardScaled) [fit on train]")

    return cell_embs_final_train, cell_embs_final_val, cell_embs_final_test
