#run_scgpt_sccctr_transformer
import os
import argparse
import random
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment

from Transformer import TransformerModel


class ScGPTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def evaluate(y_true, y_pred):
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        "ACC": cluster_acc(y_true, y_pred),
    }


def load_scgpt_h5(path):
    with h5py.File(os.path.expanduser(path), "r") as f:
        X = np.asarray(f["X"][:], dtype=np.float32)
        y = np.asarray(f["Y"][:], dtype=np.int64)
        n_classes = int(f.attrs["n_classes"])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, y, n_classes


def load_pseudo_labels(path):
    df = pd.read_csv(path)
    if "pred_cluster" not in df.columns:
        raise ValueError("Pseudo-label file must contain column: pred_cluster")
    return df["pred_cluster"].values.astype(np.int64)


def train_transformer(
    X,
    pseudo_y,
    true_y,
    num_classes,
    out_dir,
    seed=42,
    idx_test=None,
    epochs=80,
    batch_size=64,
    lr=1e-4,
    num_heads=8,
    num_layers=2,
    d_model=256,
    patience=10,
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = ScGPTDataset(X, pseudo_y)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    full_loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=batch_size, shuffle=False)

    class_counts = np.bincount(pseudo_y, minlength=num_classes)
    weights = class_counts.sum() / np.maximum(class_counts, 1)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model = TransformerModel(
        input_dim=X.shape[1],
        num_classes=num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        d_model=d_model,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)

        train_loss = total_loss / max(total_count, 1)

        model.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)
                val_count += xb.size(0)

        val_loss = val_loss / max(val_count, 1)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []

    with torch.no_grad():
        for xb in full_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.extend(logits.argmax(dim=1).cpu().numpy())

    preds = np.asarray(preds, dtype=np.int64)

    # PATCH: test-only evaluation
    yt = true_y[idx_test] if idx_test is not None else true_y
    pt = preds[idx_test] if idx_test is not None else preds
    scores = evaluate(yt, pt)

    pd.DataFrame({
        "cell_index": np.arange(len(true_y)),
        "true_label": true_y,
        "pseudo_label": pseudo_y,
        "transformer_pred": preds,
    }).to_csv(os.path.join(out_dir, "scgpt_sccctr_transformer_predictions.csv"), index=False)

    print("\nFinal Transformer result:")
    print(
        "ARI %.4f NMI %.4f AMI %.4f ACC %.4f"
        % (scores["ARI"], scores["NMI"], scores["AMI"], scores["ACC"])
    )

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--pseudo_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits_npz", type=str, default=None, help="PATCH: scREFINE split file")
    args = parser.parse_args()

    X, y, n_classes = load_scgpt_h5(args.data_path)
    pseudo_y = load_pseudo_labels(args.pseudo_path)

    idx_test = None
    if args.splits_npz:
        sp = np.load(args.splits_npz, allow_pickle=True)
        assert X.shape[0] == len(sp['y_full']), f"cell count mismatch"
        idx_test = sp['idx_test']
        print(f"PATCH: scREFINE splits loaded, test={len(idx_test)}")

    if len(pseudo_y) != len(y):
        raise ValueError(f"Pseudo labels length {len(pseudo_y)} != true labels length {len(y)}")

    num_classes = args.num_classes if args.num_classes is not None else n_classes

    print("=" * 70)
    print("scGPT + scCCTR Transformer stage")
    print("data:", args.data_path)
    print("pseudo:", args.pseudo_path)
    print("X:", X.shape)
    print("true classes:", n_classes)
    print("num_classes:", num_classes)
    print("=" * 70)

    train_transformer(
        X=X,
        pseudo_y=pseudo_y,
        true_y=y,
        num_classes=num_classes,
        out_dir=args.out_dir,
        seed=args.seed,
        idx_test=idx_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_model=args.d_model,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()