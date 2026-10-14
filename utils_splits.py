import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def create_stratified_splits(
    n_samples: int,
    labels: np.ndarray,
    labeled_ratio: float = 0.05,      # fraction of ALL cells
    val_ratio: float = 0.1,           # fraction of ALL cells
    test_ratio: float = 0.2,          # fraction of ALL cells
    random_state: int = 42,
    min_samples_per_class: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    unique_classes, class_counts = np.unique(labels, return_counts=True)
    rare_classes = unique_classes[class_counts < min_samples_per_class]

    valid_indices = np.arange(n_samples)
    labels_filtered = labels
    n_samples_filtered = n_samples

    def map_back(filtered_indices):
        return filtered_indices

    # First: test set
    idx_temp_filtered, idx_test_filtered = train_test_split(
        np.arange(n_samples_filtered),
        test_size=test_ratio,
        stratify=labels_filtered,
        random_state=random_state,
    )

    # Second: validation from remaining
    val_size_adjusted = val_ratio / (1 - test_ratio)
    idx_train_full_filtered, idx_val_filtered = train_test_split(
        idx_temp_filtered,
        test_size=val_size_adjusted,
        stratify=labels_filtered[idx_temp_filtered],
        random_state=random_state + 1,
    )

    # Third: labeled = labeled_ratio of ALL cells -> convert to fraction of train
    train_fraction = 1.0 - test_ratio - val_ratio
    labeled_size_adjusted = labeled_ratio / train_fraction
    assert labeled_size_adjusted < 1.0, "labeled_ratio must be smaller than train fraction"

    idx_train_labeled_filtered, idx_train_unlabeled_filtered = train_test_split(
        idx_train_full_filtered,
        test_size=1 - labeled_size_adjusted,
        stratify=labels_filtered[idx_train_full_filtered],
        random_state=random_state + 2,
    )

    idx_train_labeled = map_back(idx_train_labeled_filtered)
    idx_train_unlabeled = map_back(idx_train_unlabeled_filtered)
    idx_val = map_back(idx_val_filtered)
    idx_test = map_back(idx_test_filtered)

    print(f"📊 Data splits (labeled_ratio={labeled_ratio:.3f} of ALL cells):")
    print(f"   Labeled train:   {len(idx_train_labeled)} ({len(idx_train_labeled)/n_samples:.1%})")
    print(f"   Unlabeled train: {len(idx_train_unlabeled)} ({len(idx_train_unlabeled)/n_samples:.1%})")
    print(f"   Validation:      {len(idx_val)} ({len(idx_val)/n_samples:.1%})")
    print(f"   Test:            {len(idx_test)} ({len(idx_test)/n_samples:.1%})")

    return idx_train_labeled, idx_train_unlabeled, idx_val, idx_test


def create_partial_labels(y_full, idx_labeled, idx_unlabeled, unknown_label: int = -1):
    y_partial = np.full(len(y_full), unknown_label, dtype=int)
    y_partial[idx_labeled] = y_full[idx_labeled]
    return y_partial


def verify_no_leakage(idx_train_labeled, idx_train_unlabeled, idx_val, idx_test) -> bool:
    all_indices = np.concatenate([idx_train_labeled, idx_train_unlabeled, idx_val, idx_test])
    unique_indices = np.unique(all_indices)
    if len(all_indices) != len(unique_indices):
        raise ValueError("❌ Data leakage detected: overlapping indices in splits!")
    for name, idx in [("labeled", idx_train_labeled),
                      ("unlabeled", idx_train_unlabeled),
                      ("val", idx_val),
                      ("test", idx_test)]:
        if len(idx) == 0:
            raise ValueError(f"❌ Empty split: {name}")
    print("✅ No data leakage detected between splits")
    return True
