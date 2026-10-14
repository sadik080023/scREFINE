"""
ssd_clustering.py
SSD Structural Layer for new architecture:
    - Builds local neighborhood graph on refined/intermediate embeddings
    - Estimates per-edge stability (not scalar map)
    - Outputs soft structural weights (not hard pseudo-labels)
    - Identifies trustworthy regions of embedding space
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


class SSDStructuralLayer:
    """
    SSD Structural Layer: Per-edge / neighbor stability estimation.

    Builds local neighborhood graph on refined embeddings.
    Outputs soft structural weights (not hard pseudo-labels).
    """

    def __init__(
        self,
        knn_k: int = 15,
        stability_temp: float = 0.1,
        min_confidence: float = 0.3,
        device: str = "cpu",
    ):
        """
        Args:
            knn_k: Number of neighbors in graph
            stability_temp: Temperature for stability sigmoid (lower = sharper threshold)
            min_confidence: Minimum cosine similarity for "stable" edge
            device: torch device
        """
        self.knn_k = knn_k
        self.stability_temp = stability_temp
        self.min_confidence = min_confidence
        self.device = device

    def build_graph(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-NN graph on embeddings.

        Args:
            embeddings: (N, D) array

        Returns:
            neighbor_indices: (N, K) array of neighbor indices
            neighbor_distances: (N, K) array of cosine distances
        """
        nnbr = NearestNeighbors(n_neighbors=self.knn_k + 1, metric="cosine")
        nnbr.fit(embeddings)
        distances, indices = nnbr.kneighbors(embeddings)

        # Remove self (first neighbor is always self)
        return indices[:, 1:], distances[:, 1:]

    def compute_edge_stability(
        self,
        embeddings: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
    ) -> torch.Tensor:
        """
        Original stability computation from cosine distances.
        Kept for backward compatibility.

        Stability = sigmoid((cosine_similarity - min_confidence) / temp)

        Args:
            embeddings: (N, D) array
            neighbor_indices: (N, K) neighbor indices
            neighbor_distances: (N, K) cosine distances

        Returns:
            stability: (N, K) tensor
        """
        cosine_sims = 1.0 - torch.tensor(
            neighbor_distances, dtype=torch.float32, device=self.device
        )
        stability = torch.sigmoid(
            (cosine_sims - self.min_confidence) / self.stability_temp
        )
        return stability

    def compute_edge_stability_from_embeddings(
        self,
        embeddings: np.ndarray,
        batch_anchor_indices: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute edge stability directly from embeddings in refined space.

        This is the preferred method for the structure-aware version because it
        keeps the graph/stability/structure components in the same representation space.

        Args:
            embeddings: Full embedding matrix, shape (N, D)
            batch_anchor_indices: Global indices of anchors for the current batch, shape (B,)
            neighbor_indices: Neighbor indices for each anchor, shape (B, K)
            neighbor_distances: Distances (unused, kept for API compatibility)

        Returns:
            edge_stability: Tensor of shape (B, K) with nonnegative stability weights
        """
        n_anchors = len(batch_anchor_indices)
        k = neighbor_indices.shape[1]
        edge_stability = np.zeros((n_anchors, k), dtype=np.float32)

        for i, anchor_global_idx in enumerate(batch_anchor_indices):
            anchor_emb = embeddings[anchor_global_idx]
            neighbors = neighbor_indices[i]

            valid_mask = (neighbors >= 0) & (neighbors < embeddings.shape[0])
            if valid_mask.sum() == 0:
                continue

            neighbors_valid = neighbors[valid_mask]

            # Cosine similarity in embedding space
            anchor_norm = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-8)
            neighbor_embs = embeddings[neighbors_valid]
            neighbor_norms = neighbor_embs / (
                np.linalg.norm(neighbor_embs, axis=1, keepdims=True) + 1e-8
            )

            similarities = np.dot(neighbor_norms, anchor_norm)

            # Clamp negatives so stability stays nonnegative
            similarities = np.maximum(similarities, 0.0)

            edge_stability[i, valid_mask] = similarities

        return torch.tensor(edge_stability, dtype=torch.float32, device=self.device)

    def compute_local_consistency(
        self,
        embeddings: np.ndarray,
        neighbor_indices: np.ndarray,
        edge_stability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute local consistency scores for each node.

        Args:
            embeddings: (N, D) array
            neighbor_indices: (N, K) neighbor indices
            edge_stability: (N, K) edge weights

        Returns:
            consistency_scores: (N,) weighted average distance to neighbors
            total_weights: (N,) sum of edge weights per node
        """
        embeddings_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        n_samples, _ = neighbor_indices.shape

        total_weight = torch.zeros(n_samples, device=self.device)
        consistency_scores = torch.zeros(n_samples, device=self.device)

        for i in range(n_samples):
            z_i = embeddings_t[i : i + 1]
            neighbors = neighbor_indices[i]
            z_neighbors = embeddings_t[neighbors]

            dists = 1.0 - F.cosine_similarity(z_i, z_neighbors, dim=1)
            weights = edge_stability[i]

            if weights.sum() > 0:
                consistency = (weights * dists).sum() / (weights.sum() + 1e-8)
                total_weight[i] = weights.sum()
                consistency_scores[i] = consistency
            else:
                total_weight[i] = 0.0
                consistency_scores[i] = 0.0

        return consistency_scores, total_weight

    def get_trustworthy_mask(
        self,
        embeddings: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Get boolean mask of trustworthy (high stability) nodes.

        Args:
            embeddings: (N, D) array
            neighbor_indices: (N, K) neighbor indices
            neighbor_distances: (N, K) cosine distances
            threshold: Minimum average edge stability to be "trustworthy"

        Returns:
            mask: (N,) boolean array
        """
        edge_stability = self.compute_edge_stability(
            embeddings, neighbor_indices, neighbor_distances
        )
        avg_stability = edge_stability.mean(dim=1).cpu().numpy()
        return avg_stability >= threshold