"""
Model components for scREFINE.

Includes the refinement head, semi-supervised objective, and
structure-aware training with SSD neighborhood regularization.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _class_weights_from_labels(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for CE."""
    valid = y[y != -1]
    if len(valid) == 0:
        return torch.ones(1, dtype=torch.float32)

    classes, counts = np.unique(valid, return_counts=True)
    freq = counts.astype(np.float32) / counts.sum()
    w = 1.0 / (freq + 1e-8)
    w = w / w.mean()
    out = torch.ones(int(classes.max() + 1), dtype=torch.float32)
    out[: len(w)] = torch.from_numpy(w)
    return out


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device

        valid_mask = labels != -1
        if valid_mask.sum() <= 1:
            return features.sum() * 0.0

        features = features[valid_mask]
        labels = labels[valid_mask]

        labels = labels.contiguous().view(-1, 1)
        n = features.shape[0]

        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.div(features @ features.T, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(n, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        loss = -mean_log_prob_pos.mean()
        return loss


class RefinementHead(nn.Module):
    """Refinement Head: scGPT -> 128-d."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.projector = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = F.normalize(self.projector(h), dim=1)
        return z


class SemiSupervisedLoss(nn.Module):
    """
    L_supervised = triplet + proto + pairwise + SupCon + CE
    L_structure = stability-weighted neighborhood consistency
    """

    def __init__(
        self,
        n_classes: int,
        temp: float = 0.07,
        margin: float = 0.5,
        w_triplet: float = 1.0,
        w_proto: float = 0.5,
        w_pairwise: float = 0.1,
        w_supcon: float = 0.5,
        w_ce: float = 1.0,
        w_structure: float = 0.3,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_classes = n_classes
        self.temp = temp
        self.margin = margin
        self.w_triplet = w_triplet
        self.w_proto = w_proto
        self.w_pairwise = w_pairwise
        self.w_supcon = w_supcon
        self.w_ce = w_ce
        self.w_structure = w_structure
        self.device = device

        self.prototypes = nn.Parameter(torch.randn(n_classes, 128, device=device) * 0.1)
        nn.init.xavier_uniform_(self.prototypes)

        self.supcon = SupConLoss(temperature=temp)

    def triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask_labeled: torch.Tensor,
    ) -> torch.Tensor:
        if mask_labeled.sum() < 3:
            return embeddings.sum() * 0.0

        z_labeled = embeddings[mask_labeled]
        y_labeled = labels[mask_labeled]

        loss = embeddings.sum() * 0.0
        n_triplets = 0

        for i in range(len(z_labeled)):
            anchor = z_labeled[i : i + 1]
            anchor_label = y_labeled[i]

            pos_mask = y_labeled == anchor_label
            pos_mask[i] = False
            neg_mask = y_labeled != anchor_label

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_idx = torch.where(pos_mask)[0]
            neg_idx = torch.where(neg_mask)[0]

            rand_pos = torch.randint(0, len(pos_idx), (1,), device=self.device)
            rand_neg = torch.randint(0, len(neg_idx), (1,), device=self.device)
            pos = z_labeled[pos_idx[rand_pos]]
            neg = z_labeled[neg_idx[rand_neg]]

            pos_dist = 1.0 - F.cosine_similarity(anchor, pos, dim=1)
            neg_dist = 1.0 - F.cosine_similarity(anchor, neg, dim=1)

            loss += F.relu(pos_dist - neg_dist + self.margin).mean()
            n_triplets += 1

        if n_triplets == 0:
            return embeddings.sum() * 0.0
        return loss / n_triplets

    def proto_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask_labeled: torch.Tensor,
    ) -> torch.Tensor:
        if mask_labeled.sum() == 0:
            return embeddings.sum() * 0.0

        z_labeled = embeddings[mask_labeled]
        y_labeled = labels[mask_labeled].to(self.device)

        proto_selected = self.prototypes[y_labeled]
        loss = 1.0 - F.cosine_similarity(z_labeled, proto_selected, dim=1).mean()

        proto_sim = torch.mm(self.prototypes, self.prototypes.t())
        eye_mask = torch.eye(self.n_classes, device=self.device)
        proto_sim_masked = proto_sim - eye_mask * 2.0
        push_loss = F.relu(proto_sim_masked).mean()

        return loss + 0.1 * push_loss

    def pairwise_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask_labeled: torch.Tensor,
    ) -> torch.Tensor:
        if mask_labeled.sum() < 2:
            return embeddings.sum() * 0.0

        z_labeled = embeddings[mask_labeled]
        y_labeled = labels[mask_labeled]

        sim_matrix = torch.mm(z_labeled, z_labeled.t())
        label_matrix = (y_labeled.unsqueeze(0) == y_labeled.unsqueeze(1)).float()

        pos_loss = (1.0 - sim_matrix) * label_matrix
        neg_loss = F.relu(sim_matrix) * (1.0 - label_matrix)

        n_pos = label_matrix.sum()
        n_neg = (1.0 - label_matrix).sum()

        loss = pos_loss.sum() / (n_pos + 1e-8) + neg_loss.sum() / (n_neg + 1e-8)
        return loss

    def ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask_labeled: torch.Tensor,
    ) -> torch.Tensor:
        if mask_labeled.sum() == 0:
            return logits.sum() * 0.0

        logits_labeled = logits[mask_labeled]
        labels_labeled = labels[mask_labeled]
        return F.cross_entropy(logits_labeled, labels_labeled)

    def structure_loss(
        self,
        embeddings: torch.Tensor,
        anchor_indices: np.ndarray,
        neighbor_indices: np.ndarray,
        edge_stability: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stability-weighted neighborhood consistency.
        Return is correctly outside the loop.
        """
        n_samples = neighbor_indices.shape[0]

        anchor_indices_t = torch.tensor(anchor_indices, dtype=torch.long, device=self.device)
        neighbor_indices_t = torch.tensor(neighbor_indices, dtype=torch.long, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        total_weight = torch.tensor(0.0, device=self.device)

        for i in range(n_samples):
            anchor_idx = anchor_indices_t[i]
            z_i = embeddings[anchor_idx : anchor_idx + 1]

            neighbors = neighbor_indices_t[i]
            weights = edge_stability[i]

            valid_mask = (neighbors >= 0) & (neighbors < embeddings.shape[0])
            if valid_mask.sum() == 0:
                continue

            neighbors_valid = neighbors[valid_mask]
            weights_valid = weights[valid_mask]

            z_neighbors = embeddings[neighbors_valid]
            dists = 2.0 * (1.0 - F.cosine_similarity(z_i, z_neighbors, dim=1))

            weighted_dist = (weights_valid * dists).sum()
            weight_sum = weights_valid.sum()

            total_loss += weighted_dist
            total_weight += weight_sum + 1e-8

        return total_loss / total_weight if total_weight.item() > 0 else embeddings.sum() * 0.0

    def forward(
        self,
        embeddings: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        neighbor_indices: np.ndarray = None,
        edge_stability: torch.Tensor = None,
        full_embeddings: torch.Tensor = None,
        anchor_indices: np.ndarray = None,
    ):
        mask_labeled = labels >= 0

        l_triplet = self.triplet_loss(embeddings, labels, mask_labeled)
        l_proto = self.proto_loss(embeddings, labels, mask_labeled)
        l_pairwise = self.pairwise_loss(embeddings, labels, mask_labeled)
        l_supcon = self.supcon(embeddings, labels) if self.w_supcon > 0 else embeddings.sum() * 0.0
        l_ce = self.ce_loss(logits, labels, mask_labeled)

        l_supervised = (
            self.w_triplet * l_triplet
            + self.w_proto * l_proto
            + self.w_pairwise * l_pairwise
            + self.w_supcon * l_supcon
            + self.w_ce * l_ce
        )

        if neighbor_indices is not None and edge_stability is not None and anchor_indices is not None:
            structure_source = full_embeddings if full_embeddings is not None else embeddings
            l_structure = self.w_structure * self.structure_loss(
                structure_source,
                anchor_indices,
                neighbor_indices,
                edge_stability,
            )
        else:
            l_structure = embeddings.sum() * 0.0

        total_loss = l_supervised + l_structure

        return {
            "total": total_loss,
            "supervised": l_supervised,
            "structure": l_structure,
            "triplet": l_triplet,
            "proto": l_proto,
            "pairwise": l_pairwise,
            "supcon": l_supcon,
            "ce": l_ce,
        }


class FusionTrainer:
    """Structure-aware trainer for the scREFINE model."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        ssd_layer,
        hidden_dim: int = 256,
        output_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.output_dim = output_dim
        self.device = device
        self.seed = seed
        self.ssd_layer = ssd_layer

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.refinement_head = RefinementHead(input_dim, hidden_dim, output_dim).to(device)
        self.classifier = nn.Linear(output_dim, n_classes).to(device)
        self.criterion = SemiSupervisedLoss(n_classes, device=device)

        params = (
            list(self.refinement_head.parameters())
            + list(self.classifier.parameters())
            + list(self.criterion.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def fit_transform(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        val_embeddings: np.ndarray = None,
        val_labels: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        structure-aware training with single forward pass per batch.
        Graph and stability are both used consistently with refined space.
        """
        n_samples = embeddings.shape[0]

        embeddings_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)

        print("Building initial SSD neighborhood graph...")
        neighbor_indices, neighbor_distances = self.ssd_layer.build_graph(embeddings)

        best_val_loss = float("inf")
        patience_counter = 0

        self.refinement_head.train()
        self.classifier.train()

        for epoch in range(epochs):
            perm = torch.randperm(n_samples, device=self.device)

            epoch_losses = {
                "total": 0.0,
                "supervised": 0.0,
                "structure": 0.0,
                "triplet": 0.0,
                "proto": 0.0,
                "pairwise": 0.0,
                "supcon": 0.0,
                "ce": 0.0,
            }
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                z_all = self.refinement_head(embeddings_t)

                batch_idx = perm[i : i + batch_size]
                z_batch = z_all[batch_idx]
                logits_batch = self.classifier(z_batch)
                batch_labels = labels_t[batch_idx]

                anchor_indices = batch_idx.detach().cpu().numpy()
                batch_neighbor_idx = neighbor_indices[anchor_indices]
                batch_neighbor_dist = neighbor_distances[anchor_indices]

                with torch.no_grad():
                    z_all_np = z_all.detach().cpu().numpy()
                    edge_stability = self.ssd_layer.compute_edge_stability_from_embeddings(
                        z_all_np,
                        anchor_indices,
                        batch_neighbor_idx,
                        batch_neighbor_dist,
                    )

                loss_dict = self.criterion(
                    z_batch,
                    logits_batch,
                    batch_labels,
                    neighbor_indices=batch_neighbor_idx,
                    edge_stability=edge_stability,
                    full_embeddings=z_all,
                    anchor_indices=anchor_indices,
                )

                self.optimizer.zero_grad()
                loss_dict["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.refinement_head.parameters()) + list(self.classifier.parameters()),
                    1.0,
                )
                self.optimizer.step()

                for k in epoch_losses:
                    val = loss_dict[k]
                    epoch_losses[k] += float(val.detach().item()) if torch.is_tensor(val) else float(val)
                n_batches += 1

            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            val_loss = None
            if val_embeddings is not None and val_labels is not None:
                self.refinement_head.eval()
                self.classifier.eval()

                with torch.no_grad():
                    val_emb_t = torch.tensor(val_embeddings, dtype=torch.float32, device=self.device)
                    val_labels_t = torch.tensor(val_labels, dtype=torch.long, device=self.device)
                    z_val = self.refinement_head(val_emb_t)
                    logits_val = self.classifier(z_val)
                    val_loss = F.cross_entropy(logits_val, val_labels_t).item()

                self.refinement_head.train()
                self.classifier.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                log_str = f"Epoch {epoch + 1}/{epochs} | "
                log_str += f"Total: {epoch_losses['total']:.4f} | "
                log_str += f"Sup: {epoch_losses['supervised']:.4f} | "
                log_str += f"Str: {epoch_losses['structure']:.4f}"
                if val_loss is not None:
                    log_str += f" | Val: {val_loss:.4f}"
                print(log_str)

            if (epoch + 1) % 20 == 0:
                self.refinement_head.eval()
                with torch.no_grad():
                    z_np = self.refinement_head(embeddings_t).cpu().numpy()
                self.refinement_head.train()

                print(f"Updating SSD graph at epoch {epoch + 1}...")
                neighbor_indices, neighbor_distances = self.ssd_layer.build_graph(z_np)

        self.refinement_head.eval()
        with torch.no_grad():
            final_embeddings = self.refinement_head(embeddings_t).cpu().numpy()

        return final_embeddings