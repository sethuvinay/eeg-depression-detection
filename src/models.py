"""PyTorch models: per-modality CNN encoders, cross-modal GNN fusion,
self-supervised pretraining head, and the supervised classifier.

Data flow for one windowed sample ``x`` of shape ``(B, M, T)``
(B = batch, M = modalities, T = time steps):

    per-modality 1D-CNN encoders  ->  (B, M, D) embeddings
    cross-modal GNN (message passing over learned modality affinities)
                                  ->  (B, D') fused representation
    linear classifier             ->  (B, n_classes) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ModalityEncoder(nn.Module):
    """1D-CNN encoder turning one modality's window into a fixed embedding.

    Three strided conv blocks halve the temporal resolution each time, then
    adaptive average pooling collapses whatever remains to a single vector —
    so the encoder is robust to the exact window length.
    """

    def __init__(self, emb_dim: int = 128, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.projection = nn.Linear(base_channels * 4, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) -> (B, emb_dim)
        x = x.unsqueeze(1)  # add channel dim
        return self.projection(self.net(x))


class CrossModalGNN(nn.Module):
    """Graph fusion over modalities.

    Each modality embedding is a graph node. Edges carry *learned* affinities
    (a softmax-normalised parameter matrix), so the model discovers which
    modality pairs inform each other — e.g. ECG <-> respiration coupling under
    stress — instead of us hard-coding the graph. Two GCN layers propagate
    information, then mean pooling over nodes gives the fused vector.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 n_nodes: int = 4):
        super().__init__()
        self.n_nodes = n_nodes
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        # Learnable dense affinity matrix between modality nodes.
        self.edge_logits = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        self.dropout = nn.Dropout(dropout)

    def _edge_index_and_weight(self, device) -> tuple:
        affinity = F.softmax(self.edge_logits, dim=-1)  # rows sum to 1
        idx = torch.arange(self.n_nodes, device=device)
        edge_index = torch.stack(
            [idx.repeat_interleave(self.n_nodes), idx.repeat(self.n_nodes)]
        )
        edge_weight = affinity.reshape(-1)
        return edge_index, edge_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) modality embeddings -> (B, hidden_dim) fused
        edge_index, edge_weight = self._edge_index_and_weight(x.device)
        outs = []
        for b in range(x.size(0)):
            h = x[b]
            for conv in self.convs:
                h = F.relu(conv(h, edge_index, edge_weight))
                h = self.dropout(h)
            outs.append(h.mean(dim=0))  # mean pool over modality nodes
        return torch.stack(outs)


class ProjectionHead(nn.Module):
    """Non-linear projection used only during contrastive pretraining."""

    def __init__(self, in_dim: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalAffectNet(nn.Module):
    """Full model: encoders -> GNN fusion -> classifier (+ SSL head)."""

    def __init__(self, modalities=("ECG", "EDA", "EMG", "Resp"),
                 emb_dim: int = 128, gnn_hidden: int = 128,
                 n_classes: int = 3):
        super().__init__()
        self.modalities = tuple(modalities)
        self.encoders = nn.ModuleDict(
            {m: ModalityEncoder(emb_dim=emb_dim) for m in self.modalities}
        )
        self.gnn = CrossModalGNN(emb_dim, hidden_dim=gnn_hidden,
                                 n_nodes=len(self.modalities))
        self.classifier = nn.Linear(gnn_hidden, n_classes)
        self.ssl_head = ProjectionHead(gnn_hidden)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M, T) -> (B, gnn_hidden) fused representation."""
        embs = torch.stack(
            [self.encoders[m](x[:, i]) for i, m in enumerate(self.modalities)],
            dim=1,
        )
        return self.gnn(embs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M, T) -> (B, n_classes) logits."""
        return self.classifier(self.encode(x))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Contrastive projection of the fused representation."""
        return F.normalize(self.ssl_head(self.encode(x)), dim=-1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """Normalised temperature-scaled cross-entropy (SimCLR-style).

    ``z1`` and ``z2`` are L2-normalised projections of two augmented views of
    the same batch; the i-th pair is positive, all others negative.
    """
    batch = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                       # (2B, D)
    sim = torch.mm(z, z.t()) / temperature                # (2B, 2B)
    # Mask out self-similarity.
    mask = torch.eye(2 * batch, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)
    # Positive for row i is row (i + B) % 2B.
    pos_idx = (torch.arange(2 * batch, device=z.device) + batch) % (2 * batch)
    positives = sim[torch.arange(2 * batch, device=z.device), pos_idx]
    logsumexp = torch.logsumexp(sim, dim=1)
    return (logsumexp - positives).mean()
