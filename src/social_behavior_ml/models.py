from __future__ import annotations

import torch
from torch import nn


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return self.linear(adj_norm @ x)


class GCNLinkPredictor(nn.Module):
    """Small graph auto-encoder for user interaction prediction."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.gcn1(x, adj_norm))
        h = self.dropout(h)
        z = self.gcn2(h, adj_norm)
        return z

    def score_edges(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        return (z[src] * z[dst]).sum(dim=1)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, adj_norm)
        return self.score_edges(z, edge_index)


class EventMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    n = adj.shape[0]
    adj = adj + torch.eye(n, device=adj.device)
    degree = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(degree.clamp(min=1.0), -0.5)
    return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]

