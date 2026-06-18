from __future__ import annotations

import numpy as np


def node_structural_features(adj: np.ndarray) -> np.ndarray:
    """Build simple node features from graph structure."""
    out_degree = adj.sum(axis=1)
    in_degree = adj.sum(axis=0)
    total_degree = out_degree + in_degree

    two_hop = (adj @ adj).sum(axis=1)
    clustering_proxy = np.zeros_like(total_degree)
    for i in range(adj.shape[0]):
        neighbors = np.where((adj[i] + adj[:, i]) > 0)[0]
        k = len(neighbors)
        if k > 1:
            sub = adj[np.ix_(neighbors, neighbors)]
            clustering_proxy[i] = sub.sum() / (k * (k - 1))

    features = np.stack([in_degree, out_degree, total_degree, two_hop, clustering_proxy], axis=1)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    return ((features - mean) / std).astype(np.float32)


def pair_heuristic_features(adj: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Classic interpretable features for link prediction baselines."""
    undirected = ((adj + adj.T) > 0).astype(np.float32)
    degree = undirected.sum(axis=1)
    rows = []
    for i, j in pairs:
        ni = undirected[i] > 0
        nj = undirected[j] > 0
        common = float(np.logical_and(ni, nj).sum())
        union = float(np.logical_or(ni, nj).sum())
        jaccard = common / union if union else 0.0
        pref_attach = float(degree[i] * degree[j])
        cosine = common / np.sqrt(max(degree[i] * degree[j], 1.0))
        rows.append([common, jaccard, pref_attach, cosine, degree[i], degree[j]])
    x = np.asarray(rows, dtype=np.float32)
    return x


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    return (train_x - mean) / std, (test_x - mean) / std

