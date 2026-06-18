from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GraphData:
    num_nodes: int
    edges: list[tuple[int, int]]
    node_names: list[str]


def load_edge_csv(path: str | Path) -> GraphData:
    """Load an edge list CSV with at least source and target columns."""
    path = Path(path)
    node_to_id: dict[str, int] = {}
    edges: list[tuple[int, int]] = []

    def get_id(name: str) -> int:
        if name not in node_to_id:
            node_to_id[name] = len(node_to_id)
        return node_to_id[name]

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "source" not in reader.fieldnames or "target" not in reader.fieldnames:
            raise ValueError("CSV must contain source and target columns.")
        for row in reader:
            s = row["source"].strip()
            t = row["target"].strip()
            if not s or not t or s == t:
                continue
            edges.append((get_id(s), get_id(t)))

    node_names = [""] * len(node_to_id)
    for name, idx in node_to_id.items():
        node_names[idx] = name
    return GraphData(len(node_names), sorted(set(edges)), node_names)


def make_synthetic_social_graph(
    num_nodes: int = 180,
    communities: int = 4,
    p_in: float = 0.08,
    p_out: float = 0.012,
    seed: int = 42,
) -> GraphData:
    """Generate a directed social graph with community structure."""
    rng = random.Random(seed)
    labels = [i % communities for i in range(num_nodes)]
    edges: set[tuple[int, int]] = set()

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                edges.add((i, j))

    # Add a few hub-like users to mimic active accounts.
    hubs = rng.sample(range(num_nodes), k=max(3, num_nodes // 40))
    for h in hubs:
        for _ in range(num_nodes // 4):
            j = rng.randrange(num_nodes)
            if h != j:
                edges.add((h, j))

    node_names = [f"u{i:03d}" for i in range(num_nodes)]
    return GraphData(num_nodes, sorted(edges), node_names)


def split_edges(
    edges: list[tuple[int, int]],
    val_ratio: float = 0.1,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    rng = random.Random(seed)
    shuffled = edges[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = math.floor(n * test_ratio)
    n_val = math.floor(n * val_ratio)
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


def sample_negative_edges(
    num_nodes: int,
    positive_edges: list[tuple[int, int]],
    count: int,
    seed: int = 42,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    positives = set(positive_edges)
    negatives: set[tuple[int, int]] = set()
    while len(negatives) < count:
        i = rng.randrange(num_nodes)
        j = rng.randrange(num_nodes)
        if i != j and (i, j) not in positives:
            negatives.add((i, j))
    return list(negatives)


def adjacency_matrix(num_nodes: int, edges: list[tuple[int, int]], undirected: bool = True) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        adj[i, j] = 1.0
        if undirected:
            adj[j, i] = 1.0
    return adj


def make_event_dataset(num_events: int = 240, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic event-level structural features.

    Features: node_count, edge_count, max_depth, max_width, avg_out_degree,
    early_growth, leaf_ratio, density.
    Label 1 means a bursty event; label 0 means a normal event.
    """
    rng = np.random.default_rng(seed)
    xs: list[list[float]] = []
    ys: list[int] = []

    for _ in range(num_events):
        burst = int(rng.random() > 0.5)
        if burst:
            node_count = int(rng.normal(85, 18))
            max_depth = int(rng.normal(4, 1.2))
            max_width = int(rng.normal(38, 10))
            early_growth = rng.normal(0.72, 0.12)
            leaf_ratio = rng.normal(0.68, 0.08)
        else:
            node_count = int(rng.normal(55, 14))
            max_depth = int(rng.normal(9, 2.0))
            max_width = int(rng.normal(16, 5))
            early_growth = rng.normal(0.35, 0.10)
            leaf_ratio = rng.normal(0.49, 0.10)

        node_count = max(node_count, 8)
        max_depth = max(max_depth, 2)
        max_width = max(max_width, 3)
        edge_count = max(node_count - 1 + int(rng.normal(4, 2)), node_count - 1)
        avg_out_degree = edge_count / node_count
        density = edge_count / max(node_count * (node_count - 1), 1)
        xs.append([node_count, edge_count, max_depth, max_width, avg_out_degree, early_growth, leaf_ratio, density])
        ys.append(burst)

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return x, y

