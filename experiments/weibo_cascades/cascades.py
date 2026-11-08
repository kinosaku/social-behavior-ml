"""Read the supplied synthetic cascade files and measure tree structure."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch


SPLITS = ("train", "val", "test")
FEATURES = (
    "node_count",
    "max_depth",
    "mean_depth",
    "max_width_fraction",
    "root_child_fraction",
    "max_out_degree_fraction",
    "leaf_fraction",
)


@dataclass
class Cascade:
    x: torch.Tensor
    edges: torch.Tensor
    label: int
    parents: np.ndarray
    depth: np.ndarray

    def structural_features(self) -> list[float]:
        n = len(self.parents)
        degree = np.bincount(self.parents[1:], minlength=n)
        return [
            n,
            int(self.depth.max()),
            float(self.depth.mean()),
            float(np.bincount(self.depth).max() / n),
            float(degree[0] / (n - 1)),
            float(degree.max() / (n - 1)),
            float(np.mean(degree == 0)),
        ]


def read_cascade(record: dict) -> Cascade:
    x, edges, label = record["x"], record["edge_index"], record["label"]
    n = x.shape[0]
    if x.shape != (n, 32) or x.dtype != torch.float32 or n < 2:
        raise ValueError("Expected at least two nodes and 32 float32 features per node.")
    if not np.isfinite(x.numpy()).all():
        raise ValueError("Node features contain non-finite values.")
    if edges.shape != (2, n - 1) or edges.dtype != torch.int64:
        raise ValueError("Expected n-1 directed int64 edges.")
    if label.shape != (1,) or label.dtype != torch.int64 or label.item() not in (0, 1):
        raise ValueError("Expected one int64 class label, 0 or 1.")

    src, dst = edges.numpy()
    order = np.arange(1, n)
    if not np.array_equal(np.sort(dst), order) or np.any(src < 0) or np.any(src >= dst):
        raise ValueError("Every non-root node must have exactly one earlier parent.")
    parents = np.full(n, -1, dtype=np.int64)
    parents[dst] = src
    depth = np.zeros(n, dtype=np.int64)
    for child in order:
        depth[child] = depth[parents[child]] + 1
    return Cascade(x, edges, int(label.item()), parents, depth)


def load_dataset(folder: Path) -> tuple[dict, dict]:
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    datasets, metadata, seen = {}, {}, set()
    for split in SPLITS:
        records = torch.load(folder / f"{split}.pt", map_location="cpu", weights_only=True)
        graphs, digest = [], sha256()
        for index, record in enumerate(records):
            try:
                graph = read_cascade(record)
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
                raise ValueError(f"{split}[{index}]: {error}") from error
            content = sha256(graph.x.numpy().astype("<f4", copy=False).tobytes())
            content.update(graph.parents.astype("<i8", copy=False).tobytes())
            fingerprint = content.digest()
            if fingerprint in seen:
                raise ValueError(f"{split}[{index}]: duplicate graph and feature vectors.")
            seen.add(fingerprint)
            digest.update(fingerprint + bytes([graph.label]))
            graphs.append(graph)
        burst = sum(g.label for g in graphs)
        counts = {"graphs": len(graphs), "normal": len(graphs) - burst, "burst": burst}
        if not counts["normal"] or not burst:
            raise ValueError(f"{split} must contain both classes.")
        if manifest and counts != manifest["splits"][split]:
            raise ValueError(f"{split} counts differ from the supplied manifest.")
        sizes = [len(g.parents) for g in graphs]
        metadata[split] = {
            **counts,
            "nodes": sum(sizes),
            "edges": sum(sizes) - len(graphs),
            "min_nodes": min(sizes),
            "max_nodes": max(sizes),
            "content_sha256": digest.hexdigest(),
        }
        datasets[split] = graphs
        print(f"{split}: {len(graphs):,} graphs, {burst:,} burst, {sum(sizes):,} nodes", flush=True)
    return datasets, metadata


def matrices(graphs: list[Cascade]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    structure = np.asarray([g.structural_features() for g in graphs])
    vectors = np.stack([g.x.numpy().mean(axis=0) for g in graphs])
    labels = np.asarray([g.label for g in graphs])
    return structure, vectors, labels


def describe_structure(structure: np.ndarray, labels: np.ndarray) -> dict:
    summary = {}
    for label in (0, 1):
        rows = structure[labels == label]
        quantiles = np.quantile(rows, [0.25, 0.5, 0.75], axis=0)
        summary[str(label)] = {
            feature: {
                "mean": float(rows[:, i].mean()),
                "median": float(quantiles[1, i]),
                "q1": float(quantiles[0, i]),
                "q3": float(quantiles[2, i]),
            }
            for i, feature in enumerate(FEATURES)
        }
    return summary
