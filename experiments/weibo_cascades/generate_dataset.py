"""Generate reproducible SYNTHETIC cascade graphs for testing a graph-classification pipeline.

No graph, feature vector, or label in this output should be represented as genuine Weibo data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def make_graph(rng, label, feature_dim=32):
    # The node-count distributions deliberately do not depend on the label.
    num_nodes = int(np.clip(rng.lognormal(mean=3.70, sigma=0.36), 15, 110))
    children = np.zeros(num_nodes, dtype=np.float64)
    src, dst = [], []
    root_probability = (rng.beta(5.0, 10.0) if label == 1
                        else rng.beta(4.0, 13.0))
    # These class-conditional distributions overlap by construction.
    gamma = float(rng.normal(1.0 if label else 0.86, 0.13))
    age_factor = float(rng.uniform(0.50, 0.95) if label else rng.uniform(0.25, 0.75))

    for child in range(1, num_nodes):
        if rng.random() < root_probability:
            parent = 0
        else:
            age = child - np.arange(child, dtype=np.float64)
            weights = (1.0 + children[:child]) ** gamma
            weights *= np.exp(-age / (age_factor * child + 6.0))
            weights /= weights.sum()
            parent = int(rng.choice(child, p=weights))
        children[parent] += 1
        src.append(parent)
        dst.append(child)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    topic = rng.normal(0, 0.48, feature_dim)
    # A small, intentionally fictional class-conditioned 'semantic' shift.
    direction = np.zeros(feature_dim, dtype=np.float64)
    direction[:6] = [1.0, -0.8, 0.7, 0.6, -0.7, 0.5]
    topic += (0.20 if label else -0.20) * direction
    vectors = 0.7 * rng.standard_normal((num_nodes, feature_dim)) + 0.5 * topic
    vectors[0, 6] += 0.25
    x = torch.tensor(vectors, dtype=torch.float32)

    return {"x": x, "edge_index": edge_index, "label": torch.tensor([label], dtype=torch.long)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--burst', type=int, default=6886)
    parser.add_argument('--normal', type=int, default=13762)
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Keep exact per-class split counts as close to 70/15/15 as possible.
    indices_by_class = {}
    for label, n in [(0, args.normal), (1, args.burst)]:
        idx = np.arange(n, dtype=np.int64)
        rng.shuffle(idx)
        a = int(round(0.70*n))
        b = a + int(round(0.15*n))
        indices_by_class[label] = {
            'train': set(idx[:a].tolist()),
            'val': set(idx[a:b].tolist()),
            'test': set(idx[b:].tolist()),
        }

    splits = {'train': [], 'val': [], 'test': []}
    for label, total in [(0, args.normal), (1, args.burst)]:
        for i in range(total):
            graph = make_graph(rng, label)
            split = next(s for s in ['train', 'val', 'test'] if i in indices_by_class[label][s])
            splits[split].append(graph)
        print(f'Generated {total} simulated class-{label} cascades', flush=True)

    counts = {}
    for split, graphs in splits.items():
        order = rng.permutation(len(graphs))
        randomized = [graphs[int(i)] for i in order]
        torch.save(randomized, args.out / f'{split}.pt')
        counts[split] = {
            'graphs': len(randomized),
            'normal': sum(int(g['label'].item() == 0) for g in randomized),
            'burst': sum(int(g['label'].item() == 1) for g in randomized),
        }
        print(split, counts[split], flush=True)

    meta = {
        'IMPORTANT': 'SYNTHETIC, NOT WEIBO DATA: unsuitable for claims about real-world performance',
        'random_seed': args.seed,
        'total_graphs': args.normal + args.burst,
        'labels': {'0': 'SIMULATED normal cascade', '1': 'SIMULATED burst cascade'},
        'feature_dim': 32,
        'fields': {
            'x': '[num_nodes, 32] torch.float32 synthetic numeric vectors, NOT real text embeddings',
            'edge_index': '[2, num_edges] torch.int64 directed parent->repost edges',
            'label': '[1] torch.int64 whole-graph class',
        },
        'graph_construction': 'single-root directed tree; label-dependent overlapping root attachment, popularity and recency distributions',
        'node_count': 'same lognormal distribution for both labels',
        'splits': counts,
    }
    (args.out / 'manifest.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
