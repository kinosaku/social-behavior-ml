from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from social_behavior_ml.data import adjacency_matrix, make_event_dataset, make_synthetic_social_graph, sample_negative_edges, split_edges
from social_behavior_ml.features import pair_heuristic_features, standardize_train_test
from social_behavior_ml.metrics import accuracy, average_precision, roc_auc_score


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_logistic_regression(x: np.ndarray, y: np.ndarray, steps: int = 900, lr: float = 0.08) -> np.ndarray:
    x_aug = np.c_[np.ones(len(x)), x]
    w = np.zeros(x_aug.shape[1], dtype=np.float64)
    y = y.astype(np.float64)
    for _ in range(steps):
        p = sigmoid(x_aug @ w)
        grad = x_aug.T @ (p - y) / len(y) + 1e-4 * w
        w -= lr * grad
    return w


def predict_lr(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sigmoid(np.c_[np.ones(len(x)), x] @ w)


def run_link_prediction(seed: int, out_dir: Path) -> dict[str, float]:
    graph = make_synthetic_social_graph(seed=seed)
    train_edges, _, test_edges = split_edges(graph.edges, seed=seed)
    adj = adjacency_matrix(graph.num_nodes, train_edges)
    train_neg = sample_negative_edges(graph.num_nodes, graph.edges, len(train_edges), seed=seed + 1)
    test_neg = sample_negative_edges(graph.num_nodes, graph.edges, len(test_edges), seed=seed + 2)

    train_pairs = train_edges + train_neg
    test_pairs = test_edges + test_neg
    train_y = np.asarray([1] * len(train_edges) + [0] * len(train_neg))
    test_y = np.asarray([1] * len(test_edges) + [0] * len(test_neg))

    train_x = pair_heuristic_features(adj, train_pairs)
    test_x = pair_heuristic_features(adj, test_pairs)
    train_x, test_x = standardize_train_test(train_x, test_x)

    w = fit_logistic_regression(train_x, train_y)
    scores = predict_lr(w, test_x)

    plt.figure(figsize=(7.2, 4.2))
    plt.hist(scores[test_y == 1], bins=20, alpha=0.72, label="真实互动边")
    plt.hist(scores[test_y == 0], bins=20, alpha=0.72, label="随机负样本")
    plt.xlabel("预测互动概率")
    plt.ylabel("样本数")
    plt.title("链路预测得分分布")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "link_prediction_scores.png", dpi=180)
    plt.close()

    return {
        "auc": roc_auc_score(test_y, scores),
        "average_precision": average_precision(test_y, scores),
    }


def run_event_classifier(seed: int, out_dir: Path) -> dict[str, float]:
    x, y = make_event_dataset(seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    train_idx = idx[: int(0.75 * len(idx))]
    test_idx = idx[int(0.75 * len(idx)) :]
    train_x, test_x = standardize_train_test(x[train_idx], x[test_idx])
    train_y = y[train_idx]
    test_y = y[test_idx]

    w = fit_logistic_regression(train_x, train_y, steps=1200, lr=0.06)
    scores = predict_lr(w, test_x)
    pred = (scores >= 0.5).astype(int)

    plt.figure(figsize=(6.4, 4.6))
    labels = np.array(["普通事件", "突发事件"])
    for cls in [0, 1]:
        mask = y == cls
        plt.scatter(x[mask, 2], x[mask, 3], s=24, alpha=0.72, label=labels[cls])
    plt.xlabel("最大传播深度")
    plt.ylabel("最大传播宽度")
    plt.title("事件传播结构特征")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "event_feature_scatter.png", dpi=180)
    plt.close()

    return {
        "accuracy": accuracy(test_y, pred),
        "auc": roc_auc_score(test_y, scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="runs/demo")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "link_prediction": run_link_prediction(args.seed, out_dir),
        "event_classification": run_event_classifier(args.seed, out_dir),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
