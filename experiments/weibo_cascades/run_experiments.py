"""Reproduce both experiments using the provided train/val/test cascade files."""
import argparse
import csv
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
import torch_geometric

from cascades import describe_structure, load_dataset, matrices
from link_prediction import run_link_prediction
from train_gnn import run_gin


HERE = Path(__file__).resolve().parent


def classification_metrics(y, scores):
    predicted = scores > 0.5
    return {
        "accuracy": accuracy_score(y, predicted),
        "macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
        "burst_precision": precision_score(y, predicted, zero_division=0),
        "burst_recall": recall_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def event_classification(datasets, out, args):
    features = {split: matrices(graphs) for split, graphs in datasets.items()}
    labels = {split: row[2] for split, row in features.items()}
    scores = {
        "majority": {split: np.zeros(len(labels[split])) for split in ("val", "test")}
    }
    for name, combine in (
        ("size_only", lambda structure, vectors: structure[:, :1]),
        ("structure", lambda structure, vectors: structure),
        ("node_mean", lambda structure, vectors: vectors),
        ("structure_and_node_mean", lambda structure, vectors: np.column_stack((structure, vectors))),
    ):
        x = {split: combine(structure, vectors) for split, (structure, vectors, _) in features.items()}
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=400))
        model.fit(x["train"], labels["train"])
        scores[name] = {split: model.predict_proba(x[split])[:, 1] for split in ("val", "test")}

    scores["gin"], gin_details = run_gin(
        datasets, labels, args.epochs, args.batch_size, args.seed, out, torch.device(args.device)
    )
    with (out / "event_test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("graph_index", "true_label", *(name + "_burst_score" for name in scores)))
        for index, label in enumerate(labels["test"]):
            writer.writerow((index, label, *(f"{v['test'][index]:.8f}" for v in scores.values())))
    result = {
        "decision_threshold": 0.5,
        "gin": gin_details,
        "metrics": {
            split: {name: classification_metrics(labels[split], values[split]) for name, values in scores.items()}
            for split in ("val", "test")
        },
    }
    return result, describe_structure(features["train"][0], labels["train"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path, default=HERE / "run")
    parser.add_argument("--task", choices=("all", "event", "link"), default="all")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("--epochs, --batch-size and --threads must be positive.")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    datasets, metadata = load_dataset(args.data_dir)

    results = {
        "data_origin": "synthetic Weibo-style post cascades",
        "settings": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                     "device": args.device, "torch_threads": args.threads},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                     "torch_geometric": torch_geometric.__version__},
        "data": metadata,
    }
    with threadpool_limits(limits=1):
        if args.task in ("all", "event"):
            results["event_classification"], results["training_structure"] = event_classification(datasets, args.out, args)
        if args.task in ("all", "link"):
            results["link_prediction"] = run_link_prediction(datasets, args.out, args.seed)
    path = args.out / "metrics.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Results: {path}", flush=True)


if __name__ == "__main__":
    main()
