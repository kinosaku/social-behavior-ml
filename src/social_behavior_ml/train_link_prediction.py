from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import adjacency_matrix, load_edge_csv, make_synthetic_social_graph, sample_negative_edges, split_edges
from .features import node_structural_features
from .metrics import average_precision, roc_auc_score
from .models import GCNLinkPredictor, normalize_adjacency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="runs/link_prediction")
    return parser.parse_args()


def evaluate(model, x, adj_norm, positives, negatives) -> dict[str, float]:
    model.eval()
    pairs = positives + negatives
    y = np.asarray([1] * len(positives) + [0] * len(negatives))
    with torch.no_grad():
        logits = model(x, adj_norm, torch.tensor(pairs, dtype=torch.long, device=x.device))
        scores = torch.sigmoid(logits).cpu().numpy()
    return {
        "auc": roc_auc_score(y, scores),
        "average_precision": average_precision(y, scores),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    graph = load_edge_csv(args.edge_csv) if args.edge_csv else make_synthetic_social_graph(seed=args.seed)
    train_edges, val_edges, test_edges = split_edges(graph.edges, seed=args.seed)
    train_adj_np = adjacency_matrix(graph.num_nodes, train_edges)
    features = node_structural_features(train_adj_np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(features, dtype=torch.float32, device=device)
    adj = torch.tensor(train_adj_np, dtype=torch.float32, device=device)
    adj_norm = normalize_adjacency(adj)

    model = GCNLinkPredictor(in_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    all_positive = graph.edges
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        neg_edges = sample_negative_edges(graph.num_nodes, all_positive, len(train_edges), seed=args.seed + epoch)
        pairs = train_edges + neg_edges
        labels = torch.tensor([1.0] * len(train_edges) + [0.0] * len(neg_edges), device=device)
        edge_index = torch.tensor(pairs, dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(x, adj_norm, edge_index)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            val_neg = sample_negative_edges(graph.num_nodes, all_positive, len(val_edges), seed=args.seed + 1000 + epoch)
            val_metrics = evaluate(model, x, adj_norm, val_edges, val_neg)
            history.append({"epoch": epoch, "loss": float(loss.item()), **val_metrics})
            print(f"epoch={epoch:03d} loss={loss.item():.4f} val_auc={val_metrics['auc']:.3f}")

    test_neg = sample_negative_edges(graph.num_nodes, all_positive, len(test_edges), seed=args.seed + 999)
    test_metrics = evaluate(model, x, adj_norm, test_edges, test_neg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps({"test": test_metrics, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), out_dir / "gcn_link_predictor.pt")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

