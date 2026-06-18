from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import make_event_dataset
from .metrics import accuracy
from .models import EventMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="runs/event_classifier")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    x, y = make_event_dataset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y))
    train_idx = idx[: int(0.75 * len(idx))]
    test_idx = idx[int(0.75 * len(idx)) :]

    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.tensor(x[train_idx], dtype=torch.float32, device=device)
    train_y = torch.tensor(y[train_idx], dtype=torch.long, device=device)
    test_x = torch.tensor(x[test_idx], dtype=torch.float32, device=device)
    test_y_np = y[test_idx]

    model = EventMLP(in_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(test_x).argmax(dim=1).cpu().numpy()
    result = {"test_accuracy": accuracy(test_y_np, pred)}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "event_mlp.pt")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

