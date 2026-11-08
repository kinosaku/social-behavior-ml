"""GIN classifier for the synthetic propagation trees."""
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool


class Classifier(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()

        def mlp(width: int):
            return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.convs = nn.ModuleList([GINConv(mlp(channels)), GINConv(mlp(hidden))])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, batch):
        x = batch.x
        for conv in self.convs:
            x = F.relu(conv(x, batch.edge_index))
        return self.head(global_mean_pool(x, batch.batch))


def graph_loader(graphs, batch_size, shuffle=False):
    items = []
    for graph in graphs:
        root = torch.zeros((len(graph.parents), 1), dtype=graph.x.dtype)
        root[0] = 1
        # Both ends receive messages. The root indicator preserves its identity.
        edges = torch.cat((graph.edges, graph.edges.flip(0)), dim=1)
        items.append(Data(x=torch.cat((graph.x, root), dim=1), edge_index=edges, y=graph.label))
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    return torch.cat([model(batch.to(device)).softmax(dim=1)[:, 1].cpu() for batch in loader]).numpy()


def run_gin(datasets, labels, epochs, batch_size, seed, out, device):
    torch.manual_seed(seed)
    train = graph_loader(datasets["train"], batch_size, shuffle=True)
    val = graph_loader(datasets["val"], batch_size)
    model = Classifier(datasets["train"][0].x.shape[1] + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, best_f1, best_epoch = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        scores = predict(model, val, device)
        f1 = f1_score(labels["val"], scores > 0.5, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train.dataset), "val_macro_f1": float(f1)})
        print(f"GIN {epoch:02d}/{epochs:02d}  loss={history[-1]['train_loss']:.4f}  val_macro_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, out / "gin_model.pt")
    return {
        "val": predict(model, val, device),
        "test": predict(model, graph_loader(datasets["test"], batch_size), device),
    }, {"best_epoch": best_epoch, "history": history}
