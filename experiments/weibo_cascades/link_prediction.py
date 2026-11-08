"""Rank possible parents of each incoming post using only its observed prefix."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CANDIDATE_FEATURES = ("is_root", "log_out_degree", "depth", "relative_age", "log_age")
NEGATIVES_PER_POST = 8


def queries(graph):
    parents, depth = graph.parents, graph.depth
    observed = (7 * len(parents) + 9) // 10
    degree = np.bincount(parents[1:observed], minlength=len(parents))
    for child in range(observed, len(parents)):
        parent = parents[child]
        yield child, parent, degree[:child], depth[:child]
        # Reveal the answer only after predicting this post's parent.
        degree[parent] += 1


def candidate_features(child, degree, depth, candidates):
    age = child - candidates
    return np.column_stack((
        candidates == 0,
        np.log1p(degree[candidates]),
        depth[candidates],
        age / child,
        np.log1p(age),
    ))


def fit_ranker(graphs, seed):
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for graph in graphs:
        for child, parent, degree, depth in queries(graph):
            negative = rng.choice(child - 1, size=min(NEGATIVES_PER_POST, child - 1), replace=False)
            negative += negative >= parent
            candidates = np.r_[parent, negative]
            blocks.append(candidate_features(child, degree, depth, candidates))
            labels.extend([1] + [0] * len(negative))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=400))
    return model.fit(np.concatenate(blocks), labels)


def evaluate_ranker(graphs, model, writer=None):
    names = ("uniform", "recency", "popularity", "logistic")
    sums = np.zeros((len(names), 3))
    by_parent = np.zeros((2, 3))
    parent_counts = np.zeros(2, dtype=int)
    max_nodes = max(len(g.parents) for g in graphs)
    all_nodes = np.arange(max_nodes)
    harmonic = np.r_[0, np.cumsum(1.0 / np.arange(1, max_nodes + 1))]
    scaler, logistic = model.steps[0][1], model.steps[1][1]
    weights = logistic.coef_[0] / scaler.scale_
    # Centering and the intercept add the same offset to each candidate's score.
    query_count, candidate_count = 0, 0
    for graph_index, graph in enumerate(graphs):
        for child, parent, degree, depth in queries(graph):
            candidates = all_nodes[:child]
            learned = candidate_features(child, degree, depth, candidates) @ weights
            scores = np.stack((np.zeros(child), candidates, degree, learned))
            truth = scores[:, [parent]]
            higher = np.count_nonzero(scores > truth, axis=1)
            tied = np.count_nonzero(scores == truth, axis=1)
            metrics = np.column_stack((
                np.clip(1 - higher, 0, tied) / tied,
                np.clip(5 - higher, 0, tied) / tied,
                (harmonic[higher + tied] - harmonic[higher]) / tied,
            ))
            sums += metrics
            group = int(parent != 0)
            by_parent[group] += metrics[-1]
            parent_counts[group] += 1
            query_count += 1
            candidate_count += child
            if writer:
                writer.writerow((graph_index, child, parent, int(learned.argmax()), int(higher[-1] + 1),
                                 int(higher[-1] + tied[-1])))

    def summary(values):
        return dict(zip(("hits_at_1", "hits_at_5", "mrr"), values.tolist()))

    return {
        "queries": query_count,
        "candidate_pairs": candidate_count,
        "mean_candidates": candidate_count / query_count,
        "methods": {name: summary(row / query_count) for name, row in zip(names, sums)},
        "logistic_by_parent": {
            name: {"queries": int(count), **(summary(row / count) if count else {})}
            for name, count, row in zip(("root", "non_root"), parent_counts, by_parent)
        },
    }


def run_link_prediction(datasets, out, seed):
    import csv

    print("Fitting repost parent ranker", flush=True)
    model = fit_ranker(datasets["train"], seed)
    val = evaluate_ranker(datasets["val"], model)
    with (out / "link_test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("graph_index", "child", "true_parent", "top_parent", "rank_min", "rank_max"))
        test = evaluate_ranker(datasets["test"], model, writer)
    return {
        "observed_prefix_fraction": 0.7,
        "training_negatives_per_post": NEGATIVES_PER_POST,
        "candidate_features": list(CANDIDATE_FEATURES),
        "val": val,
        "test": test,
    }
