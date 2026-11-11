from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sklearn.cluster import MiniBatchKMeans

from ..scores.md import nmd_direction_from_means, score_nmd_direction, unit_mean

def semantic_cluster_labels(
    candidate_embeddings: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    # Partition the candidate representation space for T coverage correction.
    embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or not 1 < n_clusters <= len(embeddings):
        raise ValueError("candidate embeddings and n_clusters are incompatible")
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=4096,
        n_init=3,
    ).fit_predict(embeddings).astype(np.int32)

def core_plus_cluster_indices(
    target_scores: np.ndarray,
    *,
    cluster_labels: np.ndarray,
    budget: int,
    supplement_fraction: float,
    eligible_indices: np.ndarray | None = None,
) -> np.ndarray:
    # Apply the T coverage wrapper to a candidate domain without source labels.
    if not 0.0 <= supplement_fraction < 1.0:
        raise ValueError("supplement_fraction must be in [0, 1)")
    if budget <= 0:
        raise ValueError("budget must be positive")
    scores = np.asarray(target_scores, dtype=np.float32)
    labels = np.asarray(cluster_labels, dtype=np.int32)
    if scores.ndim != 1 or len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("target scores and cluster labels are incompatible")
    domain = (
        np.arange(len(scores), dtype=np.int64)
        if eligible_indices is None
        else np.asarray(eligible_indices, dtype=np.int64)
    )
    if len(domain) < budget or len(np.unique(domain)) != len(domain):
        raise ValueError("eligible indices cannot satisfy the requested budget")
    if (domain < 0).any() or (domain >= len(scores)).any():
        raise ValueError("eligible indices are outside the target-score range")

    core_budget = round((1.0 - supplement_fraction) * budget)
    core = domain[np.argsort(scores[domain], kind="mergesort")[-core_budget:][::-1]]
    selected = list(map(int, core))
    selected_set = set(selected)
    supplement_budget = budget - len(selected)
    if supplement_budget == 0:
        return np.asarray(selected, dtype=np.int64)

    cluster_ids = np.unique(labels[domain])
    ranked_by_cluster = []
    for cluster_id in cluster_ids:
        members = domain[labels[domain] == cluster_id]
        ranked_by_cluster.append(members[np.argsort(scores[members], kind="mergesort")[::-1]])
    cursors = np.zeros(len(cluster_ids), dtype=np.int64)
    while len(selected) < budget:
        progressed = False
        for position, ranked in enumerate(ranked_by_cluster):
            while cursors[position] < len(ranked):
                index = int(ranked[cursors[position]])
                cursors[position] += 1
                if index not in selected_set:
                    selected.append(index)
                    selected_set.add(index)
                    progressed = True
                    break
            if len(selected) == budget:
                break
        if not progressed:
            raise RuntimeError("cluster supplement could not fill the requested budget")
    return np.asarray(selected, dtype=np.int64)

def build_t(candidate_embeddings, anchor_embeddings, background_embeddings, *, progress=None):
    # Score candidates against the supplied task's anchor representation mean.
    candidates = np.asarray(candidate_embeddings)
    anchors = np.asarray(anchor_embeddings)
    if candidates.ndim != 2 or anchors.ndim != 2 or not len(anchors) or anchors.shape[1] != candidates.shape[1]:
        raise ValueError('Expected candidate and task-anchor matrices of matching width')
    direction = nmd_direction_from_means(unit_mean(anchors), unit_mean(background_embeddings))
    scores = score_nmd_direction(candidates, direction)
    positive = np.maximum(scores, 0.0)
    order = np.argsort(positive, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    return scores, ranks / max(1, len(scores)-1)
