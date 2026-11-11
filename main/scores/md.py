from __future__ import annotations

import numpy as np

def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

def unit_mean(embeddings: np.ndarray, *, chunk_size: int = 8192) -> np.ndarray:
    # Return the exact mean of row-normalized embeddings without sampling.
    values = np.asarray(embeddings)
    if values.ndim != 2 or len(values) == 0 or chunk_size <= 0:
        raise ValueError("embeddings must be a non-empty matrix and chunk_size positive")
    total = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        total += _unit_rows(values[start : start + chunk_size]).sum(axis=0, dtype=np.float64)
    return (total / len(values)).astype(np.float32)

def nmd_direction_from_means(anchor_unit_mean: np.ndarray, background_unit_mean: np.ndarray) -> np.ndarray:
    # Return the unit nMD contrast direction from normalized representation means.
    direction = np.asarray(anchor_unit_mean, dtype=np.float32) - np.asarray(
        background_unit_mean, dtype=np.float32
    )
    return direction / max(float(np.linalg.norm(direction)), 1e-12)

def score_nmd_direction(
    candidate_embeddings: np.ndarray,
    direction: np.ndarray,
    *,
    chunk_size: int = 8192,
) -> np.ndarray:
    # Project row-normalized candidate embeddings onto a fixed nMD direction.
    candidates = np.asarray(candidate_embeddings)
    if candidates.ndim != 2 or len(candidates) == 0:
        raise ValueError("candidate embeddings must be a non-empty matrix")
    scores = np.empty(len(candidates), dtype=np.float32)
    for start in range(0, len(candidates), chunk_size):
        stop = min(start + chunk_size, len(candidates))
        scores[start:stop] = _unit_rows(candidates[start:stop]) @ direction
    return scores
