from __future__ import annotations

import numpy as np

def topk_energy(last_hidden: np.ndarray, *, percent: float = 10.0) -> np.ndarray:
    # Return fraction of squared energy in largest absolute coordinates (Eq. 2.4).
    values = _as_matrix(last_hidden)
    if not 0 < percent <= 100:
        raise ValueError("percent must be in (0, 100]")
    count = max(1, int(values.shape[1] * percent / 100.0))
    squared = np.square(values.astype(np.float32, copy=False))
    largest = np.partition(squared, -count, axis=1)[:, -count:].sum(axis=1)
    return largest / np.maximum(squared.sum(axis=1), 1e-12)

def _as_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError("last_hidden must have shape [samples, hidden_size]")
    if not np.isfinite(array).all():
        raise ValueError("last_hidden must be finite")
    return array
