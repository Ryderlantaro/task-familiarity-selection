import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

from scipy.stats import wasserstein_distance

def mean_familiarity_selection(gate_indices, gate_energy, anchor_energy, *, budget,
                               comparison_energy=None, anchor_languages=None, progress=None):
    # Locate a fixed-budget window by mean anchor familiarity percentile.
    ids = np.asarray(gate_indices, dtype=np.int64)
    raw = np.asarray(gate_energy, dtype=np.float64)
    anchor = np.asarray(anchor_energy, dtype=np.float64)
    scores = raw if comparison_energy is None else np.asarray(comparison_energy, dtype=np.float64)
    if (ids.ndim != 1 or raw.shape != ids.shape or scores.shape != ids.shape
            or len(np.unique(ids)) != len(ids) or (ids < 0).any()
            or not 0 < budget <= len(ids) or anchor.ndim != 1 or not len(anchor)
            or not all(np.isfinite(x).all() for x in (raw, anchor, scores))):
        raise ValueError("Invalid indices, budget, or Sparsity scores")
    order = np.lexsort((ids, scores))
    sorted_scores = scores[order]
    anchor_f = 1 - np.searchsorted(sorted_scores, anchor, side="right") / len(ids)
    candidate_f = 1 - np.searchsorted(sorted_scores, sorted_scores, side="right") / len(ids)
    groups = [anchor]
    target = float(anchor_f.mean())
    if anchor_languages is not None:
        languages = np.asarray(anchor_languages)
        if languages.shape != anchor.shape:
            raise ValueError("Anchor languages must align with anchor scores")
        keys = sorted(set(languages))
        groups = [anchor[languages == language] for language in keys]
        target = float(np.mean([anchor_f[languages == language].mean() for language in keys]))
    cumulative = np.r_[0., np.cumsum(candidate_f)]
    window_means = (cumulative[budget:] - cumulative[:-budget]) / budget
    start = int(np.argmin(np.abs(window_means - target)))
    selected = order[start:start + budget]
    if progress is not None:
        progress(f"Mean familiarity: anchor={target:.6f}; selected={window_means[start]:.6f}")
    return ids[selected], {
        "method": "continuous window matching mean familiarity percentile",
        "anchor_mean_familiarity": target,
        "selected_mean_familiarity": float(window_means[start]),
        "familiarity_mean_gap": float(abs(window_means[start] - target)),
        "w1": float(np.mean([wasserstein_distance(scores[selected], group) for group in groups])),
        "language_groups": len(groups),
        "raw_energy_range": [float(raw[selected].min()), float(raw[selected].max())],
        "selected_raw_mean": float(raw[selected].mean()),
        "anchor_raw_mean": float(anchor.mean()),
        "window_start_ascending_comparison_score": start,
        "reference": "gate empirical CDF; equal language means after per-anchor percentile conversion",
    }

def language_offsets(candidate, anchor, candidate_languages, anchor_languages, minimum_count=50):
    # Apply gate-estimated language offsets to candidates and anchors alike.
    candidate = np.asarray(candidate, dtype=np.float64).copy()
    anchor = np.asarray(anchor, dtype=np.float64).copy()
    raw = candidate.copy()
    offsets = {}
    for language in sorted(set(anchor_languages)):
        mask = np.asarray(candidate_languages) == language
        if mask.sum() >= minimum_count:
            offset = float(raw[mask].mean() - raw.mean())
            candidate[mask] -= offset
            anchor[np.asarray(anchor_languages) == language] -= offset
            offsets[language] = offset
    return candidate, anchor, offsets
