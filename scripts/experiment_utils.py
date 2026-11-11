# File identity and score-order helpers for experiment entry points.
import hashlib
import json
from pathlib import Path

import numpy as np


TASKS = ('gsm8k', 'truthfulqa', 'mmlu', 'bbh', 'tydiqa')
ENGLISH_TASKS = ('gsm8k', 'truthfulqa', 'bbh', 'mmlu')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def load_analysis(selection):
    selection = Path(selection)
    manifest = read_json(selection / 'manifest.json')
    path = selection / 'analysis_inputs.npz'
    if digest(path) != manifest['analysis_sha256']:
        raise ValueError('Analysis input hash mismatch')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    ids = data['gate_indices']
    if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids) or np.any(ids < 0)):
        raise ValueError('Invalid gate indices')
    for name in ('gate_energy', 'comparison_energy', 'anchor_energy'):
        values = data[name]
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f'Invalid {name}')
        if name != 'anchor_energy' and values.shape != ids.shape:
            raise ValueError(f'Gate alignment mismatch: {name}')
    if ('anchor_languages' in data
            and data['anchor_languages'].shape != data['anchor_energy'].shape):
        raise ValueError('Anchor language alignment mismatch')
    return data, manifest


def window_statistics(ids, scores, budget):
    ids, scores = np.asarray(ids), np.asarray(scores, dtype=float)
    if (ids.ndim != 1 or scores.shape != ids.shape or not np.isfinite(scores).all()
            or len(np.unique(ids)) != len(ids) or not 0 < budget <= len(ids)):
        raise ValueError('Invalid window inputs')
    order = np.lexsort((ids, scores))
    ordered = scores[order]
    familiarity = 1 - np.searchsorted(ordered, ordered, side='right') / len(ids)
    sums = np.r_[0., np.cumsum(familiarity)]
    means = (sums[budget:] - sums[:-budget]) / budget
    return order, ordered, means


def anchor_percentiles(ordered_energy, anchor_energy):
    return 1 - np.searchsorted(ordered_energy, anchor_energy, side='right') / len(ordered_energy)


def reference_mean(percentiles, languages=None):
    if languages is None:
        return float(np.mean(percentiles))
    return float(np.mean([np.mean(percentiles[languages == language])
                          for language in np.unique(languages)]))


def nearest_starts(means, references):
    # The window means decrease with ascending comparison Energy.
    key = -np.asarray(means)
    refs = np.atleast_1d(references)
    right = np.clip(np.searchsorted(key, -refs), 0, len(means) - 1)
    left = np.maximum(right - 1, 0)
    best = np.where(abs(means[left] - refs) <= abs(means[right] - refs), left, right)
    return np.searchsorted(key, key[best], side='left')
