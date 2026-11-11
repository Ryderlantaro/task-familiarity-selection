# Sample scoring inputs or construct paired response variants.
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import random

import numpy as np

from scripts.experiment_utils import digest, save_json


def source_sample(rows, size, seed):
    if not 0 < size <= len(rows):
        raise ValueError('Sample size exceeds pool')
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['dataset']].append(index)
    names = sorted(groups)
    quotas = np.asarray([len(groups[name]) * size / len(rows) for name in names])
    counts = np.floor(quotas).astype(int)
    for i in np.argsort(-(quotas - counts), kind='mergesort')[:size - int(counts.sum())]:
        counts[i] += 1
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(groups[name], count, replace=False)
                                   for name, count in zip(names, counts)]))


def response_variants(row, seed):
    positions = [i for i, message in enumerate(row['messages']) if message['role'] == 'assistant']
    if not positions:
        raise ValueError('Missing assistant response')
    position = positions[-1]
    words = row['messages'][position]['content'].split()
    if len(words) < 30:
        raise ValueError('Paired response inputs require at least 30 words')
    shuffled = words.copy()
    random.Random(seed).shuffle(shuffled)
    if shuffled == words:
        raise ValueError('Word shuffling did not change the response')
    variants = {}
    for condition, text in [('original', row['messages'][position]['content']),
                            ('first_3_words', ' '.join(words[:3])),
                            ('shuffled_words', ' '.join(shuffled))]:
        variant = copy.deepcopy(row)
        variant['messages'][position]['content'] = text
        variants[condition] = variant
    return variants


def main():
    parser = argparse.ArgumentParser(description='Prepare content-scoring inputs')
    parser.add_argument('mode', choices=('sample', 'perturb'))
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--size', type=int)
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines()]
    if not rows:
        raise ValueError('Empty scoring input')
    if args.mode == 'sample':
        if args.size is None:
            parser.error('--size is required for sample')
        indices = source_sample(rows, args.size, args.seed)
        output = [rows[int(i)] for i in indices]
        metadata = dict(source_pool_rows=indices.tolist())
    else:
        output = []
        seen = set()
        for row in rows:
            key = (row['task_selection'], row['pool_index'])
            if key in seen:
                raise ValueError('Duplicate original source')
            seen.add(key)
            for condition, variant in response_variants(row, args.seed + int(row['pool_index'])).items():
                variant['condition'] = condition
                output.append(variant)
        metadata = dict(original_sources=len(rows), conditions=['original', 'first_3_words', 'shuffled_words'])
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / 'inputs.jsonl'
    with path.open('w', encoding='utf-8') as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    save_json(args.output_dir / 'manifest.json', dict(mode=args.mode, seed=args.seed,
        input_sha256=digest(args.input), output_sha256=digest(path), rows=len(output), **metadata))


if __name__ == '__main__':
    main()
