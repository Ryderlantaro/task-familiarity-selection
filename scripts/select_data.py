# Select training data using target alignment and anchor familiarity.

import argparse
import json
from pathlib import Path
import time

import numpy as np

from main.adapters.target import build_t, semantic_cluster_labels, core_plus_cluster_indices
from main.adapters.sparsity_alignment import language_offsets, mean_familiarity_selection
from main.scores.sparsity import topk_energy
from main.pipeline import write_indices, materialize_jsonl
from main.tools.extract_dual_last_layer import _sha256, _write_json
from main.tools.language_cache import LANGUAGES, gate_languages
from main.tools.cache_extraction import validate_vectors


def energy_batched(values):
    result = np.empty(len(values), dtype=np.float32)
    for start in range(0, len(values), 4096):
        block = values[start:start + 4096]
        validate_vectors(block, len(block), values.shape[1])
        result[start:start + len(block)] = topk_energy(block, percent=10)
    return result


def cached_energy(values, path, representation_sha256):
    # Bind derived scores to the validated representation file and statistic.
    metadata_path = path.with_suffix('.manifest.json')
    expected = {'representation_sha256': representation_sha256,
                'rows': len(values), 'top_percent': 10, 'version': 1}
    if path.exists() or metadata_path.exists():
        if not path.is_file() or not metadata_path.is_file():
            raise ValueError(f'Incomplete Energy cache: {path}')
        saved = json.loads(metadata_path.read_text())
        if saved['spec'] != expected or _sha256(path) != saved['output_sha256']:
            raise ValueError(f'Energy cache identity mismatch: {path}')
        result = np.load(path, allow_pickle=False)
        if result.shape != (len(values),) or result.dtype != np.float32:
            raise ValueError(f'Unexpected Energy cache format: {path}')
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError(f'Invalid Energy scores: {path}')
        return result, True
    result = energy_batched(values)
    np.save(path, result)
    _write_json(metadata_path, {'spec': expected, 'output_sha256': _sha256(path)})
    return result, False


def main():
    parser = argparse.ArgumentParser(description='Select training data using target alignment and anchor familiarity.')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--task', choices=('gsm8k', 'truthfulqa', 'mmlu', 'bbh', 'tydiqa'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=10000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a new selection directory')
    manifest = json.loads((args.cache / f'manifest_{args.task}.json').read_text())
    spec = manifest['spec']
    if (manifest['status'] != 'complete' or spec['smoke_limit'] != 0 or spec['task'] != args.task
            or set(spec['sources']) != {'pool', 'anchors'}):
        raise ValueError('A complete task cache is required')
    if args.budget < 1 or 6 * args.budget > spec['sources']['pool']['count']:
        raise ValueError('Pool must contain at least six times the positive budget')
    vectors, representation_hashes = {}, {}
    for view in ('t', 'm'):
        vectors[view] = {}
        for name, source in spec['sources'].items():
            filename = 'pool.npy' if name == 'pool' else f'{args.task}_anchors.npy'
            path = args.cache / view / filename
            saved = json.loads(path.with_suffix('.manifest.json').read_text())
            expected_spec = {**spec['identity'], **source, **spec[view], 'view': view}
            if saved['spec'] != expected_spec or _sha256(path) != saved['output_sha256']:
                raise ValueError(f'Cache identity mismatch: {path}')
            if _sha256(Path(source['path'])) != source['sha256']:
                raise ValueError(f'Source changed: {source["path"]}')
            values = np.load(path, mmap_mode='r')
            if values.shape != (source['count'], spec['identity']['hidden_size']):
                raise ValueError('Unexpected cache shape')
            for start in range(0, len(values), 4096):
                validate_vectors(values[start:start + 4096], min(4096, len(values)-start), values.shape[1])
            vectors[view][name] = values
            representation_hashes[view, name] = saved['output_sha256']
    started = time.monotonic()
    out = args.output_dir
    out.mkdir(parents=True)
    nmd, t = build_t(vectors['t']['pool'], vectors['t']['anchors'], vectors['t']['pool'])
    labels = semantic_cluster_labels(vectors['t']['pool'], n_clusters=30, seed=42)
    gate = core_plus_cluster_indices(t, cluster_labels=labels, budget=6 * args.budget,
                                     supplement_fraction=.15)
    energy, pool_reused = cached_energy(vectors['m']['pool'], args.cache / 'm/pool_top10_energy.npy',
                                       representation_hashes['m', 'pool'])
    anchor, anchor_reused = cached_energy(vectors['m']['anchors'],
        args.cache / f'm/{args.task}_anchors_top10_energy.npy', representation_hashes['m', 'anchors'])
    raw = energy[gate]
    comparison, anchor_languages, offsets = raw, None, {}
    if args.task == 'tydiqa':
        candidate_languages = gate_languages(spec['sources']['pool']['path'], gate, out / 'gate_languages.npz')
        with Path(spec['sources']['anchors']['path']).open() as stream:
            anchor_languages = np.asarray([LANGUAGES[json.loads(line)['id'].split('-', 1)[0]] for line in stream])
        comparison, anchor, offsets = language_offsets(raw, anchor, candidate_languages, anchor_languages)
    selected, report = mean_familiarity_selection(gate, raw, anchor, budget=args.budget,
                                                 comparison_energy=comparison,
                                                 anchor_languages=anchor_languages)
    write_indices(gate, out / 'gate_indices.npy')
    write_indices(selected, out / f'{args.task}_indices.npy')
    materialize_jsonl(spec['sources']['pool']['path'], selected, out / f'{args.task}.jsonl')
    np.savez_compressed(out / 'scores.npz', nmd=nmd, target_factor=t, energy=energy,
                        cluster_labels=labels)
    analysis = dict(gate_indices=gate, gate_energy=raw, comparison_energy=comparison,
                    anchor_energy=anchor)
    if anchor_languages is not None:
        analysis['anchor_languages'] = anchor_languages
    np.savez_compressed(out / 'analysis_inputs.npz', **analysis)
    _write_json(out / 'manifest.json', {'task': args.task, 'budget': args.budget,
        'cache_manifest_sha256': _sha256(args.cache / f'manifest_{args.task}.json'),
        'pool_sha256': spec['sources']['pool']['sha256'],
        'anchors_sha256': spec['sources']['anchors']['sha256'],
        'analysis_sha256': _sha256(out / 'analysis_inputs.npz'),
        'scores_sha256': _sha256(out / 'scores.npz'),
        'train_sha256': _sha256(out / f'{args.task}.jsonl')})
    _write_json(out / 'summary.json', {**report, 'task': args.task, 'language_offsets': offsets,
                                      'energy_cache_reused': {'pool': pool_reused, 'anchors': anchor_reused},
                                      'elapsed_seconds': time.monotonic()-started})


if __name__ == '__main__':
    main()
