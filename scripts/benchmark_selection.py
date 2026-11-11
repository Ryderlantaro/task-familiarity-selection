# Measure selection from existing representation and Energy caches.
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from main.adapters.target import build_t, semantic_cluster_labels, core_plus_cluster_indices
from main.adapters.sparsity_alignment import language_offsets, mean_familiarity_selection
from main.pipeline import materialize_jsonl
from scripts.experiment_utils import TASKS, digest, save_json


def load_array(path, key=None):
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            if key is None:
                raise ValueError('An archive key is required for NPZ inputs')
            if 'pool_indices' in archive:
                np.testing.assert_array_equal(archive['pool_indices'], np.arange(len(archive[key])))
            return archive[key]
    return np.load(path, mmap_mode='r', allow_pickle=False)


def valid_vectors(values):
    if values.ndim != 2 or not len(values):
        raise ValueError('Expected a nonempty representation matrix')
    for start in range(0, len(values), 4096):
        block = np.asarray(values[start:start + 4096], dtype=np.float32)
        if not np.isfinite(block).all() or (np.linalg.norm(block, axis=1) == 0).any():
            raise ValueError('Invalid or zero representation')


def main():
    parser = argparse.ArgumentParser(description='Benchmark cached data selection')
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--candidate-t', type=Path, required=True)
    parser.add_argument('--anchor-t', type=Path, required=True)
    parser.add_argument('--candidate-energy', type=Path, required=True)
    parser.add_argument('--anchor-energy', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--anchor-t-key')
    parser.add_argument('--candidate-energy-key')
    parser.add_argument('--anchor-energy-key')
    parser.add_argument('--candidate-languages', type=Path)
    parser.add_argument('--anchor-languages', type=Path)
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.budget < 1 or args.repeats < 1:
        parser.error('Budget and repeats must be positive')
    if args.task == 'tydiqa' and (args.candidate_languages is None or args.anchor_languages is None):
        parser.error('TyDiQA requires pool-aligned and anchor-aligned language arrays')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    paths = [args.pool, args.candidate_t, args.anchor_t, args.candidate_energy, args.anchor_energy]
    paths += [p for p in (args.candidate_languages, args.anchor_languages) if p is not None]
    input_hashes = None
    for repeat in range(args.repeats):
        folder = args.output_dir / f'repeat_{repeat + 1}'
        folder.mkdir()
        times = {}
        began = time.perf_counter()
        start = time.perf_counter()
        hashes = {str(path.resolve()): digest(path) for path in paths}
        if input_hashes is not None and input_hashes != hashes:
            raise ValueError('Benchmark inputs changed between repetitions')
        input_hashes = hashes
        candidate = load_array(args.candidate_t)
        anchor_t = load_array(args.anchor_t, args.anchor_t_key)
        energy = load_array(args.candidate_energy, args.candidate_energy_key)
        anchor_energy = load_array(args.anchor_energy, args.anchor_energy_key)
        valid_vectors(candidate)
        valid_vectors(anchor_t)
        if (anchor_t.shape[1] != candidate.shape[1] or energy.shape != (len(candidate),)
                or anchor_energy.shape != (len(anchor_t),) or len(candidate) < 6 * args.budget
                or not np.isfinite(energy).all() or not np.isfinite(anchor_energy).all()
                or np.any((energy < 0) | (energy > 1))
                or np.any((anchor_energy < 0) | (anchor_energy > 1))):
            raise ValueError('Representation and Energy inputs do not align')
        times['cache_loading_validation'] = time.perf_counter() - start
        print(f"repeat {repeat + 1}: validated caches", flush=True)
        start = time.perf_counter()
        _, scores = build_t(candidate, anchor_t, candidate)
        times['target_scoring'] = time.perf_counter() - start
        print(f"repeat {repeat + 1}: T scoring {times['target_scoring']:.3f}s", flush=True)
        start = time.perf_counter()
        labels = semantic_cluster_labels(candidate, n_clusters=30, seed=42)
        times['semantic_clustering'] = time.perf_counter() - start
        print(f"repeat {repeat + 1}: clustering {times['semantic_clustering']:.3f}s", flush=True)
        start = time.perf_counter()
        gate = core_plus_cluster_indices(scores, cluster_labels=labels, budget=6 * args.budget,
                                         supplement_fraction=.15)
        times['gate_selection'] = time.perf_counter() - start
        start = time.perf_counter()
        raw = energy[gate]
        comparison, anchor, languages = raw, anchor_energy, None
        if args.task == 'tydiqa':
            pool_languages = load_array(args.candidate_languages)
            languages = load_array(args.anchor_languages)
            if pool_languages.shape != (len(candidate),) or languages.shape != anchor_energy.shape:
                raise ValueError('Language arrays do not align')
            comparison, anchor, _ = language_offsets(raw, anchor_energy, pool_languages[gate], languages)
        times['language_processing'] = time.perf_counter() - start
        start = time.perf_counter()
        selected, report = mean_familiarity_selection(gate, raw, anchor, budget=args.budget,
            comparison_energy=comparison, anchor_languages=languages)
        times['window_matching'] = time.perf_counter() - start
        start = time.perf_counter()
        np.save(folder / 'indices.npy', selected, allow_pickle=False)
        materialize_jsonl(args.pool, selected, folder / 'selected.jsonl')
        times['export'] = time.perf_counter() - start
        times['total'] = time.perf_counter() - began
        with (folder / 'selected.jsonl').open(encoding='utf-8') as stream:
            count = sum(1 for _ in stream)
        if len(np.unique(selected)) != args.budget or count != args.budget or not np.isin(selected, gate).all():
            raise ValueError('Export budget or gate membership check failed')
        if repeat:
            np.testing.assert_array_equal(selected, np.load(args.output_dir / 'repeat_1/indices.npy'))
        record = dict(repeat=repeat + 1, seconds=times, selection_report=report,
                      indices_sha256=digest(folder / 'indices.npy'))
        records.append(record)
        save_json(folder / 'timing.json', record)
        print(json.dumps(dict(repeat=repeat + 1, seconds=times)), flush=True)
    summary = dict(task=args.task, budget=args.budget, pool_count=len(candidate),
        repetitions=records, input_sha256=input_hashes,
        stages={key: dict(mean=statistics.mean(row['seconds'][key] for row in records),
                         median=statistics.median(row['seconds'][key] for row in records),
                         sample_sd=statistics.stdev(row['seconds'][key] for row in records)
                         if len(records) > 1 else None) for key in records[0]['seconds']},
        environment=dict(platform=platform.platform(), processor=platform.processor(),
                         numpy=np.__version__, cpu_count=os.cpu_count(),
                         omp_threads=os.environ.get('OMP_NUM_THREADS'),
                         openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS')),
        scope='Measured CPU selection with ready T representations and Energy scores; includes input validation and export; extraction is excluded')
    save_json(args.output_dir / 'summary.json', summary)


if __name__ == '__main__':
    main()
