# Select control datasets and write their training/evaluation plan.
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from main.adapters.target import core_plus_cluster_indices
from main.pipeline import materialize_jsonl
from scripts.experiment_utils import (
    TASKS, ENGLISH_TASKS, digest, read_json, save_json, load_analysis,
    window_statistics, anchor_percentiles, reference_mean, nearest_starts,
)


def band_selections(gate, comparison, scores, labels, budget):
    # Cut the sorted population into equal-count bands, from low to high familiarity.
    if len(gate) % 5 or len(gate) // 5 < budget:
        raise ValueError('Each of five equal-count bands must supply the budget')
    order = np.lexsort((gate, -np.asarray(comparison)))
    result = []
    for members in np.split(np.asarray(gate)[order], 5):
        result.append(core_plus_cluster_indices(
            scores, cluster_labels=labels, budget=budget,
            supplement_fraction=.15, eligible_indices=members))
    return result


def shuffled_scores(anchors, seed):
    if set(anchors) != set(ENGLISH_TASKS):
        raise ValueError('Reassignment requires the four English tasks')
    joined = np.concatenate([anchors[task] for task in ENGLISH_TASKS])
    assignment = np.random.default_rng(seed).permutation(len(joined))
    permuted = joined[assignment]
    result, start = {}, 0
    for task in ENGLISH_TASKS:
        result[task] = permuted[start:start + len(anchors[task])]
        start += len(anchors[task])
    return result, assignment


def gate_random(gate, budget, seed):
    gate = np.asarray(gate)
    if (gate.ndim != 1 or len(np.unique(gate)) != len(gate)
            or not 0 < budget <= len(gate)):
        raise ValueError('Invalid random-sampling inputs')
    return np.random.default_rng(seed).choice(gate, budget, replace=False)


def shifted_windows(gate, comparison, budget, reference, shift=.2):
    if not np.isfinite(reference) or not 0 <= reference <= 1:
        raise ValueError('Invalid reference')
    if not np.isfinite(shift) or not 0 < shift < 1:
        raise ValueError('Window shift must lie between zero and one')
    order, ordered, means = window_statistics(gate, comparison, budget)
    step = int(round(len(gate) * shift))
    if step < 1:
        raise ValueError('Window shift is smaller than one candidate')
    baseline = int(nearest_starts(means, reference)[0])
    result = {}
    for name, offset in (('lower_window', step), ('higher_window', -step)):
        start = int(np.clip(baseline + offset, 0, len(gate) - budget))
        positions = order[start:start + budget]
        indices = np.asarray(gate)[positions]
        familiarity = anchor_percentiles(ordered, np.asarray(comparison)[positions])
        result[name] = (indices, dict(
            reference=float(reference), requested_shift_fraction=float(shift),
            baseline_start=baseline, window_start=start,
            shift_rows=start - baseline, boundary_clipped=start != baseline + offset,
            selected_mean_familiarity=float(means[start]),
            familiarity_min=float(familiarity.min()), familiarity_max=float(familiarity.max())))
    return result


def main():
    parser = argparse.ArgumentParser(description='Prepare experimental selections')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    tasks = config['tasks']
    conditions = config.get('conditions', ['t_only', 'bands', 'references', 'gate_random', 'shifted_windows'])
    if not tasks or not set(tasks) <= set(TASKS) or not conditions or not set(conditions) <= {'t_only', 'bands', 'references', 'gate_random', 'shifted_windows'}:
        raise ValueError('Invalid tasks or conditions')
    if len(conditions) != len(set(conditions)):
        raise ValueError('Duplicate conditions')
    budget = config.get('budget', 10000)
    seed = config.get('training_seed', 42)
    repeats = config.get('evaluation_repeats', 1)
    if not isinstance(budget, int) or budget < 1 or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('Budget and evaluation repeats must be positive integers')
    if 'references' in conditions and not set(ENGLISH_TASKS) <= set(tasks):
        raise ValueError('Reference controls require all four English tasks')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    prepared, jobs = {}, []
    for task, entry in tasks.items():
        cache = (args.config.parent / entry['cache']).resolve()
        baseline = root / task / 'task_reference'
        subprocess.run([sys.executable, '-m', 'scripts.select_data', '--cache', str(cache),
                        '--task', task, '--budget', str(budget), '--output-dir', str(baseline)], check=True)
        data, manifest = load_analysis(baseline)
        cache_manifest = cache / f'manifest_{task}.json'
        if digest(cache_manifest) != manifest['cache_manifest_sha256']:
            raise ValueError('Cache manifest changed')
        spec = read_json(cache_manifest)['spec']
        score_path = baseline / 'scores.npz'
        if digest(score_path) != manifest['scores_sha256']:
            raise ValueError('Selection scores changed')
        with np.load(score_path, allow_pickle=False) as archive:
            scores, labels = archive['target_factor'], archive['cluster_labels']
        order, ordered, means = window_statistics(data['gate_indices'], data['comparison_energy'], budget)
        af = anchor_percentiles(ordered, data['anchor_energy'])
        prepared[task] = dict(data=data, spec=spec, scores=scores, labels=labels,
                              order=order, ordered=ordered, means=means,
                              reference=reference_mean(af, data.get('anchor_languages')))

        def export(condition, indices, extra=None):
            folder = root / task / condition
            folder.mkdir(parents=True, exist_ok=False)
            indices = np.asarray(indices, dtype=np.int64)
            if len(indices) != budget or len(np.unique(indices)) != budget:
                raise ValueError('Selection does not satisfy the unique-example budget')
            np.save(folder / f'{task}_indices.npy', indices)
            path = folder / f'{task}.jsonl'
            materialize_jsonl(spec['sources']['pool']['path'], indices, path)
            save_json(folder / 'manifest.json', dict(
                task=task, condition=condition, budget=budget, train_sha256=digest(path),
                indices_sha256=digest(folder / f'{task}_indices.npy'),
                baseline_manifest_sha256=digest(baseline / 'manifest.json'), **(extra or {})))
            return folder

        def job(condition, folder):
            path = folder / f'{task}.jsonl'
            jobs.append(dict(task=task, condition=condition, seed=seed,
                             base_model=spec['identity']['model'], train_file=str(path),
                             train_sha256=digest(path), evaluation_repeats=repeats,
                             run_name=f'{task}_{condition}_seed{seed}'))

        job('task_reference', baseline)
        if 'gate_random' in conditions:
            sampling_seed = config.get('sampling_seed', 42)
            indices = gate_random(data['gate_indices'], budget, sampling_seed)
            job('gate_random', export('gate_random', indices, dict(
                sampling_seed=sampling_seed, sampling='uniform without replacement')))
        if 'shifted_windows' in conditions:
            windows = shifted_windows(data['gate_indices'], data['comparison_energy'],
                budget, prepared[task]['reference'], config.get('window_shift', .2))
            for condition, (indices, details) in windows.items():
                job(condition, export(condition, indices, details))
        if 't_only' in conditions:
            indices = np.argsort(scores, kind='mergesort')[-budget:][::-1]
            job('t_only', export('t_only', indices))
        if 'bands' in conditions:
            bands = band_selections(data['gate_indices'], data['comparison_energy'], scores, labels, budget)
            for band, indices in enumerate(bands):
                name = f'band_{band}'
                folder = export(name, indices, dict(
                    band=band, familiarity_bounds=[band / 5, (band + 1) / 5],
                    population_per_band=len(data['gate_indices']) // 5,
                    tie_rule='pool index ascending at equal comparison Energy'))
                job(name, folder)
    if 'references' in conditions:
        pool_ids = {p['spec']['sources']['pool']['sha256'] for p in prepared.values()}
        models = {json.dumps(p['spec']['identity'], sort_keys=True) for p in prepared.values()}
        if len(pool_ids) != 1 or len(models) != 1:
            raise ValueError('Reference controls require the same pool and model identity')
        anchors = {task: prepared[task]['data']['anchor_energy'] for task in ENGLISH_TASKS}
        shuffled, assignment = shuffled_scores(anchors, config.get('shuffle_seed', 2601))
        np.save(root / 'reference_assignment.npy', assignment)
        uniform = prepared['gsm8k']['reference']
        for task in ENGLISH_TASKS:
            item = prepared[task]
            reference = float(anchor_percentiles(item['ordered'], shuffled[task]).mean())
            for condition, value in (('reassigned_reference', reference),
                                     ('fixed_centre', .5), ('gsm8k_reference', uniform)):
                start = int(nearest_starts(item['means'], value)[0])
                indices = item['data']['gate_indices'][item['order'][start:start + budget]]
                folder = root / task / condition
                folder.mkdir()
                path = folder / f'{task}.jsonl'
                np.save(folder / f'{task}_indices.npy', indices)
                materialize_jsonl(item['spec']['sources']['pool']['path'], indices, path)
                save_json(folder / 'manifest.json', dict(task=task, condition=condition,
                    budget=budget, reference=value, train_sha256=digest(path),
                    indices_sha256=digest(folder / f'{task}_indices.npy'),
                    baseline_manifest_sha256=digest(root / task / 'task_reference/manifest.json')))
                jobs.append(dict(task=task, condition=condition, seed=seed,
                    base_model=item['spec']['identity']['model'], train_file=str(path),
                    train_sha256=digest(path), evaluation_repeats=repeats,
                    run_name=f'{task}_{condition}_seed{seed}'))
    save_json(root / 'runs.json', dict(config_sha256=digest(args.config), jobs=jobs,
        band_training_jobs=sum(job['condition'].startswith('band_') for job in jobs),
        budget=budget, training_seed=seed))
    print(json.dumps(dict(jobs=len(jobs), band_training_jobs=sum(
        job['condition'].startswith('band_') for job in jobs)), indent=2))


if __name__ == '__main__':
    main()
