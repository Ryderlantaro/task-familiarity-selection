# Analyze cached selections, representations, ratings, and predictions.
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from scripts.experiment_utils import (
    TASKS, digest, read_json, save_json, load_analysis, window_statistics,
    anchor_percentiles, reference_mean, nearest_starts,
)
from main.tools.data_schema import normalize_record


FIELDS = ('Completeness', 'Information Richness', 'Rarity', 'Complexity', 'Overall Score')


def bootstrap(data, budget, replicates, seed):
    if replicates < 2:
        raise ValueError('At least two bootstrap replicates are required')
    ids = data['gate_indices']
    order, ordered, means = window_statistics(ids, data['comparison_energy'], budget)
    values = anchor_percentiles(ordered, data['anchor_energy'])
    languages = data.get('anchor_languages')
    reference = reference_mean(values, languages)
    start = int(nearest_starts(means, reference)[0])
    groups = [values] if languages is None else [values[languages == name]
                                                for name in np.unique(languages)]
    rng = np.random.default_rng(seed)
    references = np.zeros(replicates)
    for group in groups:
        # Bound temporary bootstrap index allocations for large anchor sets.
        chunk = max(1, min(100, 1000000 // len(group)))
        for begin in range(0, replicates, chunk):
            end = min(begin + chunk, replicates)
            draw = rng.integers(0, len(group), size=(end - begin, len(group)))
            references[begin:end] += group[draw].mean(axis=1) / len(groups)
    starts = nearest_starts(means, references)
    retained = np.maximum(0, budget - abs(starts - start)) / budget
    result = dict(reference=reference, reference_ci95=np.quantile(references, [.025, .975]).tolist(),
        retained_fraction_quantiles=np.quantile(retained, [.025, .5, .975]).tolist(),
        jaccard_quantiles=np.quantile(retained / (2 - retained), [.025, .5, .975]).tolist(),
        anchor_count=len(values), replicates=replicates, seed=seed,
        original_window_start=start, language_groups=len(groups),
        interval_target='Mean familiarity reference under anchor resampling at fixed scores and offsets')
    return result, ids[order[start:start + budget]]


def analyze_bootstrap(args):
    data, manifest = load_analysis(args.selection)
    result, indices = bootstrap(data, manifest['budget'], args.replicates, args.seed)
    expected = np.load(args.selection / f"{manifest['task']}_indices.npy", allow_pickle=False)
    if not np.array_equal(indices, expected):
        raise ValueError('The reconstructed baseline window differs from the saved selection')
    result.update(task=manifest['task'], input_sha256=digest(args.selection / 'analysis_inputs.npz'))
    return result, [args.selection / name for name in
                    ('manifest.json', 'analysis_inputs.npz', f"{manifest['task']}_indices.npy")]


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected a finite representation matrix')
    norms = np.linalg.norm(values, axis=1)
    if (norms <= 0).any():
        raise ValueError('Zero representation')
    return values / norms[:, None]


def effective_clusters(labels, count):
    frequencies = np.bincount(labels, minlength=count)
    probabilities = frequencies[frequencies > 0] / len(labels)
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def pairwise_distance(values):
    if len(values) < 2:
        raise ValueError('At least two vectors are required')
    gram = np.clip(values @ values.T, -1, 1)
    np.fill_diagonal(gram, 1)
    return float((1 - gram).sum() / (len(values) * (len(values) - 1)))


def describe(values):
    return dict(mean=float(np.mean(values)),
                subsample_interval95=np.quantile(values, [.025, .975]).tolist())


def analyze(embeddings, languages, sample_size=128, repeats=30,
            cluster_counts=(10, 20, 30), seeds=(42, 43, 44), within_size=14):
    from sklearn.cluster import MiniBatchKMeans

    if sample_size < 2 or repeats < 1 or not seeds or not cluster_counts:
        raise ValueError('Invalid sampling settings')
    if any(len(x) < sample_size for x in embeddings.values()):
        raise ValueError('Each task must supply the requested sample size')
    if any(k < 2 or k > sample_size * len(embeddings) for k in cluster_counts):
        raise ValueError('Invalid cluster count')
    draws, occupancy = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        fit = np.concatenate([x[rng.choice(len(x), sample_size, replace=False)]
                              for x in embeddings.values()])
        for k in cluster_counts:
            model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                                    batch_size=256, max_iter=100).fit(fit)
            for task, values in embeddings.items():
                labels = model.predict(values)
                frequencies = np.bincount(labels, minlength=k) / len(labels)
                occupancy.append(dict(task=task, k=k, seed=seed,
                    cluster_fractions=frequencies.tolist(),
                    cumulative_coverage=np.cumsum(np.sort(frequencies)[::-1]).tolist()))
                for repeat in range(repeats):
                    ids = rng.choice(len(values), sample_size, replace=False)
                    row = dict(task=task, k=k, seed=seed, repeat=repeat,
                               effective_clusters=effective_clusters(labels[ids], k))
                    if k == 30 or (30 not in cluster_counts and k == max(cluster_counts)):
                        row['cosine_distance'] = pairwise_distance(values[ids])
                    draws.append(row)
    summaries = []
    for task in embeddings:
        for k in cluster_counts:
            group = [row for row in draws if row['task'] == task and row['k'] == k]
            row = dict(task=task, k=k,
                       effective_clusters=describe([r['effective_clusters'] for r in group]))
            if 'cosine_distance' in group[0]:
                row['cosine_distance'] = describe([r['cosine_distance'] for r in group])
            summaries.append(row)
    within = {}
    for task, labels in languages.items():
        x = embeddings[task]
        if labels.shape != (len(x),) or within_size < 2:
            raise ValueError('Language labels must match representation rows')
        groups = [np.flatnonzero(labels == name) for name in np.unique(labels)]
        if any(len(group) < within_size for group in groups):
            raise ValueError('Each language must supply the requested within-language sample')
        rng = np.random.default_rng(2141)
        values = [np.mean([pairwise_distance(x[rng.choice(group, within_size, replace=False)])
                           for group in groups]) for _ in range(repeats)]
        within[task] = dict(cosine_distance=describe(values), languages=len(groups),
                            sample_per_language=within_size)
    return dict(tasks=summaries, cluster_occupancy=occupancy, draws=draws,
                within_language=within, interval_target='Empirical subsample variation, not population confidence intervals')


def analyze_concentration(args):
    tasks = read_json(args.config)['tasks']
    if not tasks or not set(tasks) <= set(TASKS):
        raise ValueError('Invalid task list')
    embeddings, languages, paths = {}, {}, [args.config]
    for task, entry in tasks.items():
        path = (args.config.parent / entry['representations']).resolve()
        embeddings[task] = normalize(np.load(path, allow_pickle=False))
        paths.append(path)
        if 'languages' in entry:
            path = (args.config.parent / entry['languages']).resolve()
            if path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as archive:
                    languages[task] = archive[entry.get('language_key', 'anchor_languages')]
            else:
                languages[task] = np.load(path, allow_pickle=False)
            paths.append(path)
    if len({x.shape[1] for x in embeddings.values()}) != 1:
        raise ValueError('Representation widths differ')
    result = analyze(embeddings, languages, args.sample_size, args.repeats)
    result.update(config_sha256=digest(args.config), sample_size=args.sample_size, repeats=args.repeats)
    return result, paths


def analyze_composition(args):
    data, manifest = load_analysis(args.selection)
    if digest(args.pool) != manifest['pool_sha256']:
        raise ValueError('Pool identity mismatch')
    score_path = args.selection / 'scores.npz'
    if digest(score_path) != manifest['scores_sha256']:
        raise ValueError('Selection scores changed')
    with np.load(score_path, allow_pickle=False) as archive:
        target = archive['target_factor']
    gate = data['gate_indices']
    if (target.ndim != 1 or not np.isfinite(target).all()
            or np.any(gate >= len(target))):
        raise ValueError('Invalid target scores or gate row mapping')
    familiarity = np.full(len(target), np.nan)
    familiarity[gate] = anchor_percentiles(np.sort(data['comparison_energy']), data['comparison_energy'])
    sources, digits = [], []
    with args.pool.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            response = normalize_record(row)['response']
            sources.append(str(row.get('dataset', 'unknown')))
            digits.append(any(char.isdecimal() for char in response))
    if len(sources) != len(target):
        raise ValueError('Pool rows do not match cached scores')
    sources, digits = np.asarray(sources), np.asarray(digits)
    paths = [args.pool, score_path, args.selection / 'manifest.json',
             args.selection / 'analysis_inputs.npz']
    lengths = None
    if args.response_lengths:
        lengths = np.load(args.response_lengths, allow_pickle=False)
        if (lengths.shape != target.shape or not np.isfinite(lengths).all()
                or (lengths < 0).any()):
            raise ValueError('Response token lengths must align with pool rows')
        paths.append(args.response_lengths)
    sets, summaries = {}, {}
    for entry in args.indices:
        name, separator, raw_path = entry.partition('=')
        if not separator or not name or name in sets:
            raise ValueError('Use unique NAME=PATH index inputs')
        path = Path(raw_path)
        ids = np.load(path, allow_pickle=False)
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
                or len(ids) != manifest['budget'] or len(np.unique(ids)) != len(ids)
                or not np.isin(ids, gate).all()):
            raise ValueError('Each selection must contain budget unique gate indices')
        sets[name] = ids
        row = dict(n=len(ids), mean_target_factor=float(target[ids].mean()),
                   mean_familiarity=float(familiarity[ids].mean()),
                   familiarity_range=[float(familiarity[ids].min()), float(familiarity[ids].max())],
                   source_fractions={key: count / len(ids) for key, count in
                                     sorted(Counter(sources[ids]).items())},
                   digit_response_fraction=float(digits[ids].mean()))
        if lengths is not None:
            row.update(mean_response_tokens=float(lengths[ids].mean()),
                       median_response_tokens=float(np.median(lengths[ids])))
        summaries[name] = row
        paths.append(path)
    overlap = []
    for i, (left, a) in enumerate(sets.items()):
        for right, b in list(sets.items())[i + 1:]:
            common = len(np.intersect1d(a, b))
            overlap.append(dict(left=left, right=right, common=common,
                retained_fraction=common / len(a), jaccard=common / (len(a) + len(b) - common)))
    return dict(task=manifest['task'], selections=summaries, overlap=overlap), paths


def load_rows(path, trusted_pt=False):
    if path.suffix == '.pt':
        if not trusted_pt:
            raise ValueError('Loading a local pickle requires --trust-local-pt')
        import torch
        rows = torch.load(path, map_location='cpu', weights_only=False)
    elif path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    else:
        rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty row list')
    return rows


def score_matrix(rows):
    matrix = np.asarray([[row['qwen_scores'][key] for key in FIELDS] for row in rows], dtype=float)
    if not np.isfinite(matrix).all() or np.any((matrix < 1) | (matrix > 10)):
        raise ValueError('Expected finite content ratings in [1, 10]')
    return matrix


def gate_partition(matrix, threshold=7):
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[1] != 5 or not np.isfinite(matrix).all()
            or not 1 <= threshold <= 10):
        raise ValueError('Expected five content rating columns')
    content = (matrix[:, [0, 1, 4]] >= threshold).all(axis=1)
    extra = (matrix[:, [2, 3]] >= threshold).all(axis=1)
    return dict(n=len(matrix), passes_content3=int(content.sum()),
                passes_all5=int((content & extra).sum()),
                rejected_by_rarity_or_complexity=int((content & ~extra).sum()),
                fails_content3=int((~content).sum()))


def matched_contrast(matrix, sources, lengths, selected, gate,
                     cutpoints=(3, 15, 63, 255, 1023), minimum=5):
    strata = defaultdict(lambda: [[], []])
    for i in np.flatnonzero(gate):
        key = (sources[i], int(np.searchsorted(cutpoints, lengths[i], side='left')))
        strata[key][0 if selected[i] else 1].append(i)
    eligible = [(a, b) for a, b in strata.values() if len(a) >= minimum and len(b) >= minimum]
    supported = sum(len(a) for a, _ in eligible)
    result = dict(selected_n=int(selected.sum()), supported_n=supported, strata=len(eligible))
    if supported:
        treated = sum(len(a) / supported * matrix[a].mean(axis=0) for a, _ in eligible)
        control = sum(len(a) / supported * matrix[b].mean(axis=0) for a, b in eligible)
        result.update(selected_scores=dict(zip(FIELDS, treated.tolist())),
                      reweighted_control_scores=dict(zip(FIELDS, control.tolist())),
                      selected_minus_control=dict(zip(FIELDS, (treated - control).tolist())))
    return result


def paired_effect(original, changed, seed=20261030, replicates=5000):
    original, changed = np.asarray(original, float), np.asarray(changed, float)
    if original.ndim != 1 or original.shape != changed.shape or not len(original):
        raise ValueError('Paired ratings must align')
    difference = changed - original
    if not np.isfinite(difference).all() or replicates < 2:
        raise ValueError('Invalid paired bootstrap inputs')
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    chunk = max(1, min(100, 1000000 // len(difference)))
    for begin in range(0, replicates, chunk):
        end = min(begin + chunk, replicates)
        means[begin:end] = rng.choice(difference, (end - begin, len(difference)), replace=True).mean(axis=1)
    return dict(n_sources=len(difference), mean_delta=float(difference.mean()),
                bootstrap_percentile95=np.quantile(means, [.025, .975]).tolist())


def analyze_ratings(args):
    rows = load_rows(args.scores, args.trust_local_pt)
    matrix = score_matrix(rows)
    if args.pool_rows:
        mapping = read_json(args.pool_rows)
        pool_ids = np.asarray(mapping['source_pool_rows'] if isinstance(mapping, dict) else mapping)
    else:
        pool_ids = np.asarray([row['pool_index'] for row in rows])
    if (pool_ids.shape != (len(rows),) or not np.issubdtype(pool_ids.dtype, np.integer)
            or len(np.unique(pool_ids)) != len(pool_ids) or np.any(pool_ids < 0)):
        raise ValueError('Scored rows must have unique aligned pool indices')
    selected_ids = np.load(args.selected_indices, allow_pickle=False)
    gate_ids = np.load(args.gate_indices, allow_pickle=False)
    if not np.isin(selected_ids, gate_ids).all():
        raise ValueError('Selected indices must belong to the supplied gate')
    selected, gate = np.isin(pool_ids, selected_ids), np.isin(pool_ids, gate_ids)
    responses = [next((m['content'] for m in reversed(row['messages'])
                       if m['role'] == 'assistant'), '') for row in rows]
    lengths = np.asarray([len(text.split()) for text in responses])
    sources = np.asarray([row['dataset'] for row in rows])
    result = dict(all_scored=gate_partition(matrix, args.threshold),
                  selected_overlap=gate_partition(matrix[selected], args.threshold),
                  matched_gate_contrast=matched_contrast(matrix, sources, lengths, selected, gate),
                  matched_gate_contrast_finer=matched_contrast(matrix, sources, lengths, selected, gate,
                      cutpoints=(3, 7, 15, 31, 63, 127, 255, 511, 1023)),
                  threshold=args.threshold, length_within_source={})
    passed = (matrix >= args.threshold).all(axis=1)
    for source in np.unique(sources):
        mask = sources == source
        result['length_within_source'][str(source)] = dict(
            passed_n=int((mask & passed).sum()), rejected_n=int((mask & ~passed).sum()),
            passed_median_words=float(np.median(lengths[mask & passed])) if (mask & passed).any() else None,
            rejected_median_words=float(np.median(lengths[mask & ~passed])) if (mask & ~passed).any() else None)
    return result, [args.scores, args.selected_indices, args.gate_indices] + ([args.pool_rows] if args.pool_rows else [])


def analyze_paired(args):
    rows = load_rows(args.scores)
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(row['task_selection'], row['pool_index'])][row['condition']].append(row['scores'])
    conditions = ('original', 'first_3_words', 'shuffled_words')
    if any(set(group) != set(conditions) or any(len(group[c]) != args.judge_repeats for c in conditions)
           for group in groups.values()):
        raise ValueError('Every source must have all conditions and the requested judge repetitions')
    for group in groups.values():
        score_matrix([{'qwen_scores': ratings} for condition in conditions for ratings in group[condition]])
    result = {}
    for task in sorted({key[0] for key in groups}):
        task_groups = [group for (name, _), group in groups.items() if name == task]
        result[task] = {}
        for condition in conditions[1:]:
            result[task][condition] = {
                key: paired_effect(
                    [np.mean([r[key] for r in group['original']]) for group in task_groups],
                    [np.mean([r[key] for r in group[condition]]) for group in task_groups],
                    replicates=args.replicates) for key in FIELDS}
    return dict(paired_ratings=result, judge_repeats=args.judge_repeats), [args.scores]


def read_predictions(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty predictions')
    keys = [key for key in rows[0] if key.endswith(' truth acc')]
    if len(keys) != 1:
        raise ValueError('Expected one evaluated response column')
    response = keys[0][:-len(' truth acc')]
    result = []
    for row in rows:
        truth, info = int(row[f'{response} truth acc']), int(row[f'{response} info acc'])
        joint = int(row[f'{response} truth-info acc'])
        if truth not in (0, 1) or info not in (0, 1) or joint != truth * info:
            raise ValueError('Invalid truth/info flags')
        result.append(dict(question=row['Question'], best_answer=row['Best Answer'],
                           response=row[response].strip(), truth=truth, info=info))
    return result


def prediction_summary(rows):
    return dict(n=len(rows), true_info=sum(r['truth'] * r['info'] for r in rows),
                false_info=sum((1 - r['truth']) * r['info'] for r in rows),
                no_info=sum(1 - r['info'] for r in rows),
                no_comment=sum(r['response'] == 'I have no comment.' for r in rows),
                empty=sum(not r['response'] for r in rows))


def analyze_predictions(args):
    before, after = read_predictions(args.before), read_predictions(args.after)
    if [(r['question'], r['best_answer']) for r in before] != [(r['question'], r['best_answer']) for r in after]:
        raise ValueError('Question identities or ordering differ')
    left, right = prediction_summary(before), prediction_summary(after)
    return dict(before=left, after=right,
                changes={key: right[key] - left[key] for key in left if key != 'n'}), [args.before, args.after]


def analyze_subtasks(args):
    before = [read_json(path) for path in args.before]
    after = [read_json(path) for path in args.after]
    groups = before + after
    names = set(groups[0]) - {'average_exact_match'}
    if not names or any(set(row) - {'average_exact_match'} != names for row in groups):
        raise ValueError('Subtask lists differ between metric files')
    multiplier = 100 if args.unit == 'fraction' else 1
    results = []
    for name in sorted(names):
        values = np.array([row[name] for row in groups], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 100 / multiplier):
            raise ValueError('Invalid subtask score')
        left = float(np.mean([row[name] for row in before]) * multiplier)
        right = float(np.mean([row[name] for row in after]) * multiplier)
        results.append(dict(subtask=name, before=left, after=right, difference=right - left))
    differences = np.array([row['difference'] for row in results])
    return dict(subtasks=results, mean_difference=float(differences.mean()),
                improved=int((differences > 1e-10).sum()), declined=int((differences < -1e-10).sum()),
                unchanged=int((abs(differences) <= 1e-10).sum()),
                before_evaluations=len(before), after_evaluations=len(after)), args.before + args.after


def main():
    parser = argparse.ArgumentParser(description='Analyze cached selections and saved experiment outputs')
    sub = parser.add_subparsers(dest='action', required=True)
    stability = sub.add_parser('bootstrap')
    stability.add_argument('--selection', required=True, type=Path)
    stability.add_argument('--replicates', type=int, default=5000)
    stability.add_argument('--seed', type=int, default=20261111)
    concentration = sub.add_parser('concentration')
    concentration.add_argument('--config', required=True, type=Path)
    concentration.add_argument('--sample-size', type=int, default=128)
    concentration.add_argument('--repeats', type=int, default=30)
    composition = sub.add_parser('composition')
    composition.add_argument('--selection', required=True, type=Path)
    composition.add_argument('--pool', required=True, type=Path)
    composition.add_argument('--indices', required=True, action='append')
    composition.add_argument('--response-lengths', type=Path)
    ratings = sub.add_parser('ratings')
    ratings.add_argument('--scores', type=Path, required=True)
    ratings.add_argument('--pool-rows', type=Path)
    ratings.add_argument('--selected-indices', type=Path, required=True)
    ratings.add_argument('--gate-indices', type=Path, required=True)
    ratings.add_argument('--threshold', type=float, default=7)
    ratings.add_argument('--trust-local-pt', action='store_true')
    paired = sub.add_parser('paired-ratings')
    paired.add_argument('--scores', type=Path, required=True)
    paired.add_argument('--judge-repeats', type=int, default=3)
    paired.add_argument('--replicates', type=int, default=5000)
    predictions = sub.add_parser('predictions')
    predictions.add_argument('--before', type=Path, required=True)
    predictions.add_argument('--after', type=Path, required=True)
    subtasks = sub.add_parser('subtasks')
    subtasks.add_argument('--before', type=Path, nargs='+', required=True)
    subtasks.add_argument('--after', type=Path, nargs='+', required=True)
    subtasks.add_argument('--unit', choices=('percent', 'fraction'), default='percent')
    for entry in (stability, concentration, composition, ratings, paired, predictions, subtasks):
        entry.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    result, paths = {'bootstrap': analyze_bootstrap, 'concentration': analyze_concentration,
                     'composition': analyze_composition,
                     'ratings': analyze_ratings, 'paired-ratings': analyze_paired,
                     'predictions': analyze_predictions, 'subtasks': analyze_subtasks}[args.action](args)
    result.update(source_sha256={str(path): digest(path) for path in paths})
    args.output_dir.mkdir(parents=True)
    save_json(args.output_dir / 'summary.json', result)


if __name__ == '__main__':
    main()
