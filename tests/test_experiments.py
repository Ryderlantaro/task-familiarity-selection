# Synthetic CPU checks for experiment planning and cached analyses.
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
from scripts import select_controls as prepare
from scripts import train_eval as runner
from scripts import analyze_experiments as boot
from scripts import analyze_experiments as distributions
from scripts import analyze_experiments as content
from scripts import rating_inputs
from scripts import evaluate_answer_cue as answer_cue
from scripts import benchmark_selection as benchmark
from scripts.experiment_utils import digest, save_json, read_json, window_statistics, nearest_starts
from main.adapters.sparsity_alignment import mean_familiarity_selection


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_nearest_window_matches_exhaustive_with_ties(self):
        rng = np.random.default_rng(7)
        scores = rng.integers(0, 10, 200).astype(float)
        _, _, means = window_statistics(np.arange(200), scores, 20)
        references = np.r_[rng.random(100), means[0], means[-1], means[20]]
        expected = [int(np.argmin(abs(means - value))) for value in references]
        np.testing.assert_array_equal(nearest_starts(means, references), expected)

    def test_bands_equal_count_disjoint_and_task_gate(self):
        gate = np.arange(10, 70)
        bands = prepare.band_selections(gate, np.linspace(1, 0, 60),
                    np.arange(100, dtype=float), np.arange(100) % 4, 10)
        self.assertEqual(len(bands), 5)
        self.assertTrue(all(len(ids) == 10 for ids in bands))
        self.assertEqual(len(np.unique(np.concatenate(bands))), 50)
        self.assertTrue(all(np.isin(ids, gate[i * 12:(i + 1) * 12]).all()
                            for i, ids in enumerate(bands)))

    def test_small_bands_rejected(self):
        with self.assertRaises(ValueError):
            prepare.band_selections(np.arange(60), np.arange(60), np.arange(60), np.arange(60) % 4, 13)

    def test_gate_random_matches_seeded_sampling(self):
        gate = np.arange(10, 70)
        selected = prepare.gate_random(gate, 10, 42)
        np.testing.assert_array_equal(selected, np.random.default_rng(42).choice(gate, 10, replace=False))
        self.assertEqual(len(np.unique(selected)), 10)
        self.assertTrue(np.isin(selected, gate).all())
        with self.assertRaises(ValueError):
            prepare.gate_random(gate, 61, 42)

    def test_training_queue_keeps_original_prompt(self):
        train = self.root / 'train.jsonl'
        train.write_text('{}\n')
        plan = self.root / 'runs.json'
        save_json(plan, dict(budget=10000, jobs=[dict(
            task='truthfulqa', condition='task_reference', seed=42,
            base_model='test-model', train_file=str(train), train_sha256=digest(train),
            evaluation_repeats=1, run_name='truthfulqa_task_reference_seed42')]))
        with patch.object(sys, 'argv', ['run', '--plan', str(plan), '--execute']), \
             patch.dict(os.environ, {'TRUTHFULQA_PROMPT': 'answer_cue',
                                     'TRUTHFULQA_RUNNER': '/tmp/custom_runner.py'}), \
             patch.object(runner.subprocess, 'run') as run, \
             contextlib.redirect_stdout(io.StringIO()):
            runner.main()
        self.assertEqual(run.call_count, 2)
        env = run.call_args_list[-1].kwargs['env']
        self.assertEqual(env['TRUTHFULQA_PROMPT'], 'original')
        self.assertEqual(env['TRUTHFULQA_RUNNER'], '')
        self.assertEqual(env['DATA_PROTOCOL'], 'heldout')

    def test_shifted_windows_move_rows_and_preserve_budget(self):
        gate = np.arange(10, 70)
        energy = np.linspace(.1, .9, 60)
        windows = prepare.shifted_windows(gate, energy, 10, .5)
        for name, offset, expected_mean in [('lower_window', 12, .3), ('higher_window', -12, .7)]:
            ids, report = windows[name]
            self.assertEqual(report['shift_rows'], offset)
            self.assertFalse(report['boundary_clipped'])
            self.assertEqual(len(np.unique(ids)), 10)
            self.assertTrue(np.isin(ids, gate).all())
            self.assertTrue(np.all(np.diff(ids) == 1))
            self.assertLess(abs(report['selected_mean_familiarity'] - expected_mean), 1 / 60)

    def test_shifted_windows_report_boundaries_and_reject_bad_shift(self):
        gate = np.arange(60)
        energy = np.linspace(.1, .9, 60)
        ids, report = prepare.shifted_windows(gate, energy, 10, 0.)['lower_window']
        self.assertTrue(report['boundary_clipped'])
        self.assertEqual(report['window_start'], 50)
        np.testing.assert_array_equal(ids, gate[-10:])
        for shift in (0, -.2, 1., float('nan')):
            with self.assertRaises(ValueError):
                prepare.shifted_windows(gate, energy, 10, .5, shift)

    def test_shuffle_preserves_population_and_task_counts(self):
        anchors = {task: np.arange(i + 2) + i * 10 for i, task in enumerate(prepare.ENGLISH_TASKS)}
        shuffled, assignment = prepare.shuffled_scores(anchors, 2601)
        self.assertEqual({k: len(v) for k, v in anchors.items()}, {k: len(v) for k, v in shuffled.items()})
        np.testing.assert_array_equal(np.sort(np.concatenate(list(anchors.values()))),
                                      np.sort(np.concatenate(list(shuffled.values()))))
        np.testing.assert_array_equal(np.sort(assignment), np.arange(len(assignment)))

    def test_bootstrap_reconstructs_and_bounds_overlap(self):
        data = dict(gate_indices=np.arange(60), comparison_energy=np.linspace(.1, .9, 60),
                    anchor_energy=np.array([.3, .4, .5, .7]))
        result, selected = boot.bootstrap(data, 10, 100, 4)
        expected, _ = mean_familiarity_selection(data['gate_indices'], data['comparison_energy'],
                                                 data['anchor_energy'], budget=10)
        np.testing.assert_array_equal(selected, expected)
        self.assertTrue(all(0 <= value <= 1 for value in result['retained_fraction_quantiles']))

    def test_bootstrap_language_weights(self):
        data = dict(gate_indices=np.arange(60), comparison_energy=np.linspace(.1, .9, 60),
                    anchor_energy=np.array([.2, .2, .2, .8]),
                    anchor_languages=np.array(['en', 'en', 'en', 'ru']))
        result, _ = boot.bootstrap(data, 10, 20, 3)
        self.assertEqual(result['language_groups'], 2)
        self.assertAlmostEqual(result['reference_ci95'][0], result['reference_ci95'][1])

    def test_concentration_measures(self):
        vectors = distributions.normalize(np.array([[1., 0.], [1., 0.]]))
        self.assertAlmostEqual(distributions.pairwise_distance(vectors), 0)
        self.assertAlmostEqual(distributions.effective_clusters(np.array([0, 0, 1, 1]), 2), 2)
        with self.assertRaises(ValueError):
            distributions.normalize(np.zeros((2, 3)))

    def composition_fixture(self):
        selection = self.root / 'selection'
        selection.mkdir()
        pool = self.root / 'pool.jsonl'
        rows = [dict(dataset='a' if i < 4 else 'b', instruction='question',
                     output='1' if i % 2 else 'answer') for i in range(8)]
        pool.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        np.savez(selection / 'analysis_inputs.npz', gate_indices=np.arange(6),
                 gate_energy=np.arange(6, dtype=float), comparison_energy=np.arange(6, dtype=float),
                 anchor_energy=np.array([2., 3.]))
        np.savez(selection / 'scores.npz', target_factor=np.arange(8, dtype=float))
        save_json(selection / 'manifest.json', dict(task='gsm8k', budget=2,
            pool_sha256=digest(pool), analysis_sha256=digest(selection / 'analysis_inputs.npz'),
            scores_sha256=digest(selection / 'scores.npz')))
        a, b = self.root / 'a.npy', self.root / 'b.npy'
        np.save(a, [0, 1])
        np.save(b, [1, 4])
        lengths = self.root / 'lengths.npy'
        np.save(lengths, np.arange(8))
        return SimpleNamespace(selection=selection, pool=pool,
            indices=[f'a={a}', f'b={b}'], response_lengths=lengths)

    def test_composition_overlap_sources_scores_and_lengths(self):
        args = self.composition_fixture()
        result, paths = content.analyze_composition(args)
        a = result['selections']['a']
        self.assertEqual(a['source_fractions'], {'a': 1.})
        self.assertEqual(a['digit_response_fraction'], .5)
        self.assertEqual(a['mean_target_factor'], .5)
        self.assertEqual(a['mean_response_tokens'], .5)
        self.assertAlmostEqual(a['mean_familiarity'], .75)
        self.assertEqual(result['overlap'][0]['retained_fraction'], .5)
        self.assertAlmostEqual(result['overlap'][0]['jaccard'], 1 / 3)
        self.assertIn(args.response_lengths, paths)

    def test_composition_rejects_changed_pool_and_bad_indices(self):
        args = self.composition_fixture()
        path = self.root / 'a.npy'
        for ids in ([0, 0], [0, 7], [0., 1.]):
            np.save(path, ids)
            with self.assertRaises(ValueError):
                content.analyze_composition(args)
        np.save(path, [0, 1])
        args.pool.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'Pool identity'):
            content.analyze_composition(args)

    def test_composition_cli_and_output_protection(self):
        args = self.composition_fixture()
        out = self.root / 'analysis'
        argv = ['analysis', 'composition', '--selection', str(args.selection),
                '--pool', str(args.pool), '--output-dir', str(out)]
        for item in args.indices:
            argv.extend(['--indices', item])
        with patch.object(sys, 'argv', argv):
            content.main()
        result = read_json(out / 'summary.json')
        self.assertEqual(result['overlap'][0]['common'], 1)
        self.assertEqual(result['source_sha256'][str(args.pool)], digest(args.pool))
        with patch.object(sys, 'argv', argv), self.assertRaises(FileExistsError):
            content.main()

    def test_bootstrap_cli_reconstructs_saved_window(self):
        args = self.composition_fixture()
        data, _ = content.load_analysis(args.selection)
        _, ids = content.bootstrap(data, 2, 20, 4)
        np.save(args.selection / 'gsm8k_indices.npy', ids)
        out = self.root / 'bootstrap'
        with patch.object(sys, 'argv', ['analysis', 'bootstrap', '--selection', str(args.selection),
                '--replicates', '20', '--seed', '4', '--output-dir', str(out)]):
            content.main()
        result = read_json(out / 'summary.json')
        self.assertEqual(result['replicates'], 20)
        self.assertEqual(result['task'], 'gsm8k')

    def test_concentration_cli_loads_representation_configuration(self):
        vectors = self.root / 'vectors.npy'
        np.save(vectors, [[1., 0.], [0., 1.]])
        config = self.root / 'concentration.json'
        save_json(config, dict(tasks=dict(gsm8k=dict(representations=vectors.name))))
        out = self.root / 'concentration'
        with patch.object(sys, 'argv', ['analysis', 'concentration', '--config', str(config),
                '--output-dir', str(out)]), \
             patch.object(content, 'analyze', return_value=dict(tasks=[])) as analyze:
            content.main()
        np.testing.assert_array_equal(analyze.call_args.args[0]['gsm8k'], [[1., 0.], [0., 1.]])
        self.assertIn(str(vectors), read_json(out / 'summary.json')['source_sha256'])

    def test_task_distributions_run_on_synthetic_arrays(self):
        rng = np.random.default_rng(3)
        values = {task: distributions.normalize(rng.normal(size=(30, 5))) for task in ('gsm8k', 'bbh')}
        result = distributions.analyze(values, {}, sample_size=10, repeats=2,
                                       cluster_counts=(2, 3), seeds=(42,), within_size=2)
        self.assertEqual(len(result['tasks']), 4)
        self.assertTrue(all(abs(row['cumulative_coverage'][-1] - 1) < 1e-10
                            for row in result['cluster_occupancy']))

    def test_filter_partitions(self):
        matrix = np.array([[8, 8, 8, 8, 8], [8, 8, 6, 8, 8], [6, 8, 8, 8, 8]])
        result = content.gate_partition(matrix)
        self.assertEqual(result['passes_all5'], 1)
        self.assertEqual(result['rejected_by_rarity_or_complexity'], 1)
        self.assertEqual(result['fails_content3'], 1)
        with self.assertRaises(ValueError):
            content.gate_partition(matrix, float('nan'))

    def test_matched_scores_and_paired_effect(self):
        selected = np.array([True] * 5 + [False] * 5)
        matrix = np.array([[8.] * 5] * 5 + [[7.] * 5] * 5)
        result = content.matched_contrast(matrix, np.array(['a'] * 10),
                    np.array([30] * 10), selected, np.ones(10, dtype=bool))
        self.assertTrue(all(value == 1 for value in result['selected_minus_control'].values()))
        result = content.paired_effect([8, 8, 8], [5, 5, 5], replicates=20)
        self.assertEqual(result['bootstrap_percentile95'], [-3., -3.])

    def test_source_sample_and_response_variants(self):
        rows = [dict(dataset='a' if i < 8 else 'b') for i in range(10)]
        ids = rating_inputs.source_sample(rows, 5, 42)
        self.assertEqual(sum(rows[i]['dataset'] == 'a' for i in ids), 4)
        original = dict(messages=[dict(role='user', content='question'),
                                  dict(role='assistant', content=' '.join(str(i) for i in range(40)))],
                        generated_content={'Task': 'reasoning'})
        variants = rating_inputs.response_variants(original, 4)
        self.assertEqual(len(variants['first_3_words']['messages'][-1]['content'].split()), 3)
        self.assertEqual(sorted(variants['shuffled_words']['messages'][-1]['content'].split()),
                         sorted(original['messages'][-1]['content'].split()))
        self.assertEqual(variants['original'], original)
        self.assertEqual(variants['first_3_words']['generated_content'], original['generated_content'])

    def test_saved_subtask_comparison(self):
        a, b = self.root / 'a.json', self.root / 'b.json'
        save_json(a, dict(task1=40, task2=60, average_exact_match=50))
        save_json(b, dict(task1=60, task2=50, average_exact_match=55))
        result, _ = content.analyze_subtasks(SimpleNamespace(before=[a], after=[b], unit='percent'))
        self.assertEqual(result['mean_difference'], 5)
        self.assertEqual(result['improved'], 1)
        self.assertEqual(result['declined'], 1)
        save_json(b, dict(task3=40))
        with self.assertRaises(ValueError):
            content.analyze_subtasks(SimpleNamespace(before=[a], after=[b], unit='percent'))

    def test_saved_prediction_comparison(self):
        import csv
        before, after = self.root / 'before.csv', self.root / 'after.csv'
        fields = ['Question', 'Best Answer', 'model', 'model truth acc',
                  'model info acc', 'model truth-info acc']
        for path, truth, info in [(before, 1, 1), (after, 1, 0)]:
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, ['q', 'a', 'answer', truth, info, truth * info])))
        result, _ = content.analyze_predictions(SimpleNamespace(before=before, after=after))
        self.assertEqual(result['changes']['no_info'], 1)
        self.assertEqual(result['changes']['true_info'], -1)

    def cue_modules(self):
        package = ModuleType('eval')
        task = ModuleType('eval.truthfulqa')
        utilities = ModuleType('eval.truthfulqa.utilities')
        utilities.format_prompt = lambda question, **kwargs: 'Q: ' + question
        task.utilities = utilities
        package.truthfulqa = task
        return utilities, {'eval': package, 'eval.truthfulqa': task,
                            'eval.truthfulqa.utilities': utilities}

    def test_answer_cue_changes_only_prompt(self):
        utilities, modules = self.cue_modules()
        original = utilities.format_prompt
        argv = ['cue', '--model_name_or_path', 'model', '--metrics', 'truth', 'info', '--preset', 'qa']
        recorded = {}

        def evaluate(*args, **kwargs):
            recorded['prompt'] = utilities.format_prompt('question', preset='qa')
            recorded['argv'] = sys.argv.copy()

        with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv), \
             patch.dict('os.environ', {'TRUTHFULQA_LOW_MEMORY': '0'}), \
             patch.object(answer_cue.runpy, 'run_module', side_effect=evaluate) as run:
            answer_cue.main()
            self.assertIs(sys.argv, argv)
        self.assertEqual(recorded['prompt'], 'Q: question\nA:')
        self.assertEqual(recorded['argv'][-3:], ['--metrics', 'truth', 'info'])
        self.assertIn('--preset', recorded['argv'])
        self.assertIs(utilities.format_prompt, original)
        run.assert_called_once_with('eval.truthfulqa.run_eval', run_name='__main__')

    def test_answer_cue_low_memory_and_cleanup_on_error(self):
        utilities, modules = self.cue_modules()
        original = utilities.format_prompt
        argv = ['cue', '--model_name_or_path', 'model']
        with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv), \
             patch.dict('os.environ', {'TRUTHFULQA_LOW_MEMORY': '1'}), \
             patch.object(answer_cue.runpy, 'run_path', side_effect=RuntimeError('test stop')) as run:
            with self.assertRaises(RuntimeError):
                answer_cue.main()
            self.assertIs(sys.argv, argv)
        self.assertIs(utilities.format_prompt, original)
        self.assertEqual(Path(run.call_args.args[0]).name, 'evaluate_truthfulqa_low_memory.py')

    def test_answer_cue_rejects_mc_and_chat(self):
        for extra in (['--metrics', 'mc'], ['--use_chat_format']):
            with patch.object(sys, 'argv', ['cue', *extra]), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 patch.object(answer_cue.runpy, 'run_module') as run:
                with self.assertRaises(SystemExit) as error:
                    answer_cue.main()
                self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def shell_eval(self, mode):
        backend = self.root / 'backend'
        (backend / 'model_finetune/eval').mkdir(parents=True, exist_ok=True)
        (backend / 'data/eval/split_eval').mkdir(parents=True, exist_ok=True)
        model = self.root / 'model'
        model.mkdir(exist_ok=True)
        executable = self.root / 'mock_python'
        executable.write_text('#!' + sys.executable + '\n'
            'import json,os,sys\n'
            'with open(os.environ["CAPTURE"], "a") as stream:\n'
            '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n')
        executable.chmod(0o700)
        capture = self.root / f'{mode}_commands.jsonl'
        env = dict(os.environ, PROJECT=str(BUNDLE), ROOT=str(self.root),
                   BACKEND_ROOT=str(backend), MODEL_DIR=str(model), PYTHON=str(executable),
                   RUN_NAME=mode, TASKS='truthfulqa', CAPTURE=str(capture),
                   RESULTS_ROOT=str(self.root / mode), HF_HOME=str(self.root / 'hf'),
                   MERGED_DIR=str(self.root / 'temporary_merge' / mode),
                   TRUTHFULQA_PROMPT=mode, TRUTHFULQA_LOW_MEMORY='0', OVERWRITE='0',
                   METRICS_ROOT=str(BUNDLE / 'compat_metrics'))
        env.pop('TRUTHFULQA_RUNNER', None)
        subprocess.run(['bash', str(BUNDLE / 'scripts/evaluate_tasks.sh')], env=env,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return [json.loads(line) for line in capture.read_text().splitlines()]

    def test_shell_keeps_original_evaluation_default(self):
        commands = self.shell_eval('original')
        command = next(row for row in commands if row[:2] == ['-m', 'eval.truthfulqa.run_eval'])
        self.assertIn('mc', command)
        self.assertTrue(any('/split_eval/truthfulqa' in argument for argument in command))

    def test_shell_routes_answer_cue_without_mc(self):
        commands = self.shell_eval('answer_cue')
        command = next(row for row in commands if row[0].endswith('/evaluate_answer_cue.py'))
        self.assertEqual(command[command.index('--metrics') + 1:command.index('--preset')], ['truth', 'info'])
        self.assertNotIn('mc', command)

    def test_cached_selection_benchmark(self):
        rng = np.random.default_rng(4)
        pool = self.root / 'pool.jsonl'
        pool.write_text(''.join(json.dumps(dict(messages=[dict(role='user', content=str(i)),
                    dict(role='assistant', content='answer')])) + '\n' for i in range(120)))
        values = dict(candidate_t=rng.normal(size=(120, 8)).astype(np.float16),
                      anchor_t=rng.normal(size=(4, 8)).astype(np.float16),
                      candidate_energy=np.linspace(.1, .9, 120), anchor_energy=np.array([.3, .4, .5, .6]))
        argv = ['benchmark', '--pool', str(pool), '--task', 'gsm8k', '--budget', '10',
                '--repeats', '2', '--output-dir', str(self.root / 'timing')]
        for key, array in values.items():
            path = self.root / f'{key}.npy'
            np.save(path, array)
            argv.extend(['--' + key.replace('_', '-'), str(path)])
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            benchmark.main()
        result = read_json(self.root / 'timing/summary.json')
        self.assertEqual(len(result['repetitions']), 2)
        self.assertGreater(result['stages']['total']['mean'], 0)
        np.testing.assert_array_equal(np.load(self.root / 'timing/repeat_1/indices.npy'),
                                      np.load(self.root / 'timing/repeat_2/indices.npy'))

    def test_benchmark_rejects_invalid_vectors_and_row_mapping(self):
        with self.assertRaises(ValueError):
            benchmark.valid_vectors(np.zeros((4, 8)))
        path = self.root / 'energy.npz'
        np.savez(path, energy=np.array([.3, .4]), pool_indices=np.array([1, 0]))
        with self.assertRaises(AssertionError):
            benchmark.load_array(path, 'energy')

    def test_five_tasks_produce_25_band_jobs(self):
        pool = self.root / 'pool.jsonl'
        pool.write_text(''.join(json.dumps(dict(messages=[dict(role='user', content=str(i)),
                    dict(role='assistant', content='answer')])) + '\n' for i in range(120)))
        cache = self.root / 'cache'
        cache.mkdir()
        for task in prepare.TASKS:
            save_json(cache / f'manifest_{task}.json', dict(spec=dict(identity=dict(model='test-model'),
                sources=dict(pool=dict(path=str(pool), sha256=digest(pool))))))
        config = self.root / 'config.json'
        save_json(config, dict(budget=10, conditions=['t_only', 'bands', 'references', 'gate_random', 'shifted_windows'],
                              tasks={task: dict(cache='cache') for task in prepare.TASKS}))
        output = self.root / 'selections'

        def fake_selection(command, **kwargs):
            task = command[command.index('--task') + 1]
            out = Path(command[command.index('--output-dir') + 1])
            out.mkdir(parents=True)
            start = prepare.TASKS.index(task) * 5
            gate = np.arange(start, start + 60)
            energy = np.linspace(.1, .9, 120)
            anchor = np.array([.3, .4, .5, .6])
            selected, report = mean_familiarity_selection(gate, energy[gate], anchor, budget=10)
            analysis = dict(gate_indices=gate, gate_energy=energy[gate],
                            comparison_energy=energy[gate], anchor_energy=anchor)
            if task == 'tydiqa':
                analysis['anchor_languages'] = np.array(['en', 'en', 'ru', 'ru'])
            np.savez(out / 'analysis_inputs.npz', **analysis)
            np.savez(out / 'scores.npz', target_factor=np.linspace(0, 1, 120), cluster_labels=np.arange(120) % 4)
            np.save(out / f'{task}_indices.npy', selected)
            lines = pool.read_text().splitlines()
            (out / f'{task}.jsonl').write_text(''.join(lines[i] + '\n' for i in selected))
            save_json(out / 'manifest.json', dict(task=task, budget=10,
                analysis_sha256=digest(out / 'analysis_inputs.npz'),
                scores_sha256=digest(out / 'scores.npz'),
                cache_manifest_sha256=digest(cache / f'manifest_{task}.json')))

        with patch.object(sys, 'argv', ['prepare', '--config', str(config), '--output-dir', str(output)]), \
             patch.object(prepare.subprocess, 'run', side_effect=fake_selection), contextlib.redirect_stdout(io.StringIO()):
            prepare.main()
        plan = read_json(output / 'runs.json')
        self.assertEqual(plan['band_training_jobs'], 25)
        self.assertEqual(len(plan['jobs']), 62)
        self.assertEqual(len({row['run_name'] for row in plan['jobs']}), 62)
        for row in plan['jobs']:
            ids = np.load(Path(row['train_file']).with_name(f"{row['task']}_indices.npy"))
            self.assertEqual(len(np.unique(ids)), 10)
            self.assertEqual(digest(row['train_file']), row['train_sha256'])
            if row['condition'].startswith('band_') or row['condition'] in ('gate_random', 'lower_window', 'higher_window'):
                gate = np.load(output / row['task'] / 'task_reference/analysis_inputs.npz')['gate_indices']
                self.assertTrue(np.isin(ids, gate).all())
        with patch.object(sys, 'argv', ['run', '--plan', str(output / 'runs.json'), '--condition', 'band_0']), \
             patch.object(runner.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            runner.main()
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
