# Experimental Controls and Analyses

Run commands from the package root. Configuration paths are relative to the
configuration file. Use new output directories.

| Script | Purpose |
|---|---|
| `select_controls.py` | Select T-only, random, band, and window-control datasets |
| `train_eval.py` | Train and evaluate planned adapters |
| `analyze_experiments.py` | Analyze anchor stability, task concentration, selection composition, ratings, and predictions |
| `rating_inputs.py` | Sample scoring inputs and construct response variants |
| `aggregate_scores.py` | Summarize per-seed and cross-seed scores |
| `evaluate_answer_cue.py` | Evaluate TruthfulQA with an appended answer cue |
| `benchmark_selection.py` | Time selection from ready representation and Energy caches |

## Selection Controls

First extract candidate and anchor representations as described in the README.
Repeat the extraction command for each task, using its task examples. Point
`configs/controls.json` to the resulting cache directories.

```bash
python -m scripts.select_controls \
  --config configs/controls.json --output-dir selections
```

For each task, this command builds the T gate and the task-reference baseline,
then exports the requested controls:

| Condition | Selection |
|---|---|
| `task_reference` | The TxM anchor-matched window within that task's T gate |
| `gate_random` | Uniform sampling without replacement from the task's gate |
| `lower_window`, `higher_window` | Continuous windows shifted toward lower or higher familiarity |
| `t_only` | The highest-T examples from the full candidate pool |
| `band_0` to `band_4` | Five equal-count familiarity bands within that task's gate, each reduced to the training budget by T ranking and 15% cluster coverage |
| `reassigned_reference` | The window using reassigned English anchor scores |
| `fixed_centre` | The window closest to mean familiarity 0.5 |
| `gsm8k_reference` | The window using GSM8K's mean familiarity reference |

Each task has five band adapters, evaluated on that task. Bands run from low to
high familiarity. Ties in comparison Energy are ordered by pool index when
dividing the population into equal-count groups.

Set `window_shift` to `0.2` to move the anchor window by 20% of the gate's
sorted rows in each direction. At 60,000 candidates this is 12,000 rows.
Windows retain the same budget and comparison-Energy ordering. Their manifests
record actual mean familiarity, endpoints, row displacement, and any clipping
at the gate boundaries. `sampling_seed` controls gate-random selection.

Reference reassignment permutes the concatenated English anchor Energy scores
and redistributes them while retaining each task's anchor count. Each resulting
reference is computed against the recipient task's gate. TyDiQA's band and
task-reference selections use calibrated Energy and language-equal anchor means;
the reassignment controls use the four English tasks.

Set `training_seed` and `evaluation_repeats` in the configuration. Training uses a
10,000-example budget. `runs.json` records the queue and training-file hashes.

```bash
# Preview without training or evaluation.
python -m scripts.train_eval --plan selections/runs.json

# Train and evaluate the band controls in sequence.
BACKEND_ROOT=/path/to/backend PYTHON=/path/to/train_eval/python \
ACCELERATE=/path/to/train_eval/accelerate \
python -m scripts.train_eval --plan selections/runs.json \
  --condition band_0 --condition band_1 --condition band_2 \
  --condition band_3 --condition band_4 --execute
```

Other conditions can be selected with `--condition`; `--task` restricts the task.
Each adapter is trained before its evaluations run. Existing run directories are
not overwritten. The planned evaluations use the original TruthfulQA prompt;
run the answer-cue diagnostic separately on the trained weights.

### TruthfulQA Answer Cue

To evaluate a trained adapter with `A:` after the final question, use a new
run name and output directory. Apply the same command to the main task-reference
and T-only adapters for the prompt diagnostic, or to band adapters for the band
comparison:

```bash
PROJECT="$PWD" BACKEND_ROOT=/path/to/backend \
PYTHON=/path/to/train_eval/python BASE_MODEL=/path/to/base_model \
ADAPTER_DIR=/path/to/truthfulqa_task_reference_adapter \
RUN_NAME=truthfulqa_task_reference_answer_cue TASKS=truthfulqa \
RESULTS_ROOT=/path/to/new_answer_cue_results \
MERGED_DIR=/path/to/new_temporary_merge \
TRUTHFULQA_PROMPT=answer_cue TRUTHFULQA_LOW_MEMORY=1 \
bash scripts/evaluate_tasks.sh
```

The wrapper appends `\nA:` and uses the existing truth/info judges. Keep all other
evaluation settings unchanged when comparing prompts. The default
`TRUTHFULQA_PROMPT=original` retains the original prompt and metrics. Model and
judge paths are configured as in the README.
This entry evaluates existing weights without training. Use separate run names
and output directories for each adapter and prompt.

## Anchor Bootstrap

```bash
python -m scripts.analyze_experiments bootstrap \
  --selection selections/gsm8k/task_reference \
  --output-dir analyses/gsm8k_bootstrap --replicates 5000
```

The command resamples anchors at fixed candidate scores and language offsets.
TyDiQA retains within-language resampling and equal language weights. Outputs
include bootstrap intervals for the mean familiarity reference and quantiles of
selected-set retention and Jaccard overlap.

## Task Concentration

Set `configs/task_concentration.json` to the anchor T representations and TyDiQA
language labels. The language labels can be read directly from a selection's
`analysis_inputs.npz`.

```bash
python -m scripts.analyze_experiments concentration \
  --config configs/task_concentration.json --output-dir analyses/task_concentration
```

The analysis uses normalized representations, 128 anchors per task per draw,
30 draws, and clustering seeds 42/43/44 at 10/20/30 clusters. It reports effective
cluster counts, mean pairwise cosine distances, and cumulative cluster coverage.
Clustering fits equal-sized task samples. The TyDiQA control averages distances
within languages using 14 anchors per language. Intervals summarize subsample
variation.

## Content Scores and Response Behaviour

Use the dependency repository's Qwen scorer and prompts for content ratings.
The following commands prepare scoring inputs and analyze its saved outputs.

Prepare a source-proportional pool sample for a 20k or 50k scoring run:

```bash
python -m scripts.rating_inputs sample --input /path/to/pool.jsonl \
  --size 50000 --output-dir rating_inputs
```

The output retains the original records. `manifest.json` maps sample rows back
to pool indices. Supply that mapping when analyzing the corresponding saved scores:

```bash
python -m scripts.analyze_experiments ratings \
  --scores /path/to/scored_data.pt --trust-local-pt \
  --pool-rows rating_inputs/manifest.json \
  --selected-indices selections/gsm8k/task_reference/gsm8k_indices.npy \
  --gate-indices selections/gsm8k/task_reference/gate_indices.npy \
  --output-dir analyses/content_ratings
```

JSON or JSONL score lists are also accepted. Each row contains `messages`,
`dataset`, and `qwen_scores` with the five scorer field names: `Completeness`,
`Information Richness`, `Rarity`, `Complexity`, and `Overall Score`. Without a
mapping file, each row must include its original `pool_index`. Pickle-based `.pt`
files should only be loaded when their origin is trusted.

The analysis reports threshold pass counts, rarity/complexity exclusions among
content-passing examples, answer lengths within sources, and selected-minus-control
rating gaps. Controls are reweighted by source and answer-length strata with at
least five selected and five unselected gate examples per supported stratum.

For the paired scoring check, provide the chosen original responses as JSONL.
Each record needs `task_selection`, `pool_index`, `messages`, and the propagated
`generated_content` tags consumed by the scorer. Responses must contain at least
30 words. The transformation retains the questions and tags:

```bash
python -m scripts.rating_inputs perturb --input /path/to/originals.jsonl \
  --output-dir paired_inputs
```

The output contains `original`, `first_3_words`, and `shuffled_words` conditions.
Score each condition with the same external scorer and three judge repetitions.
Save a JSON list with `task_selection`, `pool_index`, `condition`, and `scores`
(the five rating fields) for each judgment. Then run:

```bash
python -m scripts.analyze_experiments paired-ratings \
  --scores /path/to/paired_scores.json --output-dir analyses/paired_ratings
```

The script averages judgments per source and condition before computing paired
differences and 5,000 source-level bootstrap replicates.

Compare existing TruthfulQA predictions without rerunning the judges:

```bash
python -m scripts.analyze_experiments predictions \
  --before /path/to/t_only/predictions.csv \
  --after /path/to/task_reference/predictions.csv \
  --output-dir analyses/response_changes
```

CSV inputs use the evaluation backend's `Question`, `Best Answer`, generated
response, and stored truth/info/joint-accuracy columns. Question identities and
order must agree. Outputs count true-and-informative, false-but-informative,
uninformative, empty, and `I have no comment.` responses.

For BBH, compare per-subtask scores from the evaluator's saved metric files:

```bash
python -m scripts.analyze_experiments subtasks \
  --before /path/to/reassigned/metrics.json \
  --after /path/to/task_reference/metrics.json \
  --output-dir analyses/subtask_changes
```

Each side accepts multiple metric files and averages evaluations per subtask
before comparing the two adapters. The overall `average_exact_match` field is
excluded. Subtask lists must match. The output reports score differences and
counts of improved, declined, and unchanged subtasks. Use `--unit fraction` if
the stored scores use [0, 1].

## Selection Overlap and Composition

Compare saved subsets against the same task gate and cached scores:

```bash
python -m scripts.analyze_experiments composition \
  --selection selections/gsm8k/task_reference --pool /path/to/pool.jsonl \
  --indices anchor=selections/gsm8k/task_reference/gsm8k_indices.npy \
  --indices t_only=selections/gsm8k/t_only/gsm8k_indices.npy \
  --indices higher=selections/gsm8k/higher_window/gsm8k_indices.npy \
  --output-dir analyses/gsm8k_composition
```

Add band or gate-random index files with further `--indices NAME=PATH` arguments.
The output reports pairwise overlap and Jaccard similarity, source fractions,
mean target score, gate-relative familiarity, and the fraction of responses
containing decimal digits. All subsets must contain the same budget of unique
gate members. The command verifies pool and score hashes before analysis.
Optional `--response-lengths /path/to/response_tokens.npy` adds mean and median
answer-token counts from a cached array ordered by original pool row.

## Score Aggregation

Use `scripts.aggregate_scores` with the saved per-run metric paths as described
in the README. Keep model families and experimental conditions separate using
the `model` and `method` fields. Repeated evaluations require an explicit
aggregation policy.

## Cached Selection Timing

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -m scripts.benchmark_selection \
  --pool /path/to/pool.jsonl --candidate-t /path/to/pool_t.npy \
  --anchor-t /path/to/anchor_t.npy \
  --candidate-energy /path/to/pool_energy.npy \
  --anchor-energy /path/to/anchor_energy.npy --task gsm8k \
  --repeats 3 --output-dir timing
```

NPZ inputs require the corresponding `--anchor-t-key`, `--candidate-energy-key`,
or `--anchor-energy-key`. Timing covers input loading and validation, T scoring,
clustering, gate selection, language processing, window matching, and export.
Representations and Energy scores must already exist. Each repetition writes its
selection and timings; the summary reports means, medians, and sample standard
deviations. The command runs selection, not model extraction or training.
