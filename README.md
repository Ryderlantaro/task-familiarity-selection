# Task Relevance Meets Model Familiarity

Code for **Task Relevance Meets Model Familiarity: Efficient Data Selection for Instruction Tuning**.

TxM selects instruction-tuning data in two stages. Target alignment (T) builds a relevant candidate gate. Model familiarity (M) selects a fixed-budget window whose mean familiarity is closest to the task anchors. Candidate representations are cached for reuse.

## Setup

Prepare the base model, candidate pool, task anchors, and benchmark data. Pool and anchor files use messages-format JSONL with no blank lines. Selected examples retain their original text.

Use Python 3.10 with `requirements.txt` for extraction and selection, and Python 3.12 with `locks/train_eval-py312.txt` for training and evaluation. Environment details are recorded in `locks/environment.json`. PyTorch and FlashAttention binaries must match the Python and CUDA versions.

Training and evaluation use the [TADS backend](https://github.com/tmlr-group/TADS). Check out the commit in `dependencies/backend.json`, then check and apply `dependencies/backend_local_changes.patch` from the backend directory using `git apply --check` and `git apply`.

Run the commands below from this package's root, replacing the example paths:

```bash
BACKEND_ROOT=/path/to/backend_checkout
MODEL=/path/to/base_model
POOL=/path/to/pool_messages.jsonl
ANCHORS=/path/to/gsm8k_anchor_messages.jsonl
TASK=gsm8k
CACHE=/path/to/tm_cache
SELECTION=/path/to/new_gsm8k_selection
```

## Extract and Select

```bash
python -m scripts.extract_representations \
  --model "$MODEL" --pool "$POOL" --task "$TASK" --anchors "$ANCHORS" \
  --output-dir "$CACHE" --layer 28

python -m scripts.select_data \
  --cache "$CACHE" --task "$TASK" --budget 10000 --output-dir "$SELECTION"
```

T uses position-weighted mean representations from transformer block 29 (`--layer 28`, zero-based), with a 5,120-token limit. M uses the final post-normalization state at the last retained token, with a 2,048-token limit and head-and-tail truncation. The two views are extracted separately.

The T gate contains six times the training budget: 85% by relevance ranking and 15% through coverage across 30 clusters. M uses an anchor-matched familiarity window. TyDiQA includes language calibration and equal weighting across anchor languages.

Supported tasks are `gsm8k`, `truthfulqa`, `mmlu`, `bbh`, and `tydiqa`. Selection requires a new output directory and writes indices, the selected JSONL, and a manifest. Cache reuse checks model, input, and extraction identities.

Energy scores are cached beside the M representations and reused after their identities and hashes are checked.

## Train and Evaluate

```bash
PROJECT="$PWD" ROOT=/path/to/workspace BACKEND_ROOT="$BACKEND_ROOT" \
PYTHON=/path/to/train_venv/bin/python \
ACCELERATE=/path/to/train_venv/bin/accelerate \
BASE_MODEL="$MODEL" TRAIN_FILE="$SELECTION/gsm8k.jsonl" \
RUN_NAME=txm_gsm8k RUN_ROOT=/path/to/new_gsm8k_run \
SEED=42 GPU_IDS=0 bash scripts/train_lora.sh

PROJECT="$PWD" ROOT=/path/to/workspace BACKEND_ROOT="$BACKEND_ROOT" \
PYTHON=/path/to/eval_venv/bin/python BASE_MODEL="$MODEL" \
ADAPTER_DIR=/path/to/new_gsm8k_run/training/lora \
RUN_NAME=txm_gsm8k TASKS=gsm \
RESULTS_ROOT=/path/to/new_gsm8k_eval MERGED_DIR=/path/to/new_temporary_merge \
bash scripts/evaluate_tasks.sh
```

Training expects 10,000 examples and defaults to three epochs, global batch size 128, and LoRA rank 64. Set `MODE=preflight` to validate inputs without training. Evaluation reads `data/eval/split_eval/` under the backend. Its GSM8K task argument is `gsm`; other task names are unchanged. The temporary merge directory must not already exist.

For TruthfulQA, set `TRUTH_MODEL` and `INFO_MODEL` to the judge model paths. Set `TRUTHFULQA_LOW_MEMORY=1` for sequential judge loading.

## Aggregate Scores

Create a JSON list with one record per evaluation. Each record specifies `model`,
`method`, `task`, and training `seed`, together with a metrics file and its JSON key:

```json
[
  {"model": "llama", "method": "txm", "task": "gsm8k", "seed": 42,
   "metrics_file": "runs/seed42/metrics.json", "metric": "exact_match", "unit": "percent"},
  {"model": "llama", "method": "txm", "task": "gsm8k", "seed": 43,
   "metrics_file": "runs/seed43/metrics.json", "metric": "exact_match", "unit": "percent"}
]
```

Paths are relative to the manifest. Dotted keys address nested JSON objects.
Use `unit: "fraction"` for metrics in [0, 1]; the output uses percentages.
For a registered scalar, replace `metrics_file` with `score` and keep the metric name.

```bash
python -m scripts.aggregate_scores --manifest scores.json --output-dir score_summary
```

The script writes `per_seed.csv`, `summary.csv`, and `summary.json`. It computes
the mean and sample standard deviation across training seeds. A single seed has
no sample standard deviation and leaves that field empty. Repeated evaluations
of one seed require an explicit `--repeat-policy mean` or `--repeat-policy max`.
The JSON output records the rule, input scores, and metrics-file hashes.

## Audit the Candidate Pool

```bash
python -m scripts.audit_candidate_pool \
  --pool "$POOL" --source-parquet /path/to/train.parquet \
  --benchmark-root "$BACKEND_ROOT/data/eval/split_eval" \
  --output-dir /path/to/new_pool_audit
```

The audit compares the source Parquet with the JSONL row by row and checks normalized question matches against the five benchmarks. It records input hashes, match counts, and row indices, separating MMLU question stems from complete multiple-choice questions. Matching covers text equality after normalization, not semantic paraphrases.

Parquet comparison requires `pyarrow`. Omit `--source-parquet` for a standard-library-only question audit. To check the complete original benchmark files, use the backend's `data/eval/` as `--benchmark-root`. An optional `--expected-source-sha256` verifies a known source-file hash.

## Experimental Controls and Analyses

[EXPERIMENTS.md](EXPERIMENTS.md) describes T-only, task-specific familiarity bands,
reference controls, anchor bootstrap, task concentration, selection composition,
and content-score analyses. Cached analyses use the subcommands of
`scripts.analyze_experiments`.
`configs/` contains example input configurations. `scripts.train_eval` previews
its queue by default; training requires `--execute`.

## Files

- `scripts/`: extraction, selection, training, evaluation, and audit entry points.
- `main/scores/` and `main/adapters/`: scoring and selection rules.
- `main/tools/`: input parsing, representation extraction, and cache validation.
- `dependencies/` and `locks/`: backend identity, patches, and dependency versions.
- `SOURCE_MAP.json`: source files and export transformations.

Third-party sources and licenses are listed in [NOTICE.md](NOTICE.md).
