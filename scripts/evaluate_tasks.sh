#!/usr/bin/env bash
set -euo pipefail

# Evaluate an existing LoRA adapter with the evaluators.
PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ROOT:-$(dirname "$PROJECT")}
BACKEND_ROOT=${BACKEND_ROOT:?Set BACKEND_ROOT to the dependency checkout}
MODEL_FINETUNE=$BACKEND_ROOT/model_finetune
PYTHON=${PYTHON:-$ROOT/open-instruct/.venv/bin/python}
MERGE_SCRIPT=${MERGE_SCRIPT:-$PROJECT/scripts/merge_lora.py}
BASE_MODEL=${BASE_MODEL:-$ROOT/models/Meta-Llama-3.1-8B}
MODEL_DIR=${MODEL_DIR:-}
RUN_NAME=${RUN_NAME:-txm_seed42}
ADAPTER_DIR=${ADAPTER_DIR:-$PROJECT/runs/$RUN_NAME/training/lora}
RESULTS_ROOT=${RESULTS_ROOT:-$PROJECT/runs/$RUN_NAME/evaluation}
MERGED_DIR=${MERGED_DIR:-$PROJECT/runs/$RUN_NAME/training/merged_for_eval}
EVAL_GPU=${EVAL_GPU:-0}
EVAL_GPUS=${EVAL_GPUS:-$EVAL_GPU}
TASKS=${TASKS:-mmlu,gsm,bbh,truthfulqa,tydiqa}
# Evaluation is restricted to the split_eval partition.
DATA_PROTOCOL=${DATA_PROTOCOL:-heldout}
KEEP_MERGED=${KEEP_MERGED:-0}
MERGED_BY_WRAPPER=0
OWNER_SCRIPT=$PROJECT/scripts/merge_directory.py
MERGE_TOKEN=

GSM_MAX_EXAMPLES=${GSM_MAX_EXAMPLES-200}
BBH_MAX_PER_TASK=${BBH_MAX_PER_TASK-40}
TYDIQA_MAX_PER_LANG=${TYDIQA_MAX_PER_LANG-100}
MMLU_MAX_PER_SUBJECT=${MMLU_MAX_PER_SUBJECT-}
TRUTHFULQA_METRICS=${TRUTHFULQA_METRICS:-truth,info,mc}
TRUTHFULQA_PROMPT=${TRUTHFULQA_PROMPT:-original}
case "$TRUTHFULQA_PROMPT" in
  original|answer_cue) ;;
  *) echo "TRUTHFULQA_PROMPT must be original or answer_cue" >&2; exit 2 ;;
esac
TRUTHFULQA_EVAL_BATCH_SIZE=${TRUTHFULQA_EVAL_BATCH_SIZE:-40}

TRUTH_MODEL=${TRUTH_MODEL:-$ROOT/models/allenai/truthfulqa-truth-judge-llama2-7B}
INFO_MODEL=${INFO_MODEL:-$ROOT/models/allenai/truthfulqa-info-judge-llama2-7B}
METRICS_ROOT=${METRICS_ROOT:-$PROJECT/compat_metrics}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=${HF_HOME:-$ROOT/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-$HF_HOME/modules}
export HF_METRICS_CACHE=${HF_METRICS_CACHE:-$HF_HOME/metrics}
export OPENAI_API_KEY=${OPENAI_API_KEY:-fake}
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

[[ "$DATA_PROTOCOL" == heldout ]] || {
  echo "Only DATA_PROTOCOL=heldout (split_eval) is supported" >&2; exit 2
}
EVAL_DATA_ROOT=$BACKEND_ROOT/data/eval/split_eval
TYDIQA_DATA_DIR=$EVAL_DATA_ROOT/tydiqa
[[ -d "$EVAL_DATA_ROOT" ]] || { echo "Missing split_eval: $EVAL_DATA_ROOT" >&2; exit 1; }

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
if [[ -n "$MODEL_DIR" ]]; then
  [[ -d "$MODEL_DIR" ]] || { echo "Missing evaluation model: $MODEL_DIR" >&2; exit 1; }
else
  [[ -d "$BASE_MODEL" ]] || { echo "Missing base model: $BASE_MODEL" >&2; exit 1; }
  [[ -f "$MERGE_SCRIPT" ]] || { echo "Missing merge wrapper: $MERGE_SCRIPT" >&2; exit 1; }
  [[ -f "$ADAPTER_DIR/adapter_model.safetensors" ]] || {
    echo "Missing LoRA adapter: $ADAPTER_DIR" >&2
    exit 1
  }
fi
[[ -d "$MODEL_FINETUNE/eval" ]] || { echo "Missing evaluators" >&2; exit 1; }
[[ -f "$METRICS_ROOT/exact_match/exact_match.py" ]] || {
  echo "Missing exact-match compatibility metric: $METRICS_ROOT/exact_match/exact_match.py" >&2
  exit 1
}
[[ -f "$METRICS_ROOT/squad/squad.py" ]] || {
  echo "Missing SQuAD compatibility metric: $METRICS_ROOT/squad/squad.py" >&2
  exit 1
}

mkdir -p "$RESULTS_ROOT/logs" "$(dirname "$MERGED_DIR")" "$HF_HOME" \
  "$HF_MODULES_CACHE" "$HF_METRICS_CACHE"
exec > >(tee -a "$RESULTS_ROOT/logs/driver.log") 2>&1

contains_task() {
  [[ ",$TASKS," == *",$1,"* ]]
}

ensure_fresh_output() {
  local task=$1
  local output=$RESULTS_ROOT/$task/$RUN_NAME
  if [[ -e "$output" && "${OVERWRITE:-0}" != 1 ]]; then
    echo "Output already exists: $output" >&2
    echo "Use a new RUN_NAME, or set OVERWRITE=1 explicitly." >&2
    exit 1
  fi
  if [[ "${OVERWRITE:-0}" == 1 ]]; then
    rm -rf "$output"
  fi
}

cleanup() {
  if [[ "$MERGED_BY_WRAPPER" == 1 && "$KEEP_MERGED" != 1 ]]; then
    "$PYTHON" "$OWNER_SCRIPT" remove --path "$MERGED_DIR" --token "$MERGE_TOKEN" || {
      echo "Merge cleanup refused; directory preserved: $MERGED_DIR" >&2; return 1;
    }
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for task in mmlu gsm bbh truthfulqa tydiqa; do
  contains_task "$task" && ensure_fresh_output "$task"
done

if [[ -z "$MODEL_DIR" ]]; then
  echo "[merge] adapter-tokenizer-safe LoRA merge"
  MERGE_TOKEN=$("$PYTHON" "$OWNER_SCRIPT" token)
  MERGED_DIR=$("$PYTHON" "$OWNER_SCRIPT" reserve --path "$MERGED_DIR" --token "$MERGE_TOKEN" \
    --protect "$BASE_MODEL" "$ADAPTER_DIR" "$PROJECT" "$ROOT" "$BACKEND_ROOT" "$RESULTS_ROOT")
  # Mark ownership before writing so the trap removes a partial merge on error.
  MERGED_BY_WRAPPER=1
  "$PYTHON" "$MERGE_SCRIPT" \
    --base_model_name_or_path "$BASE_MODEL" \
    --lora_model_name_or_path "$ADAPTER_DIR" \
    --output_dir "$MERGED_DIR"
  MODEL_DIR=$MERGED_DIR
else
  echo "[model] evaluating existing model directory: $MODEL_DIR"
fi

run_task() {
  local task=$1
  local gpu=$2
  cd "$METRICS_ROOT"

  case "$task" in
    mmlu)
      echo "[eval gpu=$gpu] MMLU: official 0-shot accuracy"
      local mmlu_limit=()
      [[ -n "$MMLU_MAX_PER_SUBJECT" ]] && mmlu_limit=(--n_instances "$MMLU_MAX_PER_SUBJECT")
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$MODEL_FINETUNE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m eval.mmlu.run_eval \
        --ntrain 0 --data_dir "$EVAL_DATA_ROOT/mmlu" \
        --save_dir "$RESULTS_ROOT/mmlu/$RUN_NAME" \
        --model_name_or_path "$MODEL_DIR" --tokenizer_name_or_path "$MODEL_DIR" \
        --eval_batch_size 8 "${mmlu_limit[@]}"
      ;;
    gsm)
      echo "[eval gpu=$gpu] GSM8K: official 8-shot exact match"
      local gsm_limit=()
      [[ -n "$GSM_MAX_EXAMPLES" ]] && gsm_limit=(--max_num_examples "$GSM_MAX_EXAMPLES")
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$MODEL_FINETUNE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m eval.gsm.run_eval \
        --data_dir "$EVAL_DATA_ROOT/gsm" --save_dir "$RESULTS_ROOT/gsm/$RUN_NAME" \
        --model_name_or_path "$MODEL_DIR" --tokenizer_name_or_path "$MODEL_DIR" \
        --n_shot 8 --use_vllm "${gsm_limit[@]}"
      ;;
    bbh)
      echo "[eval gpu=$gpu] BBH: official CoT exact match"
      local bbh_limit=()
      [[ -n "$BBH_MAX_PER_TASK" ]] && bbh_limit=(--max_num_examples_per_task "$BBH_MAX_PER_TASK")
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$MODEL_FINETUNE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m eval.bbh.run_eval \
        --data_dir "$EVAL_DATA_ROOT/bbh" --save_dir "$RESULTS_ROOT/bbh/$RUN_NAME" \
        --model_name_or_path "$MODEL_DIR" --tokenizer_name_or_path "$MODEL_DIR" \
        --use_vllm "${bbh_limit[@]}"
      ;;
    truthfulqa)
      echo "[eval gpu=$gpu] TruthfulQA: official truth-info accuracy and MC metrics"
      local truthfulqa_metrics=()
      local truthfulqa_runner=(-m eval.truthfulqa.run_eval)
      if [[ "${TRUTHFULQA_LOW_MEMORY:-0}" == 1 ]]; then
        truthfulqa_runner=("$PROJECT/scripts/evaluate_truthfulqa_low_memory.py")
      fi
      if [[ -n "${TRUTHFULQA_RUNNER:-}" ]]; then
        truthfulqa_runner=("$TRUTHFULQA_RUNNER")
      fi
      IFS=',' read -r -a truthfulqa_metrics <<< "$TRUTHFULQA_METRICS"
      if [[ "$TRUTHFULQA_PROMPT" == answer_cue ]]; then
        [[ -z "${TRUTHFULQA_RUNNER:-}" ]] || {
          echo "Answer-cue mode cannot use a custom runner" >&2; return 2;
        }
        truthfulqa_runner=("$PROJECT/scripts/evaluate_answer_cue.py")
        truthfulqa_metrics=(truth info)
      fi
      echo "[prompt] $TRUTHFULQA_PROMPT"
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$MODEL_FINETUNE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "${truthfulqa_runner[@]}" \
        --data_dir "$EVAL_DATA_ROOT/truthfulqa" --save_dir "$RESULTS_ROOT/truthfulqa/$RUN_NAME" \
        --model_name_or_path "$MODEL_DIR" --tokenizer_name_or_path "$MODEL_DIR" \
        --metrics "${truthfulqa_metrics[@]}" --preset qa \
        --hf_truth_model_name_or_path "$TRUTH_MODEL" \
        --hf_info_model_name_or_path "$INFO_MODEL" --eval_batch_size "$TRUTHFULQA_EVAL_BATCH_SIZE" --load_in_8bit
      ;;
    tydiqa)
      echo "[eval gpu=$gpu] TyDiQA: official 1-shot macro-F1"
      local tydiqa_limit=()
      [[ -n "$TYDIQA_MAX_PER_LANG" ]] && tydiqa_limit=(--max_num_examples_per_lang "$TYDIQA_MAX_PER_LANG")
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$MODEL_FINETUNE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m eval.tydiqa.run_eval \
        --data_dir "$TYDIQA_DATA_DIR" --n_shot 1 --max_context_length 512 \
        --save_dir "$RESULTS_ROOT/tydiqa/$RUN_NAME" \
        --model_name_or_path "$MODEL_DIR" --tokenizer_name_or_path "$MODEL_DIR" \
        --eval_batch_size 40 --load_in_8bit "${tydiqa_limit[@]}"
      ;;
  esac
}

IFS=',' read -r -a EVAL_GPU_SLOTS <<< "$EVAL_GPUS"
(( ${#EVAL_GPU_SLOTS[@]} > 0 )) || { echo "EVAL_GPUS must list at least one GPU" >&2; exit 2; }

selected_tasks=()
for task in mmlu gsm bbh truthfulqa tydiqa; do
  contains_task "$task" && selected_tasks+=("$task")
done

# One official evaluator owns one GPU. Run up to one task per configured GPU,
# then wait before scheduling the next wave to avoid model-device conflicts.
for ((offset = 0; offset < ${#selected_tasks[@]}; offset += ${#EVAL_GPU_SLOTS[@]})); do
  pids=()
  task_names=()
  for ((slot = 0; slot < ${#EVAL_GPU_SLOTS[@]} && offset + slot < ${#selected_tasks[@]}; slot++)); do
    task=${selected_tasks[$((offset + slot))]}
    gpu=${EVAL_GPU_SLOTS[$slot]}
    run_task "$task" "$gpu" 2>&1 | tee "$RESULTS_ROOT/logs/${task}.log" &
    pids+=("$!")
    task_names+=("$task")
  done
  for ((slot = 0; slot < ${#pids[@]}; slot++)); do
    wait "${pids[$slot]}" || { cat "$RESULTS_ROOT/logs/${task_names[$slot]}.log" >&2; exit 1; }
  done
done

echo "[summary] read_results.py"
(
  cd "$MODEL_FINETUNE"
  "$PYTHON" read_results.py \
    --root_result_path "$RESULTS_ROOT" \
    --baseline_tag "$RUN_NAME"
)

echo "[done] $RESULTS_ROOT"
