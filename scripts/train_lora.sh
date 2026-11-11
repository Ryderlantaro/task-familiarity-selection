#!/usr/bin/env bash
set -euo pipefail

# Train the selected 10k subset with the finetuning code.
PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ROOT:-$(dirname "$PROJECT")}
BACKEND_ROOT=${BACKEND_ROOT:?Set BACKEND_ROOT to the dependency checkout}
MODEL_FINETUNE=$BACKEND_ROOT/model_finetune
PYTHON=${PYTHON:-$ROOT/open-instruct/.venv/bin/python}
ACCELERATE=${ACCELERATE:-$ROOT/open-instruct/.venv/bin/accelerate}

BASE_MODEL=${BASE_MODEL:-$ROOT/models/Meta-Llama-3.1-8B}
TRAIN_FILE=${TRAIN_FILE:?Set TRAIN_FILE to an exported selection}
RUN_NAME=${RUN_NAME:-txm_seed42}
RUN_ROOT=${RUN_ROOT:-$PROJECT/runs/$RUN_NAME}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/training/lora}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
MANIFEST=${MANIFEST:-$RUN_ROOT/train_manifest.json}

GPU_IDS=${GPU_IDS:-0}
SEED=${SEED:-42}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
EPOCHS=${EPOCHS:-3}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-1}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
OVERWRITE=${OVERWRITE:-0}
MODE=${MODE:-train}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=${HF_HOME:-$ROOT/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export WANDB_DISABLED=true
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -x "$ACCELERATE" ]] || { echo "Missing accelerate: $ACCELERATE" >&2; exit 1; }
[[ -f "$MODEL_FINETUNE/finetune.py" ]] || { echo "Missing finetune.py" >&2; exit 1; }
[[ -d "$BASE_MODEL" ]] || { echo "Missing base model: $BASE_MODEL" >&2; exit 1; }
[[ -f "$TRAIN_FILE" ]] || { echo "Missing train file: $TRAIN_FILE" >&2; exit 1; }
[[ "$MODE" == train || "$MODE" == preflight ]] || {
  echo "MODE must be train or preflight" >&2
  exit 2
}

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}
DENOMINATOR=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE))
if (( NUM_GPUS < 1 || TOTAL_BATCH_SIZE % DENOMINATOR != 0 )); then
  echo "TOTAL_BATCH_SIZE must be divisible by GPU count times PER_DEVICE_BATCH_SIZE" >&2
  exit 2
fi
GRAD_ACCUM=$((TOTAL_BATCH_SIZE / DENOMINATOR))

if [[ -e "$OUTPUT_DIR" ]]; then
  if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    :
  elif [[ "$OVERWRITE" != 1 ]]; then
    echo "Output already exists: $OUTPUT_DIR" >&2
    echo "Use a new RUN_NAME, resume from a checkpoint, or set OVERWRITE=1 explicitly." >&2
    exit 1
  else
    rm -rf "$OUTPUT_DIR"
  fi
fi

mkdir -p "$LOG_DIR" "$HF_HOME"

"$PYTHON" - "$TRAIN_FILE" "$MANIFEST" "$BASE_MODEL" "$BACKEND_ROOT" "$GPU_IDS" \
  "$NUM_GPUS" "$GRAD_ACCUM" "$SEED" "$TOTAL_BATCH_SIZE" "$PER_DEVICE_BATCH_SIZE" \
  "$MAX_SEQ_LENGTH" "$EPOCHS" "$USE_FLASH_ATTN" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

train_file = Path(sys.argv[1])
manifest = Path(sys.argv[2])
base_model, backend, gpu_ids = sys.argv[3:6]
num_gpus, grad_accum, seed, total_batch = map(int, sys.argv[6:10])
per_device_batch, max_seq_length, epochs, use_flash_attn = map(int, sys.argv[10:14])

digest = hashlib.sha256()
rows = 0
with train_file.open("rb") as handle:
    for line in handle:
        digest.update(line)
        if line.strip():
            record = json.loads(line)
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise SystemExit(f"Invalid messages at non-empty row {rows}")
            rows += 1
if rows != 10_000:
    raise SystemExit(f"Expected 10000 training records, found {rows}")

payload = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "trainer": "model_finetune/finetune.py",
    "backend_commit": subprocess.check_output(
        ["git", "-C", backend, "rev-parse", "HEAD"], text=True
    ).strip(),
    "base_model": base_model,
    "train_file": str(train_file.resolve()),
    "train_rows": rows,
    "train_sha256": digest.hexdigest(),
    "gpu_ids": gpu_ids,
    "num_gpus": num_gpus,
    "seed": seed,
    "hyperparameters": {
        "precision": "bf16",
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "max_sequence_length": max_seq_length,
        "epochs": epochs,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": 1e-4,
        "lr_scheduler": "linear",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "per_device_batch_size": per_device_batch,
        "total_batch_size": total_batch,
        "gradient_accumulation_steps": grad_accum,
        "preprocessing_workers": 16,
        "gradient_checkpointing": True,
        "flash_attention_2": bool(use_flash_attn),
    },
}
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

if [[ "$MODE" == preflight ]]; then
  echo "[done] preflight passed; manifest: $MANIFEST"
  exit 0
fi

flash_flag=()
[[ "$USE_FLASH_ATTN" == 1 ]] && flash_flag=(--use_flash_attn)
resume_flag=()
[[ -n "$RESUME_FROM_CHECKPOINT" ]] && resume_flag=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")

echo "[train] finetune.py"
echo "[train] GPUs=$GPU_IDS global_batch=$TOTAL_BATCH_SIZE grad_accum=$GRAD_ACCUM"
(
  cd "$MODEL_FINETUNE"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$ACCELERATE" launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    finetune.py \
    --model_name_or_path "$BASE_MODEL" \
    --tokenizer_name "$BASE_MODEL" \
    --use_slow_tokenizer \
    --train_file "$TRAIN_FILE" \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --gradient_checkpointing \
    "${flash_flag[@]}" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs "$EPOCHS" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    "${resume_flag[@]}"
) 2>&1 | tee "$LOG_DIR/train.log"

[[ -f "$OUTPUT_DIR/adapter_model.safetensors" ]] || {
  echo "Training finished without adapter_model.safetensors" >&2
  exit 1
}
echo "[done] adapter: $OUTPUT_DIR"
echo "[next] RUN_NAME=$RUN_NAME ADAPTER_DIR=$OUTPUT_DIR bash $PROJECT/scripts/evaluate_tasks.sh"
