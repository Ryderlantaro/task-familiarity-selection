#!/usr/bin/env python3
# Merge a LoRA adapter while preserving its training tokenizer.

import argparse
import json
import math
from pathlib import Path

from peft import PeftConfig, PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--lora_model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    return parser.parse_args()


def adapter_vocab_size(adapter_dir: Path) -> int | None:
    weights = adapter_dir / "adapter_model.safetensors"
    if not weights.exists():
        return None
    with safe_open(str(weights), framework="pt") as handle:
        for key in handle.keys():
            if key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight"):
                return handle.get_tensor(key).shape[0]
    return None


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.lora_model_name_or_path)
    output_dir = Path(args.output_dir)

    # The adapter directory contains the tokenizer used during fine-tuning.
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=args.use_fast_tokenizer)
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_path = args.base_model_name_or_path or peft_config.base_model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto")

    base_vocab_size = model.get_input_embeddings().weight.shape[0]
    saved_vocab_size = adapter_vocab_size(adapter_dir)
    required_vocab_size = max(base_vocab_size, len(tokenizer), saved_vocab_size or 0)
    required_vocab_size = math.ceil(required_vocab_size / 8) * 8
    model.resize_token_embeddings(required_vocab_size)

    merged = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    merged.config.bos_token_id = tokenizer.bos_token_id
    merged.config.eos_token_id = tokenizer.eos_token_id
    merged.config.pad_token_id = tokenizer.pad_token_id
    merged.generation_config.bos_token_id = tokenizer.bos_token_id
    merged.generation_config.eos_token_id = tokenizer.eos_token_id
    merged.generation_config.pad_token_id = tokenizer.pad_token_id
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    manifest = {
        "base_model": str(Path(base_path).resolve()),
        "adapter": str(adapter_dir.resolve()),
        "tokenizer": str(adapter_dir.resolve()),
        "tokenizer_vocab_size": len(tokenizer),
        "merged_vocab_size": merged.get_input_embeddings().weight.shape[0],
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    (output_dir / "merge_tokenizer_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
