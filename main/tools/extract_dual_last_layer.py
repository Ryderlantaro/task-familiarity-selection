from __future__ import annotations

import argparse

import hashlib

import json

from itertools import islice

from pathlib import Path

import numpy as np

import torch

from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

from .data_schema import selection_text, text_parts

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)

def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def _encode_t(tokenizer, text: str, max_tokens: int) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    if len(ids) > max_tokens:
        head = max_tokens // 2
        ids = ids[:head] + ids[-(max_tokens - head):]
    return torch.tensor(ids, dtype=torch.long)

def _encode_sparsity(tokenizer, text: str, max_tokens: int) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    if len(ids) > max_tokens:
        head = max_tokens // 2
        ids = ids[:head] + ids[-(max_tokens - head):]
    return torch.tensor(ids, dtype=torch.long)

def _batches(encoded: list[torch.Tensor], token_budget: int):
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]))
    batch: list[int] = []
    longest = 0
    for index in order:
        new_longest = max(longest, len(encoded[index]))
        if batch and new_longest * (len(batch) + 1) > token_budget:
            yield batch
            batch = []
            longest = 0
        batch.append(index)
        longest = max(longest, len(encoded[index]))
    if batch:
        yield batch

def _extract_view(model, tokenizer, texts: list[str], *, max_tokens: int, token_budget: int, pool: str):
    encode = _encode_t if pool == "weighted_mean" else _encode_sparsity
    encoded = [encode(tokenizer, text, max_tokens) for text in texts]
    if any(len(ids) == 0 for ids in encoded):
        raise ValueError("empty token sequence in extraction input")
    device = next(model.parameters()).device
    with torch.inference_mode():
        for batch in _batches(encoded, token_budget):
            lengths = [len(encoded[index]) for index in batch]
            input_ids = pad_sequence(
                [encoded[index].flip(0) for index in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ).flip(1).to(device)
            positions = torch.arange(input_ids.shape[1], device=device)[None, :]
            attention_mask = (positions >= input_ids.shape[1] - torch.tensor(lengths, device=device)[:, None]).long()
            # LlamaModel's returned last_hidden_state includes its final norm.
            hidden = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
            for local, index in enumerate(batch):
                if pool == "weighted_mean":
                    length = lengths[local]
                    weights = torch.arange(1, length + 1, device=device, dtype=torch.float32)
                    vector = (hidden[local, -length:].float() * (weights / weights.sum())[:, None]).sum(dim=0)
                else:
                    vector = hidden[local, -1].float()
                yield index, vector.cpu().to(torch.float16).numpy()

def _load_model(path: Path, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
    ).to(device).eval()
    return model, tokenizer
