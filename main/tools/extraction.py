from __future__ import annotations

import json

from pathlib import Path

from typing import Any

import numpy as np

import torch

from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

from .data_schema import selection_text

class _StopForward(Exception):
    pass

def load_model_for_extraction(model_path: str | Path, *, attn_implementation: str = "sdpa") -> tuple[Any, Any]:
    # Load a causal LM and left-padding tokenizer for representation extraction.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return model, tokenizer

def _transformer_layers(model: Any) -> Any:
    return model.model.layers

def _truncate(input_ids: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if max_tokens <= 0 or len(input_ids) <= max_tokens:
        return input_ids
    head = max_tokens // 2
    return torch.cat([input_ids[:head], input_ids[-(max_tokens - head):]], dim=0)

def _batches(encoded: list[torch.Tensor], token_budget: int) -> list[list[torch.Tensor]]:
    batches: list[list[torch.Tensor]] = []
    batch: list[torch.Tensor] = []
    max_length = 0
    for item in encoded:
        proposed_max = max(max_length, len(item))
        if batch and proposed_max * (len(batch) + 1) > token_budget:
            batches.append(batch)
            batch, max_length = [item], len(item)
        else:
            batch.append(item)
            max_length = proposed_max
    if batch:
        batches.append(batch)
    return batches

def extract_weighted_mean(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    layer: int,
    text_view: str = "canonical",
    max_tokens: int = 5120,
    token_budget: int = 4096,
) -> np.ndarray:
    # Extract weighted-mean hidden-state representations at one transformer layer.
    if not records:
        return np.empty((0, model.config.hidden_size), dtype=np.float16)
    encoded = [
        _truncate(tokenizer(selection_text(record, text_view), return_tensors="pt")["input_ids"].squeeze(0), max_tokens)
        for record in records
    ]
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]))
    batches = _batches([encoded[index] for index in order], token_budget)
    output = np.empty((len(records), model.config.hidden_size), dtype=np.float16)
    device = next(model.parameters()).device
    captured: dict[str, torch.Tensor] = {}

    def capture(_, __, layer_output):
        captured["hidden"] = (layer_output[0] if isinstance(layer_output, tuple) else layer_output).detach()
        raise _StopForward()

    handle = _transformer_layers(model)[layer].register_forward_hook(capture)
    position = 0
    try:
        for batch in tqdm(batches, desc="Extract representations", unit="batch"):
            lengths = [len(item) for item in batch]
            padded = pad_sequence(
                [item.flip(0) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id
            ).flip(1).to(device)
            mask = (padded != tokenizer.pad_token_id).long()
            captured.clear()
            try:
                with torch.no_grad():
                    model(input_ids=padded, attention_mask=mask)
            except _StopForward:
                pass
            hidden = captured["hidden"]
            for local_index, length in enumerate(lengths):
                weights = torch.arange(1, length + 1, device=device, dtype=torch.float32)
                weights = weights / weights.sum()
                output[order[position]] = (
                    hidden[local_index, -length:].float().mul(weights[:, None]).sum(dim=0).cpu().to(torch.float16).numpy()
                )
                position += 1
    finally:
        handle.remove()
        torch.cuda.empty_cache()
    invalid = ~np.isfinite(output).all(axis=1) | (np.linalg.norm(output.astype(np.float32), axis=1) == 0)
    if invalid.any():
        raise ValueError(f'Invalid extracted representations at rows {np.flatnonzero(invalid).tolist()}')
    return output
