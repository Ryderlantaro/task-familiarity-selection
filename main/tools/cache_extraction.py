from __future__ import annotations

import argparse

import gc

import json

import time

from itertools import islice

from pathlib import Path

import numpy as np

import torch

from main.tools.data_schema import text_parts

from main.tools.extract_dual_last_layer import _extract_view, _load_model, _rows, _sha256, _write_json


from main.tools.extraction import extract_weighted_mean, load_model_for_extraction

def validate_vectors(vectors, count, width):
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape != (count, width):
        raise ValueError(f"Unexpected vector shape: {values.shape}")
    if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError("Non-finite or zero representation")

def extract_source(source, output, spec, model, tokenizer, *, view, chunk_size, token_budget):
    # Commit a chunk only after every row has passed numerical validation.
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_file = output.with_suffix(".progress.json")
    final_manifest = output.with_suffix(".manifest.json")
    partial = output.with_suffix(".partial.npy")
    count, width = spec["count"], spec["hidden_size"]
    if final_manifest.exists():
        saved = json.loads(final_manifest.read_text())
        if saved["spec"] != spec or not output.is_file():
            raise ValueError(f"Completed cache configuration mismatch: {output}")
        if _sha256(output) != saved["output_sha256"]:
            raise ValueError(f"Completed cache hash mismatch: {output}")
        cached = np.load(output, mmap_mode="r")
        for offset in range(0, count, chunk_size):
            validate_vectors(cached[offset:offset + chunk_size], min(chunk_size, count - offset), width)
        return
    if progress_file.exists():
        state = json.loads(progress_file.read_text())
        if state["spec"] != spec:
            raise ValueError(f"Partial cache configuration mismatch: {output}")
        start = state["next_row"]
        if not 0 <= start <= count:
            raise ValueError("Invalid resume position")
        cache = np.load(partial if partial.exists() else output, mmap_mode="r+")
        if cache.shape != (count, width) or cache.dtype != np.float16:
            raise ValueError("Invalid partial cache layout")
        for offset in range(0, start, chunk_size):
            validate_vectors(cache[offset:min(offset + chunk_size, start)], min(chunk_size, start - offset), width)
    else:
        if output.exists() or partial.exists():
            raise FileExistsError(f"Untracked cache exists: {output}")
        start = 0
        state = {"spec": spec, "next_row": 0, "elapsed_seconds": 0.0}
        cache = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(count, width))
        _write_json(progress_file, state)
    source_rows = islice(_rows(source), start, count)
    began = time.monotonic()
    previous_seconds = state["elapsed_seconds"]
    for offset in range(start, count, chunk_size):
        records = list(islice(source_rows, min(chunk_size, count - offset)))
        if len(records) != min(chunk_size, count - offset):
            raise ValueError("Source ended before its declared row count")
        if view == "t":
            vectors = extract_weighted_mean(
                model, tokenizer, records, layer=spec["layer"], text_view="canonical",
                max_tokens=5120, token_budget=token_budget,
            )
        else:
            texts = ["\n\n".join(part for part in text_parts(row) if part) for row in records]
            if any(not text for text in texts):
                raise ValueError("Empty M input")
            vectors = np.empty((len(records), width), dtype=np.float16)
            seen = set()
            for index, vector in _extract_view(
                model, tokenizer, texts, max_tokens=2048, token_budget=token_budget, pool="last_token"
            ):
                if index in seen or not 0 <= index < len(records):
                    raise ValueError("Duplicate or invalid chunk row index")
                vectors[index] = vector
                seen.add(index)
            if len(seen) != len(records):
                raise ValueError("Missing M vectors")
        validate_vectors(vectors, len(records), width)
        cache[offset:offset + len(records)] = vectors
        cache.flush()
        elapsed = previous_seconds + time.monotonic() - began
        state = {"spec": spec, "next_row": offset + len(records), "elapsed_seconds": elapsed}
        _write_json(progress_file, state)
        rate = state["next_row"] / max(elapsed, 1e-9)
        print(f"[{view}:{source.name}] {state['next_row']}/{count} "
              f"{rate:.2f} rows/s ETA={(count-state['next_row'])/rate/3600:.2f}h", flush=True)
    del cache
    if partial.exists():
        partial.replace(output)
    _write_json(final_manifest, {"spec": spec, "elapsed_seconds": state["elapsed_seconds"],
                                 "output_sha256": _sha256(output)})
