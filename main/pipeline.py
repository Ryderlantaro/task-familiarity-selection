from __future__ import annotations

from pathlib import Path

import numpy as np





def write_indices(indices: np.ndarray, output_path: str | Path) -> Path:
    # Persist the authoritative selection artifact before materializing data.
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(indices, dtype=np.int64), allow_pickle=False)
    return output

def materialize_jsonl(pool_path: str | Path, indices: np.ndarray, output_path: str | Path) -> Path:
    # Write selected raw JSONL rows in ranking order without changing their payloads.
    requested = np.asarray(indices, dtype=np.int64)
    if len(np.unique(requested)) != len(requested) or (requested < 0).any():
        raise ValueError("indices must be unique non-negative pool row indices")
    rank = {int(index): position for position, index in enumerate(requested)}
    rows: list[str | None] = [None] * len(requested)
    with Path(pool_path).open(encoding="utf-8") as source:
        for row_index, line in enumerate(source):
            position = rank.get(row_index)
            if position is not None:
                rows[position] = line.rstrip("\n")
    if any(row is None for row in rows):
        missing = sum(row is None for row in rows)
        raise ValueError(f"{missing} selected indices are outside the pool")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(f"{row}\n")
    temporary.replace(output)
    return output
