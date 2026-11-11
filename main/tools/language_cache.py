import json

from concurrent.futures import ProcessPoolExecutor

from pathlib import Path

import numpy as np

from langdetect import DetectorFactory, detect_langs

from .data_schema import text_parts

LANGUAGES = {"arabic": "ar", "bengali": "bn", "english": "en", "finnish": "fi",
             "indonesian": "id", "korean": "ko", "russian": "ru", "swahili": "sw", "telugu": "te"}

def detect_language(text):
    DetectorFactory.seed = 42
    try:
        prediction = detect_langs(text[:4000])[0]
        return prediction.lang if prediction.prob >= .8 else "unknown"
    except Exception:
        return "unknown"

def gate_languages(pool_path, gate_indices, cache_path, *, progress=print):
    # Persist language labels with a pool hash supplied by the caller's run manifest.
    path = Path(cache_path)
    if path.exists():
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["gate_indices"], gate_indices)
            return saved["languages"].copy()
    wanted = {int(index): position for position, index in enumerate(gate_indices)}
    texts = [None] * len(gate_indices)
    with Path(pool_path).open() as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                texts[wanted[index]] = "\n\n".join(p for p in text_parts(json.loads(line)) if p)
    if any(text is None for text in texts):
        raise ValueError("Gate indices exceed pool rows")
    detected = []
    with ProcessPoolExecutor(max_workers=8) as workers:
        for language in workers.map(detect_language, texts, chunksize=64):
            detected.append(language)
            if len(detected) % 5000 == 0:
                progress(f"Language detection: {len(detected)}/{len(texts)}")
    labels = np.asarray(detected)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, gate_indices=gate_indices, languages=labels)
    return labels
