# Candidate provenance and normalized benchmark-question overlap checks.
import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
import unicodedata


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    operators = set("+-*/=<>%^|()[]{}")
    return " ".join("".join(
        " " if unicodedata.category(c).startswith("P") and c not in operators
        and not (c == "." and i > 0 and i + 1 < len(text)
                 and text[i - 1].isdigit() and text[i + 1].isdigit()) else c
        for i, c in enumerate(text)).split())


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError(f"Blank JSONL row in {path}")
            yield json.loads(line)


def benchmark_questions(root):
    for row in jsonl(root / "gsm/test.jsonl"):
        yield "gsm8k", row["question"], row["question"]
    with (root / "truthfulqa/TruthfulQA.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield "truthfulqa", row["Question"], row["Question"]
    paths = sorted((root / "mmlu/test").glob("*_test.csv"))
    if not paths:
        raise FileNotFoundError("No MMLU test CSV files")
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) != 6:
                    raise ValueError(f"Expected six MMLU columns: {path}")
                full = row[0] + "\n" + "\n".join(f"{c}. {a}" for c, a in zip("ABCD", row[1:5]))
                yield "mmlu", full, row[0]
    paths = sorted((root / "bbh/bbh").glob("*.json"))
    if not paths:
        raise FileNotFoundError("No BBH JSON files")
    for path in paths:
        for row in json.loads(path.read_text(encoding="utf-8"))["examples"]:
            yield "bbh", row["input"], row["input"]
    data = json.loads((root / "tydiqa/tydiqa-goldp-v1.1-dev.json").read_text(encoding="utf-8"))
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                yield "tydiqa", qa["question"], qa["question"]


def original_rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=["id", "dataset", "messages"]):
        yield from batch.to_pylist()


def main():
    parser = argparse.ArgumentParser(description="Audit candidate provenance and normalized question overlap")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must not exist")
    if args.expected_source_sha256 and not args.source_parquet:
        parser.error("Expected source hash requires --source-parquet")
    source_hash = sha256(args.source_parquet) if args.source_parquet else None
    source_hash_matches = (source_hash.lower() == args.expected_source_sha256.lower()
                           if args.expected_source_sha256 else None)
    lookup, sizes = defaultdict(set), defaultdict(int)
    for task, full, stem in benchmark_questions(args.benchmark_root):
        index = sizes[task]
        sizes[task] += 1
        lookup[normalize(full)].add((task, "full", index))
        lookup[normalize(stem)].add((task, "stem", index))
    missing = object()
    pairs = (itertools.zip_longest(jsonl(args.pool), original_rows(args.source_parquet), fillvalue=missing)
             if args.source_parquet else ((r, None) for r in jsonl(args.pool)))
    matched_pool, matched_questions = defaultdict(set), defaultdict(set)
    mismatches, count = 0, 0
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "matches.jsonl").open("x", encoding="utf-8") as output:
        for index, (row, original) in enumerate(pairs):
            if row is missing or original is missing:
                raise ValueError("Source Parquet and pool row counts differ")
            if original is not None and any(row.get(k) != v for k, v in original.items()):
                mismatches += 1
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"Pool row {index} requires messages format")
            turns = [(m["role"], m["content"]) for m in messages if m.get("content", "").strip()]
            end = next((i for i in range(len(turns)-1, -1, -1) if turns[i][0] == "assistant"), len(turns))
            question = next((v for role, v in reversed(turns[:end]) if role == "user"), "")
            views = {question.strip()}
            if "Question:" in question:
                views.add(question.rsplit("Question:", 1)[-1].removesuffix("Answer:").strip())
            if "\n\nQ:" in question:
                views.add(question.rsplit("\n\nQ:", 1)[-1].removesuffix("A:").strip())
            hits = set()
            for view in views:
                hits.update(lookup.get(normalize(view), ()))
            for task, kind, target in sorted(hits):
                matched_pool[task, kind].add(index)
                matched_questions[task, kind].add(target)
                output.write(json.dumps(dict(pool_index=index, task=task, kind=kind,
                                             benchmark_row=target, source=row.get("dataset"))) + "\n")
            count += 1
    report = dict(pool_sha256=sha256(args.pool), benchmark_root=str(args.benchmark_root.resolve()),
                  benchmark_input_sha256={str(p.relative_to(args.benchmark_root)): sha256(p)
                                          for p in sorted(args.benchmark_root.rglob("*"))
                                          if p.is_file() and p.suffix in (".json", ".jsonl", ".csv")},
                  source_sha256=source_hash, expected_source_sha256=args.expected_source_sha256,
                  source_hash_matches=source_hash_matches, pool_rows=count,
                  conversion_mismatched_rows=mismatches if args.source_parquet else None,
                  normalization="NFKC, casefold, whitespace and punctuation normalization; preserve numbers and operators",
                  scope="Question equality and question-stem equality, not semantic paraphrase matching",
                  tasks={t: dict(benchmark_rows=n, **{
                      kind: dict(pool_rows=len(matched_pool[t, kind]),
                                 benchmark_rows=len(matched_questions[t, kind]))
                      for kind in ("full", "stem")}) for t, n in sizes.items()})
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if mismatches or source_hash_matches is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
