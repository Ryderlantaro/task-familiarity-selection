# Aggregate evaluation metrics by training seed, then across seeds.
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics


def read_records(manifest):
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a nonempty JSON list")
    records, seen = [], set()
    for entry in entries:
        record = {name: str(entry[name]).strip()
                  for name in ("model", "method", "task", "seed")}
        if not all(record.values()):
            raise ValueError("Model, method, task, and seed must be nonempty")
        if ("score" in entry) == ("metrics_file" in entry):
            raise ValueError("Provide either score or metrics_file")
        if "metrics_file" in entry:
            path = (manifest.parent / entry["metrics_file"]).resolve()
            key = entry["metric"]
            identity = (str(path), key)
            if identity in seen:
                raise ValueError(f"Duplicate metric input: {identity}")
            seen.add(identity)
            raw = path.read_bytes()
            value = json.loads(raw)
            for part in key.split("."):
                value = value[part]
            record.update(metrics_file=str(path), metric=key,
                          sha256=hashlib.sha256(raw).hexdigest())
        else:
            value = entry["score"]
            record["metric"] = entry.get("metric", "score")
        if isinstance(value, bool):
            raise ValueError("Boolean metric is not a score")
        value = float(value)
        unit = entry.get("unit", "percent")
        if unit not in ("percent", "fraction"):
            raise ValueError("Unit must be percent or fraction")
        upper = 1 if unit == "fraction" else 100
        if not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f"Score outside {unit} range: {value}")
        record["score"] = value * (100 if unit == "fraction" else 1)
        records.append(record)
    return records


def aggregate(records, repeat_policy):
    if repeat_policy not in ("error", "mean", "max"):
        raise ValueError("Unknown repeat policy")
    groups = defaultdict(list)
    metrics = defaultdict(set)
    for row in records:
        key = tuple(row[k] for k in ("model", "method", "task"))
        metrics[key].add(row["metric"])
        groups[key + (row["seed"],)].append(row["score"])
    if any(len(names) != 1 for names in metrics.values()):
        raise ValueError("A model/method/task group contains different metrics")
    per_seed, by_task = [], defaultdict(list)
    for key, scores in sorted(groups.items()):
        if len(scores) > 1 and repeat_policy == "error":
            raise ValueError(f"Repeated evaluations for {key}; choose mean or max")
        score = max(scores) if repeat_policy == "max" else statistics.mean(scores)
        row = dict(zip(("model", "method", "task", "seed"), key))
        row.update(score=score, evaluations=len(scores))
        per_seed.append(row)
        by_task[key[:3]].append(score)
    summary = []
    for key, scores in sorted(by_task.items()):
        row = dict(zip(("model", "method", "task"), key))
        row.update(metric=next(iter(metrics[key])), seeds=len(scores),
                   mean=statistics.mean(scores),
                   sample_sd=statistics.stdev(scores) if len(scores) > 1 else None)
        summary.append(row)
    return per_seed, summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation scores")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeat-policy", choices=("error", "mean", "max"),
                        default="error")
    args = parser.parse_args()
    records = read_records(args.manifest)
    per_seed, summary = aggregate(records, args.repeat_policy)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "per_seed.csv", per_seed)
    write_csv(args.output_dir / "summary.csv", summary)
    report = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "repeat_policy": args.repeat_policy,
        "score_unit": "percent",
        "seed_aggregation": "arithmetic mean and sample standard deviation (ddof=1)",
        "inputs": records, "per_seed": per_seed, "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
