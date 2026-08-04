"""Aggregate REM JSONL artifacts across seeds without manual transcription.

Example:

    python -m experiments.aggregate_results outputs/cifar_eval_* \
        --group-by config.training_config.mode,config.training_config.mobility,steps \
        --metrics fid,sampling_wall_seconds,precision,recall \
        --json-out outputs/cifar_aggregate.json \
        --csv-out outputs/cifar_aggregate.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any, Iterable

from rem.metrics import bootstrap_confidence_interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Run directories or metrics.jsonl files")
    parser.add_argument("--split", default="test")
    parser.add_argument("--group-by", default="variant,task,dt,steps")
    parser.add_argument("--metrics", default="", help="Comma-separated metric allowlist")
    parser.add_argument("--where", action="append", default=[], help="Exact key=value filter")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--csv-out", default="")
    return parser.parse_args()


def flatten(value: Any, *, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, prefix=name))
    else:
        result[prefix] = value
    return result


def discover(inputs: Iterable[str]) -> list[Path]:
    discovered: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            discovered.add(path.resolve())
        elif path.is_dir():
            discovered.update(item.resolve() for item in path.rglob("metrics.jsonl"))
        else:
            matched = [Path(item) for item in glob.glob(raw)]
            discovered.update(item.resolve() for item in matched if item.is_file())
            discovered.update(
                metric.resolve()
                for item in matched
                if item.is_dir()
                for metric in item.rglob("metrics.jsonl")
            )
    if not discovered:
        raise FileNotFoundError("no metrics.jsonl files found")
    return sorted(discovered)


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in paths:
        config_path = metrics_path.parent / "resolved_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        flat_config = {f"config.{key}": value for key, value in flatten(config).items()}
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = record.pop("metrics", {})
            row = {
                "run_id": metrics_path.parent.name,
                "artifact_path": str(metrics_path.parent),
                **flat_config,
                **flatten(record),
                **metrics,
            }
            rows.append(row)
    return rows


def parse_where(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--where values must have key=value form")
        key, expected = value.split("=", 1)
        result[key] = expected
    return result


def matches(row: dict[str, Any], filters: dict[str, str]) -> bool:
    return all(str(row.get(key)) == expected for key, expected in filters.items())


def aggregate(
    rows: list[dict[str, Any]],
    *,
    group_fields: list[str],
    metric_allowlist: set[str],
    confidence: float,
    n_resamples: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        groups.setdefault(key, []).append(row)

    excluded = {
        "step",
        "timestamp",
        "run_id",
        "artifact_path",
        "split",
        *group_fields,
    }
    output = []
    for key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        candidates = metric_allowlist or {
            name
            for row in group_rows
            for name, value in row.items()
            if name not in excluded
            and not name.startswith("config.")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
        summaries = {}
        for metric in sorted(candidates):
            values = [float(row[metric]) for row in group_rows if metric in row]
            values = [value for value in values if math.isfinite(value)]
            if not values:
                continue
            lower, upper = bootstrap_confidence_interval(
                values,
                confidence=confidence,
                n_resamples=n_resamples,
            )
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                if len(values) > 1
                else 0.0
            )
            summaries[metric] = {
                "count": len(values),
                "values": values,
                "mean": mean,
                "std": variance**0.5,
                "ci": [lower, upper],
            }
        output.append(
            {
                "group": dict(zip(group_fields, key)),
                "run_ids": sorted({row["run_id"] for row in group_rows}),
                "metrics": summaries,
            }
        )
    return output


def write_csv(path: str, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        for metric, summary in result["metrics"].items():
            rows.append(
                {
                    **result["group"],
                    "metric": metric,
                    "count": summary["count"],
                    "mean": summary["mean"],
                    "std": summary["std"],
                    "ci_lower": summary["ci"][0],
                    "ci_upper": summary["ci"][1],
                }
            )
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    paths = discover(args.inputs)
    rows = load_rows(paths)
    filters = parse_where(args.where)
    rows = [row for row in rows if row.get("split") == args.split and matches(row, filters)]
    if not rows:
        raise ValueError("no artifact records remain after filtering")
    group_fields = [value.strip() for value in args.group_by.split(",") if value.strip()]
    metrics = {value.strip() for value in args.metrics.split(",") if value.strip()}
    results = aggregate(
        rows,
        group_fields=group_fields,
        metric_allowlist=metrics,
        confidence=args.confidence,
        n_resamples=args.bootstrap_resamples,
    )
    payload = {
        "source_files": [str(path) for path in paths],
        "split": args.split,
        "group_by": group_fields,
        "filters": filters,
        "groups": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")
    if args.csv_out:
        write_csv(args.csv_out, results)
    print(rendered)


if __name__ == "__main__":
    main()
