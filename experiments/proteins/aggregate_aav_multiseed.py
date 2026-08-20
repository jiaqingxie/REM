"""Aggregate nested AAV training-seed and sampling-seed comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch


TASK_PROTOCOL = {
    "medium": {"sigma": 10.0, "mobility_active_until": 1.7},
    "hard": {"sigma": 20.0, "mobility_active_until": 1.3},
}
METRICS = ("fitness", "diversity", "novelty", "unique_sequences")


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",") if item.strip()]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("provide at least two distinct training seeds")
    return seeds


def sample_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(payload: dict, *, task: str, mode: str) -> None:
    task_protocol = TASK_PROTOCOL[task]
    expected = {
        "modes": [mode],
        "sigmas": [task_protocol["sigma"]],
        "n_samples": 512,
        "top_k": 128,
        "eval_seeds": 5,
        "base_seed": 42,
        "dt": 0.01,
        "epsilon_max": 0.1,
        "time_cutoff": 0.9,
        "mobility_strength": 1.0,
        "mobility_active_until": task_protocol["mobility_active_until"],
        "divergence_samples": 1,
        "divergence_method": "exact",
    }
    if payload.get("task") != task:
        raise RuntimeError(f"evaluation task mismatch: {payload.get('task')!r}")
    protocol = payload.get("protocol", {})
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RuntimeError(
                f"{task}/{mode} protocol mismatch for {key}: "
                f"{protocol.get(key)!r} != {value!r}"
            )


def rows_by_seed(payload: dict, *, task: str, mode: str) -> dict[int, dict]:
    validate_protocol(payload, task=task, mode=mode)
    rows = {}
    for row in payload.get("rows", []):
        if row.get("mode") != mode:
            raise RuntimeError(f"unexpected row mode: {row!r}")
        seed = int(row["seed"])
        if seed in rows:
            raise RuntimeError(f"duplicate evaluation seed {seed} for {task}/{mode}")
        rows[seed] = row
    expected_seeds = [int(seed) for seed in payload.get("evaluation_seeds", [])]
    if len(expected_seeds) != 5 or set(rows) != set(expected_seeds):
        raise RuntimeError(f"incomplete evaluation rows for {task}/{mode}")
    return rows


def checkpoint_metadata(training_dir: Path, *, task: str, seed: int) -> dict:
    summary_path = training_dir / "summary.json"
    configuration_path = training_dir / "config.json"
    checkpoint_path = training_dir / "best.pt"
    if (
        not summary_path.is_file()
        or not configuration_path.is_file()
        or not checkpoint_path.is_file()
    ):
        raise RuntimeError(f"missing completed training artifacts in {training_dir}")
    summary = json.loads(summary_path.read_text())
    run_configuration = json.loads(configuration_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    expected = {
        "task": task,
        "seed": seed,
        "mobility_kind": "full",
    }
    for key, value in expected.items():
        if config.get(key) != value or run_configuration.get(key) != value:
            raise RuntimeError(
                f"training metadata mismatch for {key}: "
                f"checkpoint={config.get(key)!r}, "
                f"run={run_configuration.get(key)!r}, expected={value!r}"
            )
    if int(run_configuration.get("steps", -1)) != 10_000:
        raise RuntimeError(f"training target was not 10k: {training_dir}")
    if int(summary.get("last_step", -1)) != 10_000:
        raise RuntimeError(f"training did not reach 10k: {training_dir}")
    best_step = int(summary["best_step"])
    if int(checkpoint["step"]) != best_step:
        raise RuntimeError(f"summary/best checkpoint mismatch: {training_dir}")
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256(checkpoint_path),
        "best_step": best_step,
        "last_training_step": 10_000,
        "best_validation_transport_mse": float(
            summary["best_validation_transport_mse"]
        ),
    }


def aggregate_task(
    *,
    task: str,
    training_seeds: list[int],
    training_template: str,
    evaluation_root: Path,
) -> tuple[dict, list[dict]]:
    baseline_payload = json.loads(
        (evaluation_root / task / "em" / "summary.json").read_text()
    )
    baseline = rows_by_seed(baseline_payload, task=task, mode="em")
    sampling_seeds = [int(seed) for seed in baseline_payload["evaluation_seeds"]]
    paired_rows: list[dict] = []
    per_training_seed = {}
    checkpoint_rows = {}

    for training_seed in training_seeds:
        training_dir = Path(
            training_template.format(task=task, seed=training_seed)
        )
        checkpoint_rows[str(training_seed)] = checkpoint_metadata(
            training_dir, task=task, seed=training_seed
        )
        rem_payload = json.loads(
            (
                evaluation_root
                / task
                / f"rem_trainseed{training_seed}"
                / "summary.json"
            ).read_text()
        )
        rem = rows_by_seed(rem_payload, task=task, mode="rem")
        if [int(seed) for seed in rem_payload["evaluation_seeds"]] != sampling_seeds:
            raise RuntimeError(f"sampling seed mismatch for {task}/train{training_seed}")

        seed_rows = []
        for sampling_seed in sampling_seeds:
            row = {
                "task": task,
                "training_seed": training_seed,
                "sampling_seed": sampling_seed,
            }
            for metric in METRICS:
                em_value = float(baseline[sampling_seed][metric])
                rem_value = float(rem[sampling_seed][metric])
                row[f"em_{metric}"] = em_value
                row[f"rem_{metric}"] = rem_value
                row[f"delta_{metric}"] = rem_value - em_value
            seed_rows.append(row)
            paired_rows.append(row)

        per_training_seed[str(training_seed)] = {
            metric: {
                "em": sample_stats([row[f"em_{metric}"] for row in seed_rows]),
                "rem": sample_stats([row[f"rem_{metric}"] for row in seed_rows]),
                "paired_delta_rem_minus_em": sample_stats(
                    [row[f"delta_{metric}"] for row in seed_rows]
                ),
                "rem_wins": sum(row[f"delta_{metric}"] > 0 for row in seed_rows),
            }
            for metric in METRICS
        }

    across_training_seeds = {}
    for metric in METRICS:
        rem_sampling_means = [
            per_training_seed[str(seed)][metric]["rem"]["mean"]
            for seed in training_seeds
        ]
        delta_sampling_means = [
            per_training_seed[str(seed)][metric]["paired_delta_rem_minus_em"][
                "mean"
            ]
            for seed in training_seeds
        ]
        across_training_seeds[metric] = {
            "em_sampling_seed_stats": sample_stats(
                [float(baseline[seed][metric]) for seed in sampling_seeds]
            ),
            "rem_training_seed_stats_of_sampling_means": sample_stats(
                rem_sampling_means
            ),
            "paired_delta_training_seed_stats_of_sampling_means": sample_stats(
                delta_sampling_means
            ),
            "pooled_rem_15_runs_diagnostic": sample_stats(
                [row[f"rem_{metric}"] for row in paired_rows]
            ),
        }

    return {
        "task": task,
        "training_seeds": training_seeds,
        "sampling_seeds": sampling_seeds,
        "checkpoints": checkpoint_rows,
        "per_training_seed": per_training_seed,
        "across_training_seeds": across_training_seeds,
    }, paired_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-seeds", type=parse_seeds, default=[0, 1, 2])
    parser.add_argument(
        "--training-template",
        required=True,
        help="Path template containing {task} and {seed} placeholders.",
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if "{task}" not in args.training_template or "{seed}" not in args.training_template:
        raise ValueError("--training-template must contain {task} and {seed}")
    args.output.mkdir(parents=True, exist_ok=True)
    results = {
        "status": "complete",
        "protocol": {
            "training_steps": 10_000,
            "training_seeds": args.training_seeds,
            "sampling_seeds_per_training_seed": 5,
            "generated_per_sampling_seed": 512,
            "top_k": 128,
            "checkpoint_selection": "validation-best",
            "mobility": "full SPD, determinant-one",
            "divergence": "exact",
        },
        "tasks": {},
    }
    all_rows = []
    for task in TASK_PROTOCOL:
        task_summary, task_rows = aggregate_task(
            task=task,
            training_seeds=args.training_seeds,
            training_template=args.training_template,
            evaluation_root=args.evaluation_root,
        )
        results["tasks"][task] = task_summary
        all_rows.extend(task_rows)

    csv_path = args.output / "paired_nested_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = args.output / "summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(json.dumps(results, indent=2) + "\n")
    os.replace(temporary_summary, summary_path)
    (args.output / "aggregate.done").write_text("complete\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
