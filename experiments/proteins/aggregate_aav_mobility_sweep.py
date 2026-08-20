"""Aggregate the paired AAV mobility-strength/cutoff screening sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


TASK_GRID = {
    "medium": {
        "sigma": 10.0,
        "cutoffs": (0.9, 1.0, 1.3, 1.7),
    },
    "hard": {
        "sigma": 20.0,
        "cutoffs": (0.9, 1.0, 1.15, 1.3),
    },
}
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
METRICS = ("fitness", "diversity", "novelty", "unique_sequences", "wall_seconds")


def label(value: float) -> str:
    # Match the labels produced from the literal shell grid. The grid spells
    # one as ``1.0`` while zero is spelled ``0``.
    rendered = "1.0" if value == 1.0 else f"{value:g}"
    return rendered.replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_stats(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load_evaluation(
    path: Path,
    *,
    task: str,
    mode: str,
    sigma: float,
    strength: float,
    cutoff: float,
) -> tuple[dict, dict[int, dict]]:
    payload = json.loads((path / "summary.json").read_text())
    protocol = payload.get("protocol", {})
    expected = {
        "modes": [mode],
        "sigmas": [sigma],
        "n_samples": 512,
        "top_k": 128,
        "eval_seeds": 3,
        "base_seed": 42,
        "dt": 0.01,
        "epsilon_max": 0.1,
        "time_cutoff": 0.9,
        "mobility_strength": strength,
        "mobility_active_until": cutoff,
        "divergence_samples": 1,
        "divergence_method": "exact",
    }
    if payload.get("status") != "complete" or payload.get("task") != task:
        raise RuntimeError(f"incomplete or mismatched evaluation: {path}")
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RuntimeError(
                f"protocol mismatch at {path} for {key}: "
                f"{protocol.get(key)!r} != {value!r}"
            )
    rows: dict[int, dict] = {}
    for row in payload.get("rows", []):
        if row.get("mode") != mode or float(row.get("sigma")) != sigma:
            raise RuntimeError(f"row mismatch at {path}: {row!r}")
        seed = int(row["seed"])
        if seed in rows:
            raise RuntimeError(f"duplicate sampling seed {seed} at {path}")
        rows[seed] = row
    expected_seeds = [int(seed) for seed in payload.get("evaluation_seeds", [])]
    if len(expected_seeds) != 3 or set(rows) != set(expected_seeds):
        raise RuntimeError(f"expected three complete sampling seeds at {path}")
    return payload, rows


def nondominated(points: list[dict]) -> list[dict]:
    """Return points not dominated in mean fitness and mean diversity."""
    frontier = []
    for candidate in points:
        dominated = any(
            other is not candidate
            and other["fitness_mean"] >= candidate["fitness_mean"]
            and other["diversity_mean"] >= candidate["diversity_mean"]
            and (
                other["fitness_mean"] > candidate["fitness_mean"]
                or other["diversity_mean"] > candidate["diversity_mean"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (item["fitness_mean"], item["diversity_mean"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_seed_rows: list[dict] = []
    all_points: list[dict] = []
    task_summaries = {}
    checkpoint_metadata = {}

    for task, grid in TASK_GRID.items():
        checkpoint = args.checkpoint_root / f"{task}_seed0" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_metadata[task] = {
            "path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
        }
        terminal_time = max(grid["cutoffs"])
        baseline_payload, baseline_rows = load_evaluation(
            args.evaluation_root / task / "em",
            task=task,
            mode="em",
            sigma=grid["sigma"],
            strength=0.0,
            cutoff=terminal_time,
        )
        sampling_seeds = [int(seed) for seed in baseline_payload["evaluation_seeds"]]
        task_points = []

        for strength in STRENGTHS:
            for cutoff in grid["cutoffs"]:
                point_dir = (
                    args.evaluation_root
                    / task
                    / f"strength_{label(strength)}_cutoff_{label(cutoff)}"
                )
                _, rem_rows = load_evaluation(
                    point_dir,
                    task=task,
                    mode="rem",
                    sigma=grid["sigma"],
                    strength=strength,
                    cutoff=cutoff,
                )
                point_seed_rows = []
                for seed in sampling_seeds:
                    row = {
                        "task": task,
                        "strength": strength,
                        "cutoff": cutoff,
                        "sampling_seed": seed,
                    }
                    for metric in METRICS:
                        em_value = float(baseline_rows[seed][metric])
                        rem_value = float(rem_rows[seed][metric])
                        row[f"em_{metric}"] = em_value
                        row[f"rem_{metric}"] = rem_value
                        row[f"delta_{metric}"] = rem_value - em_value
                    point_seed_rows.append(row)
                    all_seed_rows.append(row)

                point = {"task": task, "strength": strength, "cutoff": cutoff}
                for metric in METRICS:
                    rem_mean, rem_std = sample_stats(
                        [row[f"rem_{metric}"] for row in point_seed_rows]
                    )
                    delta_mean, delta_std = sample_stats(
                        [row[f"delta_{metric}"] for row in point_seed_rows]
                    )
                    point[f"{metric}_mean"] = rem_mean
                    point[f"{metric}_std"] = rem_std
                    point[f"delta_{metric}_mean"] = delta_mean
                    point[f"delta_{metric}_std"] = delta_std
                task_points.append(point)
                all_points.append(point)

        frontier = nondominated(task_points)
        task_summaries[task] = {
            "sampling_seeds": sampling_seeds,
            "baseline": {
                metric: dict(zip(("mean", "std"), sample_stats(
                    [float(baseline_rows[seed][metric]) for seed in sampling_seeds]
                )))
                for metric in METRICS
            },
            "pareto_frontier": frontier,
        }

    write_csv(args.output / "per_seed.csv", all_seed_rows)
    write_csv(args.output / "points.csv", all_points)
    frontier_rows = [
        point
        for task_summary in task_summaries.values()
        for point in task_summary["pareto_frontier"]
    ]
    write_csv(args.output / "pareto_frontier.csv", frontier_rows)
    summary = {
        "status": "complete",
        "protocol": {
            "training_seed": 0,
            "sampling_seeds": 3,
            "generated_per_seed": 512,
            "selected_top_k": 128,
            "strengths": list(STRENGTHS),
            "task_grid": TASK_GRID,
            "paired_baseline": "EM with identical sampling seeds",
            "pareto_objectives": ["fitness_mean", "diversity_mean"],
            "divergence": "exact",
        },
        "checkpoints": checkpoint_metadata,
        "aggregator_sha256": sha256(Path(__file__)),
        "tasks": task_summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
