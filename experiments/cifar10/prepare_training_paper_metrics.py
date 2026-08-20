"""Prepare one paper-facing final training record per CIFAR run.

Raw ``metrics.jsonl`` files are append-only.  A quota-recovery resume therefore
adds a second step-149999 row whose wall clock covers only the resumed tail.
This utility preserves those raw records and writes a derived, auditable input
for aggregation: final model diagnostics come from the latest row, while the
wall clock is reconstructed from the complete algorithmic training path.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Training run directories")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--total-steps", type=int, default=150_000)
    parser.add_argument("--global-batch-size", type=int, default=128)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def training_wall(row: dict[str, Any]) -> float:
    return float(row["metrics"]["training_wall_seconds"])


def summary_metric(summary: dict[str, Any], name: str) -> float:
    try:
        return float(summary["metrics"][name]["mean"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing summary metric {name!r}") from error


def resume_start(history: list[dict[str, Any]]) -> tuple[int, str] | None:
    if not history:
        return None
    latest = history[-1]
    checkpoint = str(latest.get("config", {}).get("resume", ""))
    match = re.search(r"checkpoint_(\d+)\.pt$", checkpoint)
    if not match:
        return None
    created_at = str(latest.get("environment", {}).get("created_at", ""))
    return int(match.group(1)), created_at


def prepare_run(
    run_dir: Path,
    output_root: Path,
    *,
    total_steps: int,
    global_batch_size: int,
) -> None:
    metrics_path = run_dir / "metrics.jsonl"
    rows = read_jsonl(metrics_path)
    config_path = run_dir / "resolved_config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    wall_budgeted = float(config.get("max_training_seconds", 0.0) or 0.0) > 0.0

    history_path = run_dir / "resume_history.jsonl"
    history = read_jsonl(history_path) if history_path.exists() else []
    if wall_budgeted:
        train_rows = [
            row
            for row in rows
            if row.get("split") == "train"
            and "training_wall_seconds" in row.get("metrics", {})
        ]
        if not train_rows:
            raise ValueError(f"missing wall-clock-budget train rows: {run_dir}")
        latest_final = max(train_rows, key=lambda row: str(row.get("timestamp", "")))
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise ValueError(f"missing wall-clock-budget summary: {run_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise ValueError(f"wall-clock-budget run is not complete: {run_dir}")
        paper_wall = summary_metric(summary, "training_wall_seconds")
        completed_steps = summary_metric(summary, "completed_steps")
        examples_seen = summary_metric(summary, "examples_seen")
        method = "completed wall-clock budget summary; latest logged diagnostics"
    else:
        final_rows = [
            row
            for row in rows
            if row.get("split") == "train"
            and int(row.get("step", -1)) == total_steps - 1
        ]
        if not final_rows:
            raise ValueError(f"missing step-{total_steps - 1} train row: {run_dir}")
        latest_final = max(final_rows, key=lambda row: str(row.get("timestamp", "")))
        resume = resume_start(history)
        method = "single uninterrupted final row"
        paper_wall = training_wall(latest_final)

        # If the original run reached the final logged step before only the final
        # checkpoint write failed, its wall clock is the complete algorithmic cost.
        if len(final_rows) > 1:
            paper_wall = max(training_wall(row) for row in final_rows)
            method = "maximum complete step-149999 wall clock before quota recovery"
        elif resume is not None:
            # If the original run stopped before the final step, join the prefix up
            # to the last valid checkpoint with the resumed tail.  Timestamp keeps
            # post-resume rows out of the prefix; the one-step logging granularity
            # around a checkpoint changes the result by less than one iteration.
            start_step, resumed_at = resume
            prefix_rows = [
                row
                for row in rows
                if row.get("split") == "train"
                and int(row.get("step", -1)) <= start_step
                and str(row.get("timestamp", "")) < resumed_at
                and "training_wall_seconds" in row.get("metrics", {})
            ]
            if not prefix_rows:
                raise ValueError(f"cannot reconstruct pre-resume wall clock: {run_dir}")
            paper_wall = max(training_wall(row) for row in prefix_rows) + training_wall(
                latest_final
            )
            method = "pre-checkpoint prefix plus quota-recovery resumed tail"
        completed_steps = float(total_steps)
        examples_seen = float(total_steps * global_batch_size)

    derived = json.loads(json.dumps(latest_final))
    derived_metrics = derived["metrics"]
    derived_metrics["training_wall_seconds"] = paper_wall
    derived_metrics["training_gpu_hours"] = paper_wall / 3600.0
    derived_metrics["completed_steps"] = completed_steps
    derived_metrics["examples_seen"] = examples_seen
    derived_metrics["quota_recovery_tail_wall_seconds"] = (
        training_wall(latest_final) if history and not wall_budgeted else 0.0
    )
    derived["paper_wall_seconds_method"] = method

    target = output_root / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        shutil.copyfile(config_path, target / "resolved_config.json")
    (target / "metrics.jsonl").write_text(
        json.dumps(derived, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "source_metrics": str(metrics_path.resolve()),
        "source_resume_history": str(history_path.resolve()) if history else None,
        "selected_final_timestamp": latest_final.get("timestamp"),
        "selected_diagnostic_step": latest_final.get("step"),
        "completed_steps": completed_steps,
        "paper_training_wall_seconds": paper_wall,
        "method": method,
    }
    (target / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for raw in args.inputs:
        prepare_run(
            Path(raw),
            output_root,
            total_steps=args.total_steps,
            global_batch_size=args.global_batch_size,
        )


if __name__ == "__main__":
    main()
