"""Audit and summarize the five-seed non-oracle warp experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path("outputs/warp_minibatch_ot_20260923")
TRAIN_FILES = (ROOT / "main/results.json", ROOT / "main_extra/results.json")
SAMPLING_FILES = (
    ROOT / "sampling_main.json", ROOT / "sampling_extra.json",
    ROOT / "sampling_nodiv_main.json", ROOT / "sampling_nodiv_extra.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values),
            "values": values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    train_files = tuple(args.root / p.relative_to(ROOT) for p in TRAIN_FILES)
    sampling_files = tuple(args.root / p.relative_to(ROOT) for p in SAMPLING_FILES)
    train = [row for path in train_files
             for row in json.loads(path.read_text())["records"]]
    sample = [row for path in sampling_files
              for row in json.loads(path.read_text())["records"]]
    keys = [(row["curvature"], row["seed"], row["variant"]) for row in train]
    assert len(keys) == len(set(keys)), "duplicate training record"
    keys = [(row["seed"], row["variant"]) for row in sample]
    assert len(keys) == len(set(keys)), "duplicate sampling record"
    for row in train:
        if "checkpoint" in row:
            assert digest(Path(row["checkpoint"])) == row["checkpoint_sha256"]
    for row in sample:
        if row["checkpoint_sha256"] is not None:
            matching = [fit for fit in train
                        if fit["curvature"] == 1.2 and fit["seed"] == row["seed"]
                        and fit["variant"] == row["variant"].removesuffix("-no-div")]
            assert len(matching) == 1
            assert row["checkpoint_sha256"] == matching[0]["checkpoint_sha256"]
    summary = {}
    for variant in ("identity", "diagonal", "full", "full-no-div"):
        field = [row for row in train if row["curvature"] == 1.2
                 and row["variant"] == variant]
        dynamics = [row for row in sample if row["variant"] == variant]
        assert [row["seed"] for row in sorted(dynamics, key=lambda x: x["seed"])] == list(range(5))
        dynamics.sort(key=lambda x: x["seed"])
        result = {}
        if field:
            field.sort(key=lambda x: x["seed"])
            assert [row["seed"] for row in field] == list(range(5))
            result["relative_velocity_mse"] = describe(
                [row["test"]["relative_velocity_mse"] for row in field])
            result["descent_violation_fraction"] = describe(
                [row["test"]["descent_violation_fraction"] for row in field])
        for key in ("median_first_passage_steps", "latent_mmd",
                    "mode_max_absolute_occupancy_error", "mean_mode_indicator_ess",
                    "sample_wall_seconds"):
            result[key] = describe([row["metrics"][key] for row in dynamics])
        summary[variant] = result
    comparison = {}
    for baseline in ("identity", "diagonal"):
        comparison[baseline] = {}
        for metric in ("median_first_passage_steps", "latent_mmd",
                       "mode_max_absolute_occupancy_error"):
            full = summary["full"][metric]["values"]
            other = summary[baseline][metric]["values"]
            comparison[baseline][metric] = {
                "paired_difference": describe([a - b for a, b in zip(full, other)]),
                "improved_seeds": sum(a < b for a, b in zip(full, other)),
            }
    report = {
        "protocol": "frozen 2D warped GMM; exact minibatch OT chords; five independent fits and corrected 64-chain sampling streams at c=1.2, dt=.002, burn=4, draw=12",
        "source_files_sha256": {str(path): digest(path) for path in (*train_files, *sampling_files)},
        "summary": summary, "paired_comparisons": comparison,
        "zero_curvature_field": {
            variant: describe([row["test"]["relative_velocity_mse"]
                               for row in sorted(train, key=lambda x: x["seed"])
                               if row["curvature"] == 0 and row["variant"] == variant])
            for variant in ("identity", "diagonal", "full")
        },
    }
    destination = args.root / "aggregate.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({variant: {
        key: round(value["mean"], 6) for key, value in metrics.items()
        if key in ("relative_velocity_mse", "median_first_passage_steps", "latent_mmd")}
        for variant, metrics in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
