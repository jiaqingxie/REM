"""Aggregate the matched CIFAR additive-residual reviewer experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviewer-aggregate",
        default="outputs/reviewer_expansion_20260818/paper_aggregate.json",
    )
    parser.add_argument(
        "--results-root", default="outputs/additive_residual_cifar_20260828"
    )
    parser.add_argument(
        "--diagnostic",
        default="outputs/additive_residual_cifar_20260828/heldout_diagnostic.json",
    )
    parser.add_argument(
        "--rem-diagnostic",
        default="outputs/reviewer_diagnostics_20260820/cifar10_image_mobility/summary.json",
    )
    parser.add_argument(
        "--output", default="outputs/additive_residual_cifar_20260828/aggregate.json"
    )
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, object]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def metric(summary: dict, name: str) -> float:
    return float(summary["metrics"][f"steps=325/{name}"]["mean"])


def main() -> None:
    args = parse_args()
    reviewer = json.loads(Path(args.reviewer_aggregate).read_text(encoding="utf-8"))
    root = Path(args.results_root)
    identity_records = {
        int(record["sampling_seed"]): record
        for record in reviewer["records"]
        if record["dataset"] == "cifar10" and record["variant"] == "identity"
    }
    rem_records = {
        (int(record["mobility_training_seed"]), int(record["sampling_seed"])): record
        for record in reviewer["records"]
        if record["dataset"] == "cifar10"
        and record["variant"] == "rem_phase_gated"
    }
    if len(identity_records) != 3 or len(rem_records) != 9:
        raise RuntimeError("the verified identity/REM reviewer grid is incomplete")

    additive_records = {}
    for training_seed in range(3):
        for sampling_seed in range(3):
            path = root / (
                f"cifar_additive_train{training_seed}_sample{sampling_seed}_50k"
            ) / "summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete" or payload.get("failure_reason") is not None:
                raise RuntimeError(f"incomplete additive evaluation: {path}")
            additive_records[(training_seed, sampling_seed)] = {
                "path": str(path.resolve()),
                "fid": metric(payload, "fid"),
                "kid_mean": metric(payload, "kid_mean"),
                "steady_state_wall_seconds": metric(payload, "steady_state_wall_seconds"),
                "images_per_second": metric(payload, "images_per_second"),
                "peak_memory_bytes": metric(payload, "peak_memory_bytes"),
                "mobility_drift_seconds_per_step": metric(
                    payload, "mobility_drift_seconds_per_step"
                ),
            }

    identity_fid = {seed: float(record["fid"]) for seed, record in identity_records.items()}
    identity_kid = {
        seed: float(record["kid_mean"]) for seed, record in identity_records.items()
    }
    rem_fit_fid = [
        statistics.fmean(rem_records[(fit, stream)]["fid"] for stream in range(3))
        for fit in range(3)
    ]
    rem_fit_kid = [
        statistics.fmean(rem_records[(fit, stream)]["kid_mean"] for stream in range(3))
        for fit in range(3)
    ]
    additive_fit_fid = [
        statistics.fmean(additive_records[(fit, stream)]["fid"] for stream in range(3))
        for fit in range(3)
    ]
    additive_fit_kid = [
        statistics.fmean(
            additive_records[(fit, stream)]["kid_mean"] for stream in range(3)
        )
        for fit in range(3)
    ]
    additive_delta_identity = [
        statistics.fmean(
            additive_records[(fit, stream)]["fid"] - identity_fid[stream]
            for stream in range(3)
        )
        for fit in range(3)
    ]
    additive_delta_rem = [
        statistics.fmean(
            additive_records[(fit, stream)]["fid"]
            - float(rem_records[(fit, stream)]["fid"])
            for stream in range(3)
        )
        for fit in range(3)
    ]

    diagnostic_path = Path(args.diagnostic)
    heldout = (
        json.loads(diagnostic_path.read_text(encoding="utf-8"))
        if diagnostic_path.is_file()
        else None
    )
    rem_heldout = json.loads(Path(args.rem_diagnostic).read_text(encoding="utf-8"))
    payload = {
        "status": "complete",
        "design": {
            "energy": "same frozen CIFAR EM 147k checkpoint",
            "supervision": "same minibatch-OT straight chords",
            "learned_parameters": {"additive_residual": 57635, "diagonal_rem": 57635},
            "training_steps": 150000,
            "phase_cutoff": 1.0,
            "heun_intervals": 325,
            "energy_gradient_evaluations": 650,
            "samples_per_stream": 50000,
            "training_fits": 3,
            "sampling_streams_per_fit": 3,
            "heldout_diagnostic": (
                str(diagnostic_path.resolve()) if heldout is not None else "not run"
            ),
        },
        "identity": {
            "fid": stats(list(identity_fid.values())),
            "kid_mean": stats(list(identity_kid.values())),
            "heldout_relative_euclidean_velocity_mse": rem_heldout[
                "identity_relative_euclidean_velocity_mse"
            ],
        },
        "rem": {
            "fid_nested": stats(rem_fit_fid),
            "kid_nested": stats(rem_fit_kid),
            "heldout_relative_euclidean_velocity_mse_nested": rem_heldout[
                "rem_relative_euclidean_velocity_mse_nested"
            ],
        },
        "additive_residual": {
            "fid_nested": stats(additive_fit_fid),
            "kid_nested": stats(additive_fit_kid),
            "fid_delta_vs_identity_nested": stats(additive_delta_identity),
            "fid_delta_vs_rem_nested": stats(additive_delta_rem),
            "fid_pair_wins_vs_identity": sum(
                additive_records[(fit, stream)]["fid"] < identity_fid[stream]
                for fit in range(3)
                for stream in range(3)
            ),
            "fid_pair_wins_vs_rem": sum(
                additive_records[(fit, stream)]["fid"]
                < float(rem_records[(fit, stream)]["fid"])
                for fit in range(3)
                for stream in range(3)
            ),
            "heldout_relative_euclidean_velocity_mse_nested": (
                heldout["additive_relative_euclidean_velocity_mse_nested"]
                if heldout is not None
                else None
            ),
            "inference": {
                name: stats(
                    [record[name] for record in additive_records.values()]
                )
                for name in (
                    "steady_state_wall_seconds",
                    "images_per_second",
                    "peak_memory_bytes",
                    "mobility_drift_seconds_per_step",
                )
            },
        },
        "records": [
            {
                "training_seed": fit,
                "sampling_seed": stream,
                **additive_records[(fit, stream)],
            }
            for fit in range(3)
            for stream in range(3)
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
