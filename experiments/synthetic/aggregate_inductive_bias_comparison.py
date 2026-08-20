"""Validate and aggregate the controlled inductive-bias experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


VARIANTS = (
    "identity",
    "rem-full",
    "pem-style-residual",
    "pem-helmholtz-residual",
    "learned-mirror",
    "learned-preconditioner",
    "oracle-full",
)
REPRESENTATION_METRICS = (
    "relative_velocity_mse",
    "mean_velocity_cosine",
    "predicted_descent_violation_rate",
    "trainable_parameters",
)
OPTIONAL_REPRESENTATION_METRICS = (
    "normalized_divergence_mse",
    "divergence_rms",
)
EQUILIBRIUM_METRICS = (
    "latent_mmd",
    "stationary_expectation_error",
    "mode_transitions_per_1000_steps",
    "median_first_passage_steps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, nargs="+")
    parser.add_argument("--expected-seeds", default="0,1,2,3,4")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate(records: list[dict], expected_seeds: list[int]) -> dict:
    by_key: dict[tuple[int, str], dict] = {}
    for record in records:
        key = (int(record["seed"]), str(record["variant"]))
        if key in by_key:
            raise ValueError(f"duplicate inductive-bias record: {key}")
        by_key[key] = record

    expected = {(seed, variant) for seed in expected_seeds for variant in VARIANTS}
    found = set(by_key)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    if missing or unexpected:
        raise ValueError(
            f"incomplete inductive-bias grid; missing={missing}, unexpected={unexpected}"
        )

    for seed in expected_seeds:
        for variant in VARIANTS:
            corrected = bool(by_key[(seed, variant)]["equilibrium_correction"])
            if corrected != (
                variant not in {"pem-style-residual", "pem-helmholtz-residual"}
            ):
                raise ValueError(
                    f"unexpected equilibrium correction for seed={seed}, variant={variant}"
                )

    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in by_key.values():
        variant = record["variant"]
        for metric in REPRESENTATION_METRICS:
            values[(variant, "representation", metric)].append(
                float(record["representation"][metric])
            )
        for metric in OPTIONAL_REPRESENTATION_METRICS:
            if metric in record["representation"]:
                values[(variant, "representation", metric)].append(
                    float(record["representation"][metric])
                )
        for metric in EQUILIBRIUM_METRICS:
            values[(variant, "equilibrium", metric)].append(
                float(record["equilibrium"][metric])
            )

    methods = []
    for variant in VARIANTS:
        methods.append(
            {
                "variant": variant,
                "representation": {
                    metric: _mean_std(values[(variant, "representation", metric)])
                    for metric in REPRESENTATION_METRICS
                }
                | {
                    metric: _mean_std(values[(variant, "representation", metric)])
                    for metric in OPTIONAL_REPRESENTATION_METRICS
                    if values[(variant, "representation", metric)]
                },
                "equilibrium": {
                    metric: _mean_std(values[(variant, "equilibrium", metric)])
                    for metric in EQUILIBRIUM_METRICS
                },
            }
        )

    parameter_counts = {
        item["variant"]: item["representation"]["trainable_parameters"]["mean"]
        for item in methods
    }
    rem_parameters = parameter_counts["rem-full"]
    residual_ratio = parameter_counts["pem-style-residual"] / rem_parameters
    helmholtz_residual_ratio = (
        parameter_counts["pem-helmholtz-residual"] / rem_parameters
    )
    mirror_ratio = parameter_counts["learned-mirror"] / rem_parameters
    return {
        "complete": True,
        "expected_seeds": expected_seeds,
        "record_count": len(records),
        "methods": methods,
        "capacity_check": {
            "rem_parameters": rem_parameters,
            "pem_residual_to_rem_ratio": residual_ratio,
            "pem_helmholtz_residual_to_rem_ratio": helmholtz_residual_ratio,
            "mirror_to_rem_ratio": mirror_ratio,
            "parameter_matched_within_5_percent": abs(residual_ratio - 1) <= 0.05
            and abs(helmholtz_residual_ratio - 1) <= 0.05
            and abs(mirror_ratio - 1) <= 0.05,
        },
    }


def main() -> None:
    args = parse_args()
    records = []
    for record_path in args.records:
        records.extend(json.loads(Path(record_path).read_text(encoding="utf-8")))
    expected_seeds = [
        int(item.strip()) for item in args.expected_seeds.split(",") if item.strip()
    ]
    payload = aggregate(records, expected_seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
