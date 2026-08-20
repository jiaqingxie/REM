"""Aggregate nested mobility-training and sampling-seed image evaluations.

The paper's original CIFAR result varied only the sampling stream for one
trained mobility.  This aggregator keeps the two randomness levels separate:
it first forms paired REM-minus-identity deltas within a mobility-training seed
and then reports variation across independently trained mobilities.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--require-complete-reviewer-grid",
        action="store_true",
        help="Fail unless all predeclared reviewer-expansion runs are present.",
    )
    return parser.parse_args()


def _metric(summary: dict, name: str, *, required: bool = True) -> float | None:
    matches = [
        value["mean"]
        for key, value in summary["metrics"].items()
        if key.endswith(f"/{name}")
    ]
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected one {name} metric, found {len(matches)}")
    return float(matches[0])


def _variant(config: dict) -> str:
    strength = float(config.get("mobility_strength", 1.0))
    active_until = float(config.get("mobility_active_until", -1.0))
    if strength == 0.0 or active_until == 0.0:
        return "identity"
    if active_until == 1.0:
        checkpoint = str(config.get("checkpoint", ""))
        label = str(config.get("variant_label", ""))
        if "em_const_seed" in checkpoint or "constant-gated" in label:
            return "learned_constant_gated"
        return "rem_phase_gated"
    return "other"


def _training_seed(config: dict, variant: str) -> int | None:
    if variant == "identity":
        return None
    explicit = int(config.get("mobility_training_seed", -1))
    if explicit >= 0:
        return explicit
    checkpoint = str(config.get("checkpoint", ""))
    match = re.search(r"rem_diagonal_seed(\d+)", checkpoint)
    return int(match.group(1)) if match else None


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
    }


def _equal(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


def protocol_violations(config: dict, variant: str) -> list[str]:
    """Return deviations from the preregistered matched image protocol."""

    dataset = str(config.get("dataset", "cifar10"))
    expected = {
        "batch_size": 64,
        "clamp_every_step": False,
        "divergence_correction": True,
        "energy_state_key": "energy",
        "integrator": "heun",
        "mobility_state_key": "mobility_ema",
        "mobility_strength": 1.0,
        "samples": 50_000,
        "temperature": 0.01,
        "temperature_ramp_end": 1.0,
        "temperature_schedule": "official",
        "time_cutoff": 1.0,
        "use_ema": True,
    }
    if dataset == "cifar10":
        expected.update({"step_counts": [325], "t_end": 3.25})
        energy_name = "cifar10_main_training_147000.pt"
    elif dataset == "imagenet32":
        expected.update(
            {
                "expected_fid_real_samples": 1_281_167,
                "image_quantization": "floor",
                "kid": False,
                "step_counts": [250],
                "t_end": 2.5,
            }
        )
        energy_name = "imagenet32x32_main_training_641000.pt"
    else:
        return [f"unsupported dataset={dataset!r}"]

    expected["mobility_active_until"] = 0.0 if variant == "identity" else 1.0
    violations = [
        f"{name}={config.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if not _equal(config.get(name), value)
    ]

    training = config.get("training_config")
    if not isinstance(training, dict):
        violations.append("missing training_config")
        return violations
    if Path(str(training.get("energy_checkpoint", ""))).name != energy_name:
        violations.append(
            "training_config.energy_checkpoint="
            f"{training.get('energy_checkpoint')!r}, expected basename {energy_name!r}"
        )
    for name, value in (("mode", "frozen"), ("total_steps", 150_000)):
        if not _equal(training.get(name), value):
            violations.append(
                f"training_config.{name}={training.get(name)!r}, expected {value!r}"
            )

    checkpoint = str(config.get("checkpoint", ""))
    inferred_training_seed = _training_seed(config, variant)
    if dataset == "imagenet32":
        expected_training_seed = 0 if variant == "identity" else inferred_training_seed
        if expected_training_seed is None:
            violations.append("could not infer mobility-training seed")
            return violations
        expected_checkpoint = (
            f"/rem_diagonal_seed{expected_training_seed}/checkpoints/latest.pt"
        )
        expected_mobility = "diagonal"
        stats_name = "imagenet32_train_1281167_floor_fid_stats.pt"
        if Path(str(config.get("fid_real_stats", ""))).name != stats_name:
            violations.append(
                f"fid_real_stats must end in {stats_name!r}"
            )
    elif variant == "learned_constant_gated":
        expected_checkpoint = "/em_const_seed0/checkpoints/checkpoint_150000.pt"
        expected_training_seed = 0
        expected_mobility = "constant"
    else:
        expected_training_seed = 0 if variant == "identity" else inferred_training_seed
        if expected_training_seed is None:
            violations.append("could not infer mobility-training seed")
            return violations
        expected_checkpoint = (
            f"/rem_diagonal_seed{expected_training_seed}/checkpoints/checkpoint_150000.pt"
        )
        expected_mobility = "diagonal"
    if not checkpoint.endswith(expected_checkpoint):
        violations.append(
            f"checkpoint={checkpoint!r}, expected suffix {expected_checkpoint!r}"
        )
    if not _equal(training.get("seed"), expected_training_seed):
        violations.append(
            f"training_config.seed={training.get('seed')!r}, expected {expected_training_seed!r}"
        )
    if not _equal(training.get("mobility"), expected_mobility):
        violations.append(
            f"training_config.mobility={training.get('mobility')!r}, expected {expected_mobility!r}"
        )
    return violations


def collect_records(roots: list[str]) -> list[dict]:
    records_by_key: dict[tuple, dict] = {}
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for summary_path in root.rglob("summary.json"):
            config_path = summary_path.with_name("resolved_config.json")
            if not config_path.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text())
                config = json.loads(config_path.read_text())
                if summary.get("status") != "complete":
                    continue
                variant = _variant(config)
                if variant == "other" or int(config.get("samples", 0)) != 50_000:
                    continue
                record = {
                    "dataset": str(config.get("dataset", "cifar10")),
                    "status": summary.get("status"),
                    "failure_reason": summary.get("failure_reason"),
                    "samples": int(config["samples"]),
                    "sampling_seed": int(config["seed"]),
                    "mobility_training_seed": _training_seed(config, variant),
                    "variant": variant,
                    "fid": _metric(summary, "fid"),
                    "kid_mean": _metric(summary, "kid_mean", required=False),
                    "path": str(summary_path.resolve()),
                    "protocol_violations": protocol_violations(config, variant),
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            key = (
                record["dataset"],
                record["sampling_seed"],
                record["mobility_training_seed"],
                record["variant"],
            )
            if key in records_by_key:
                raise ValueError(
                    "duplicate reviewer record key "
                    f"{key}: {records_by_key[key]['path']} and {record['path']}"
                )
            records_by_key[key] = record
    return sorted(
        records_by_key.values(),
        key=lambda item: (
            item["dataset"],
            item["variant"],
            -1 if item["mobility_training_seed"] is None else item["mobility_training_seed"],
            item["sampling_seed"],
        ),
    )


def aggregate(records: list[dict]) -> dict:
    identities = {
        (record["dataset"], record["sampling_seed"]): record
        for record in records
        if record["variant"] == "identity"
    }
    rem_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        training_seed = record["mobility_training_seed"]
        if record["variant"] == "rem_phase_gated" and training_seed is not None:
            rem_groups[(record["dataset"], training_seed)].append(record)

    per_training_seed = []
    for (dataset, training_seed), group in sorted(rem_groups.items()):
        pairs = []
        for rem_record in sorted(group, key=lambda item: item["sampling_seed"]):
            identity = identities.get((dataset, rem_record["sampling_seed"]))
            if identity is None:
                continue
            kid_delta = None
            if rem_record["kid_mean"] is not None and identity["kid_mean"] is not None:
                kid_delta = rem_record["kid_mean"] - identity["kid_mean"]
            pairs.append(
                {
                    "sampling_seed": rem_record["sampling_seed"],
                    "identity_fid": identity["fid"],
                    "rem_fid": rem_record["fid"],
                    "identity_kid_mean": identity["kid_mean"],
                    "rem_kid_mean": rem_record["kid_mean"],
                    "fid_delta_rem_minus_identity": rem_record["fid"] - identity["fid"],
                    "kid_delta_rem_minus_identity": kid_delta,
                }
            )
        if not pairs:
            continue
        fid_values = [item["rem_fid"] for item in pairs]
        fid_deltas = [item["fid_delta_rem_minus_identity"] for item in pairs]
        kid_deltas = [
            item["kid_delta_rem_minus_identity"]
            for item in pairs
            if item["kid_delta_rem_minus_identity"] is not None
        ]
        kid_values = [
            item["rem_kid_mean"]
            for item in pairs
            if item["rem_kid_mean"] is not None
        ]
        per_training_seed.append(
            {
                "dataset": dataset,
                "mobility_training_seed": training_seed,
                "sampling_seeds": [item["sampling_seed"] for item in pairs],
                "rem_fid": _mean_std(fid_values),
                "rem_kid_mean": _mean_std(kid_values) if kid_values else None,
                "paired_fid_delta": _mean_std(fid_deltas),
                "paired_kid_delta": _mean_std(kid_deltas) if kid_deltas else None,
                "pairs": pairs,
            }
        )

    across_training_seed = []
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for item in per_training_seed:
        by_dataset[item["dataset"]].append(item)
    for dataset, group in sorted(by_dataset.items()):
        fid_means = [item["rem_fid"]["mean"] for item in group]
        delta_means = [item["paired_fid_delta"]["mean"] for item in group]
        kid_delta_means = [
            item["paired_kid_delta"]["mean"]
            for item in group
            if item["paired_kid_delta"] is not None
        ]
        kid_means = [
            item["rem_kid_mean"]["mean"]
            for item in group
            if item["rem_kid_mean"] is not None
        ]
        across_training_seed.append(
            {
                "dataset": dataset,
                "mobility_training_seeds": [
                    item["mobility_training_seed"] for item in group
                ],
                "training_seed_count": len(group),
                "mean_rem_fid_across_training_seeds": _mean_std(fid_means),
                "mean_rem_kid_across_training_seeds": _mean_std(kid_means)
                if kid_means
                else None,
                "mean_paired_fid_delta_across_training_seeds": _mean_std(delta_means),
                "mean_paired_kid_delta_across_training_seeds": _mean_std(
                    kid_delta_means
                )
                if kid_delta_means
                else None,
            }
        )

    baseline_summaries = []
    baseline_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record["variant"] not in {"identity", "rem_phase_gated"}:
            baseline_groups[(record["dataset"], record["variant"])].append(record)
    for (dataset, variant), group in sorted(baseline_groups.items()):
        pairs = []
        for record in sorted(group, key=lambda item: item["sampling_seed"]):
            identity = identities.get((dataset, record["sampling_seed"]))
            if identity is None:
                continue
            pairs.append(
                {
                    "sampling_seed": record["sampling_seed"],
                    "identity_fid": identity["fid"],
                    "baseline_fid": record["fid"],
                    "identity_kid_mean": identity["kid_mean"],
                    "baseline_kid_mean": record["kid_mean"],
                    "fid_delta_baseline_minus_identity": record["fid"]
                    - identity["fid"],
                }
            )
        if pairs:
            baseline_summaries.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "sampling_seeds": [item["sampling_seed"] for item in pairs],
                    "baseline_fid": _mean_std(
                        [item["baseline_fid"] for item in pairs]
                    ),
                    "baseline_kid_mean": _mean_std(
                        [
                            item["baseline_kid_mean"]
                            for item in pairs
                            if item["baseline_kid_mean"] is not None
                        ]
                    )
                    if any(item["baseline_kid_mean"] is not None for item in pairs)
                    else None,
                    "paired_fid_delta": _mean_std(
                        [item["fid_delta_baseline_minus_identity"] for item in pairs]
                    ),
                    "pairs": pairs,
                }
            )

    completeness = reviewer_grid_completeness(records)
    protocol_failures = [
        {
            "dataset": record["dataset"],
            "variant": record["variant"],
            "mobility_training_seed": record["mobility_training_seed"],
            "sampling_seed": record["sampling_seed"],
            "path": record["path"],
            "violations": record.get("protocol_violations", []),
        }
        for record in records
        if record.get("protocol_violations")
    ]
    execution_failures = [
        {
            "dataset": record["dataset"],
            "variant": record["variant"],
            "mobility_training_seed": record["mobility_training_seed"],
            "sampling_seed": record["sampling_seed"],
            "path": record["path"],
            "status": record.get("status"),
            "failure_reason": record.get("failure_reason"),
        }
        for record in records
        if record.get("status") != "complete" or record.get("failure_reason") not in (None, "")
    ]
    return {
        "records": records,
        "per_mobility_training_seed": per_training_seed,
        "across_mobility_training_seeds": across_training_seed,
        "baseline_summaries": baseline_summaries,
        "reviewer_grid_completeness": completeness,
        "protocol_check": {
            "complete": not protocol_failures,
            "checked_record_count": len(records),
            "failures": protocol_failures,
        },
        "execution_check": {
            "complete": not execution_failures,
            "checked_record_count": len(records),
            "failures": execution_failures,
        },
    }


def reviewer_grid_completeness(records: list[dict]) -> dict:
    found = {
        (
            record["dataset"],
            record["variant"],
            record["mobility_training_seed"],
            record["sampling_seed"],
        )
        for record in records
    }
    expected = set()
    for sampling_seed in (0, 1, 2):
        expected.add(("cifar10", "identity", None, sampling_seed))
        expected.add(("cifar10", "learned_constant_gated", 0, sampling_seed))
        for training_seed in (0, 1, 2):
            expected.add(
                ("cifar10", "rem_phase_gated", training_seed, sampling_seed)
            )
    for sampling_seed in (0, 7, 42):
        expected.add(("imagenet32", "identity", None, sampling_seed))
        for training_seed in (0, 1, 2):
            expected.add(
                ("imagenet32", "rem_phase_gated", training_seed, sampling_seed)
            )
    missing = sorted(
        expected - found,
        key=lambda item: (
            item[0],
            item[1],
            -1 if item[2] is None else item[2],
            item[3],
        ),
    )
    unexpected = sorted(
        found - expected,
        key=lambda item: (
            item[0],
            item[1],
            -1 if item[2] is None else item[2],
            item[3],
        ),
    )

    def describe(item: tuple) -> dict:
        return {
            "dataset": item[0],
            "variant": item[1],
            "mobility_training_seed": item[2],
            "sampling_seed": item[3],
        }

    return {
        "complete": not missing and not unexpected,
        "expected_record_count": len(expected),
        "found_expected_record_count": len(expected & found),
        "unexpected_record_count": len(unexpected),
        "missing": [describe(item) for item in missing],
        "unexpected": [describe(item) for item in unexpected],
    }


def main() -> None:
    args = parse_args()
    payload = aggregate(collect_records(args.roots))
    if (
        args.require_complete_reviewer_grid
        and not payload["reviewer_grid_completeness"]["complete"]
    ):
        raise SystemExit(
            "reviewer expansion is incomplete: "
            + json.dumps(payload["reviewer_grid_completeness"])
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
