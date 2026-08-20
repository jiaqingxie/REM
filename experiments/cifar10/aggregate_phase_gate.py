"""Aggregate phase-gate FID/KID runs across sampling seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def metric(summary: dict, name: str, *, required: bool = True) -> float | None:
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


def variant(config: dict) -> str:
    active_until = float(config.get("mobility_active_until", -1.0))
    strength = float(config.get("mobility_strength", 1.0))
    if strength == 0.0 or active_until == 0.0:
        return "identity"
    if active_until < 0:
        return "full"
    return f"cutoff={active_until:g}"


def main() -> None:
    args = parse_args()
    records_by_key: dict[tuple, dict] = {}
    for root_value in args.roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for summary_path in root.rglob("summary.json"):
            config_path = summary_path.with_name("resolved_config.json")
            if not config_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "complete":
                continue
            config = json.loads(config_path.read_text())
            try:
                record = {
                    "path": str(summary_path.resolve()),
                    "dataset": config.get("dataset", "cifar10"),
                    "samples": int(config["samples"]),
                    "seed": int(config["seed"]),
                    "variant": variant(config),
                    "mobility_active_until": float(
                        config.get("mobility_active_until", -1.0)
                    ),
                    "fid": metric(summary, "fid"),
                    "kid_mean": metric(summary, "kid_mean", required=False),
                    "kid_std": metric(summary, "kid_std", required=False),
                }
            except (KeyError, TypeError, ValueError):
                continue
            key = (
                record["dataset"],
                record["samples"],
                record["seed"],
                record["variant"],
            )
            records_by_key[key] = record

    records = sorted(
        records_by_key.values(),
        key=lambda item: (
            item["dataset"],
            item["samples"],
            item["mobility_active_until"],
            item["seed"],
        ),
    )
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["samples"], record["variant"])].append(
            record
        )

    aggregates = []
    for (dataset, samples, name), group in sorted(grouped.items()):
        fids = [item["fid"] for item in group]
        kids = [item["kid_mean"] for item in group if item["kid_mean"] is not None]
        aggregates.append(
            {
                "dataset": dataset,
                "samples": samples,
                "variant": name,
                "n": len(group),
                "seeds": [item["seed"] for item in group],
                "fid_mean": statistics.fmean(fids),
                "fid_std": statistics.stdev(fids) if len(fids) > 1 else None,
                "kid_mean": statistics.fmean(kids) if kids else None,
                "kid_across_seed_std": (
                    statistics.stdev(kids) if len(kids) > 1 else None
                ),
            }
        )

    paired = []
    records_by_pair = {
        (record["dataset"], record["samples"], record["seed"], record["variant"]): record
        for record in records
    }
    pair_groups: dict[tuple, list[dict]] = defaultdict(list)
    for dataset, samples, seed, name in records_by_pair:
        if name != "identity":
            continue
        identity = records_by_pair[(dataset, samples, seed, "identity")]
        gated = records_by_pair.get((dataset, samples, seed, "cutoff=1"))
        if gated is None:
            continue
        pair_groups[(dataset, samples)].append(
            {
                "seed": seed,
                "fid_delta_gated_minus_identity": gated["fid"] - identity["fid"],
                "kid_delta_gated_minus_identity": (
                    gated["kid_mean"] - identity["kid_mean"]
                    if gated["kid_mean"] is not None
                    and identity["kid_mean"] is not None
                    else None
                ),
            }
        )
    for (dataset, samples), pairs in sorted(pair_groups.items()):
        fid_deltas = [item["fid_delta_gated_minus_identity"] for item in pairs]
        kid_deltas = [
            item["kid_delta_gated_minus_identity"]
            for item in pairs
            if item["kid_delta_gated_minus_identity"] is not None
        ]
        paired.append(
            {
                "dataset": dataset,
                "samples": samples,
                "n": len(pairs),
                "seeds": [item["seed"] for item in pairs],
                "fid_delta_mean": statistics.fmean(fid_deltas),
                "fid_delta_std": (
                    statistics.stdev(fid_deltas) if len(fid_deltas) > 1 else None
                ),
                "kid_delta_mean": (
                    statistics.fmean(kid_deltas) if kid_deltas else None
                ),
                "kid_delta_std": (
                    statistics.stdev(kid_deltas) if len(kid_deltas) > 1 else None
                ),
                "per_seed": pairs,
            }
        )

    payload = {"records": records, "aggregates": aggregates, "paired": paired}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
