"""Generate or execute reproducible multi-seed experiment manifests."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--energy-checkpoint", default="")
    parser.add_argument(
        "--frozen-checkpoint-template",
        default="{output}/rem_diagonal_seed{seed}/checkpoints/latest.pt",
    )
    parser.add_argument("--em-large-multiplier", type=float, default=0.0)
    parser.add_argument("--compute-matched-seconds", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def option(name: str, value) -> list[str]:
    flag = "--" + name
    if isinstance(value, bool):
        return [flag if value else "--no-" + name]
    return [flag, str(value)]


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    commands = []
    for experiment in manifest["experiments"]:
        for seed in manifest["seeds"]:
            if args.gpus > 1:
                command = [
                    "torchrun",
                    "--standalone",
                    f"--nproc_per_node={args.gpus}",
                    "-m",
                    manifest["module"],
                ]
            else:
                command = [sys.executable, "-m", manifest["module"]]
            metadata_keys = {"name", "requires-parameter-match", "requires-compute-budget"}
            resolved = {
                **manifest.get("shared", {}),
                **{key: value for key, value in experiment.items() if key not in metadata_keys},
                "seed": seed,
                "output": args.output,
                "experiment-id": f"{experiment['name']}_seed{seed}",
            }
            if args.energy_checkpoint and resolved.get("mode") in {"frozen", "joint"}:
                resolved["energy-checkpoint"] = args.energy_checkpoint
            if resolved.get("mode") == "joint":
                resolved["rem-init-checkpoint"] = args.frozen_checkpoint_template.format(
                    output=args.output,
                    seed=seed,
                )
            if experiment.get("requires-parameter-match"):
                if args.em_large_multiplier <= 0 and args.execute:
                    raise ValueError(
                        "EM-Large execution requires --em-large-multiplier from match_capacity"
                    )
                if args.em_large_multiplier > 0:
                    resolved["energy-width-multiplier"] = args.em_large_multiplier
            if experiment.get("requires-compute-budget"):
                if args.compute_matched_seconds <= 0 and args.execute:
                    raise ValueError(
                        "compute-matched execution requires --compute-matched-seconds"
                    )
                if args.compute_matched_seconds > 0:
                    resolved["max-training-seconds"] = args.compute_matched_seconds
            for name, value in resolved.items():
                command.extend(option(name, value))
            commands.append(command)

    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
