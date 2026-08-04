"""Generate or execute CIFAR sampler sweeps from checkpoint globs."""

from __future__ import annotations

import argparse
import glob
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from experiments.run_ablation_suite import option


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--training-output", default="outputs")
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--evaluation-seed-offset", type=int, default=10_000)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def training_seed(path: str) -> int:
    matches = re.findall(r"seed(\d+)", path)
    if not matches:
        raise ValueError(f"cannot infer training seed from checkpoint path: {path}")
    return int(matches[-1])


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    commands: list[list[str]] = []
    for evaluation in manifest["evaluations"]:
        pattern = evaluation["checkpoint-glob"].format(
            training_output=args.training_output
        )
        checkpoints = sorted(glob.glob(pattern))
        if not checkpoints:
            message = f"no checkpoints matched {pattern}"
            if args.execute:
                raise FileNotFoundError(message)
            print(f"# {message}")
            continue
        for checkpoint in checkpoints:
            seed = training_seed(checkpoint)
            resolved = {
                **manifest.get("shared", {}),
                **{
                    key: value
                    for key, value in evaluation.items()
                    if key not in {"name", "checkpoint-glob"}
                },
                "checkpoint": checkpoint,
                "seed": args.evaluation_seed_offset + seed,
                "output": args.output,
                "experiment-id": f"eval_{evaluation['name']}_trainseed{seed}",
                "variant-label": evaluation["name"],
            }
            command = [sys.executable, "-m", manifest["module"]]
            for name, value in resolved.items():
                command.extend(option(name, value))
            commands.append(command)

    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
