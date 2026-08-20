"""Find an EM-Large width multiplier matching REM trainable parameters."""

from __future__ import annotations

import argparse
import copy
import gc
import json

from experiments.cifar10.train_rem import build_energy
from rem.networks import build_mobility
from rem.training import trainable_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mobility",
        choices=["constant", "scalar", "diagonal", "unfixed-diagonal", "low-rank"],
        default="diagonal",
    )
    parser.add_argument("--mobility-hidden", type=int, default=32)
    parser.add_argument("--mobility-depth", type=int, default=3)
    parser.add_argument("--mobility-rank", type=int, default=4)
    parser.add_argument("--mobility-log-bound", type=float, default=1.5)
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--channel-mult", default="1,2,2,2")
    parser.add_argument("--attention-resolutions", default="16")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-head-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=8)
    parser.add_argument("--output-scale", type=float, default=1000.0)
    parser.add_argument("--energy-clamp", type=float, default=None)
    parser.add_argument("--lower", type=float, default=1.0)
    parser.add_argument("--upper", type=float, default=1.5)
    parser.add_argument("--iterations", type=int, default=12)
    return parser.parse_args()


def energy_count(
    args: argparse.Namespace,
    multiplier: float,
    *,
    enlarged: bool = True,
) -> int:
    candidate = copy.copy(args)
    # Count the true baseline separately.  ``build_energy`` uses the
    # ``em-large`` mode only to select the architecture; treating its default
    # multiplier as an implicit 1.1 makes the width search non-monotone.
    candidate.mode = "em-large" if enlarged and multiplier > 1.0 else "baseline"
    candidate.energy_width_multiplier = multiplier
    model = build_energy(candidate)
    count = trainable_parameter_count(model)
    del model
    gc.collect()
    return count


def main() -> None:
    args = parse_args()
    base_energy = energy_count(args, 1.0, enlarged=False)
    mobility = build_mobility(
        args.mobility,
        (3, 32, 32),
        hidden_dim=args.mobility_hidden,
        depth=args.mobility_depth,
        log_bound=args.mobility_log_bound,
        rank=args.mobility_rank,
    )
    mobility_parameters = trainable_parameter_count(mobility)
    target = base_energy + mobility_parameters
    del mobility

    lower, upper = args.lower, args.upper
    candidates: dict[float, int] = {
        lower: energy_count(args, lower),
        upper: energy_count(args, upper),
    }
    for _ in range(args.iterations):
        middle = 0.5 * (lower + upper)
        count = energy_count(args, middle)
        candidates[middle] = count
        if count < target:
            lower = middle
        else:
            upper = middle
    multiplier, matched = min(
        candidates.items(), key=lambda item: abs(item[1] - target)
    )
    print(
        json.dumps(
            {
                "base_energy_parameters": base_energy,
                "mobility_parameters": mobility_parameters,
                "target_total_parameters": target,
                "recommended_energy_width_multiplier": multiplier,
                "matched_energy_parameters": matched,
                "relative_parameter_error": abs(matched - target) / target,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
