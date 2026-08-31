"""Held-out OT diagnostics for capacity-matched additive image transports."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from experiments.cifar10.train_rem import (
    build_dataset,
    build_energy,
    configure_energy_training_mode,
    configure_training_attention_backend,
)
from rem.artifacts import seed_everything
from rem.networks import build_mobility
from rem.training import energy_gradient, trainable_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--batches-per-seed", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--phase-bins", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def flatten(value: Tensor) -> Tensor:
    return value.reshape(value.shape[0], -1)


def mean_std(values: list[float]) -> dict[str, object]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "values": values,
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=True)) if tensor.numel() > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(args.checkpoints) < 2 or not seeds:
        raise ValueError("multiple checkpoints and at least one held-out seed are required")
    if min(args.batches_per_seed, args.batch_size, args.phase_bins) < 1:
        raise ValueError("batch and phase counts must be positive")

    device = torch.device(args.device)
    configure_training_attention_backend(device)
    states = [torch.load(path, map_location="cpu") for path in args.checkpoints]
    configs = [dict(state["config"]) for state in states]
    first_args = argparse.Namespace(**configs[0])
    energy = build_energy(first_args)
    energy.load_state_dict(states[0]["energy"], strict=True)
    energy.to(device).eval()
    configure_energy_training_mode(energy, frozen=True)

    transports = []
    records = []
    for path, state, config in zip(args.checkpoints, states, configs):
        if config["mobility"] != "additive-residual":
            raise ValueError(f"not an additive-residual checkpoint: {path}")
        transport = build_mobility(
            config["mobility"],
            (3, 32, 32),
            hidden_dim=int(config["mobility_hidden"]),
            depth=int(config["mobility_depth"]),
            log_bound=float(config["mobility_log_bound"]),
            rank=int(config["mobility_rank"]),
        )
        state_key = "mobility_ema" if state.get("mobility_ema") else "mobility"
        transport.load_state_dict(state[state_key], strict=True)
        transport.to(device).eval()
        for parameter in transport.parameters():
            parameter.requires_grad_(False)
        transports.append(transport)
        records.append(
            {
                "checkpoint": str(Path(path).resolve()),
                "training_seed": int(config["seed"]),
                "state_key": state_key,
                "parameters": trainable_parameter_count(transport),
            }
        )

    dataset_args = argparse.Namespace(**configs[0])
    dataset_args.dataset = "cifar10"
    dataset_args.data_root = args.data_root
    dataset = build_dataset(dataset_args, distributed=False, rank=0)
    try:
        from torchcfm.conditional_flow_matching import (
            ExactOptimalTransportConditionalFlowMatcher,
        )
    except ImportError as error:
        raise RuntimeError("diagnostic requires torchcfm") from error
    matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    fit_count = len(transports)
    target_sq = 0.0
    identity_sq = 0.0
    residual_sq = [0.0] * fit_count
    correction_sq = [0.0] * fit_count
    improved = [0] * fit_count
    correction_correlations = {pair: [] for pair in combinations(range(fit_count), 2)}
    phases = [
        {"target_sq": 0.0, "identity_sq": 0.0, "residual_sq": [0.0] * fit_count}
        for _ in range(args.phase_bins)
    ]
    sample_count = 0

    for seed in seeds:
        seed_everything(seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(80_000_000 + seed),
        )
        for batch_index, (real, _) in enumerate(loader):
            if batch_index >= args.batches_per_seed:
                break
            real = real.to(device, non_blocking=True)
            noise = torch.randn_like(real)
            t, x_t, target = matcher.sample_location_and_conditional_flow(noise, real)
            gradient = energy_gradient(energy, x_t, t).detach()
            target = target.detach()
            identity = -gradient - target
            per_target = flatten(target).square().sum(dim=1)
            per_identity = flatten(identity).square().sum(dim=1)
            target_sq += float(per_target.sum())
            identity_sq += float(per_identity.sum())
            sample_count += real.shape[0]

            corrections = []
            per_fit_sq = []
            with torch.no_grad():
                for fit_index, transport in enumerate(transports):
                    correction = transport.residual(x_t)
                    error = -gradient + correction - target
                    per_error = flatten(error).square().sum(dim=1)
                    corrections.append(flatten(correction))
                    per_fit_sq.append(per_error)
                    residual_sq[fit_index] += float(per_error.sum())
                    correction_sq[fit_index] += float(flatten(correction).square().sum())
                    improved[fit_index] += int((per_error < per_identity).sum())

            for pair in correction_correlations:
                left = corrections[pair[0]]
                right = corrections[pair[1]]
                left = left - left.mean(dim=1, keepdim=True)
                right = right - right.mean(dim=1, keepdim=True)
                corr = (left * right).sum(dim=1) / (
                    left.norm(dim=1) * right.norm(dim=1)
                ).clamp_min(1e-12)
                correction_correlations[pair].append(corr.cpu())

            phase_index = torch.clamp(
                torch.floor(t.reshape(-1) * args.phase_bins).long(),
                min=0,
                max=args.phase_bins - 1,
            )
            for bin_index, phase in enumerate(phases):
                mask = phase_index == bin_index
                if not bool(mask.any()):
                    continue
                phase["target_sq"] += float(per_target[mask].sum())
                phase["identity_sq"] += float(per_identity[mask].sum())
                for fit_index in range(fit_count):
                    phase["residual_sq"][fit_index] += float(per_fit_sq[fit_index][mask].sum())

    identity_relative = identity_sq / target_sq
    relative = [value / target_sq for value in residual_sq]
    for record, value, correction, count in zip(
        records, relative, correction_sq, improved
    ):
        record.update(
            {
                "heldout_relative_euclidean_velocity_mse": value,
                "relative_reduction_vs_identity": 1.0 - value / identity_relative,
                "sample_improvement_fraction": count / sample_count,
                "correction_energy_over_target_energy": correction / target_sq,
            }
        )
    result = {
        "status": "complete",
        "supervision": "held-out standard minibatch-OT straight chords",
        "sample_count": sample_count,
        "heldout_draw_count": len(seeds),
        "identity_relative_euclidean_velocity_mse": identity_relative,
        "additive_relative_euclidean_velocity_mse_nested": mean_std(relative),
        "fits": records,
        "phase_bins": [
            {
                "t_interval": [index / args.phase_bins, (index + 1) / args.phase_bins],
                "identity_relative_euclidean_velocity_mse": phase["identity_sq"] / phase["target_sq"],
                "additive_relative_euclidean_velocity_mse": [
                    value / phase["target_sq"] for value in phase["residual_sq"]
                ],
            }
            for index, phase in enumerate(phases)
        ],
        "cross_fit_correction_correlation": [
            {
                "fit_indices": list(pair),
                "mean": float(torch.cat(values).mean()),
                "std": float(torch.cat(values).std(unbiased=True)),
            }
            for pair, values in correction_correlations.items()
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
