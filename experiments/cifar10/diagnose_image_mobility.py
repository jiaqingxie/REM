"""Held-out diagnostics for learned image mobilities.

This diagnostic closes the gap between pointwise SPD compatibility and the
restricted diagonal neural mobility used for images.  On held-out minibatch-OT
pairs it reports

* Euclidean velocity residuals for the frozen energy and each REM fit;
* diagonal eigenvalue distributions and saturation rates;
* activation and residual reduction across OT-time bins; and
* pairwise correlation of log-mobility maps across independent fits.

The energy and mobility states are loaded exactly as in paper evaluation: the
raw frozen energy and the EMA mobility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from experiments.cifar10.train_rem import (
    build_dataset,
    build_energy,
    configure_energy_training_mode,
    configure_training_attention_backend,
)
from rem.artifacts import RunArtifacts, seed_everything
from rem.networks import build_mobility
from rem.training import energy_gradient


QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "imagenet32"], required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--batches-per-seed", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--phase-bins", type=int, default=5)
    parser.add_argument("--coordinates-per-image", type=int, default=64)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(value: Tensor) -> Tensor:
    return value.reshape(value.shape[0], -1)


def _quantiles(values: Tensor) -> dict[str, float]:
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    probabilities = torch.tensor(QUANTILES, dtype=torch.float64)
    result = torch.quantile(values.to(torch.float64), probabilities)
    return {
        f"q{int(round(100 * probability)):02d}": float(value)
        for probability, value in zip(QUANTILES, result)
    }


def _sample_correlation(left: Tensor, right: Tensor, eps: float = 1e-12) -> Tensor:
    """Pearson correlation between two flattened maps, one value per sample."""

    left_flat = _flatten(left).to(torch.float64)
    right_flat = _flatten(right).to(torch.float64)
    left_centered = left_flat - left_flat.mean(dim=1, keepdim=True)
    right_centered = right_flat - right_flat.mean(dim=1, keepdim=True)
    numerator = (left_centered * right_centered).sum(dim=1)
    denominator = left_centered.norm(dim=1) * right_centered.norm(dim=1)
    return numerator / denominator.clamp_min(eps)


def _mean_std(values: list[float]) -> dict[str, Any]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "values": values,
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=True)) if tensor.numel() > 1 else 0.0,
    }


def _load_models(
    checkpoints: list[str], device: torch.device
) -> tuple[nn.Module, list[nn.Module], list[dict[str, Any]]]:
    states = [torch.load(path, map_location="cpu") for path in checkpoints]
    configs = [dict(state["config"]) for state in states]
    first_args = argparse.Namespace(**configs[0])
    energy = build_energy(first_args)
    energy.load_state_dict(states[0]["energy"], strict=True)
    energy.to(device).eval()
    configure_energy_training_mode(energy, frozen=True)

    mobilities: list[nn.Module] = []
    selections: list[dict[str, Any]] = []
    for path, state, config in zip(checkpoints, states, configs):
        mobility = build_mobility(
            config["mobility"],
            (3, 32, 32),
            hidden_dim=int(config["mobility_hidden"]),
            depth=int(config["mobility_depth"]),
            log_bound=float(config["mobility_log_bound"]),
            rank=int(config["mobility_rank"]),
        )
        mobility_key = "mobility_ema" if state.get("mobility_ema") else "mobility"
        mobility.load_state_dict(state[mobility_key], strict=True)
        mobility.to(device).eval()
        for parameter in mobility.parameters():
            parameter.requires_grad_(False)
        mobilities.append(mobility)
        selections.append(
            {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(path),
                "step": int(state["step"]),
                "mobility_state": mobility_key,
                "training_seed": int(config["seed"]),
                "log_bound": float(config["mobility_log_bound"]),
                "geometry_weight": float(config["geometry_weight"]),
                "metric_weighted": bool(config["metric_weighted"]),
            }
        )
    return energy, mobilities, selections


def _new_phase_accumulator(fit_count: int, phase_bins: int) -> list[dict[str, Any]]:
    return [
        {
            "sample_count": 0,
            "target_sq": 0.0,
            "identity_residual_sq": 0.0,
            "rem_residual_sq": [0.0] * fit_count,
            "abs_log_sum": [0.0] * fit_count,
            "coordinate_count": [0] * fit_count,
        }
        for _ in range(phase_bins)
    ]


def main() -> None:
    args = parse_args()
    if (
        args.batches_per_seed < 1
        or args.batch_size < 1
        or args.num_workers < 0
        or args.phase_bins < 1
        or args.coordinates_per_image < 1
    ):
        raise ValueError("batch, phase-bin, and coordinate counts must be positive")
    if len(args.checkpoints) < 2:
        raise ValueError("at least two independent mobility checkpoints are required")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one held-out draw seed is required")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    configure_training_attention_backend(device)
    energy, mobilities, checkpoint_records = _load_models(args.checkpoints, device)
    fit_count = len(mobilities)

    dataset_args = argparse.Namespace(**torch.load(args.checkpoints[0], map_location="cpu")["config"])
    dataset_args.dataset = args.dataset
    dataset_args.data_root = args.data_root
    dataset = build_dataset(dataset_args, distributed=False, rank=0)
    try:
        from torchcfm.conditional_flow_matching import (
            ExactOptimalTransportConditionalFlowMatcher,
        )
    except ImportError as error:
        raise RuntimeError("image mobility diagnostic requires torchcfm") from error
    matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    experiment_id = args.experiment_id or (
        f"{args.dataset}_image_mobility_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    config = {
        **vars(args),
        "seeds": seeds,
        "checkpoint_records": checkpoint_records,
        "energy_state": "raw frozen state from first REM checkpoint",
        "mobility_state": "EMA state used by paper evaluation",
        "residual_definition": "sum ||prediction-u||^2 / sum ||u||^2",
        "fit_consistency": "per-image Pearson correlation of flattened log-diagonal maps",
    }
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        config,
        repository_root=Path(__file__).resolve().parents[2],
    )

    target_sq = 0.0
    identity_residual_sq = 0.0
    rem_residual_sq = [0.0] * fit_count
    rem_metric_residual = [0.0] * fit_count
    improved_samples = [0] * fit_count
    sample_count = 0
    coordinate_count = 0
    sampled_eigenvalues: list[list[Tensor]] = [[] for _ in range(fit_count)]
    per_sample_minimum: list[list[Tensor]] = [[] for _ in range(fit_count)]
    per_sample_maximum: list[list[Tensor]] = [[] for _ in range(fit_count)]
    per_sample_condition: list[list[Tensor]] = [[] for _ in range(fit_count)]
    per_sample_log_distance: list[list[Tensor]] = [[] for _ in range(fit_count)]
    saturation_count = [0] * fit_count
    pair_indices = list(combinations(range(fit_count), 2))
    pair_correlations: dict[tuple[int, int], list[Tensor]] = {
        pair: [] for pair in pair_indices
    }
    phases = _new_phase_accumulator(fit_count, args.phase_bins)

    coordinate_generator = torch.Generator().manual_seed(91_733)
    coordinate_indices = torch.randperm(3 * 32 * 32, generator=coordinate_generator)[
        : min(args.coordinates_per_image, 3 * 32 * 32)
    ]

    for seed in seeds:
        seed_everything(seed)
        loader_generator = torch.Generator().manual_seed(80_000_000 + seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            generator=loader_generator,
        )
        for batch_index, (real, _) in enumerate(loader):
            if batch_index >= args.batches_per_seed:
                break
            real = real.to(device, non_blocking=True)
            noise = torch.randn_like(real)
            t, x_t, target_velocity = matcher.sample_location_and_conditional_flow(noise, real)
            gradient = energy_gradient(energy, x_t, t).detach()
            target_velocity = target_velocity.detach()
            identity_residual = -gradient - target_velocity
            target_flat = _flatten(target_velocity)
            identity_flat = _flatten(identity_residual)
            target_batch_sq = target_flat.square().sum(dim=1)
            identity_batch_sq = identity_flat.square().sum(dim=1)
            target_sq += float(target_batch_sq.sum().cpu())
            identity_residual_sq += float(identity_batch_sq.sum().cpu())
            sample_count += int(real.shape[0])
            coordinate_count += int(real.numel())

            batch_logs: list[Tensor] = []
            batch_rem_sq: list[Tensor] = []
            with torch.no_grad():
                for fit_index, mobility in enumerate(mobilities):
                    prediction = -mobility.apply(x_t, gradient)
                    residual = prediction - target_velocity
                    residual_flat = _flatten(residual)
                    residual_sample_sq = residual_flat.square().sum(dim=1)
                    rem_residual_sq[fit_index] += float(residual_sample_sq.sum().cpu())
                    inverse_residual = mobility.inverse_apply(x_t, residual)
                    rem_metric_residual[fit_index] += float(
                        (residual_flat * _flatten(inverse_residual)).sum().cpu()
                    )
                    improved_samples[fit_index] += int(
                        (residual_sample_sq < identity_batch_sq).sum().cpu()
                    )
                    logs = mobility.log_diagonal(x_t)
                    eigenvalues = logs.exp()
                    logs_flat = _flatten(logs)
                    eigen_flat = _flatten(eigenvalues)
                    batch_logs.append(logs.detach())
                    batch_rem_sq.append(residual_sample_sq.detach())
                    sampled_eigenvalues[fit_index].append(
                        eigen_flat[:, coordinate_indices.to(device)].detach().cpu()
                    )
                    minimum = eigen_flat.min(dim=1).values
                    maximum = eigen_flat.max(dim=1).values
                    per_sample_minimum[fit_index].append(minimum.cpu())
                    per_sample_maximum[fit_index].append(maximum.cpu())
                    per_sample_condition[fit_index].append((maximum / minimum).cpu())
                    per_sample_log_distance[fit_index].append(
                        logs_flat.square().mean(dim=1).sqrt().cpu()
                    )
                    bound = checkpoint_records[fit_index]["log_bound"]
                    saturation_count[fit_index] += int(
                        (logs_flat.abs() >= 0.95 * bound).sum().cpu()
                    )

            for pair in pair_indices:
                pair_correlations[pair].append(
                    _sample_correlation(batch_logs[pair[0]], batch_logs[pair[1]]).cpu()
                )

            phase_index = torch.clamp(
                torch.floor(t.detach().reshape(-1) * args.phase_bins).long(),
                min=0,
                max=args.phase_bins - 1,
            )
            for bin_index in range(args.phase_bins):
                mask = phase_index == bin_index
                if not bool(mask.any()):
                    continue
                record = phases[bin_index]
                record["sample_count"] += int(mask.sum().cpu())
                record["target_sq"] += float(target_batch_sq[mask].sum().cpu())
                record["identity_residual_sq"] += float(identity_batch_sq[mask].sum().cpu())
                for fit_index in range(fit_count):
                    record["rem_residual_sq"][fit_index] += float(
                        batch_rem_sq[fit_index][mask].sum().cpu()
                    )
                    masked_logs = _flatten(batch_logs[fit_index][mask])
                    record["abs_log_sum"][fit_index] += float(masked_logs.abs().sum().cpu())
                    record["coordinate_count"][fit_index] += int(masked_logs.numel())
        else:
            raise RuntimeError(
                f"dataset exhausted before {args.batches_per_seed} batches for seed {seed}"
            )

    if not math.isfinite(target_sq) or target_sq <= 0:
        raise RuntimeError("invalid target-velocity norm")
    identity_relative = identity_residual_sq / target_sq
    fit_summaries = []
    relative_values = []
    for fit_index in range(fit_count):
        relative = rem_residual_sq[fit_index] / target_sq
        relative_values.append(relative)
        eigen = torch.cat(sampled_eigenvalues[fit_index]).flatten()
        minimum = torch.cat(per_sample_minimum[fit_index])
        maximum = torch.cat(per_sample_maximum[fit_index])
        condition = torch.cat(per_sample_condition[fit_index])
        log_distance = torch.cat(per_sample_log_distance[fit_index])
        fit_summaries.append(
            {
                **checkpoint_records[fit_index],
                "heldout_relative_euclidean_velocity_mse": relative,
                "relative_reduction_vs_identity": 1.0 - relative / identity_relative,
                "sample_improvement_fraction": improved_samples[fit_index] / sample_count,
                "heldout_metric_weighted_residual_over_euclidean_target_norm": (
                    rem_metric_residual[fit_index] / target_sq
                ),
                "eigenvalue_quantiles_coordinate_sample": _quantiles(eigen),
                "per_image_minimum_eigenvalue_quantiles": _quantiles(minimum),
                "per_image_maximum_eigenvalue_quantiles": _quantiles(maximum),
                "per_image_condition_number_quantiles": _quantiles(condition),
                "per_image_log_distance_quantiles": _quantiles(log_distance),
                "fraction_coordinates_near_log_bound": (
                    saturation_count[fit_index] / coordinate_count
                ),
            }
        )

    phase_summaries = []
    for bin_index, record in enumerate(phases):
        bin_target = float(record["target_sq"])
        phase_summaries.append(
            {
                "t_interval": [bin_index / args.phase_bins, (bin_index + 1) / args.phase_bins],
                "sample_count": int(record["sample_count"]),
                "identity_relative_euclidean_velocity_mse": (
                    float(record["identity_residual_sq"]) / bin_target
                ),
                "rem_relative_euclidean_velocity_mse": [
                    float(value) / bin_target for value in record["rem_residual_sq"]
                ],
                "mean_abs_log_eigenvalue": [
                    float(value) / int(count)
                    for value, count in zip(
                        record["abs_log_sum"], record["coordinate_count"]
                    )
                ],
            }
        )

    consistency = []
    for left, right in pair_indices:
        values = torch.cat(pair_correlations[(left, right)])
        consistency.append(
            {
                "fit_indices": [left, right],
                "training_seeds": [
                    checkpoint_records[left]["training_seed"],
                    checkpoint_records[right]["training_seed"],
                ],
                "per_image_log_map_correlation_mean": float(values.mean()),
                "per_image_log_map_correlation_std": float(values.std(unbiased=True)),
                "per_image_log_map_correlation_quantiles": _quantiles(values),
            }
        )

    result = {
        "status": "complete",
        "failure_reason": None,
        "dataset": args.dataset,
        "sample_count": sample_count,
        "coordinate_count": coordinate_count,
        "heldout_draw_count": len(seeds),
        "mobility_fit_count": fit_count,
        "identity_relative_euclidean_velocity_mse": identity_relative,
        "rem_relative_euclidean_velocity_mse_nested": _mean_std(relative_values),
        "fits": fit_summaries,
        "phase_bins": phase_summaries,
        "cross_fit_consistency": consistency,
        "scope": (
            "Held-out draws come from the training augmentation and minibatch-OT "
            "distribution; this diagnoses restricted diagonal fit and does not prove "
            "out-of-distribution or global representability."
        ),
    }
    (artifacts.path / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts.log(
        0,
        {
            "sample_count": sample_count,
            "mobility_fit_count": fit_count,
            "identity_relative_euclidean_velocity_mse": identity_relative,
            "rem_relative_euclidean_velocity_mse_mean": result[
                "rem_relative_euclidean_velocity_mse_nested"
            ]["mean"],
        },
        phase="held-out-image-geometry",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
