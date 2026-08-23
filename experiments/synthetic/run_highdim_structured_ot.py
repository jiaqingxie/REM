"""High-dimensional OT-chord test of structured REM and energy continuation.

The target is a correlated Gaussian whose dense anisotropy lives in a small
cosine subspace.  Its scalar energy is exact and fixed before transport
supervision is generated.  Standard independent minibatches from a spherical
noise law and the target law are coupled by minibatch optimal transport, then
interpolated by observed-space straight chords exactly as in image REM.

The experiment separates three interventions:

* diagonal REM changes coordinate-wise mobility while preserving the energy;
* structured REM adds an O(d r) off-diagonal congruence while preserving it;
* continued energy training changes the scalar potential itself.

The last row may fit the supervised velocity well, but its equilibrium is a
different Gaussian.  This directly tests the paper's ``match density once,
learn geometry later'' argument rather than comparing REM only with a frozen
identity sampler.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import (
    IdentityMobility,
    LowRankCongruenceMobility,
    minimal_distortion_regularizer,
    orthonormal_cosine_basis,
)
from rem.networks import build_mobility
from rem.training import trainable_parameter_count


VARIANTS = (
    "identity",
    "diagonal",
    "diagonal-matched",
    "structured",
    "continue-energy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="16,32,64,128")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--coupling", choices=["minibatch", "population"], default="minibatch")
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--heldout-batches", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--geometry-weight", type=float, default=1e-4)
    parser.add_argument("--orthogonal-scale", type=float, default=0.72)
    parser.add_argument("--structured-scale", type=float, default=0.18)
    parser.add_argument("--log-bound", type=float, default=1.5)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--output", default="outputs/highdim_structured_ot")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@dataclass(frozen=True)
class CorrelatedGaussian:
    dimension: int
    rank: int
    orthogonal_scale: float
    structured_scale: float
    basis: Tensor
    transport: Tensor
    covariance: Tensor
    precision: Tensor

    @classmethod
    def build(
        cls,
        dimension: int,
        rank: int,
        orthogonal_scale: float,
        structured_scale: float,
    ) -> "CorrelatedGaussian":
        if not 0 < structured_scale < orthogonal_scale < 1:
            raise ValueError("require 0 < structured_scale < orthogonal_scale < 1")
        basis = orthonormal_cosine_basis(dimension, rank, dtype=torch.float64)
        identity = torch.eye(dimension, dtype=torch.float64)
        projector = basis @ basis.transpose(0, 1)
        transport = (
            orthogonal_scale * identity
            + (structured_scale - orthogonal_scale) * projector
        )
        covariance = transport @ transport.transpose(0, 1)
        precision = torch.linalg.inv(covariance)
        return cls(
            dimension,
            rank,
            orthogonal_scale,
            structured_scale,
            basis,
            transport,
            covariance,
            precision,
        )

    def target_sample(self, count: int, *, generator: torch.Generator) -> Tensor:
        latent = torch.randn(
            count, self.dimension, generator=generator, dtype=torch.float64
        )
        return latent @ self.transport.transpose(0, 1)

    def gradient(self, x: Tensor) -> Tensor:
        precision = self.precision.to(device=x.device, dtype=x.dtype)
        return x @ precision.transpose(0, 1)


class TrainableQuadraticEnergy(nn.Module):
    """Full SPD quadratic initialized at the exact target density."""

    def __init__(self, precision: Tensor) -> None:
        super().__init__()
        cholesky = torch.linalg.cholesky(precision)
        self.strict_lower = nn.Parameter(torch.tril(cholesky, diagonal=-1))
        self.log_diagonal = nn.Parameter(torch.diagonal(cholesky).log())

    def factor(self) -> Tensor:
        return torch.tril(self.strict_lower, diagonal=-1) + torch.diag(
            self.log_diagonal.exp()
        )

    def precision(self) -> Tensor:
        factor = self.factor()
        return factor @ factor.transpose(0, 1)

    def velocity(self, x: Tensor) -> Tensor:
        return -(x @ self.precision().transpose(0, 1))


def make_chords(
    target: CorrelatedGaussian,
    *,
    count: int,
    seed: int,
    coupling: str,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        count, target.dimension, generator=generator, dtype=torch.float64
    )
    if coupling == "population":
        data = noise @ target.transport.transpose(0, 1)
    else:
        data = target.target_sample(count, generator=generator)
        from torchcfm.conditional_flow_matching import (
            ExactOptimalTransportConditionalFlowMatcher,
        )

        matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        # ``sample_location`` uses both NumPy/POT and torch randomness.  The
        # outer caller fixes all process RNGs before constructing each matched
        # evidence stream.
        _, points, velocity = matcher.sample_location_and_conditional_flow(
            noise.float(), data.float()
        )
        return points.double(), velocity.double()
    time_value = torch.rand(count, 1, generator=generator, dtype=torch.float64)
    return (1.0 - time_value) * noise + time_value * data, data - noise


def matched_batches(
    target: CorrelatedGaussian,
    *,
    batch_size: int,
    batches: int,
    seed: int,
    coupling: str,
) -> list[tuple[Tensor, Tensor]]:
    result = []
    for batch_index in range(batches):
        batch_seed = (seed * 1_000_003 + batch_index) % (2**32 - 1)
        seed_everything(batch_seed)
        result.append(
            make_chords(
                target,
                count=batch_size,
                seed=batch_seed,
                coupling=coupling,
            )
        )
    return result


def relative_velocity_mse(prediction: Tensor, target: Tensor) -> Tensor:
    return prediction.sub(target).square().mean() / target.square().mean().clamp_min(1e-12)


def mlp_parameter_count(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
) -> int:
    count = input_dim * hidden_dim + hidden_dim
    count += (depth - 1) * (hidden_dim * hidden_dim + hidden_dim)
    return count + hidden_dim * output_dim + output_dim


def capacity_matched_diagonal_width(
    dimension: int,
    rank: int,
    hidden_dim: int,
    depth: int,
) -> int:
    target = mlp_parameter_count(
        dimension, dimension * (rank + 1), hidden_dim, depth
    )
    candidates = range(4, 4 * hidden_dim + 1)
    return min(
        candidates,
        key=lambda value: abs(
            mlp_parameter_count(dimension, dimension, value, depth) - target
        ),
    )


def train_variant(
    variant: str,
    target: CorrelatedGaussian,
    batches: list[tuple[Tensor, Tensor]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, float]:
    if variant == "identity":
        return IdentityMobility().to(device), 0.0
    if variant == "continue-energy":
        model: nn.Module = TrainableQuadraticEnergy(target.precision.float()).to(device)
    else:
        mobility_kind = "diagonal" if variant == "diagonal-matched" else variant
        hidden_dim = (
            capacity_matched_diagonal_width(
                target.dimension,
                args.rank,
                args.hidden_dim,
                args.depth,
            )
            if variant == "diagonal-matched"
            else args.hidden_dim
        )
        model = build_mobility(
            mobility_kind,
            (target.dimension,),
            hidden_dim=hidden_dim,
            depth=args.depth,
            rank=args.rank,
            log_bound=args.log_bound,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    model.train()
    for step, (points_cpu, velocity_cpu) in enumerate(batches):
        points = points_cpu.to(device=device, dtype=torch.float32)
        velocity = velocity_cpu.to(device=device, dtype=torch.float32)
        if variant == "continue-energy":
            prediction = model.velocity(points)
            geometry = prediction.new_zeros(())
        else:
            gradient = target.gradient(points)
            prediction = -model.apply(points, gradient)
            geometry = minimal_distortion_regularizer(model, points)
        loss = relative_velocity_mse(prediction, velocity) + args.geometry_weight * geometry
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if args.log_every > 0 and (step + 1) % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "dimension": target.dimension,
                        "variant": variant,
                        "step": step + 1,
                        "loss": float(loss.detach().cpu()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model.eval(), time.perf_counter() - start


@torch.no_grad()
def evaluate_variant(
    variant: str,
    model: nn.Module,
    target: CorrelatedGaussian,
    batches: list[tuple[Tensor, Tensor]],
    device: torch.device,
) -> dict[str, float | int | bool]:
    predictions = []
    velocities = []
    points_all = []
    for points_cpu, velocity_cpu in batches:
        points = points_cpu.to(device=device, dtype=torch.float32)
        velocity = velocity_cpu.to(device=device, dtype=torch.float32)
        if variant == "continue-energy":
            prediction = model.velocity(points)
        elif variant == "identity":
            prediction = -target.gradient(points)
        else:
            prediction = -model.apply(points, target.gradient(points))
        predictions.append(prediction.cpu())
        velocities.append(velocity.cpu())
        points_all.append(points.cpu())
    prediction = torch.cat(predictions)
    velocity = torch.cat(velocities)
    points = torch.cat(points_all)
    cosine = torch.nn.functional.cosine_similarity(prediction, velocity, dim=1)
    gradient = target.gradient(points)
    descent_violation = (velocity.mul(gradient).sum(dim=1) >= 0).float().mean()

    equilibrium_kl = 0.0
    equilibrium_covariance_error = 0.0
    offdiagonal_fraction = 0.0
    max_abs_logdet = 0.0
    fixed_density = variant != "continue-energy"
    if variant == "continue-energy":
        precision = model.precision().detach().double().cpu()
        covariance = target.covariance
        equilibrium_kl = 0.5 * float(
            torch.trace(precision @ covariance)
            - target.dimension
            - torch.linalg.slogdet(precision).logabsdet
            - torch.linalg.slogdet(covariance).logabsdet
        )
        learned_covariance = torch.linalg.inv(precision)
        equilibrium_covariance_error = float(
            torch.linalg.norm(learned_covariance - covariance)
            / torch.linalg.norm(covariance)
        )
    elif isinstance(model, LowRankCongruenceMobility):
        diagnostic_points = points[: min(64, points.shape[0])].to(device)
        dense = model.dense_matrix(diagnostic_points).double().cpu()
        diagonal = torch.diag_embed(torch.diagonal(dense, dim1=1, dim2=2))
        offdiagonal_fraction = float(
            torch.linalg.norm(dense - diagonal, dim=(1, 2)).mean()
            / torch.linalg.norm(dense, dim=(1, 2)).mean().clamp_min(1e-12)
        )
        max_abs_logdet = float(model.logdet(diagnostic_points).abs().max().cpu())
    elif hasattr(model, "logdet"):
        diagnostic_points = points[: min(64, points.shape[0])].to(device)
        max_abs_logdet = float(model.logdet(diagnostic_points).abs().max().cpu())

    return {
        "dimension": target.dimension,
        "rank": target.rank,
        "variant": variant,
        "fixed_density": fixed_density,
        "trainable_parameters": trainable_parameter_count(model),
        "relative_velocity_mse": float(relative_velocity_mse(prediction, velocity)),
        "mean_velocity_cosine": float(cosine.mean()),
        "descent_violation_rate": float(descent_violation),
        "equilibrium_kl": equilibrium_kl,
        "equilibrium_covariance_error": equilibrium_covariance_error,
        "offdiagonal_frobenius_fraction": offdiagonal_fraction,
        "max_abs_logdet": max_abs_logdet,
    }


def aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["dimension"], record["variant"]), []).append(record)
    metrics = (
        "relative_velocity_mse",
        "mean_velocity_cosine",
        "equilibrium_kl",
        "equilibrium_covariance_error",
        "offdiagonal_frobenius_fraction",
        "training_wall_seconds",
    )
    result = []
    for (dimension, variant), values in sorted(grouped.items()):
        item: dict[str, object] = {
            "dimension": dimension,
            "variant": variant,
            "seeds": [value["seed"] for value in values],
            "fixed_density": values[0]["fixed_density"],
            "trainable_parameters": values[0]["trainable_parameters"],
        }
        for metric in metrics:
            samples = [float(value[metric]) for value in values]
            item[metric] = {
                "mean": statistics.mean(samples),
                "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            }
        result.append(item)
    return result


def main() -> None:
    args = parse_args()
    dimensions = [int(value) for value in args.dimensions.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    if not dimensions or not seeds or not variants:
        raise ValueError("dimensions, seeds, and variants must be nonempty")
    if any(value not in VARIANTS for value in variants):
        raise ValueError(f"variants must be selected from {VARIANTS}")
    if any(args.rank >= dimension for dimension in dimensions):
        raise ValueError("rank must be below every requested dimension")
    if min(args.train_steps, args.batch_size, args.heldout_batches) < 1:
        raise ValueError("training and evaluation counts must be positive")

    device = torch.device(args.device)
    experiment_id = args.experiment_id or (
        f"highdim_structured_ot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        vars(args),
        repository_root=Path(__file__).resolve().parents[2],
    )
    records: list[dict] = []
    for dimension in dimensions:
        target = CorrelatedGaussian.build(
            dimension,
            args.rank,
            args.orthogonal_scale,
            args.structured_scale,
        )
        for seed in seeds:
            seed_everything(seed)
            training_batches = matched_batches(
                target,
                batch_size=args.batch_size,
                batches=args.train_steps,
                seed=10_000 + seed,
                coupling=args.coupling,
            )
            heldout_batches = matched_batches(
                target,
                batch_size=args.batch_size,
                batches=args.heldout_batches,
                seed=90_000 + seed,
                coupling=args.coupling,
            )
            for variant_index, variant in enumerate(variants):
                seed_everything(seed * 101 + variant_index)
                model, wall_seconds = train_variant(
                    variant, target, training_batches, args, device
                )
                record = evaluate_variant(
                    variant, model, target, heldout_batches, device
                )
                record.update(seed=seed, training_wall_seconds=wall_seconds)
                records.append(record)
                artifacts.log(
                    len(records) - 1,
                    record,
                    split="heldout",
                    mode=variant,
                )
                print(json.dumps(record, sort_keys=True), flush=True)
            del training_batches, heldout_batches

    payload = {
        "status": "complete",
        "records": records,
        "aggregate": aggregate(records),
    }
    artifacts.write_json("highdim_summary.json", payload)
    artifacts.summarize(
        {
            "records": [float(len(records))],
            "dimensions": [float(len(dimensions))],
            "seeds": [float(len(seeds))],
        },
        status="complete",
    )
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
