"""Corrected stochastic sampling with OT-trained structured mobilities.

This experiment extends the high-dimensional minibatch-OT diagnostic from
field fitting to actual equilibrium sampling.  For each seed it trains the
same diagonal and structured determinant-one mobilities on observed-space
straight chords, then reuses each fitted mobility in a Langevin chain for the
exact frozen Gaussian energy.  The corrected and no-divergence structured
rows share weights, so their stationary difference isolates the Itô term.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from experiments.synthetic.run_highdim_structured_ot import (
    CorrelatedGaussian,
    matched_batches,
    train_variant,
)
from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import IdentityMobility
from rem.metrics import effective_sample_size, rbf_mmd, sliced_wasserstein, split_rhat
from rem.sampling import constant_temperature, langevin_chain, profile_langevin_chain
from rem.synthetic import QuadraticPotential


VARIANTS = (
    "identity",
    "global-covariance",
    "diagonal",
    "structured",
    "structured-no-div",
)


class ConstantFullMobility(nn.Module):
    """Fixed dense SPD mobility used by the classical global baseline."""

    divergence_probe_cost = 0

    def __init__(self, matrix: Tensor) -> None:
        super().__init__()
        matrix = 0.5 * (matrix + matrix.transpose(0, 1))
        cholesky = torch.linalg.cholesky(matrix)
        self.register_buffer("matrix", matrix)
        self.register_buffer("square_root", cholesky)
        self.register_buffer("inverse", torch.cholesky_inverse(cholesky))
        self.register_buffer("log_determinant", 2.0 * cholesky.diagonal().log().sum())

    def _apply_matrix(self, value: Tensor, matrix: Tensor) -> Tensor:
        flat = value.reshape(value.shape[0], -1)
        if flat.shape[1] != matrix.shape[0]:
            raise ValueError("constant mobility dimension does not match input")
        return (flat @ matrix.transpose(0, 1)).reshape_as(value)

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        del x
        return self._apply_matrix(vector, self.matrix)

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        del x
        return self._apply_matrix(vector, self.inverse)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        del x
        return self._apply_matrix(noise, self.square_root)

    def diagonal(self, x: Tensor) -> Tensor:
        values = self.matrix.diagonal().expand(x.shape[0], -1)
        return values.reshape_as(x)

    def logdet(self, x: Tensor) -> Tensor:
        return self.log_determinant.expand(x.shape[0])

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        del n_samples, create_graph
        return torch.zeros_like(x)


def global_covariance_mobility(
    target: CorrelatedGaussian, device: torch.device
) -> ConstantFullMobility:
    """Return the target covariance rescaled to determinant one."""

    covariance = target.covariance.float()
    _, logabsdet = torch.linalg.slogdet(covariance)
    scale = torch.exp(logabsdet / target.dimension)
    return ConstantFullMobility(covariance / scale).to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="16,64")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--coupling", choices=["minibatch", "population"], default="minibatch")
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--geometry-weight", type=float, default=1e-4)
    parser.add_argument("--orthogonal-scale", type=float, default=0.72)
    parser.add_argument("--structured-scale", type=float, default=0.18)
    parser.add_argument("--log-bound", type=float, default=1.5)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--step-sizes", default="0.002,0.005")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--divergence-probes", type=int, default=8)
    parser.add_argument("--exact-divergence-up-to", type=int, default=16)
    parser.add_argument("--output", default="outputs/highdim_corrected_sampling")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _sampling_metrics(
    draws: Tensor,
    reference: Tensor,
    target: CorrelatedGaussian,
    profile,
) -> dict[str, float]:
    flat = draws.reshape(-1, draws.shape[-1]).double().cpu()
    reference = reference.double().cpu()
    count = min(flat.shape[0], reference.shape[0], 2048)
    flat = flat[:count]
    reference = reference[:count]
    covariance = torch.cov(flat.transpose(0, 1))
    covariance_error = torch.linalg.norm(covariance - target.covariance) / torch.linalg.norm(
        target.covariance
    )
    ess = effective_sample_size(draws)
    rhat = split_rhat(draws)
    total_draws = draws.shape[0] * draws.shape[1]
    iact = total_draws / ess.clamp_min(1e-12)
    mmd_count = min(count, 1024)
    return {
        "mean_error": float((flat.mean(dim=0) - reference.mean(dim=0)).norm()),
        "relative_covariance_error": float(covariance_error),
        "mmd": float(rbf_mmd(flat[:mmd_count], reference[:mmd_count])),
        "sliced_wasserstein": float(
            sliced_wasserstein(flat, reference, n_projections=128)
        ),
        "mean_ess": float(ess.mean()),
        "min_ess": float(ess.min()),
        "mean_integrated_autocorrelation_time": float(iact.mean()),
        "max_integrated_autocorrelation_time": float(iact.max()),
        "max_rhat": float(rhat.max()),
        "mean_ess_per_energy_gradient": float(
            ess.mean() / max(profile.energy_grad_evaluations, 1)
        ),
        "mean_ess_per_second": float(ess.mean() / max(profile.wall_seconds, 1e-12)),
        **{key: float(value) for key, value in profile.to_dict().items()},
    }


def sample_variant(
    args: argparse.Namespace,
    target: CorrelatedGaussian,
    mobility: nn.Module,
    variant: str,
    seed: int,
    dt: float,
    device: torch.device,
) -> dict[str, float | int | str | bool]:
    generator = torch.Generator(device=device.type).manual_seed(
        1_000_000 + 10_000 * seed + round(1_000_000 * dt)
    )
    initial = 2.0 * torch.randn(
        args.chains, target.dimension, device=device, generator=generator
    )
    potential = QuadraticPotential(target.precision.float()).to(device)
    corrected = variant != "structured-no-div"
    divergence_method = (
        "exact" if target.dimension <= args.exact_divergence_up_to else "hutchinson"
    )
    burned, _ = langevin_chain(
        potential,
        mobility,
        initial,
        steps=args.burn_in,
        dt=dt,
        temperature=constant_temperature(args.temperature),
        include_divergence_correction=corrected,
        divergence_samples=args.divergence_probes,
        divergence_method=divergence_method,
        generator=generator,
    )
    _, trajectory, profile = profile_langevin_chain(
        potential,
        mobility,
        burned,
        steps=args.draws * args.thin,
        dt=dt,
        temperature=constant_temperature(args.temperature),
        include_divergence_correction=corrected,
        divergence_samples=args.divergence_probes,
        divergence_method=divergence_method,
        save_every=args.thin,
        generator=generator,
    )
    draws = torch.stack(trajectory[1:], dim=1)
    reference_generator = torch.Generator().manual_seed(
        2_000_000 + 10_000 * seed + round(1_000_000 * dt)
    )
    reference = target.target_sample(
        min(draws.shape[0] * draws.shape[1], 2048), generator=reference_generator
    )
    metrics: dict[str, float | int | str | bool] = {
        "dimension": target.dimension,
        "rank": target.rank,
        "variant": variant,
        "seed": seed,
        "dt": dt,
        "corrected": corrected,
        "divergence_method": divergence_method if corrected else "omitted",
        "divergence_probes": args.divergence_probes if corrected else 0,
    }
    metrics.update(_sampling_metrics(draws, reference, target, profile))
    return metrics


def _aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str, float], list[dict]] = {}
    for record in records:
        grouped.setdefault(
            (record["dimension"], record["variant"], record["dt"]), []
        ).append(record)
    names = (
        "relative_covariance_error",
        "mmd",
        "sliced_wasserstein",
        "mean_ess",
        "min_ess",
        "mean_integrated_autocorrelation_time",
        "max_rhat",
        "mean_ess_per_energy_gradient",
        "mean_ess_per_second",
        "wall_seconds",
    )
    result = []
    for (dimension, variant, dt), values in sorted(grouped.items()):
        item: dict[str, object] = {
            "dimension": dimension,
            "variant": variant,
            "dt": dt,
            "seeds": [value["seed"] for value in values],
            "corrected": values[0]["corrected"],
            "divergence_method": values[0]["divergence_method"],
        }
        for name in names:
            samples = [float(value[name]) for value in values]
            item[name] = {
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
    step_sizes = [float(value) for value in args.step_sizes.split(",") if value.strip()]
    if any(variant not in VARIANTS for variant in variants):
        raise ValueError(f"variants must be selected from {VARIANTS}")
    if any(args.rank >= dimension for dimension in dimensions):
        raise ValueError("rank must be below every requested dimension")
    if min(args.train_steps, args.batch_size, args.burn_in, args.draws, args.thin, args.chains) < 1:
        raise ValueError("training and sampling counts must be positive")
    if min(step_sizes) <= 0 or args.temperature <= 0:
        raise ValueError("step sizes and temperature must be positive")

    device = torch.device(args.device)
    experiment_id = args.experiment_id or (
        f"highdim_corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {
            **vars(args),
            "dimensions": dimensions,
            "seeds": seeds,
            "variants": variants,
            "step_sizes": step_sizes,
        },
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
            trained: dict[str, nn.Module] = {
                "identity": IdentityMobility().to(device),
                "global-covariance": global_covariance_mobility(target, device),
            }
            learned_variants = {"diagonal", "structured", "structured-no-div"}
            training_batches = None
            if learned_variants.intersection(variants):
                training_batches = matched_batches(
                    target,
                    batch_size=args.batch_size,
                    batches=args.train_steps,
                    seed=10_000 + seed,
                    coupling=args.coupling,
                )
            if "diagonal" in variants:
                assert training_batches is not None
                trained["diagonal"], _ = train_variant(
                    "diagonal", target, training_batches, args, device
                )
            if "structured" in variants or "structured-no-div" in variants:
                assert training_batches is not None
                trained["structured"], _ = train_variant(
                    "structured", target, training_batches, args, device
                )
            del training_batches

            for variant in variants:
                mobility = trained[
                    "structured" if variant == "structured-no-div" else variant
                ]
                for dt in step_sizes:
                    record = sample_variant(
                        args, target, mobility, variant, seed, dt, device
                    )
                    records.append(record)
                    artifacts.log(
                        len(records) - 1,
                        record,
                        split="sampling",
                        mode=variant,
                    )
                    print(json.dumps(record, sort_keys=True), flush=True)

    payload = {
        "status": "complete",
        "records": records,
        "aggregate": _aggregate(records),
    }
    artifacts.write_json("highdim_corrected_summary.json", payload)
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
