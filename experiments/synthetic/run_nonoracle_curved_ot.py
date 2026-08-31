"""Non-oracle curved multimodal OT experiment with corrected sampling.

The target is an exactly normalizable curved mixture embedded in a rotated
moderate-dimensional space.  The construction is used only for reference
sampling and density evaluation: mobility training receives standard
minibatch-OT straight chords and the frozen energy gradient, never an oracle
metric or analytic target field.  This makes the experiment an end-to-end
test of Algorithm 1 rather than another oracle-recovery diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from experiments.synthetic.run_highdim_structured_ot import (
    evaluate_variant,
    train_variant,
)
from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import IdentityMobility
from rem.metrics import effective_sample_size, rbf_mmd, sliced_wasserstein, split_rhat
from rem.sampling import constant_temperature, langevin_chain, profile_langevin_chain


FIELD_VARIANTS = ("identity", "diagonal-matched", "structured")
SAMPLING_VARIANTS = (
    "identity",
    "diagonal-matched",
    "structured",
    "structured-no-div",
)


class CurvedMixtureTarget(nn.Module):
    """Curved four-mode density with an exact frozen energy and sampler."""

    def __init__(
        self,
        dimension: int = 16,
        rank: int = 4,
        *,
        curvature: float = 0.2,
        component_variance: float = 0.25,
        tail_scale: float = 0.7,
        rotation_seed: int = 1729,
    ) -> None:
        super().__init__()
        if dimension < 3 or not 1 <= rank < dimension:
            raise ValueError("require dimension >= 3 and 1 <= rank < dimension")
        if curvature <= 0 or component_variance <= 0 or tail_scale <= 0:
            raise ValueError("target scales must be positive")
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.curvature = float(curvature)
        self.component_variance = float(component_variance)
        self.tail_scale = float(tail_scale)
        means = torch.tensor(
            [[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]],
            dtype=torch.float64,
        )
        generator = torch.Generator().manual_seed(rotation_seed)
        matrix = torch.randn(
            dimension, dimension, generator=generator, dtype=torch.float64
        )
        rotation, _ = torch.linalg.qr(matrix)
        if torch.linalg.det(rotation) < 0:
            rotation[:, 0] = -rotation[:, 0]
        self.register_buffer("means", means)
        self.register_buffer("rotation", rotation)

    def inverse_latent(self, x: Tensor) -> Tensor:
        flat = x.reshape(x.shape[0], -1)
        rotation = self.rotation.to(device=x.device, dtype=x.dtype)
        curved = flat @ rotation
        latent = curved.clone()
        latent[:, 1] = curved[:, 1] - self.curvature * curved[:, 0].square()
        return latent

    def forward(self, x: Tensor) -> Tensor:
        latent = self.inverse_latent(x)
        means = self.means.to(device=x.device, dtype=x.dtype)
        squared = (latent[:, None, :2] - means[None, :, :]).square().sum(dim=2)
        mixture = -torch.logsumexp(
            -squared / (2.0 * self.component_variance), dim=1
        ) + math.log(means.shape[0])
        tail = 0.5 * latent[:, 2:].square().sum(dim=1) / self.tail_scale**2
        return mixture + tail

    def gradient(self, x: Tensor) -> Tensor:
        with torch.enable_grad():
            points = x.detach().requires_grad_(True)
            value = self(points)
            return torch.autograd.grad(value.sum(), points, create_graph=False)[0]

    def sample(self, count: int, *, generator: torch.Generator) -> Tensor:
        device = self.means.device
        dtype = self.means.dtype
        indices = torch.randint(
            self.means.shape[0], (count,), generator=generator, device=device
        )
        first = self.means[indices] + self.component_variance**0.5 * torch.randn(
            count, 2, generator=generator, device=device, dtype=dtype
        )
        tail = self.tail_scale * torch.randn(
            count,
            self.dimension - 2,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        latent = torch.cat([first, tail], dim=1)
        curved = latent.clone()
        curved[:, 1] = latent[:, 1] + self.curvature * latent[:, 0].square()
        return curved @ self.rotation.transpose(0, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--coupling", choices=["minibatch", "population"], default="minibatch")
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--heldout-batches", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--geometry-weight", type=float, default=1e-4)
    parser.add_argument("--log-bound", type=float, default=1.5)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--curvature", type=float, default=0.2)
    parser.add_argument("--component-variance", type=float, default=0.25)
    parser.add_argument("--tail-scale", type=float, default=0.7)
    parser.add_argument("--step-sizes", default="0.002,0.005")
    parser.add_argument("--burn-in", type=int, default=1500)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--divergence-probes", type=int, default=8)
    parser.add_argument("--divergence-method", choices=["exact", "hutchinson"], default="exact")
    parser.add_argument("--output", default="outputs/nonoracle_curved_ot")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_chords(
    target: CurvedMixtureTarget,
    *,
    count: int,
    seed: int,
    coupling: str,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(count, target.dimension, generator=generator, dtype=torch.float64)
    data = target.sample(count, generator=generator)
    if coupling == "population":
        time_value = torch.rand(count, 1, generator=generator, dtype=torch.float64)
        return (1.0 - time_value) * noise + time_value * data, data - noise
    from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

    seed_everything(seed)
    matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    _, points, velocity = matcher.sample_location_and_conditional_flow(
        noise.float(), data.float()
    )
    return points.double(), velocity.double()


def matched_batches(
    target: CurvedMixtureTarget,
    *,
    batch_size: int,
    batches: int,
    seed: int,
    coupling: str,
) -> list[tuple[Tensor, Tensor]]:
    return [
        make_chords(
            target,
            count=batch_size,
            seed=(seed * 1_000_003 + index) % (2**32 - 1),
            coupling=coupling,
        )
        for index in range(batches)
    ]


def sampling_metrics(
    draws: Tensor,
    reference: Tensor,
    target: CurvedMixtureTarget,
    profile,
) -> dict[str, float]:
    flat = draws.reshape(-1, draws.shape[-1]).double().cpu()
    reference = reference.double().cpu()
    count = min(flat.shape[0], reference.shape[0], 2048)
    flat = flat[:count]
    reference = reference[:count]
    covariance_error = torch.linalg.norm(
        torch.cov(flat.T) - torch.cov(reference.T)
    ) / torch.linalg.norm(torch.cov(reference.T))
    ess = effective_sample_size(draws)
    rhat = split_rhat(draws)
    total_draws = draws.shape[0] * draws.shape[1]
    iact = total_draws / ess.clamp_min(1e-12)

    latent_draws = target.inverse_latent(flat)
    distances = (
        latent_draws[:, None, :2] - target.means.cpu()[None, :, :]
    ).square().sum(dim=2)
    labels = distances.argmin(dim=1)
    occupancy = torch.stack(
        [(labels == index).float().mean() for index in range(target.means.shape[0])]
    )
    chain_latent = target.inverse_latent(
        draws.reshape(-1, draws.shape[-1]).double().cpu()
    )
    chain_labels = (
        chain_latent[:, None, :2] - target.means.cpu()[None, :, :]
    ).square().sum(dim=2).argmin(dim=1).reshape(draws.shape[0], draws.shape[1])
    transitions = (chain_labels[:, 1:] != chain_labels[:, :-1]).float().sum(dim=1)
    mmd_count = min(count, 1024)
    return {
        "relative_covariance_error": float(covariance_error),
        "mmd": float(rbf_mmd(flat[:mmd_count], reference[:mmd_count])),
        "sliced_wasserstein": float(
            sliced_wasserstein(flat, reference, n_projections=128)
        ),
        "mode_max_absolute_occupancy_error": float((occupancy - 0.25).abs().max()),
        "mean_mode_transitions": float(transitions.mean()),
        "mean_ess": float(ess.mean()),
        "min_ess": float(ess.min()),
        "mean_integrated_autocorrelation_time": float(iact.mean()),
        "max_rhat": float(rhat.max()),
        "mean_ess_per_energy_gradient": float(
            ess.mean() / max(profile.energy_grad_evaluations, 1)
        ),
        "mean_ess_per_second": float(ess.mean() / max(profile.wall_seconds, 1e-12)),
        **{key: float(value) for key, value in profile.to_dict().items()},
    }


def sample_variant(
    args: argparse.Namespace,
    target: CurvedMixtureTarget,
    mobility: nn.Module,
    variant: str,
    seed: int,
    dt: float,
    device: torch.device,
) -> dict:
    generator = torch.Generator(device=device.type).manual_seed(
        3_000_000 + 10_000 * seed + round(1_000_000 * dt)
    )
    initial = 2.0 * torch.randn(
        args.chains, target.dimension, device=device, generator=generator
    )
    corrected = variant != "structured-no-div"
    burned, _ = langevin_chain(
        target,
        mobility,
        initial,
        steps=args.burn_in,
        dt=dt,
        temperature=constant_temperature(1.0),
        include_divergence_correction=corrected,
        divergence_samples=args.divergence_probes,
        divergence_method=args.divergence_method,
        generator=generator,
    )
    _, trajectory, profile = profile_langevin_chain(
        target,
        mobility,
        burned,
        steps=args.draws * args.thin,
        dt=dt,
        temperature=constant_temperature(1.0),
        include_divergence_correction=corrected,
        divergence_samples=args.divergence_probes,
        divergence_method=args.divergence_method,
        save_every=args.thin,
        generator=generator,
    )
    draws = torch.stack(trajectory[1:], dim=1)
    reference = target.sample(
        min(draws.shape[0] * draws.shape[1], 2048),
        generator=torch.Generator(device=device.type).manual_seed(
            4_000_000 + 10_000 * seed + round(1_000_000 * dt)
        ),
    )
    result = {
        "variant": variant,
        "seed": seed,
        "dt": dt,
        "corrected": corrected,
        "divergence_method": args.divergence_method if corrected else "omitted",
    }
    target.cpu()
    result.update(sampling_metrics(draws, reference, target, profile))
    return result


def aggregate(records: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        group = (record["variant"], record.get("dt"))
        grouped.setdefault(group, []).append(record)
    result = []
    for (variant, dt), values in sorted(grouped.items(), key=lambda item: str(item[0])):
        item: dict[str, object] = {
            "variant": variant,
            "dt": dt,
            "seeds": [value["seed"] for value in values],
        }
        for key in keys:
            samples = [float(value[key]) for value in values]
            item[key] = {
                "mean": statistics.mean(samples),
                "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            }
        result.append(item)
    return result


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    step_sizes = [float(value) for value in args.step_sizes.split(",") if value.strip()]
    if args.rank >= args.dimension or min(step_sizes) <= 0:
        raise ValueError("invalid dimension, rank, or step size")
    device = torch.device(args.device)
    experiment_id = args.experiment_id or (
        f"nonoracle_curved_ot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {**vars(args), "seeds": seeds, "step_sizes": step_sizes},
        repository_root=Path(__file__).resolve().parents[2],
    )
    target = CurvedMixtureTarget(
        args.dimension,
        args.rank,
        curvature=args.curvature,
        component_variance=args.component_variance,
        tail_scale=args.tail_scale,
    )
    field_records: list[dict] = []
    sampling_records: list[dict] = []
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
        trained: dict[str, nn.Module] = {"identity": IdentityMobility().to(device)}
        for variant in ("diagonal-matched", "structured"):
            trained[variant], wall = train_variant(
                variant, target, training_batches, args, device
            )
            field = evaluate_variant(
                variant, trained[variant], target, heldout_batches, device
            )
            field.update(seed=seed, training_wall_seconds=wall)
            field_records.append(field)
            artifacts.log(len(field_records) - 1, field, split="heldout", mode=variant)
        identity_field = evaluate_variant(
            "identity", trained["identity"], target, heldout_batches, device
        )
        identity_field.update(seed=seed, training_wall_seconds=0.0)
        field_records.append(identity_field)
        artifacts.log(
            len(field_records) - 1, identity_field, split="heldout", mode="identity"
        )
        del training_batches, heldout_batches

        for variant in SAMPLING_VARIANTS:
            mobility = trained[
                "structured" if variant == "structured-no-div" else variant
            ]
            for dt in step_sizes:
                record = sample_variant(
                    args, target.to(device), mobility, variant, seed, dt, device
                )
                sampling_records.append(record)
                artifacts.log(
                    len(sampling_records) - 1,
                    record,
                    split="sampling",
                    mode=variant,
                )
                print(json.dumps(record, sort_keys=True), flush=True)

    field_keys = (
        "relative_velocity_mse",
        "mean_velocity_cosine",
        "descent_violation_rate",
        "offdiagonal_frobenius_fraction",
        "max_abs_logdet",
        "training_wall_seconds",
    )
    sampling_keys = (
        "relative_covariance_error",
        "mmd",
        "sliced_wasserstein",
        "mode_max_absolute_occupancy_error",
        "mean_mode_transitions",
        "mean_ess",
        "mean_integrated_autocorrelation_time",
        "max_rhat",
        "mean_ess_per_energy_gradient",
        "mean_ess_per_second",
        "wall_seconds",
    )
    payload = {
        "status": "complete",
        "field_records": field_records,
        "field_aggregate": aggregate(field_records, field_keys),
        "sampling_records": sampling_records,
        "sampling_aggregate": aggregate(sampling_records, sampling_keys),
    }
    artifacts.write_json("nonoracle_curved_summary.json", payload)
    artifacts.summarize(
        {
            "field_records": [float(len(field_records))],
            "sampling_records": [float(len(sampling_records))],
            "seeds": [float(len(seeds))],
        },
        status="complete",
    )
    print(json.dumps(payload["sampling_aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
