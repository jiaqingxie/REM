"""Validate REM equilibrium, divergence correction, and discretization bias."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import IdentityMobility
from rem.metrics import (
    effective_sample_size,
    histogram_kl_2d,
    rbf_mmd,
    sliced_wasserstein,
    split_rhat,
)
from rem.sampling import (
    SamplingProfile,
    constant_temperature,
    langevin_chain,
    profile_langevin_chain,
)
from rem.synthetic import (
    AnalyticDiagonalMobility,
    BananaPotential,
    FixedFullMobility,
    GaussianMixturePotential,
    HutchinsonDivergenceWrapper,
    QuadraticPotential,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default=(
            "euclidean,constant,rem-exact,rem-hutch-1,rem-hutch-2,"
            "rem-hutch-4,rem-hutch-8,rem-no-div,mala"
        ),
    )
    parser.add_argument("--targets", default="gaussian,mixture,banana")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--step-sizes", default="0.0025,0.005,0.01")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--chains", type=int, default=32)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _variant(variant: str, strength: float):
    if variant == "euclidean":
        return IdentityMobility(), True, 1
    if variant == "constant":
        matrix = torch.tensor([[1.8, 0.0], [0.0, 1.0 / 1.8]])
        return FixedFullMobility(matrix), True, 1
    base = AnalyticDiagonalMobility(strength)
    if variant == "rem-exact":
        return base, True, 1
    if variant == "rem-no-div":
        return base, False, 1
    if variant.startswith("rem-hutch-"):
        probes = int(variant.rsplit("-", 1)[1])
        return HutchinsonDivergenceWrapper(base), True, probes
    raise ValueError(f"unknown stationarity variant: {variant}")


def _potential_and_gradient(potential, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        x_grad = x.detach().requires_grad_(True)
        value = potential(x_grad)
        gradient = torch.autograd.grad(value.sum(), x_grad)[0]
    return value.detach(), gradient.detach()


def mala_chain(
    potential,
    x_init: torch.Tensor,
    *,
    steps: int,
    dt: float,
    temperature: float,
    save_every: int = 0,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[torch.Tensor], SamplingProfile, float]:
    """Metropolis-adjusted Langevin reference with exact accept/reject."""

    if steps < 0 or dt <= 0 or temperature <= 0:
        raise ValueError("invalid MALA configuration")
    x = x_init.detach()
    trajectory = [x.cpu()] if save_every > 0 else []
    accepts = 0
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
    start = time.perf_counter()
    for step in range(steps):
        current_value, current_gradient = _potential_and_gradient(potential, x)
        forward_mean = x - dt * current_gradient
        noise = torch.randn(
            x.shape,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        proposal = forward_mean + (2.0 * temperature * dt) ** 0.5 * noise
        proposal_value, proposal_gradient = _potential_and_gradient(potential, proposal)
        reverse_mean = proposal - dt * proposal_gradient
        forward_residual = proposal - forward_mean
        reverse_residual = x - reverse_mean
        log_q_ratio = -(
            reverse_residual.square().reshape(x.shape[0], -1).sum(dim=1)
            - forward_residual.square().reshape(x.shape[0], -1).sum(dim=1)
        ) / (4.0 * temperature * dt)
        log_acceptance = -(
            proposal_value - current_value
        ) / temperature + log_q_ratio
        uniform = torch.rand(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        ).clamp_min(torch.finfo(x.dtype).tiny)
        accepted = uniform.log() < log_acceptance.clamp_max(0)
        accepts += int(accepted.sum())
        x = torch.where(accepted[:, None], proposal, x)
        if save_every > 0 and ((step + 1) % save_every == 0 or step + 1 == steps):
            trajectory.append(x.cpu())
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
        peak_memory = torch.cuda.max_memory_allocated(x.device)
    else:
        peak_memory = 0
    wall = time.perf_counter() - start
    profile = SamplingProfile(
        steps=steps,
        batch_size=x.shape[0],
        wall_seconds=wall,
        samples_per_second=x.shape[0] / max(wall, 1e-12),
        energy_grad_evaluations=2 * steps,
        divergence_probes=0,
        peak_memory_bytes=peak_memory,
    )
    acceptance_rate = accepts / max(steps * x.shape[0], 1)
    return x, trajectory, profile, acceptance_rate


def mixture_mixing_metrics(draws: torch.Tensor, means: torch.Tensor, thin: int) -> dict[str, float]:
    distances = (draws[:, :, None, :] - means[None, None, :, :]).square().sum(dim=3)
    labels = distances.argmin(dim=2)
    mode_count = means.shape[0]
    occupancy = torch.stack([(labels == index).float().mean() for index in range(mode_count)])
    transitions = (labels[:, 1:] != labels[:, :-1]).float().sum(dim=1)
    round_trips = []
    completed = 0
    for chain_labels in labels:
        start_mode = int(chain_labels[0])
        left = torch.nonzero(chain_labels != start_mode)
        if left.numel() == 0:
            continue
        left_index = int(left[0])
        returned = torch.nonzero(chain_labels[left_index + 1 :] == start_mode)
        if returned.numel() == 0:
            continue
        return_index = left_index + 1 + int(returned[0])
        round_trips.append(return_index * thin)
        completed += 1
    return {
        "mode_min_occupancy": float(occupancy.min()),
        "mode_max_occupancy": float(occupancy.max()),
        "mode_max_absolute_occupancy_error": float(
            (occupancy - 1.0 / mode_count).abs().max()
        ),
        "mean_mode_transitions": float(transitions.mean()),
        "round_trip_completion_rate": completed / labels.shape[0],
        "mean_round_trip_steps": float(
            torch.tensor(round_trips, dtype=torch.float32).mean()
        )
        if round_trips
        else float(labels.shape[1] * thin),
    }


def _target(name: str, temperature: float, device: torch.device):
    normalized = name.lower().replace("_", "-")
    if normalized in {"gaussian", "correlated-gaussian"}:
        covariance = torch.tensor(
            [[1.0, 0.75], [0.75, 1.0]], device=device
        )
        potential = QuadraticPotential(torch.linalg.inv(covariance)).to(device)

        def sample(count, generator):
            noise = torch.randn(count, 2, generator=generator)
            return temperature**0.5 * noise @ torch.linalg.cholesky(covariance.cpu()).T

        return "gaussian", potential, sample
    if normalized in {"mixture", "gaussian-mixture"}:
        means = torch.tensor(
            [[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]],
            device=device,
        )
        variance = 0.16
        potential = GaussianMixturePotential(
            means,
            component_variance=variance,
            equilibrium_temperature=temperature,
        ).to(device)

        def sample(count, generator):
            indices = torch.randint(means.shape[0], (count,), generator=generator)
            noise = torch.randn(count, 2, generator=generator)
            return means.cpu()[indices] + (variance * temperature) ** 0.5 * noise

        return "mixture", potential, sample
    if normalized == "banana":
        curvature = 0.25
        potential = BananaPotential(curvature).to(device)

        def sample(count, generator):
            latent = temperature**0.5 * torch.randn(count, 2, generator=generator)
            return torch.stack(
                [latent[:, 0], latent[:, 1] + curvature * latent[:, 0].square()],
                dim=1,
            )

        return "banana", potential, sample
    raise ValueError(f"unknown stationarity target: {name}")


def run(
    args: argparse.Namespace,
    artifacts: RunArtifacts,
    variant: str,
    target_name: str,
    seed: int,
    dt: float,
) -> dict[str, float]:
    seed_everything(seed)
    device = torch.device(args.device)
    target_name, potential, sample_reference = _target(
        target_name, args.temperature, device
    )
    generator = torch.Generator(device=device.type).manual_seed(seed)
    initial = 3.0 * torch.randn(
        args.chains, 2, device=device, generator=generator
    )
    acceptance_rate = float("nan")
    if variant == "mala":
        burned, _, _, _ = mala_chain(
            potential,
            initial,
            steps=args.burn_in,
            dt=dt,
            temperature=args.temperature,
            generator=generator,
        )
        final, trajectory, profile, acceptance_rate = mala_chain(
            potential,
            burned,
            steps=args.draws * args.thin,
            dt=dt,
            temperature=args.temperature,
            save_every=args.thin,
            generator=generator,
        )
    else:
        mobility, correction, probes = _variant(variant, args.strength)
        mobility.to(device)
        burned, _ = langevin_chain(
            potential,
            mobility,
            initial,
            steps=args.burn_in,
            dt=dt,
            temperature=constant_temperature(args.temperature),
            include_divergence_correction=correction,
            divergence_samples=probes,
            generator=generator,
        )
        final, trajectory, profile = profile_langevin_chain(
            potential,
            mobility,
            burned,
            steps=args.draws * args.thin,
            dt=dt,
            temperature=constant_temperature(args.temperature),
            include_divergence_correction=correction,
            divergence_samples=probes,
            save_every=args.thin,
            generator=generator,
        )
    del final
    draws = torch.stack(trajectory[1:], dim=1)
    flattened = draws.reshape(-1, 2)
    reference_generator = torch.Generator(device="cpu").manual_seed(1_000_000 + seed)
    reference = sample_reference(
        min(flattened.shape[0], 4096), reference_generator
    )
    comparison_generator = torch.Generator(device="cpu").manual_seed(2_000_000 + seed)
    comparison_indices = torch.randperm(
        flattened.shape[0], generator=comparison_generator
    )[: reference.shape[0]]
    comparison = flattened.cpu()[comparison_indices]
    ess = effective_sample_size(draws)
    rhat = split_rhat(draws)
    expectation_error = (
        (comparison.mean(dim=0) - reference.mean(dim=0)).square().mean()
        + (
            comparison.square().mean(dim=0)
            - reference.square().mean(dim=0)
        ).square().mean()
    )
    metrics = {
        "histogram_kl": float(
            histogram_kl_2d(
                comparison,
                lambda value: (
                    -potential(
                        value.to(device=device, dtype=flattened.dtype)
                    )
                    / args.temperature
                ).detach().cpu(),
            )
        ),
        "mmd": float(rbf_mmd(comparison, reference)),
        "sliced_wasserstein": float(sliced_wasserstein(comparison, reference)),
        "mean_ess": float(ess.mean()),
        "min_ess": float(ess.min()),
        "max_rhat": float(rhat.max()),
        "mean_integrated_autocorrelation_time": float(
            (draws.shape[1] / ess.clamp_min(1e-12)).mean()
        ),
        "stationary_expectation_error": float(expectation_error),
        "mean_ess_per_energy_gradient": float(
            ess.mean() / max(profile.energy_grad_evaluations, 1)
        ),
        "mean_ess_per_second": float(ess.mean() / max(profile.wall_seconds, 1e-12)),
        **{key: float(value) for key, value in profile.to_dict().items()},
    }
    if variant == "mala":
        metrics["mala_acceptance_rate"] = acceptance_rate
    if target_name == "mixture":
        metrics.update(
            mixture_mixing_metrics(draws, potential.means.detach().cpu(), args.thin)
        )
    artifacts.log(
        args.draws,
        metrics,
        variant=variant,
        target=target_name,
        seed=seed,
        dt=dt,
        split="test",
    )
    torch.save(
        {
            "variant": variant,
            "target": target_name,
            "seed": seed,
            "dt": dt,
            "draws": draws,
            "metrics": metrics,
        },
        artifacts.path
        / "samples"
        / f"{target_name}_{variant}_seed{seed}_dt{dt}.pt",
    )
    return metrics


def main() -> None:
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    step_sizes = [float(item) for item in args.step_sizes.split(",") if item.strip()]
    experiment_id = args.experiment_id or (
        f"stationarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {
            **vars(args),
            "variants": variants,
            "targets": targets,
            "seeds": seeds,
            "step_sizes": step_sizes,
        },
        repository_root=Path(__file__).resolve().parents[2],
    )
    results: dict[str, list[float]] = {}
    for target in targets:
        for variant in variants:
            for dt in step_sizes:
                for seed in seeds:
                    metrics = run(args, artifacts, variant, target, seed, dt)
                    for name, value in metrics.items():
                        results.setdefault(
                            f"{target}/{variant}/dt={dt}/{name}", []
                        ).append(value)
    summary = artifacts.summarize(results, status="complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
