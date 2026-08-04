"""Validate REM equilibrium, divergence correction, and discretization bias."""

from __future__ import annotations

import argparse
import json
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
from rem.sampling import constant_temperature, langevin_chain, profile_langevin_chain
from rem.synthetic import (
    AnalyticDiagonalMobility,
    BananaPotential,
    GaussianMixturePotential,
    HutchinsonDivergenceWrapper,
    QuadraticPotential,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default="euclidean,rem-exact,rem-hutch-1,rem-hutch-4,rem-no-div",
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
    base = AnalyticDiagonalMobility(strength)
    if variant == "rem-exact":
        return base, True, 1
    if variant == "rem-no-div":
        return base, False, 1
    if variant.startswith("rem-hutch-"):
        probes = int(variant.rsplit("-", 1)[1])
        return HutchinsonDivergenceWrapper(base), True, probes
    raise ValueError(f"unknown stationarity variant: {variant}")


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
    mobility, correction, probes = _variant(variant, args.strength)
    mobility.to(device)
    generator = torch.Generator(device=device.type).manual_seed(seed)
    initial = 3.0 * torch.randn(
        args.chains, 2, device=device, generator=generator
    )
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
    comparison = flattened[: reference.shape[0]].cpu()
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
        "stationary_expectation_error": float(expectation_error),
        **{key: float(value) for key, value in profile.to_dict().items()},
    }
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
