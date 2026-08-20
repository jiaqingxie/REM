"""Unified geometry stress test for REM mechanism, dynamics, and validity.

The experiment uses an exactly sampleable four-mode Gaussian mixture and an
area-preserving nonlinear warp.  Changing the warp curvature changes only the
coordinate geometry, not the latent equilibrium distribution.  This makes it
possible to test four paper claims with one controlled protocol:

* mechanism: robustness as the nonlinear warp becomes stronger;
* dynamics: first-passage and round-trip times at matched mode occupancy;
* validity: equilibrium preservation with the Itô divergence correction;
* ablation: identity, oracle-tuned constant SPD, diagonal/full learned REM.

The learned variants freeze the exact energy and receive the analytic
pushforward velocity ``-G_oracle grad V`` at prescribed sites.  This is an
oracle-field representability diagnostic, not minibatch-OT straight-chord
training.  The paper diagnostic uses an external Euclidean residual; an
optional flag evaluates the default inverse-mobility-weighted objective.
``oracle-full`` is an explicit ceiling and ``oracle-no-div`` isolates the
correction term without confounding metric-estimation error.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import (
    IdentityMobility,
    minimal_distortion_regularizer,
    riemannian_transport_loss,
)
from rem.metrics import effective_sample_size, rbf_mmd, sliced_wasserstein
from rem.networks import build_mobility
from rem.sampling import constant_temperature, langevin_chain
from rem.synthetic import (
    AnalyticWarpMobility,
    FixedFullMobility,
    WarpedGaussianMixturePotential,
    dense_mobility_matrix,
)


# Four equally spaced modes form a straight latent channel.  The nonlinear
# map bends that channel into a parabola without changing mixture weights or
# local latent covariances.  A single constant SPD matrix can rescale/rotate
# the channel but cannot undo its state-dependent curvature.
DEFAULT_MEANS = ((-1.5, 0.0), (-0.5, 0.0), (0.5, 0.0), (1.5, 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="main,validity")
    parser.add_argument("--curvatures", default="0,0.4,0.8,1.2")
    parser.add_argument("--validity-curvature", type=float, default=1.2)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--variants",
        default=(
            "identity,constant-spd,rem-diagonal,rem-full,"
            "rem-full-no-div,oracle-full"
        ),
    )
    parser.add_argument("--validity-variants", default="oracle-full,oracle-no-div")
    parser.add_argument("--step-size", type=float, default=0.002)
    parser.add_argument("--validity-step-sizes", default="0.001,0.002,0.004,0.008")
    parser.add_argument("--physical-burn", type=float, default=4.0)
    parser.add_argument("--physical-draw", type=float, default=12.0)
    parser.add_argument("--save-interval", type=float, default=0.01)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--component-variance", type=float, default=0.1)
    parser.add_argument("--core-radius", type=float, default=0.65)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--log-bound", type=float, default=5.0)
    parser.add_argument("--geometry-weight", type=float, default=1e-3)
    parser.add_argument(
        "--metric-weighted",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Measure the transport residual in the learned inverse-mobility "
            "norm. The controlled paper diagnostic defaults to Euclidean weighting."
        ),
    )
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_csv(value: str) -> list[float]:
    return [float(item) for item in _csv(value)]


def _int_csv(value: str) -> list[int]:
    return [int(item) for item in _csv(value)]


def _target(
    curvature: float,
    component_variance: float,
    device: torch.device,
) -> tuple[WarpedGaussianMixturePotential, AnalyticWarpMobility]:
    means = torch.tensor(DEFAULT_MEANS, device=device, dtype=torch.float32)
    potential = WarpedGaussianMixturePotential(
        means,
        curvature=curvature,
        component_variance=component_variance,
    ).to(device)
    return potential, AnalyticWarpMobility(curvature).to(device)


def _target_velocity(
    potential: nn.Module,
    oracle: nn.Module,
    points: Tensor,
) -> Tensor:
    with torch.enable_grad():
        points_for_grad = points.detach().requires_grad_(True)
        gradient = torch.autograd.grad(potential(points_for_grad).sum(), points_for_grad)[0]
    return -oracle.apply(points_for_grad, gradient).detach()


def _predicted_velocity(
    potential: nn.Module,
    mobility: nn.Module,
    points: Tensor,
) -> Tensor:
    with torch.enable_grad():
        points_for_grad = points.detach().requires_grad_(True)
        gradient = torch.autograd.grad(potential(points_for_grad).sum(), points_for_grad)[0]
        return -mobility.apply(points_for_grad, gradient).detach()


def _training_points(
    target: WarpedGaussianMixturePotential,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    # Half of each batch follows the target components; the other half fills
    # the low-density channel between adjacent modes.  Sampling an unbounded
    # broad Gaussian here is actively harmful for the quadratic shear: rare
    # |z_1| outliers create velocities outside the bounded mobility model and
    # dominate an otherwise well-conditioned transport objective.
    target_count = count // 2
    channel_count = count - target_count
    means = target.means
    standard_deviation = target.component_variance**0.5
    indices = torch.randint(
        means.shape[0],
        (target_count,),
        device=means.device,
        generator=generator,
    )
    latent_target = means[indices] + standard_deviation * torch.randn(
        target_count,
        2,
        device=means.device,
        dtype=means.dtype,
        generator=generator,
    )
    channel_limit = float(means[:, 0].abs().max()) + 0.75
    latent_target[:, 0].clamp_(-channel_limit, channel_limit)
    latent_channel = torch.empty(
        channel_count, 2, device=means.device, dtype=means.dtype
    )
    latent_channel[:, 0].uniform_(
        -channel_limit, channel_limit, generator=generator
    )
    latent_channel[:, 1] = 1.5 * standard_deviation * torch.randn(
        channel_count,
        device=means.device,
        dtype=means.dtype,
        generator=generator,
    )
    return target.warp(torch.cat([latent_target, latent_channel], dim=0))


def train_mobility(
    args: argparse.Namespace,
    artifacts: RunArtifacts,
    target: WarpedGaussianMixturePotential,
    oracle: AnalyticWarpMobility,
    kind: str,
    curvature: float,
    seed: int,
) -> nn.Module:
    checkpoint_path = (
        artifacts.path
        / "checkpoints"
        / f"curvature{curvature:g}_{kind}_seed{seed}.pt"
    )
    mobility = build_mobility(
        kind,
        (2,),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        log_bound=args.log_bound,
    ).to(target.means.device)
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=target.means.device)
        mobility.load_state_dict(checkpoint["mobility"])
        return mobility

    seed_everything(seed)
    generator = torch.Generator(device=target.means.device.type).manual_seed(
        10_000_000 + seed
    )
    optimizer = torch.optim.Adam(mobility.parameters(), lr=args.lr)
    validation_generator = torch.Generator(device=target.means.device.type).manual_seed(
        15_000_000 + seed
    )
    validation_points = _training_points(
        target, min(args.test_size, 2048), validation_generator
    )
    validation_target = _target_velocity(target, oracle, validation_points)
    best_validation = float("inf")
    best_step = -1
    best_state: dict[str, Tensor] | None = None
    started = time.perf_counter()
    final_metrics: dict[str, float] = {}
    for step in range(args.train_steps):
        points = _training_points(target, args.batch_size, generator)
        velocity = _target_velocity(target, oracle, points)
        optimizer.zero_grad(set_to_none=True)
        transport = riemannian_transport_loss(
            target,
            mobility,
            points,
            velocity,
            metric_weighted=args.metric_weighted,
        )
        distortion = minimal_distortion_regularizer(mobility, points)
        loss = transport + args.geometry_weight * distortion
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step + 1 == args.train_steps:
            validation_prediction = _predicted_velocity(
                target, mobility, validation_points
            )
            validation_relative_mse = float(
                (validation_prediction - validation_target).square().mean()
                / validation_target.square().mean().clamp_min(1e-12)
            )
            if validation_relative_mse < best_validation:
                best_validation = validation_relative_mse
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in mobility.state_dict().items()
                }
            artifacts.log(
                step,
                {
                    "loss": loss,
                    "transport_loss": transport,
                    "distortion": distortion,
                    "validation_relative_velocity_mse": validation_relative_mse,
                },
                phase="train",
                curvature=curvature,
                variant=f"rem-{kind}",
                seed=seed,
            )

    if best_state is not None:
        mobility.load_state_dict(best_state)

    test_generator = torch.Generator(device=target.means.device.type).manual_seed(
        20_000_000 + seed
    )
    test_points = _training_points(target, args.test_size, test_generator)
    target_velocity = _target_velocity(target, oracle, test_points)
    prediction = _predicted_velocity(target, mobility, test_points)
    predicted_matrix = dense_mobility_matrix(mobility, test_points).detach()
    target_matrix = oracle.matrix(test_points).detach()
    final_metrics = {
        "relative_velocity_mse": float(
            (prediction - target_velocity).square().mean()
            / target_velocity.square().mean().clamp_min(1e-12)
        ),
        "relative_metric_mse": float(
            (predicted_matrix - target_matrix).square().mean()
            / target_matrix.square().mean().clamp_min(1e-12)
        ),
        "train_wall_seconds": time.perf_counter() - started,
        "trainable_parameters": float(sum(p.numel() for p in mobility.parameters())),
        "best_validation_relative_velocity_mse": best_validation,
        "best_step": float(best_step),
    }
    artifacts.log(
        args.train_steps,
        final_metrics,
        phase="metric-test",
        curvature=curvature,
        variant=f"rem-{kind}",
        seed=seed,
    )
    torch.save(
        {
            "mobility": mobility.state_dict(),
            "kind": kind,
            "curvature": curvature,
            "seed": seed,
            "metrics": final_metrics,
        },
        checkpoint_path,
    )
    return mobility


def tuned_constant_spd(
    target: WarpedGaussianMixturePotential,
    oracle: AnalyticWarpMobility,
) -> FixedFullMobility:
    """Oracle-tune the strongest possible state-independent SPD baseline."""

    generator = torch.Generator(device=target.means.device.type).manual_seed(31_415_926)
    points = target.sample(
        32768,
        generator=generator,
        device=target.means.device,
        dtype=target.means.dtype,
    )
    matrix = oracle.matrix(points).mean(dim=0)
    matrix = matrix / torch.linalg.det(matrix).sqrt()
    return FixedFullMobility(matrix).to(target.means.device)


def _variant_mobility(
    variant: str,
    learned: dict[str, nn.Module],
    oracle: nn.Module,
    constant: nn.Module,
) -> tuple[nn.Module, bool]:
    if variant == "identity":
        return IdentityMobility(), True
    if variant == "constant-spd":
        return constant, True
    if variant == "rem-diagonal":
        return learned["diagonal"], True
    if variant == "rem-full":
        return learned["full"], True
    if variant == "rem-full-no-div":
        return learned["full"], False
    if variant == "oracle-full":
        return oracle, True
    if variant == "oracle-no-div":
        return oracle, False
    raise ValueError(f"unknown geometry-stress variant: {variant}")


def _balanced_initial(
    target: WarpedGaussianMixturePotential,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    indices = torch.arange(count, device=target.means.device) % target.means.shape[0]
    noise = 0.25 * target.component_variance**0.5 * torch.randn(
        count,
        2,
        device=target.means.device,
        dtype=target.means.dtype,
        generator=generator,
    )
    return target.warp(target.means[indices] + noise)


def _core_labels(
    latent_draws: Tensor,
    means: Tensor,
    core_radius: float,
) -> Tensor:
    distances = (
        latent_draws[:, :, None, :] - means[None, None, :, :]
    ).square().sum(dim=3)
    minimum, labels = distances.min(dim=2)
    valid = minimum <= core_radius**2
    # Outside every modal core, retain the previous committed mode.  This
    # prevents boundary jitter from being counted as repeated transitions.
    labels = torch.where(valid, labels, torch.full_like(labels, -1))
    labels[:, 0] = distances[:, 0].argmin(dim=1)
    for index in range(1, labels.shape[1]):
        labels[:, index] = torch.where(
            labels[:, index] >= 0, labels[:, index], labels[:, index - 1]
        )
    return labels


def _mixing_metrics(
    labels: Tensor, integration_stride: int, mode_count: int
) -> dict[str, float]:
    occupancy = torch.stack(
        [(labels == index).float().mean() for index in range(mode_count)]
    )
    target = torch.full_like(occupancy, 1.0 / mode_count)
    occupancy_kl = (occupancy.clamp_min(1e-12) * (
        occupancy.clamp_min(1e-12).log() - target.log()
    )).sum()
    transitions = labels[:, 1:] != labels[:, :-1]
    transition_counts = transitions.sum(dim=1)
    first_passages: list[int] = []
    round_trips: list[int] = []
    transition_matrix = torch.zeros(mode_count, mode_count, dtype=torch.float64)
    for chain_labels in labels:
        start = int(chain_labels[0])
        different = torch.nonzero(chain_labels != start).flatten()
        if different.numel():
            first = int(different[0])
            first_passages.append(first * integration_stride)
            returned = torch.nonzero(chain_labels[first + 1 :] == start).flatten()
            if returned.numel():
                round_trips.append(
                    (first + 1 + int(returned[0])) * integration_stride
                )
        changed = torch.nonzero(chain_labels[1:] != chain_labels[:-1]).flatten()
        for index in changed:
            source = int(chain_labels[int(index)])
            destination = int(chain_labels[int(index) + 1])
            transition_matrix[source, destination] += 1
    one_hot = torch.nn.functional.one_hot(labels, num_classes=mode_count).float()
    mode_ess = effective_sample_size(one_hot)
    horizon = labels.shape[1] * integration_stride
    return {
        "mode_min_occupancy": float(occupancy.min()),
        "mode_max_occupancy": float(occupancy.max()),
        "mode_max_absolute_occupancy_error": float((occupancy - target).abs().max()),
        "mode_occupancy_kl": float(occupancy_kl),
        "mean_mode_transitions": float(transition_counts.float().mean()),
        "mode_transitions_per_1000_steps": float(
            1000 * transition_counts.float().mean() / max(horizon, 1)
        ),
        "first_passage_completion_rate": len(first_passages) / labels.shape[0],
        "median_first_passage_steps": float(torch.tensor(first_passages).median())
        if first_passages
        else float(horizon),
        "round_trip_completion_rate": len(round_trips) / labels.shape[0],
        "median_round_trip_steps": float(torch.tensor(round_trips).median())
        if round_trips
        else float(horizon),
        "mean_mode_indicator_ess": float(mode_ess.mean()),
        "transition_matrix": transition_matrix.tolist(),
    }


def evaluate_chain(
    args: argparse.Namespace,
    target: WarpedGaussianMixturePotential,
    mobility: nn.Module,
    *,
    corrected: bool,
    dt: float,
    seed: int,
) -> tuple[dict[str, float], Tensor]:
    device = target.means.device
    generator = torch.Generator(device=device.type).manual_seed(40_000_000 + seed)
    initial = _balanced_initial(target, args.chains, generator)
    burn_steps = max(1, round(args.physical_burn / dt))
    draw_steps = max(1, round(args.physical_draw / dt))
    save_every = max(1, round(args.save_interval / dt))
    started = time.perf_counter()
    burned, _ = langevin_chain(
        target,
        mobility,
        initial,
        steps=burn_steps,
        dt=dt,
        temperature=constant_temperature(1.0),
        include_divergence_correction=corrected,
        divergence_method="exact",
        generator=generator,
    )
    _, trajectory = langevin_chain(
        target,
        mobility,
        burned,
        steps=draw_steps,
        dt=dt,
        temperature=constant_temperature(1.0),
        include_divergence_correction=corrected,
        divergence_method="exact",
        save_every=save_every,
        generator=generator,
    )
    wall_seconds = time.perf_counter() - started
    draws = torch.stack(trajectory[1:], dim=1)
    latent = target.inverse_warp(draws.reshape(-1, 2).to(device)).reshape_as(draws)
    labels = _core_labels(
        latent.cpu(), target.means.detach().cpu(), args.core_radius
    )
    flattened = draws.reshape(-1, 2).detach().cpu()
    latent_flattened = latent.reshape(-1, 2).detach().cpu()
    comparison_count = min(2048, flattened.shape[0])
    comparison_generator = torch.Generator(device="cpu").manual_seed(50_000_000 + seed)
    indices = torch.randperm(flattened.shape[0], generator=comparison_generator)[
        :comparison_count
    ]
    comparison = flattened[indices]
    latent_comparison = latent_flattened[indices]
    reference_generator = torch.Generator(device=device.type).manual_seed(60_000_000 + seed)
    reference = target.sample(
        comparison_count,
        generator=reference_generator,
        device=device,
        dtype=draws.dtype,
    ).detach().cpu()
    reference_latent = target.inverse_warp(reference.to(device)).detach().cpu()
    expectation_error = (
        (comparison.mean(dim=0) - reference.mean(dim=0)).square().mean()
        + (
            comparison.square().mean(dim=0)
            - reference.square().mean(dim=0)
        ).square().mean()
    )
    stride = max(1, round(args.save_interval / dt))
    metrics = {
        "observed_mmd": float(rbf_mmd(comparison, reference)),
        "latent_mmd": float(rbf_mmd(latent_comparison, reference_latent)),
        "observed_sliced_wasserstein": float(
            sliced_wasserstein(comparison, reference)
        ),
        "stationary_expectation_error": float(expectation_error),
        "sample_wall_seconds": wall_seconds,
        "integration_steps": float(burn_steps + draw_steps),
        "chains_per_second": args.chains / max(wall_seconds, 1e-12),
        **_mixing_metrics(labels, stride, target.means.shape[0]),
    }
    return metrics, draws


def _record_path(
    artifacts: RunArtifacts,
    phase: str,
    curvature: float,
    variant: str,
    seed: int,
    dt: float,
) -> Path:
    records = artifacts.path / "records"
    records.mkdir(exist_ok=True)
    return records / (
        f"{phase}_curvature{curvature:g}_{variant}_seed{seed}_dt{dt:g}.json"
    )


def run_configuration(
    args: argparse.Namespace,
    artifacts: RunArtifacts,
    *,
    phase: str,
    curvature: float,
    variant: str,
    seed: int,
    dt: float,
    target: WarpedGaussianMixturePotential,
    mobility: nn.Module,
    corrected: bool,
) -> dict[str, float]:
    record_path = _record_path(artifacts, phase, curvature, variant, seed, dt)
    if args.resume and record_path.exists():
        return json.loads(record_path.read_text(encoding="utf-8"))["metrics"]
    metrics, draws = evaluate_chain(
        args, target, mobility, corrected=corrected, dt=dt, seed=seed
    )
    record = {
        "phase": phase,
        "curvature": curvature,
        "variant": variant,
        "seed": seed,
        "dt": dt,
        "corrected": corrected,
        "metrics": metrics,
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    artifacts.log(
        -1,
        metrics,
        phase=phase,
        curvature=curvature,
        variant=variant,
        seed=seed,
        dt=dt,
        corrected=corrected,
    )
    torch.save(
        {**{key: value for key, value in record.items() if key != "metrics"}, "draws": draws},
        artifacts.path
        / "samples"
        / f"{phase}_curvature{curvature:g}_{variant}_seed{seed}_dt{dt:g}.pt",
    )
    return metrics


def main() -> None:
    args = parse_args()
    if (
        args.step_size <= 0
        or args.physical_burn <= 0
        or args.physical_draw <= 0
        or args.save_interval <= 0
        or args.chains < 4
        or args.train_steps < 0
    ):
        raise ValueError("invalid geometry stress configuration")
    phases = _csv(args.phases)
    curvatures = _float_csv(args.curvatures)
    seeds = _int_csv(args.seeds)
    variants = _csv(args.variants)
    validity_variants = _csv(args.validity_variants)
    validity_step_sizes = _float_csv(args.validity_step_sizes)
    experiment_id = args.experiment_id or (
        f"geometry_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {
            **vars(args),
            "phases": phases,
            "curvatures": curvatures,
            "seeds": seeds,
            "variants": variants,
            "validity_variants": validity_variants,
            "validity_step_sizes": validity_step_sizes,
        },
        repository_root=Path(__file__).resolve().parents[2],
        resume_existing=args.resume,
    )
    device = torch.device(args.device)
    aggregated: dict[str, list[float]] = {}

    if "main" in phases:
        for curvature in curvatures:
            for seed in seeds:
                target, oracle = _target(curvature, args.component_variance, device)
                constant = tuned_constant_spd(target, oracle)
                learned: dict[str, nn.Module] = {}
                if any("diagonal" in variant for variant in variants):
                    learned["diagonal"] = train_mobility(
                        args, artifacts, target, oracle, "diagonal", curvature, seed
                    )
                if any("rem-full" in variant for variant in variants):
                    learned["full"] = train_mobility(
                        args, artifacts, target, oracle, "full", curvature, seed
                    )
                for variant in variants:
                    mobility, corrected = _variant_mobility(
                        variant, learned, oracle, constant
                    )
                    metrics = run_configuration(
                        args,
                        artifacts,
                        phase="main",
                        curvature=curvature,
                        variant=variant,
                        seed=seed,
                        dt=args.step_size,
                        target=target,
                        mobility=mobility,
                        corrected=corrected,
                    )
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            aggregated.setdefault(
                                f"main/curvature={curvature:g}/{variant}/{name}", []
                            ).append(float(value))

    if "validity" in phases:
        curvature = args.validity_curvature
        for seed in seeds:
            target, oracle = _target(curvature, args.component_variance, device)
            constant = tuned_constant_spd(target, oracle)
            for variant in validity_variants:
                mobility, corrected = _variant_mobility(
                    variant, {}, oracle, constant
                )
                for dt in validity_step_sizes:
                    metrics = run_configuration(
                        args,
                        artifacts,
                        phase="validity",
                        curvature=curvature,
                        variant=variant,
                        seed=seed,
                        dt=dt,
                        target=target,
                        mobility=mobility,
                        corrected=corrected,
                    )
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            aggregated.setdefault(
                                f"validity/dt={dt:g}/{variant}/{name}", []
                            ).append(float(value))

    summary = artifacts.summarize(aggregated, status="complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
