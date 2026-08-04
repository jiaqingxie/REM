"""Train and evaluate REM variants on known synthetic vector fields.

Example:

    python -m experiments.synthetic.run_representability \
        --problem rotating --steps 5000 --variants em,em-large,constant,diagonal,full
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import (
    descent_condition_diagnostics,
    minimal_distortion_regularizer,
    mobility_diagnostics,
    riemannian_transport_loss,
    riemannian_velocity,
)
from rem.networks import build_mobility
from rem.synthetic import (
    ScalarPotentialMLP,
    dense_mobility_matrix,
    make_synthetic_problem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="rotating")
    parser.add_argument(
        "--variants",
        default="em,em-large,constant,scalar,diagonal,full",
        help="Comma-separated mobility variants",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--geometry-weight", type=float, default=1e-3)
    parser.add_argument("--representability-threshold", type=float, default=0.01)
    parser.add_argument("--metric-weighted", action="store_true")
    parser.add_argument(
        "--learn-energy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn V for projection-gap comparisons; disable for known-metric recovery",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def _energy_gradient(energy: nn.Module, x: Tensor) -> Tensor:
    x_grad = x.detach().requires_grad_(True)
    value = energy(x_grad)
    return torch.autograd.grad(value.sum(), x_grad, create_graph=False)[0]


def _relative_mse(prediction: Tensor, target: Tensor) -> Tensor:
    numerator = (prediction - target).square().mean()
    return numerator / target.square().mean().clamp_min(1e-12)


def _cosine_error(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=1)
    return (1 - cosine).mean()


def _relative_centered_potential_error(
    prediction: Tensor, target: Tensor
) -> Tensor:
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    return (prediction - target).square().mean() / target.square().mean().clamp_min(1e-12)


def _principal_eigenspace_angle(
    prediction: Tensor, target: Tensor
) -> Tensor:
    _, predicted_vectors = torch.linalg.eigh(prediction)
    _, target_vectors = torch.linalg.eigh(target)
    cosine = (
        predicted_vectors[:, :, -1] * target_vectors[:, :, -1]
    ).sum(dim=1).abs().clamp(0, 1)
    return torch.rad2deg(torch.acos(cosine)).mean()


def _antisymmetry_norm(field, points: Tensor, count: int = 64) -> Tensor:
    norms = []
    for point in points[:count]:
        point = point.detach().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(
            lambda value: field(value.unsqueeze(0)).squeeze(0),
            point,
            create_graph=False,
        )
        norms.append((jacobian - jacobian.transpose(0, 1)).square().mean().sqrt())
    return torch.stack(norms).mean()


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _primary_parameter_target(args: argparse.Namespace, dimension: int) -> int:
    energy = ScalarPotentialMLP(dimension, args.hidden_dim, args.depth)
    mobility = build_mobility(
        "full",
        (dimension,),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )
    return _parameter_count(energy) + _parameter_count(mobility)


def _matched_energy_hidden(args: argparse.Namespace, dimension: int) -> int:
    target = _primary_parameter_target(args, dimension)
    low, high = 1, max(args.hidden_dim * 4, 8)
    candidates: dict[int, int] = {}
    while low <= high:
        middle = (low + high) // 2
        count = _parameter_count(ScalarPotentialMLP(dimension, middle, args.depth))
        candidates[middle] = count
        if count < target:
            low = middle + 1
        else:
            high = middle - 1
    for width in {max(1, high), max(1, low)}:
        candidates[width] = _parameter_count(
            ScalarPotentialMLP(dimension, width, args.depth)
        )
    return min(candidates, key=lambda width: abs(candidates[width] - target))


def _plot_field(path: Path, points: Tensor, target: Tensor, prediction: Tensor) -> None:
    import matplotlib.pyplot as plt

    points_np = points.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    prediction_np = prediction.detach().cpu().numpy()
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, field, title in zip(
        axes, [target_np, prediction_np], ["Target field", "Fitted field"]
    ):
        axis.quiver(
            points_np[:, 0],
            points_np[:, 1],
            field[:, 0],
            field[:, 1],
            angles="xy",
            scale_units="xy",
            scale=12,
            width=0.002,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_variant(
    args: argparse.Namespace,
    artifacts: RunArtifacts,
    variant: str,
    seed: int,
) -> dict[str, float]:
    seed_everything(seed)
    device = torch.device(args.device)
    problem = make_synthetic_problem(args.problem)
    problem.potential.to(device)
    problem.mobility.to(device)

    hidden = (
        _matched_energy_hidden(args, problem.dimension)
        if variant == "em-large" and args.learn_energy
        else args.hidden_dim
    )
    if args.learn_energy:
        energy: nn.Module = ScalarPotentialMLP(problem.dimension, hidden, args.depth).to(device)
    else:
        energy = problem.potential
        for parameter in energy.parameters():
            parameter.requires_grad_(False)

    mobility_name = "identity" if variant in {"em", "em-large"} else variant
    mobility = build_mobility(
        mobility_name,
        (problem.dimension,),
        hidden_dim=hidden,
        depth=args.depth,
    ).to(device)
    parameters = [parameter for parameter in energy.parameters() if parameter.requires_grad]
    parameters.extend(parameter for parameter in mobility.parameters() if parameter.requires_grad)
    optimizer = torch.optim.Adam(parameters, lr=args.lr) if parameters else None

    start = time.perf_counter()
    final_loss = float("nan")
    for step in range(args.steps):
        if optimizer is None:
            break
        points = problem.sample_points(args.batch_size, device=device)
        target = problem.target_velocity(points).detach()
        optimizer.zero_grad(set_to_none=True)
        transport = riemannian_transport_loss(
            energy,
            mobility,
            points,
            target,
            metric_weighted=args.metric_weighted,
        )
        geometry = minimal_distortion_regularizer(mobility, points)
        loss = transport + args.geometry_weight * geometry
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if step % args.log_every == 0 or step + 1 == args.steps:
            artifacts.log(
                step,
                {
                    "loss": loss,
                    "transport_loss": transport,
                    "geometry_penalty": geometry,
                },
                variant=variant,
                seed=seed,
                split="train",
            )

    elapsed = time.perf_counter() - start
    test_generator = torch.Generator(device=device.type).manual_seed(100_000 + seed)
    test_points = problem.sample_points(
        args.test_size,
        device=device,
        generator=test_generator,
    )
    target = problem.target_velocity(test_points).detach()
    prediction = riemannian_velocity(
        energy, mobility, test_points, create_graph=False
    ).detach()
    if optimizer is None:
        final_loss = float((prediction - target).square().mean())
    gradient = _energy_gradient(energy, test_points)
    descent = descent_condition_diagnostics(gradient, target)
    geometry = mobility_diagnostics(mobility, test_points)

    predicted_matrix = dense_mobility_matrix(mobility, test_points)
    target_matrix = problem.target_matrix(test_points)
    matrix_error = (
        (predicted_matrix - target_matrix).square().mean()
        / target_matrix.square().mean().clamp_min(1e-12)
    )
    eigenspace_angle = _principal_eigenspace_angle(
        predicted_matrix.detach(), target_matrix.detach()
    )
    with torch.no_grad():
        predicted_potential = energy(test_points)
        target_potential = problem.potential(test_points)
    potential_error = _relative_centered_potential_error(
        predicted_potential, target_potential
    )
    predicted_representable = bool(
        float(descent["violation_rate"]) < args.representability_threshold
    )

    subset = test_points[: min(256, args.test_size)]
    target_curl = _antisymmetry_norm(
        lambda value: problem.target_velocity(value, create_graph=True), subset
    )
    predicted_curl = _antisymmetry_norm(
        lambda value: riemannian_velocity(
            energy, mobility, value, create_graph=True
        ),
        subset,
    )
    metrics = {
        "final_train_loss": final_loss,
        "relative_velocity_mse": float(_relative_mse(prediction, target)),
        "cosine_error": float(_cosine_error(prediction, target)),
        "descent_violation_rate": float(descent["violation_rate"]),
        "relative_metric_mse": float(matrix_error.detach()),
        "principal_eigenspace_angle_degrees": float(eigenspace_angle.detach()),
        "potential_relative_mse_up_to_constant": float(potential_error.detach()),
        "representability_label": float(problem.representable),
        "predicted_representable": float(predicted_representable),
        "representability_prediction_correct": float(
            predicted_representable == problem.representable
        ),
        "mean_logdet_abs": float(geometry["logdet"].abs().mean().detach()),
        "mean_condition_number": float(
            geometry["condition_number"].mean().detach()
        ),
        "target_antisymmetry": float(target_curl),
        "prediction_antisymmetry": float(predicted_curl),
        "train_wall_seconds": elapsed,
        "trainable_parameters": float(sum(p.numel() for p in parameters)),
    }
    if args.learn_energy and variant in {"em-large", "full"}:
        target_parameters = _primary_parameter_target(args, problem.dimension)
        metrics["parameter_match_target"] = float(target_parameters)
        metrics["relative_parameter_match_error"] = float(
            abs(sum(p.numel() for p in parameters) - target_parameters)
            / target_parameters
        )
    artifacts.log(-1, metrics, variant=variant, seed=seed, split="test")
    torch.save(
        {
            "problem": problem.name,
            "variant": variant,
            "seed": seed,
            "energy": energy.state_dict(),
            "mobility": mobility.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        artifacts.path / "checkpoints" / f"{variant}_seed{seed}.pt",
    )
    if args.plot and problem.dimension == 2:
        plot_points = test_points[:512]
        _plot_field(
            artifacts.path / "figures" / f"{variant}_seed{seed}.png",
            plot_points,
            target[:512],
            prediction[:512],
        )
    return metrics


def main() -> None:
    args = parse_args()
    if args.steps < 0 or args.batch_size < 1 or args.test_size < 1:
        raise ValueError("invalid training sizes")
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    experiment_id = args.experiment_id or (
        f"representability_{args.problem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {**vars(args), "variants": variants, "seeds": seeds},
        repository_root=Path(__file__).resolve().parents[2],
    )

    all_results: dict[str, list[dict[str, float]]] = {variant: [] for variant in variants}
    for variant in variants:
        for seed in seeds:
            all_results[variant].append(run_variant(args, artifacts, variant, seed))

    flattened: dict[str, list[float]] = {}
    for variant, results in all_results.items():
        for metric_name in results[0]:
            flattened[f"{variant}/{metric_name}"] = [
                result[metric_name] for result in results
            ]
    summary = artifacts.summarize(flattened, status="complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
