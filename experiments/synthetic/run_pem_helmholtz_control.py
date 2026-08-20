"""Frozen-energy control for PEM's Helmholtz-residual representation class.

Projected Energy Matching (PEM) jointly trains a scalar energy and an auxiliary
residual during distillation, penalizes the residual divergence, discards the
residual, and then refines the energy.  A full PEM run therefore changes the
energy and is not a controlled intervention on REM's frozen target.

This experiment isolates the Phase-2 residual class instead: for the exact same
frozen energy, supervised velocity, points, and seeds as the REM comparison, it
fits an additive residual with an exact two-dimensional divergence penalty.  In
two dimensions this is the deterministic analogue of PEM's Hutchinson trace
penalty.  We then measure representability and the equilibrium consequence of
retaining that field as an additive sampler drift.  It is a controlled
adaptation, not an official reproduction of the full PEM pipeline.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from experiments.synthetic.run_geometry_stress import (
    _target,
    _target_velocity,
    _training_points,
    evaluate_chain,
)
from experiments.synthetic.run_inductive_bias_comparison import (
    _energy_gradient,
    representation_metrics,
)
from rem.artifacts import RunArtifacts, seed_everything
from rem.networks import MLP
from rem.synthetic import AdditiveResidualSampler


VARIANT = "pem-helmholtz-residual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curvature", type=float, default=1.2)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--component-variance", type=float, default=0.1)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--divergence-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--step-size", type=float, default=0.002)
    parser.add_argument("--physical-burn", type=float, default=4.0)
    parser.add_argument("--physical-draw", type=float, default=12.0)
    parser.add_argument("--save-interval", type=float, default=0.01)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--core-radius", type=float, default=0.65)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def exact_divergence_penalty(
    field: nn.Module, points: Tensor, *, create_graph: bool
) -> tuple[Tensor, Tensor]:
    """Return PEM's normalized divergence loss and per-point divergence.

    PEM estimates ``tr(J)/D`` with a Hutchinson probe in high dimension.  The
    exact trace is cheap and lower variance for this controlled 2D problem.
    """

    differentiable = points.detach().requires_grad_(True)
    values = field(differentiable).reshape_as(differentiable)
    diagonal = []
    for coordinate in range(differentiable.shape[1]):
        jacobian_row = torch.autograd.grad(
            values[:, coordinate].sum(),
            differentiable,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        diagonal.append(jacobian_row[:, coordinate])
    divergence = torch.stack(diagonal, dim=1).sum(dim=1)
    normalized = divergence / differentiable.shape[1]
    return normalized.square().mean(), divergence


def _checkpoint_path(artifacts: RunArtifacts, curvature: float, seed: int) -> Path:
    return artifacts.path / "checkpoints" / f"curvature{curvature:g}_{VARIANT}_seed{seed}.pt"


def train_residual(
    args: argparse.Namespace,
    artifacts: RunArtifacts,
    target: nn.Module,
    oracle: nn.Module,
    seed: int,
) -> nn.Module:
    seed_everything(seed)
    device = target.means.device
    model = MLP(
        2,
        2,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        zero_output=True,
    ).to(device)
    checkpoint_path = _checkpoint_path(artifacts, args.curvature, seed)
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_generator = torch.Generator(device=device.type).manual_seed(10_000_000 + seed)
    validation_generator = torch.Generator(device=device.type).manual_seed(
        15_000_000 + seed
    )
    validation_points = _training_points(
        target, min(args.test_size, 2048), validation_generator
    )
    validation_gradient = _energy_gradient(target, validation_points)
    validation_target = _target_velocity(target, oracle, validation_points)
    denominator = validation_target.square().mean().clamp_min(1e-12)
    best_value = float("inf")
    best_step = -1
    best_state: dict[str, Tensor] | None = None

    for step in range(args.train_steps):
        points = _training_points(target, args.batch_size, train_generator)
        gradient = _energy_gradient(target, points)
        velocity = _target_velocity(target, oracle, points)
        optimizer.zero_grad(set_to_none=True)
        residual = model(points).reshape_as(gradient)
        prediction = -gradient + residual
        target_scale = velocity.square().mean().detach().clamp_min(1e-12)
        fit_mse = (prediction - velocity).square().mean()
        fit = fit_mse / target_scale
        divergence_loss, divergence = exact_divergence_penalty(
            model, points, create_graph=True
        )
        magnitude = residual.square().mean() / target_scale
        loss = (
            fit
            + args.divergence_weight * divergence_loss
            + args.residual_weight * magnitude
        )
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step + 1 == args.train_steps:
            with torch.no_grad():
                validation_residual = model(validation_points).reshape_as(
                    validation_gradient
                )
                validation_prediction = (
                    -validation_gradient
                    + validation_residual
                )
                relative_mse = float(
                    (validation_prediction - validation_target).square().mean()
                    / denominator
                )
                validation_residual_magnitude = float(
                    validation_residual.square().mean() / denominator
                )
            validation_divergence_loss, validation_divergence = (
                exact_divergence_penalty(
                    model, validation_points, create_graph=False
                )
            )
            selection_value = (
                relative_mse
                + args.divergence_weight * float(validation_divergence_loss)
                + args.residual_weight * validation_residual_magnitude
            )
            if selection_value < best_value:
                best_value = selection_value
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            artifacts.log(
                step,
                {
                    "loss": loss,
                    "velocity_mse": fit_mse,
                    "relative_velocity_mse": fit,
                    "normalized_residual_magnitude": magnitude,
                    "divergence_loss": divergence_loss,
                    "divergence_rms": divergence.square().mean().sqrt(),
                    "validation_relative_velocity_mse": relative_mse,
                    "validation_normalized_residual_magnitude": validation_residual_magnitude,
                    "validation_divergence_loss": validation_divergence_loss,
                    "validation_divergence_rms": validation_divergence.square().mean().sqrt(),
                    "validation_selection_value": selection_value,
                },
                phase="pem-helmholtz-train",
                curvature=args.curvature,
                variant=VARIANT,
                seed=seed,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model": model.state_dict(),
            "variant": VARIANT,
            "curvature": args.curvature,
            "seed": seed,
            "best_validation_selection_value": best_value,
            "best_step": best_step,
            "divergence_weight": args.divergence_weight,
            "controlled_adaptation": True,
            "full_pem_pipeline": False,
        },
        checkpoint_path,
    )
    return model


def main() -> None:
    args = parse_args()
    if (
        args.train_steps < 1
        or args.batch_size < 2
        or args.test_size < 2
        or args.log_every < 1
        or args.step_size <= 0
        or args.divergence_weight < 0
        or args.residual_weight < 0
    ):
        raise ValueError("invalid PEM Helmholtz control configuration")
    seeds = _seeds(args.seeds)
    experiment_id = args.experiment_id or (
        f"pem_helmholtz_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {**vars(args), "seeds": seeds, "variant": VARIANT},
        repository_root=Path(__file__).resolve().parents[2],
        resume_existing=args.resume,
    )
    device = torch.device(args.device)
    all_records: list[dict[str, object]] = []
    aggregated: dict[str, list[float]] = {}

    for seed in seeds:
        target, oracle = _target(args.curvature, args.component_variance, device)
        model = train_residual(args, artifacts, target, oracle, seed)
        test_generator = torch.Generator(device=device.type).manual_seed(
            20_000_000 + seed
        )
        test_points = _training_points(target, args.test_size, test_generator)
        representation = representation_metrics(
            target, oracle, "pem-style-residual", model, test_points
        )
        divergence_loss, divergence = exact_divergence_penalty(
            model, test_points, create_graph=False
        )
        representation["normalized_divergence_mse"] = float(divergence_loss)
        representation["divergence_rms"] = float(
            divergence.square().mean().sqrt()
        )
        sampler = AdditiveResidualSampler(model)
        equilibrium, _ = evaluate_chain(
            args,
            target,
            sampler,
            corrected=False,
            dt=args.step_size,
            seed=seed,
        )
        record = {
            "seed": seed,
            "curvature": args.curvature,
            "variant": VARIANT,
            "representation": representation,
            "equilibrium": equilibrium,
            "equilibrium_correction": False,
            "controlled_adaptation": True,
            "full_pem_pipeline": False,
            "pem_phase_isolated": "helmholtz-residual",
        }
        all_records.append(record)
        artifacts.log(
            -1,
            {
                **representation,
                **{f"equilibrium_{key}": value for key, value in equilibrium.items()},
            },
            phase="pem-helmholtz-test",
            curvature=args.curvature,
            variant=VARIANT,
            seed=seed,
            equilibrium_correction=False,
        )
        for family, values in (
            ("representation", representation),
            ("equilibrium", equilibrium),
        ):
            for metric, value in values.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    aggregated.setdefault(
                        f"{VARIANT}/{family}/{metric}", []
                    ).append(float(value))

    (artifacts.path / "pem_helmholtz_records.json").write_text(
        json.dumps(all_records, indent=2) + "\n", encoding="utf-8"
    )
    summary = artifacts.summarize(aggregated, status="complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
