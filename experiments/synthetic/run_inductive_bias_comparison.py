"""Controlled comparison of transport-geometry inductive biases.

Every learned method receives the same frozen energy, supervised velocity,
training points, and held-out points.  The representations differ:

* REM: a state-dependent determinant-one SPD mobility;
* free residual upper bound: an unrestricted additive vector field;
* learned mirror: inverse Hessian of a strongly convex mirror potential;
* learned preconditioner: one state-independent determinant-one SPD matrix.

The free-residual and learned-mirror rows are controlled adaptations, not claims
of reproducing the full training pipelines or published benchmark numbers of
Projected Energy Matching or CPMLA.  The experiment is designed to isolate
inductive bias: held-out velocity fit measures transport representability,
while an equilibrium chain tests whether the natural stochastic extension
preserves the unchanged energy law.
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

try:
    from experiments.synthetic.run_geometry_stress import (
        _target,
        _target_velocity,
        _training_points,
        evaluate_chain,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
    from run_geometry_stress import (
        _target,
        _target_velocity,
        _training_points,
        evaluate_chain,
    )
from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import (
    GaugeFixedFullMobility,
    IdentityMobility,
    minimal_distortion_regularizer,
)
from rem.networks import MLP, build_mobility
from rem.synthetic import (
    AdditiveResidualSampler,
    ConvexRidgeMirrorPotential,
    InverseHessianMobility,
    dense_mobility_matrix,
)


LEARNED_VARIANTS = (
    "rem-full",
    "pem-style-residual",
    "learned-mirror",
    "learned-preconditioner",
)
EVALUATED_VARIANTS = ("identity", *LEARNED_VARIANTS, "oracle-full")


class ConstantPackedMatrix(nn.Module):
    """Input-independent packed symmetric logits for a learned SPD matrix."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.packed = nn.Parameter(
            torch.zeros(dimension * (dimension + 1) // 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.packed.unsqueeze(0).expand(x.shape[0], -1)


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
    # Four learned scalars per ridge in two dimensions; 8192 ridges gives
    # 32,769 parameters, closely matching the 33k-parameter REM/residual MLPs.
    parser.add_argument("--mirror-width", type=int, default=8192)
    parser.add_argument("--log-bound", type=float, default=5.0)
    parser.add_argument("--geometry-weight", type=float, default=1e-3)
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


def _energy_gradient(potential: nn.Module, points: Tensor) -> Tensor:
    with torch.enable_grad():
        differentiable = points.detach().requires_grad_(True)
        gradient = torch.autograd.grad(potential(differentiable).sum(), differentiable)[0]
    return gradient.detach()


def _models(args: argparse.Namespace, device: torch.device) -> dict[str, nn.Module]:
    rem = build_mobility(
        "full",
        (2,),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        log_bound=args.log_bound,
    )
    residual = MLP(
        2, 2, hidden_dim=args.hidden_dim, depth=args.depth, zero_output=True
    )
    mirror = InverseHessianMobility(
        ConvexRidgeMirrorPotential(2, width=args.mirror_width)
    )
    preconditioner = GaugeFixedFullMobility(
        ConstantPackedMatrix(2), 2, log_eigenvalue_bound=args.log_bound
    )
    return {
        "rem-full": rem.to(device),
        "pem-style-residual": residual.to(device),
        "learned-mirror": mirror.to(device),
        "learned-preconditioner": preconditioner.to(device),
    }


def _prediction(
    variant: str,
    model: nn.Module,
    points: Tensor,
    gradient: Tensor,
) -> Tensor:
    if variant == "pem-style-residual":
        return -gradient + model(points).reshape_as(gradient)
    return -model.apply(points, gradient)


def _regularizer(
    variant: str,
    model: nn.Module,
    points: Tensor,
) -> Tensor:
    if variant == "pem-style-residual":
        return model(points).square().mean()
    return minimal_distortion_regularizer(model, points)


def _checkpoint_path(
    artifacts: RunArtifacts, variant: str, curvature: float, seed: int
) -> Path:
    return (
        artifacts.path
        / "checkpoints"
        / f"curvature{curvature:g}_{variant}_seed{seed}.pt"
    )


def train_models(
    args: argparse.Namespace,
    artifacts: RunArtifacts,
    target: nn.Module,
    oracle: nn.Module,
    seed: int,
) -> dict[str, nn.Module]:
    seed_everything(seed)
    device = target.means.device
    models = _models(args, device)
    if args.resume and all(
        _checkpoint_path(artifacts, name, args.curvature, seed).exists()
        for name in LEARNED_VARIANTS
    ):
        for name, model in models.items():
            checkpoint = torch.load(
                _checkpoint_path(artifacts, name, args.curvature, seed),
                map_location=device,
            )
            model.load_state_dict(checkpoint["model"])
        return models

    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=args.lr)
        for name, model in models.items()
    }
    best_values = {name: float("inf") for name in models}
    best_steps = {name: -1 for name in models}
    best_states: dict[str, dict[str, Tensor]] = {}
    train_generator = torch.Generator(device=device.type).manual_seed(
        10_000_000 + seed
    )
    validation_generator = torch.Generator(device=device.type).manual_seed(
        15_000_000 + seed
    )
    validation_points = _training_points(
        target, min(args.test_size, 2048), validation_generator
    )
    validation_gradient = _energy_gradient(target, validation_points)
    validation_target = _target_velocity(target, oracle, validation_points)

    for step in range(args.train_steps):
        points = _training_points(target, args.batch_size, train_generator)
        gradient = _energy_gradient(target, points)
        velocity = _target_velocity(target, oracle, points)
        for name, model in models.items():
            optimizers[name].zero_grad(set_to_none=True)
            prediction = _prediction(name, model, points, gradient)
            fit = (prediction - velocity).square().mean()
            regularizer = _regularizer(name, model, points)
            loss = fit + args.geometry_weight * regularizer
            loss.backward()
            optimizers[name].step()
            if step % args.log_every == 0 or step + 1 == args.train_steps:
                with torch.no_grad():
                    validation_prediction = _prediction(
                        name,
                        model,
                        validation_points,
                        validation_gradient,
                    )
                    relative_mse = float(
                        (validation_prediction - validation_target).square().mean()
                        / validation_target.square().mean().clamp_min(1e-12)
                    )
                if relative_mse < best_values[name]:
                    best_values[name] = relative_mse
                    best_steps[name] = step
                    best_states[name] = copy.deepcopy(model.state_dict())
                artifacts.log(
                    step,
                    {
                        "loss": loss,
                        "velocity_mse": fit,
                        "regularizer": regularizer,
                        "validation_relative_velocity_mse": relative_mse,
                    },
                    phase="inductive-bias-train",
                    curvature=args.curvature,
                    variant=name,
                    seed=seed,
                )

    for name, model in models.items():
        if name in best_states:
            model.load_state_dict(best_states[name])
        torch.save(
            {
                "model": model.state_dict(),
                "variant": name,
                "curvature": args.curvature,
                "seed": seed,
                "best_validation_relative_velocity_mse": best_values[name],
                "best_step": best_steps[name],
                "controlled_adaptation": name
                in {"pem-style-residual", "learned-mirror"},
            },
            _checkpoint_path(artifacts, name, args.curvature, seed),
        )
    return models


def representation_metrics(
    potential: nn.Module,
    oracle: nn.Module,
    variant: str,
    model: nn.Module,
    points: Tensor,
) -> dict[str, float]:
    gradient = _energy_gradient(potential, points)
    target_velocity = _target_velocity(potential, oracle, points)
    if variant == "identity":
        prediction = -gradient
    elif variant == "oracle-full":
        prediction = target_velocity
    else:
        prediction = _prediction(variant, model, points, gradient)
    denominator = target_velocity.square().mean().clamp_min(1e-12)
    dot = (prediction * gradient).sum(dim=1)
    cosine = torch.nn.functional.cosine_similarity(prediction, target_velocity, dim=1)
    metrics = {
        "relative_velocity_mse": float(
            (prediction - target_velocity).square().mean() / denominator
        ),
        "mean_velocity_cosine": float(cosine.mean()),
        "predicted_descent_violation_rate": float((dot >= 0).float().mean()),
        "trainable_parameters": float(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }
    if variant not in {"identity", "pem-style-residual"}:
        predicted_matrix = dense_mobility_matrix(model, points).detach()
        oracle_matrix = oracle.matrix(points).detach()
        metrics["relative_metric_mse"] = float(
            (predicted_matrix - oracle_matrix).square().mean()
            / oracle_matrix.square().mean().clamp_min(1e-12)
        )
    return metrics


def _sampling_model(
    variant: str,
    model: nn.Module,
    oracle: nn.Module,
) -> tuple[nn.Module, bool]:
    if variant == "identity":
        return IdentityMobility(), True
    if variant == "oracle-full":
        return oracle, True
    if variant == "pem-style-residual":
        return AdditiveResidualSampler(model), False
    return model, True


def main() -> None:
    args = parse_args()
    if (
        args.train_steps < 1
        or args.batch_size < 2
        or args.test_size < 2
        or args.log_every < 1
        or args.step_size <= 0
    ):
        raise ValueError("invalid inductive-bias configuration")
    seeds = _seeds(args.seeds)
    experiment_id = args.experiment_id or (
        f"inductive_bias_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {**vars(args), "seeds": seeds, "variants": EVALUATED_VARIANTS},
        repository_root=Path(__file__).resolve().parents[2],
        resume_existing=args.resume,
    )
    device = torch.device(args.device)
    aggregated: dict[str, list[float]] = {}
    all_records: list[dict[str, object]] = []

    for seed in seeds:
        target, oracle = _target(args.curvature, args.component_variance, device)
        learned = train_models(args, artifacts, target, oracle, seed)
        test_generator = torch.Generator(device=device.type).manual_seed(
            20_000_000 + seed
        )
        test_points = _training_points(target, args.test_size, test_generator)
        variant_models = {
            "identity": IdentityMobility().to(device),
            **learned,
            "oracle-full": oracle,
        }
        for variant, model in variant_models.items():
            representation = representation_metrics(
                target, oracle, variant, model, test_points
            )
            sampling_model, corrected = _sampling_model(variant, model, oracle)
            equilibrium, _ = evaluate_chain(
                args,
                target,
                sampling_model,
                corrected=corrected,
                dt=args.step_size,
                seed=seed,
            )
            record = {
                "seed": seed,
                "curvature": args.curvature,
                "variant": variant,
                "representation": representation,
                "equilibrium": equilibrium,
                "equilibrium_correction": corrected,
                "controlled_adaptation": variant
                in {"pem-style-residual", "learned-mirror"},
            }
            all_records.append(record)
            artifacts.log(
                -1,
                {**representation, **{f"equilibrium_{k}": v for k, v in equilibrium.items()}},
                phase="inductive-bias-test",
                curvature=args.curvature,
                variant=variant,
                seed=seed,
                equilibrium_correction=corrected,
            )
            for family, values in (
                ("representation", representation),
                ("equilibrium", equilibrium),
            ):
                for metric, value in values.items():
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        aggregated.setdefault(
                            f"{variant}/{family}/{metric}", []
                        ).append(float(value))

    records_path = artifacts.path / "inductive_bias_records.json"
    records_path.write_text(
        json.dumps(all_records, indent=2) + "\n", encoding="utf-8"
    )
    summary = artifacts.summarize(aggregated, status="complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
