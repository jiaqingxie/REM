"""Short non-oracle OT test on the paper's exactly sampleable 2D warp.

Only independent Gaussian/base and target draws, their exact minibatch OT
assignment, and the frozen target energy enter training. No oracle velocity or
mobility is used for fitting or checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from experiments.synthetic.run_geometry_stress import DEFAULT_MEANS
from rem.artifacts import git_commit, git_worktree_identity, seed_everything
from rem.geometry import IdentityMobility, minimal_distortion_regularizer
from rem.networks import build_mobility
from rem.synthetic import WarpedGaussianMixturePotential


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curvatures", default="0,1.2")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--test-batches", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--geometry-weight", type=float, default=1e-4)
    parser.add_argument("--log-bound", type=float, default=3.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="outputs/warp_minibatch_ot_20260923")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_target(curvature: float) -> WarpedGaussianMixturePotential:
    return WarpedGaussianMixturePotential(
        torch.tensor(DEFAULT_MEANS, dtype=torch.float32),
        curvature=curvature,
        component_variance=0.1,
    )


def make_batch(
    target: WarpedGaussianMixturePotential, count: int, seed: int
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(count, 2, generator=generator)
    data = target.sample(count, generator=generator)
    cost = torch.cdist(base, data).square().numpy()
    row, column = linear_sum_assignment(cost)
    # The assignment is a permutation; preserving base order simplifies audits.
    matched = torch.empty_like(data)
    matched[torch.from_numpy(row.copy())] = data[torch.from_numpy(column.copy())]
    time_values = torch.rand(count, 1, generator=generator)
    return (1 - time_values) * base + time_values * matched, matched - base


def make_batches(
    target: WarpedGaussianMixturePotential, count: int, batch_size: int, seed: int
) -> list[tuple[Tensor, Tensor]]:
    return [make_batch(target, batch_size, seed + index) for index in range(count)]


def gradient(target: WarpedGaussianMixturePotential, points: Tensor) -> Tensor:
    with torch.enable_grad():
        differentiable = points.detach().requires_grad_(True)
        return torch.autograd.grad(target(differentiable).sum(), differentiable)[0].detach()


def evaluate(
    target: WarpedGaussianMixturePotential,
    model: torch.nn.Module,
    batches: list[tuple[Tensor, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    squared_error = 0.0
    target_energy = 0.0
    violations = 0
    pairs = 0
    positive_margin = 0.0
    for points_cpu, velocity_cpu in batches:
        points = points_cpu.to(device)
        velocity = velocity_cpu.to(device)
        energy_gradient = gradient(target, points)
        with torch.no_grad():
            prediction = -model.apply(points, energy_gradient)
            squared_error += float((prediction - velocity).square().sum())
            target_energy += float(velocity.square().sum())
            descent = (velocity * energy_gradient).sum(dim=1)
            violations += int((descent >= 0).sum())
            positive_margin += float((-descent).clamp_min(0).sum())
            pairs += len(points)
    return {
        "relative_velocity_mse": squared_error / target_energy,
        "descent_violation_fraction": violations / pairs,
        "mean_positive_descent": positive_margin / pairs,
        "pairs": pairs,
    }


def fit(
    args: argparse.Namespace,
    target: WarpedGaussianMixturePotential,
    kind: str,
    train: list[tuple[Tensor, Tensor]],
    validation: list[tuple[Tensor, Tensor]],
    test: list[tuple[Tensor, Tensor]],
    seed: int,
    device: torch.device,
    output: Path,
) -> dict:
    seed_everything(1_000_000 + seed)
    model = build_mobility(
        kind, (2,), hidden_dim=args.hidden_dim, depth=args.depth,
        log_bound=args.log_bound,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_error = float("inf")
    best_step = -1
    best_state = None
    started = time.perf_counter()
    for step, (points_cpu, velocity_cpu) in enumerate(train, 1):
        points = points_cpu.to(device)
        velocity = velocity_cpu.to(device)
        energy_gradient = gradient(target, points)
        prediction = -model.apply(points, energy_gradient)
        relative_loss = (prediction - velocity).square().sum() / velocity.square().sum()
        regularizer = minimal_distortion_regularizer(model, points)
        loss = relative_loss + args.geometry_weight * regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == len(train):
            validation_error = evaluate(target, model, validation, device)[
                "relative_velocity_mse"
            ]
            if validation_error < best_error:
                best_error = validation_error
                best_step = step
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            print(json.dumps({"curvature": target.curvature, "seed": seed,
                              "kind": kind, "step": step,
                              "validation_relative_velocity_mse": validation_error,
                              "elapsed_seconds": round(time.perf_counter() - started, 1)}),
                  flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint = output / f"c{target.curvature:g}_{kind}_seed{seed}.pt"
    torch.save({"state_dict": best_state, "curvature": target.curvature,
                "kind": kind, "seed": seed, "best_step": best_step}, checkpoint)
    return {
        "curvature": target.curvature, "seed": seed, "variant": kind,
        "train_steps": len(train), "selected_step": best_step,
        "validation_relative_velocity_mse": best_error,
        "test": evaluate(target, model, test, device),
        "train_wall_seconds": time.perf_counter() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def main() -> None:
    args = arguments()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    metadata = {
        "config": vars(args), "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "git_commit": git_commit(source.parent),
        "git_worktree_identity": git_worktree_identity(source.parents[2]),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
    }
    records = []
    for curvature in [float(value) for value in args.curvatures.split(",")]:
        target = make_target(curvature).to(device)
        for seed in [int(value) for value in args.seeds.split(",")]:
            # Separate deterministic draws for training, selection, and test.
            train = make_batches(target.cpu(), args.train_steps, args.batch_size,
                                 10_000_000 + 1_000_000 * seed)
            validation = make_batches(target.cpu(), args.validation_batches,
                                      args.batch_size, 20_000_000 + 1_000_000 * seed)
            test = make_batches(target.cpu(), args.test_batches, args.batch_size,
                                30_000_000 + 1_000_000 * seed)
            target.to(device)
            identity = IdentityMobility().to(device)
            records.append({"curvature": curvature, "seed": seed,
                            "variant": "identity", "test": evaluate(target, identity,
                            test, device)})
            for kind in ("diagonal", "full"):
                records.append(fit(args, target, kind, train, validation,
                                   test, seed, device, output))
            report = {"metadata": metadata, "records": records}
            (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"completed_curvature": curvature, "seed": seed}),
                  flush=True)


if __name__ == "__main__":
    main()
