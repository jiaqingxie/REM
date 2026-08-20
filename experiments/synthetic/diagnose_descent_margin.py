"""Descent-margin diagnostic for the controlled coordinate-warp supervision."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from experiments.cifar10.diagnose_descent_margin import summarize_batches
from experiments.synthetic.run_geometry_stress import (
    _target,
    _target_velocity,
    _training_points,
)
from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import descent_condition_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curvatures", default="0,0.4,0.8,1.2")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--samples-per-seed", type=int, default=4096)
    parser.add_argument("--component-variance", type=float, default=0.1)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_seed < 1:
        raise ValueError("samples per seed must be positive")
    curvatures = [float(value) for value in args.curvatures.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not curvatures or not seeds:
        raise ValueError("curvatures and seeds must be nonempty")
    device = torch.device(args.device)
    experiment_id = args.experiment_id or (
        f"synthetic_descent_margin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {**vars(args), "curvatures": curvatures, "seeds": seeds},
        repository_root=Path(__file__).resolve().parents[2],
    )
    records: list[dict[str, object]] = []
    for curvature in curvatures:
        for seed in seeds:
            seed_everything(seed)
            target, oracle = _target(curvature, args.component_variance, device)
            generator = torch.Generator(device=device.type).manual_seed(80_000_000 + seed)
            points = _training_points(target, args.samples_per_seed, generator)
            target_velocity = _target_velocity(target, oracle, points)
            points_for_grad = points.detach().requires_grad_(True)
            gradient = torch.autograd.grad(target(points_for_grad).sum(), points_for_grad)[0]
            diagnostic = descent_condition_diagnostics(gradient, target_velocity)
            summary = summarize_batches(
                [diagnostic["signed_dot"].detach().cpu()],
                [-diagnostic["cosine"].detach().cpu()],
                [diagnostic["active"].detach().cpu()],
            )
            record = {"curvature": curvature, "seed": seed, **summary}
            records.append(record)
            artifacts.log(seed, summary, phase="margin", curvature=curvature, seed=seed)
    result = {"status": "complete", "failure_reason": None, "records": records}
    (artifacts.path / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
