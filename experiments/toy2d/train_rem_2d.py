"""End-to-end 2D Energy Matching -> frozen REM -> joint REM experiment."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import torch

from rem.artifacts import RunArtifacts, seed_everything
from rem.metrics import rbf_mmd, sliced_wasserstein
from rem.networks import build_mobility
from rem.sampling import langevin_chain
from rem.synthetic import ScalarPotentialMLP
from rem.training import (
    rem_objective,
    save_rem_checkpoint,
    set_trainable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobility", default="full", choices=["constant", "scalar", "diagonal", "full"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--energy-steps", type=int, default=20_000)
    parser.add_argument("--frozen-steps", type=int, default=10_000)
    parser.add_argument("--joint-steps", type=int, default=10_000)
    parser.add_argument("--lr-energy", type=float, default=1e-3)
    parser.add_argument("--lr-mobility", type=float, default=2e-3)
    parser.add_argument("--geometry-weight", type=float, default=1e-3)
    parser.add_argument("--metric-weighted", action="store_true")
    parser.add_argument("--sample-steps", type=int, default=400)
    parser.add_argument("--sample-dt", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--eval-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _dependencies():
    try:
        from torchcfm.conditional_flow_matching import (
            ExactOptimalTransportConditionalFlowMatcher,
        )
        from torchcfm.utils import sample_8gaussians, sample_moons
    except ImportError as error:
        raise RuntimeError("toy2d REM requires the torchcfm dependency") from error
    return ExactOptimalTransportConditionalFlowMatcher, sample_8gaussians, sample_moons


def _train_stage(
    *,
    artifacts: RunArtifacts,
    stage: str,
    seed: int,
    energy,
    mobility,
    matcher,
    sample_source,
    sample_target,
    steps: int,
    batch_size: int,
    lr_energy: float,
    lr_mobility: float,
    geometry_weight: float,
    metric_weighted: bool,
    log_every: int,
) -> None:
    groups = []
    energy_parameters = [p for p in energy.parameters() if p.requires_grad]
    mobility_parameters = [p for p in mobility.parameters() if p.requires_grad]
    if energy_parameters:
        groups.append({"params": energy_parameters, "lr": lr_energy})
    if mobility_parameters:
        groups.append({"params": mobility_parameters, "lr": lr_mobility})
    if not groups or steps == 0:
        return
    optimizer = torch.optim.Adam(groups)
    for step in range(steps):
        x0 = sample_source(batch_size).to(next(energy.parameters()).device)
        x1 = sample_target(batch_size).to(x0.device)
        _, x_t, target_velocity = matcher.sample_location_and_conditional_flow(x0, x1)
        optimizer.zero_grad(set_to_none=True)
        losses, _ = rem_objective(
            energy,
            mobility,
            x_t,
            target_velocity,
            geometry_weight=geometry_weight,
            metric_weighted=metric_weighted,
        )
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(energy_parameters + mobility_parameters, 10.0)
        optimizer.step()
        if step % log_every == 0 or step + 1 == steps:
            artifacts.log(
                step,
                losses.detached_metrics(),
                stage=stage,
                seed=seed,
                split="train",
            )


def run_seed(args, artifacts: RunArtifacts, seed: int) -> dict[str, float]:
    Matcher, sample_source, sample_target = _dependencies()
    seed_everything(seed)
    device = torch.device(args.device)
    energy = ScalarPotentialMLP(2, hidden_dim=256, depth=4).to(device)
    identity = build_mobility("identity", (2,)).to(device)
    matcher = Matcher(sigma=0.0)

    set_trainable(energy, True)
    _train_stage(
        artifacts=artifacts,
        stage="energy",
        seed=seed,
        energy=energy,
        mobility=identity,
        matcher=matcher,
        sample_source=sample_source,
        sample_target=sample_target,
        steps=args.energy_steps,
        batch_size=args.batch_size,
        lr_energy=args.lr_energy,
        lr_mobility=args.lr_mobility,
        geometry_weight=0.0,
        metric_weighted=False,
        log_every=args.log_every,
    )
    energy_after_em = copy.deepcopy(energy)

    mobility = build_mobility(args.mobility, (2,), hidden_dim=128, depth=3).to(device)
    set_trainable(energy, False)
    set_trainable(mobility, True)
    _train_stage(
        artifacts=artifacts,
        stage="frozen",
        seed=seed,
        energy=energy,
        mobility=mobility,
        matcher=matcher,
        sample_source=sample_source,
        sample_target=sample_target,
        steps=args.frozen_steps,
        batch_size=args.batch_size,
        lr_energy=args.lr_energy,
        lr_mobility=args.lr_mobility,
        geometry_weight=args.geometry_weight,
        metric_weighted=args.metric_weighted,
        log_every=args.log_every,
    )

    frozen_mobility = copy.deepcopy(mobility)
    set_trainable(energy, True)
    _train_stage(
        artifacts=artifacts,
        stage="joint",
        seed=seed,
        energy=energy,
        mobility=mobility,
        matcher=matcher,
        sample_source=sample_source,
        sample_target=sample_target,
        steps=args.joint_steps,
        batch_size=args.batch_size,
        lr_energy=args.lr_energy * 0.1,
        lr_mobility=args.lr_mobility,
        geometry_weight=args.geometry_weight,
        metric_weighted=args.metric_weighted,
        log_every=args.log_every,
    )

    source = sample_source(args.eval_size).to(device)
    target = sample_target(args.eval_size).to(device)
    evaluations = {}
    for name, eval_energy, eval_mobility in (
        ("em", energy_after_em, identity),
        ("frozen", energy_after_em, frozen_mobility),
        ("joint", energy, mobility),
    ):
        generated, _ = langevin_chain(
            eval_energy,
            eval_mobility,
            source,
            steps=args.sample_steps,
            dt=args.sample_dt,
            temperature=args.temperature,
            include_divergence_correction=True,
            clamp=(-6.0, 6.0),
        )
        evaluations[f"{name}_mmd"] = float(rbf_mmd(generated, target))
        evaluations[f"{name}_sliced_wasserstein"] = float(
            sliced_wasserstein(generated, target)
        )
    artifacts.log(-1, evaluations, stage="evaluation", seed=seed, split="test")
    save_rem_checkpoint(
        artifacts.path / "checkpoints" / f"toy2d_seed{seed}.pt",
        energy=energy,
        mobility=mobility,
        optimizer=None,
        step=args.energy_steps + args.frozen_steps + args.joint_steps,
        config=vars(args),
        extra={"seed": seed, "metrics": evaluations},
    )
    return evaluations


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    experiment_id = args.experiment_id or (
        f"toy2d_rem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {**vars(args), "seeds": seeds},
        repository_root=Path(__file__).resolve().parents[2],
    )
    collected: dict[str, list[float]] = {}
    for seed in seeds:
        for name, value in run_seed(args, artifacts, seed).items():
            collected.setdefault(name, []).append(value)
    print(json.dumps(artifacts.summarize(collected, status="complete"), indent=2))


if __name__ == "__main__":
    main()
