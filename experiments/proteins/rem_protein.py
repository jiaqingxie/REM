"""Frozen-energy REM training and AAV inverse-design evaluation.

This entry point follows the public Energy Matching AAV code while making the
comparison explicit: the VAE, energy, fitness predictor, oracle, latent
initialization, guidance, repulsion, and evaluation seeds are shared, and only
the mobility changes between ``em`` and ``rem``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    # The released protein modules use script-relative imports.
    sys.path.insert(0, str(SCRIPT_DIR))

from model_proteins import Unet1DModelWrapper, VAE
from oracle import BaseCNN, eval as evaluate_sequences
from utils_proteins import Encoder, ProteinDataset, check_duplicates

from rem.geometry import (
    IdentityMobility,
    mobility_diagnostics,
    riemannian_langevin_step,
    temper_mobility,
)
from rem.networks import build_mobility
from rem.training import ema_update, rem_objective


TASK_DEFAULTS = {
    "medium": {"terminal_time": 1.7, "zeta": 0.01, "predictor": "unsmoothed"},
    "hard": {"terminal_time": 1.3, "zeta": 0.009, "predictor": "smoothed"},
}
Y_MIN = 0.0
Y_MAX = 19.5365
LATENT_DIM = 16
SEQUENCE_LENGTH = 28


class StaticProteinEnergy(nn.Module):
    """Expose the released ``forward(t, x)`` model as a scalar potential."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def potential(self, x: Tensor, t: Tensor | None = None) -> Tensor:
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return self.model.potential(x, t)

    def forward(self, x: Tensor) -> Tensor:
        return self.potential(x)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_energy(device: torch.device, checkpoint: str | Path) -> nn.Module:
    model = Unet1DModelWrapper(
        dim=28,
        channels=1,
        dim_mults=(1, 2),
        dropout=0.1,
        output_scale=1000.0,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "ema_model" in state:
        state = state["ema_model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return StaticProteinEnergy(model).to(device)


def build_vae(device: torch.device, task: str) -> VAE:
    model = VAE(input_dim=SEQUENCE_LENGTH, latent_dim=LATENT_DIM).to(device)
    state = torch.load(
        SCRIPT_DIR / "vae" / f"vae_aav_{task}.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_predictor(device: torch.device, task: str, modality: str) -> BaseCNN:
    model = BaseCNN().to(device)
    state = torch.load(
        SCRIPT_DIR / "predictor" / modality / f"predictor_aav_{task}.ckpt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        {key.replace("predictor.", ""): value for key, value in state["state_dict"].items()}
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_batch(vae: VAE, batch: Tensor) -> Tensor:
    """Return deterministic VAE means in the energy model's ``(B,16,1)`` layout."""

    mean, _ = vae.encode(batch)
    return mean.unsqueeze(-1)


def ot_batch(x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

    matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    t, x_t, target = matcher.sample_location_and_conditional_flow(x0, x1)
    return t, x_t, target


@dataclass
class TrainConfiguration:
    task: str
    mobility_kind: str
    mobility_rank: int
    seed: int
    steps: int
    batch_size: int
    learning_rate: float
    geometry_weight: float
    hidden_dim: int
    depth: int
    log_bound: float
    validation_fraction: float
    validation_batches: int
    validate_every: int
    log_every: int
    ema_decay: float
    grad_clip: float
    metric_weighted: bool
    num_workers: int
    energy_checkpoint: str
    resume_checkpoint: str | None


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Publish a checkpoint atomically on the shared filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


@torch.no_grad()
def validation_transports(
    vae: VAE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Materialize one fixed validation OT problem for checkpoint selection."""

    values = []
    for batch_index, (tokens, _) in enumerate(loader):
        if batch_index >= max_batches:
            break
        x1 = encode_batch(vae, tokens.to(device))
        x0 = torch.randn_like(x1)
        time, x_t, target = ot_batch(x0, x1)
        values.append((time.cpu(), x_t.cpu(), target.cpu()))
    return values


def validate_mobility(
    energy: nn.Module,
    mobility: nn.Module,
    transport_batches: Iterable[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    mobility.eval()
    losses = []
    violations = []
    conditions = []
    for time_cpu, x_t_cpu, target_cpu in transport_batches:
        time = time_cpu.to(device)
        x_t = x_t_cpu.to(device)
        target = target_cpu.to(device)
        breakdown, _ = rem_objective(
            energy,
            mobility,
            x_t,
            target,
            time=time,
            geometry_weight=0.0,
            metric_weighted=False,
        )
        diagnostics = mobility_diagnostics(mobility, x_t)
        losses.append(float(breakdown.transport.detach()))
        violations.append(float(breakdown.descent_violation_rate.detach()))
        conditions.append(float(diagnostics["condition_number"].mean().detach()))
    mobility.train()
    return {
        "transport_mse": float(np.mean(losses)),
        "descent_violation_rate": float(np.mean(violations)),
        "condition_number": float(np.mean(conditions)),
    }


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    energy = build_energy(device, args.energy_checkpoint)
    vae = build_vae(device, args.task)
    mobility = build_mobility(
        args.mobility,
        (LATENT_DIM, 1),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        log_bound=args.log_bound,
        rank=args.mobility_rank,
    ).to(device)
    ema_mobility = copy.deepcopy(mobility).to(device).eval()

    frame = pd.read_csv(SCRIPT_DIR / "data" / f"aav_{args.task}.csv")
    dataset = ProteinDataset(frame, "aav", args.task, Encoder(), SEQUENCE_LENGTH)
    split_generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(dataset), generator=split_generator).tolist()
    validation_size = max(args.batch_size, round(len(dataset) * args.validation_fraction))
    validation_size = min(validation_size, len(dataset) - args.batch_size)
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    train_loader = DataLoader(
        Subset(dataset, training_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    fixed_validation = validation_transports(
        vae, validation_loader, device, args.validation_batches
    )
    if not fixed_validation:
        raise RuntimeError("validation split produced no batches")

    optimizer = torch.optim.Adam(mobility.parameters(), lr=args.learning_rate)
    configuration = TrainConfiguration(
        task=args.task,
        mobility_kind=args.mobility,
        mobility_rank=args.mobility_rank,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        geometry_weight=args.geometry_weight,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        log_bound=args.log_bound,
        validation_fraction=args.validation_fraction,
        validation_batches=args.validation_batches,
        validate_every=args.validate_every,
        log_every=args.log_every,
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        metric_weighted=args.metric_weighted,
        num_workers=args.num_workers,
        energy_checkpoint=str(Path(args.energy_checkpoint).resolve()),
        resume_checkpoint=(
            str(Path(args.resume_checkpoint).resolve())
            if args.resume_checkpoint
            else None
        ),
    )
    with (output / "config.json").open("w") as handle:
        json.dump(asdict(configuration), handle, indent=2)

    start_step = 1
    best_validation = math.inf
    best_step: int | None = None
    if args.resume_checkpoint:
        resume_state = torch.load(
            args.resume_checkpoint,
            map_location=device,
            weights_only=False,
        )
        resume_config = resume_state.get("config", {})
        resume_kind = str(
            resume_config.get(
                "mobility_kind",
                resume_config.get("mobility", "diagonal"),
            )
        )
        if resume_config.get("task", args.task) != args.task:
            raise ValueError("resume checkpoint task does not match --task")
        if resume_kind != args.mobility:
            raise ValueError("resume checkpoint mobility does not match --mobility")
        mobility.load_state_dict(resume_state["mobility"], strict=True)
        ema_mobility.load_state_dict(
            resume_state.get("mobility_ema") or resume_state["mobility"],
            strict=True,
        )
        if resume_state.get("optimizer") is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
        start_step = int(resume_state["step"]) + 1
        best_validation = float(
            resume_state.get("best_validation_transport_mse", math.inf)
        )
        saved_best_step = resume_state.get("best_step")
        if saved_best_step is not None:
            best_step = int(saved_best_step)
        elif Path(args.resume_checkpoint).resolve() == (output / "best.pt").resolve():
            best_step = int(resume_state["step"])
        elif (output / "best.pt").is_file():
            # Format-v1/v2 checkpoints did not record best_step in latest.pt.
            # Recover it from the separately published validation-best file.
            legacy_best = torch.load(
                output / "best.pt", map_location="cpu", weights_only=False
            )
            best_step = int(legacy_best["step"])
        print(
            json.dumps(
                {
                    "event": "resume",
                    "checkpoint": str(Path(args.resume_checkpoint).resolve()),
                    "checkpoint_step": start_step - 1,
                    "next_step": start_step,
                    "best_validation_transport_mse": best_validation,
                    "best_step": best_step,
                    "optimizer_restored": resume_state.get("optimizer") is not None,
                }
            ),
            flush=True,
        )
    if start_step > args.steps + 1:
        raise ValueError(
            f"resume checkpoint step {start_step - 1} already reaches "
            f"requested --steps {args.steps}"
        )

    iterator = iter(train_loader)
    started = time.perf_counter()
    metrics_path = output / "metrics.jsonl"
    for step in range(start_step, args.steps + 1):
        try:
            tokens, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            tokens, _ = next(iterator)
        with torch.no_grad():
            x1 = encode_batch(vae, tokens.to(device))
            x0 = torch.randn_like(x1)
            t, x_t, target = ot_batch(x0, x1)
        optimizer.zero_grad(set_to_none=True)
        breakdown, _ = rem_objective(
            energy,
            mobility,
            x_t,
            target,
            time=t,
            geometry_weight=args.geometry_weight,
            metric_weighted=args.metric_weighted,
        )
        breakdown.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(mobility.parameters(), args.grad_clip)
        optimizer.step()
        ema_update(mobility, ema_mobility, args.ema_decay)

        if step == 1 or step % args.log_every == 0:
            row = {
                "step": step,
                **breakdown.detached_metrics(),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if step % args.validate_every == 0 or step == args.steps:
            validation = validate_mobility(
                energy,
                ema_mobility,
                fixed_validation,
                device,
            )
            row = {"step": step, "split": "validation", **validation}
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            is_best = validation["transport_mse"] < best_validation
            if is_best:
                best_validation = validation["transport_mse"]
                best_step = step
            checkpoint = {
                "format_version": 2,
                "step": step,
                "config": asdict(configuration),
                "mobility": mobility.state_dict(),
                "mobility_ema": ema_mobility.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_validation_transport_mse": best_validation,
                "best_step": best_step,
            }
            atomic_torch_save(checkpoint, output / "latest.pt")
            if is_best:
                atomic_torch_save(checkpoint, output / "best.pt")

    if best_step is None or not (output / "best.pt").is_file():
        raise RuntimeError("training completed without a validation-best checkpoint")
    summary = {
        "status": "complete",
        "best_validation_transport_mse": best_validation,
        "best_step": best_step,
        "wall_seconds": time.perf_counter() - started,
        "checkpoint": str((output / "best.pt").resolve()),
        "latest_checkpoint": str((output / "latest.pt").resolve()),
        "resumed_from": configuration.resume_checkpoint,
        "start_step": start_step,
        "last_step": args.steps,
    }
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary), flush=True)


def load_mobility(device: torch.device, checkpoint: str | Path) -> nn.Module:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    mobility_kind = str(
        config.get("mobility_kind", config.get("mobility", "diagonal"))
    )
    mobility = build_mobility(
        mobility_kind,
        (LATENT_DIM, 1),
        hidden_dim=int(config["hidden_dim"]),
        depth=int(config["depth"]),
        log_bound=float(config["log_bound"]),
        rank=int(config.get("mobility_rank", 4)),
    ).to(device)
    mobility.load_state_dict(state.get("mobility_ema") or state["mobility"], strict=True)
    mobility.eval()
    for parameter in mobility.parameters():
        parameter.requires_grad_(False)
    return mobility


def epsilon_schedule(time_value: float, cutoff: float, maximum: float) -> float:
    if time_value < cutoff:
        return 0.0
    if time_value < 1.0:
        return maximum * (time_value - cutoff) / (1.0 - cutoff)
    return maximum


def interaction_energy(x: Tensor, epsilon: float, sigma: float | None) -> Tensor:
    if sigma is None or not math.isfinite(sigma):
        return x.new_zeros(())
    flat = x.flatten(start_dim=1)
    squared_distances = torch.cdist(flat, flat, p=2).square()
    # The factor 1/2 removes double counting. The diagonal is exactly zero.
    return 0.5 * squared_distances.sum() * (epsilon / sigma**2)


def sample_latents(
    *,
    energy: nn.Module,
    mobility: nn.Module,
    vae: VAE,
    predictor: BaseCNN,
    mode: str,
    seed: int,
    n_samples: int,
    terminal_time: float,
    dt: float,
    epsilon_max: float,
    time_cutoff: float,
    zeta: float,
    sigma: float | None,
    mobility_strength: float,
    mobility_active_until: float,
    divergence_samples: int,
    divergence_method: str,
) -> Tensor:
    device = next(energy.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (n_samples, LATENT_DIM, 1), device=device, generator=generator
    )
    evaluation_mobility: nn.Module
    if mode == "em":
        evaluation_mobility = IdentityMobility().to(device)
    elif mode == "rem":
        evaluation_mobility = temper_mobility(mobility, mobility_strength).to(device)
    else:
        raise ValueError(f"unknown sampling mode: {mode}")

    steps = round(terminal_time / dt)
    for step in range(steps + 1):
        time_value = min(step * dt, terminal_time)
        epsilon = epsilon_schedule(time_value, time_cutoff, epsilon_max)
        current_mobility = (
            evaluation_mobility
            if mode == "rem" and time_value <= mobility_active_until
            else IdentityMobility().to(device)
        )
        time_tensor = torch.full(
            (n_samples,), time_value, device=device, dtype=x.dtype
        )

        def conditioned_potential(value: Tensor) -> Tensor:
            logits = vae.decode(value.squeeze(-1))
            normalized_fitness = (
                predictor.forward_soft(logits) - Y_MIN
            ) / (Y_MAX - Y_MIN)
            likelihood = 0.5 * (normalized_fitness - 1.0).square()
            values = energy.potential(value, time_tensor)
            values = values + (epsilon / zeta**2) * likelihood
            repulsion = interaction_energy(value, epsilon, sigma)
            # The step implementation sums per-sample potentials.
            return values - repulsion / n_samples

        noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)
        x = riemannian_langevin_step(
            conditioned_potential,
            current_mobility,
            x,
            dt=dt,
            epsilon=epsilon,
            include_divergence_correction=True,
            divergence_samples=divergence_samples,
            divergence_method=divergence_method,
            noise=noise,
        ).clamp(-1.0, 1.0)
    return x.detach()


def decode_and_filter(
    latent: Tensor,
    vae: VAE,
    predictor: BaseCNN,
    top_k: int,
) -> list[str]:
    with torch.no_grad():
        logits = vae.decode(latent.squeeze(-1))
        sequences = Encoder().decode(logits.argmax(dim=-1))
        _, sequences = check_duplicates(sequences)
        if not sequences:
            return []
        encoded = torch.stack([Encoder().encode(sequence) for sequence in sequences]).to(
            latent.device
        )
        fitness = predictor(encoded)
        keep = min(top_k, len(sequences))
        indices = fitness.topk(keep).indices.cpu().tolist()
    return [sequences[index] for index in indices]


def parse_sigma_grid(value: str) -> list[float | None]:
    result: list[float | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        result.append(None if item in {"none", "inf", "infinity"} else float(item))
    return result


def evaluate_curve(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = torch.device(args.device)
    defaults = TASK_DEFAULTS[args.task]
    energy = build_energy(device, args.energy_checkpoint)
    vae = build_vae(device, args.task)
    predictor = build_predictor(device, args.task, defaults["predictor"])
    mobility = load_mobility(device, args.mobility_checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = np.random.RandomState(args.seed).randint(0, 1000, size=args.eval_seeds).tolist()
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    sigmas = parse_sigma_grid(args.sigmas)
    protocol = {
        "modes": modes,
        "sigmas": sigmas,
        "n_samples": args.n_samples,
        "top_k": args.top_k,
        "eval_seeds": args.eval_seeds,
        "base_seed": args.seed,
        "dt": args.dt,
        "epsilon_max": args.epsilon_max,
        "time_cutoff": args.time_cutoff,
        "mobility_strength": args.mobility_strength,
        "mobility_active_until": args.mobility_active_until,
        "divergence_samples": args.divergence_samples,
        "divergence_method": args.divergence_method,
    }
    protocol_path = output / "protocol.json"
    metrics_path = output / "metrics.jsonl"
    if protocol_path.is_file():
        previous_protocol = json.loads(protocol_path.read_text())
        if previous_protocol != protocol:
            raise RuntimeError(
                f"refusing to resume a different evaluation protocol: "
                f"{previous_protocol!r} != {protocol!r}"
            )
    elif metrics_path.is_file() and metrics_path.stat().st_size:
        raise RuntimeError("partial metrics exist without protocol.json")
    else:
        temporary_protocol = protocol_path.with_suffix(".json.tmp")
        temporary_protocol.write_text(json.dumps(protocol, indent=2) + "\n")
        os.replace(temporary_protocol, protocol_path)

    completed: dict[tuple[str, float | None, int], dict] = {}
    if metrics_path.is_file():
        for line_number, line in enumerate(metrics_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid partial metric at {metrics_path}:{line_number}"
                ) from error
            sigma = None if row.get("sigma") is None else float(row["sigma"])
            key = (str(row.get("mode")), sigma, int(row.get("seed")))
            if row.get("task") != args.task:
                raise RuntimeError(f"partial metric task mismatch: {row!r}")
            completed[key] = row
    rows: list[dict[str, float | int | str | None]] = []

    previous_cwd = Path.cwd()
    os.chdir(SCRIPT_DIR)
    try:
        for mode in modes:
            for sigma in sigmas:
                for evaluation_seed in seeds:
                    key = (mode, sigma, int(evaluation_seed))
                    if key in completed:
                        rows.append(completed[key])
                        print(
                            json.dumps(
                                {
                                    "event": "skip-completed-evaluation",
                                    "mode": mode,
                                    "sigma": sigma,
                                    "seed": evaluation_seed,
                                }
                            ),
                            flush=True,
                        )
                        continue
                    started = time.perf_counter()
                    latent = sample_latents(
                        energy=energy,
                        mobility=mobility,
                        vae=vae,
                        predictor=predictor,
                        mode=mode,
                        seed=evaluation_seed,
                        n_samples=args.n_samples,
                        terminal_time=defaults["terminal_time"],
                        dt=args.dt,
                        epsilon_max=args.epsilon_max,
                        time_cutoff=args.time_cutoff,
                        zeta=defaults["zeta"],
                        sigma=sigma,
                        mobility_strength=args.mobility_strength,
                        mobility_active_until=args.mobility_active_until,
                        divergence_samples=args.divergence_samples,
                        divergence_method=args.divergence_method,
                    )
                    sequences = decode_and_filter(
                        latent, vae, predictor, args.top_k
                    )
                    if len(sequences) < 2:
                        raise RuntimeError("protein sampler returned fewer than two unique sequences")
                    sigma_label = "inf" if sigma is None else f"{sigma:g}"
                    sample_path = (
                        output
                        / f"samples_{args.task}_{mode}_sigma{sigma_label}_seed{evaluation_seed}.csv"
                    )
                    pd.Series(sequences).to_csv(sample_path, index=False, header=False)
                    fitness, diversity, novelty = evaluate_sequences(
                        "aav", args.task, str(sample_path.resolve())
                    )
                    row = {
                        "task": args.task,
                        "mode": mode,
                        "sigma": sigma,
                        "seed": evaluation_seed,
                        "fitness": float(fitness),
                        "diversity": float(diversity),
                        "novelty": float(novelty),
                        "unique_sequences": len(sequences),
                        "wall_seconds": time.perf_counter() - started,
                    }
                    rows.append(row)
                    with metrics_path.open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
    finally:
        os.chdir(previous_cwd)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_seed.csv", index=False)
    aggregate = (
        frame.groupby(["task", "mode", "sigma"], dropna=False)[
            ["fitness", "diversity", "novelty", "wall_seconds"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in aggregate.columns
    ]
    aggregate.to_csv(output / "figure3_curve.csv", index=False)
    summary = {
        "status": "complete",
        "task": args.task,
        "evaluation_seeds": seeds,
        "protocol": protocol,
        "rows": rows,
    }
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--task", choices=TASK_DEFAULTS, required=True)
    train_parser.add_argument("--energy-checkpoint", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--steps", type=int, default=10_000)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--geometry-weight", type=float, default=1e-3)
    train_parser.add_argument(
        "--mobility",
        choices=("diagonal", "full", "low-rank"),
        default="diagonal",
    )
    train_parser.add_argument("--mobility-rank", type=int, default=4)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--depth", type=int, default=3)
    train_parser.add_argument("--log-bound", type=float, default=1.5)
    train_parser.add_argument("--validation-fraction", type=float, default=0.1)
    train_parser.add_argument("--validation-batches", type=int, default=4)
    train_parser.add_argument("--validate-every", type=int, default=500)
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--ema-decay", type=float, default=0.999)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--metric-weighted", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--resume-checkpoint")
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.set_defaults(function=train)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--task", choices=TASK_DEFAULTS, required=True)
    evaluate_parser.add_argument("--energy-checkpoint", required=True)
    evaluate_parser.add_argument("--mobility-checkpoint", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--modes", default="em,rem")
    evaluate_parser.add_argument("--sigmas", default="inf,20,10,5")
    evaluate_parser.add_argument("--n-samples", type=int, default=512)
    evaluate_parser.add_argument("--top-k", type=int, default=128)
    evaluate_parser.add_argument("--eval-seeds", type=int, default=5)
    evaluate_parser.add_argument("--dt", type=float, default=0.01)
    evaluate_parser.add_argument("--epsilon-max", type=float, default=0.1)
    evaluate_parser.add_argument("--time-cutoff", type=float, default=0.9)
    evaluate_parser.add_argument("--mobility-strength", type=float, default=1.0)
    evaluate_parser.add_argument("--mobility-active-until", type=float, default=1.0)
    evaluate_parser.add_argument("--divergence-samples", type=int, default=1)
    evaluate_parser.add_argument(
        "--divergence-method",
        choices=("hutchinson", "exact"),
        default="hutchinson",
    )
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluate_parser.set_defaults(function=evaluate_curve)
    return parser


def main() -> None:
    args = common_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
