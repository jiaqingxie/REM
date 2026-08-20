"""Measure frozen-energy SPD representability on image OT supervision.

The reported normalized descent margin is

    -u^T grad(V) / (||u|| ||grad(V)||).

Positive values satisfy the pointwise condition needed for an SPD mobility to
represent the target velocity.  This script samples held-out minibatches from
the same data augmentation and minibatch-OT distribution used for mobility
training; it does not fit or load a mobility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.cifar10.train_rem import (
    build_dataset,
    build_energy,
    configure_energy_training_mode,
    configure_training_attention_backend,
)
from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import descent_condition_diagnostics
from rem.training import energy_gradient, load_energy_state


QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "imagenet32"], required=True)
    parser.add_argument("--energy-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--batches-per-seed", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-ema-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--channel-mult", default="1,2,2,2")
    parser.add_argument("--attention-resolutions", default="16")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-head-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=8)
    parser.add_argument("--output-scale", type=float, default=1000.0)
    parser.add_argument("--energy-clamp", type=float, default=None)
    parser.add_argument("--energy-width-multiplier", type=float, default=1.0)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile_map(values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor(QUANTILES, dtype=torch.float64)
    quantiles = torch.quantile(values.to(torch.float64), probabilities)
    return {
        f"q{int(round(100 * probability)):02d}": float(value)
        for probability, value in zip(QUANTILES, quantiles)
    }


def summarize_batches(
    signed_dots: list[torch.Tensor],
    normalized_margins: list[torch.Tensor],
    active_masks: list[torch.Tensor],
) -> dict[str, object]:
    signed_dot = torch.cat(signed_dots).to(torch.float64)
    margin = torch.cat(normalized_margins).to(torch.float64)
    active = torch.cat(active_masks).bool()
    active_margin = margin[active]
    active_dot = signed_dot[active]
    if active_margin.numel() == 0:
        raise RuntimeError("no active velocity/gradient pairs in diagnostic")
    return {
        "sample_count": int(margin.numel()),
        "active_count": int(active.sum()),
        "inactive_rate": float((~active).to(torch.float64).mean()),
        "descent_violation_rate": float((active_dot >= 0).to(torch.float64).mean()),
        "normalized_margin_mean": float(active_margin.mean()),
        "normalized_margin_std": float(active_margin.std(unbiased=True)),
        "normalized_margin_quantiles": _quantile_map(active_margin),
        "fraction_margin_le_0p01": float((active_margin <= 0.01).to(torch.float64).mean()),
        "fraction_margin_le_0p05": float((active_margin <= 0.05).to(torch.float64).mean()),
        "negative_dot_quantiles": _quantile_map(-active_dot),
    }


def main() -> None:
    args = parse_args()
    if args.batches_per_seed < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch counts and batch size must be positive")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    configure_training_attention_backend(device)
    # build_energy only uses mode to select the optional width multiplier.
    args.mode = "frozen"
    energy = build_energy(args).to(device)
    load_energy_state(
        energy,
        args.energy_checkpoint,
        use_ema=args.use_ema_checkpoint,
        strict=args.strict_checkpoint,
    )
    configure_energy_training_mode(energy, frozen=True)
    dataset = build_dataset(args, distributed=False, rank=0)

    try:
        from torchcfm.conditional_flow_matching import (
            ExactOptimalTransportConditionalFlowMatcher,
        )
    except ImportError as error:
        raise RuntimeError("descent-margin diagnostic requires torchcfm") from error
    matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    experiment_id = args.experiment_id or (
        f"{args.dataset}_descent_margin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    config = {
        **vars(args),
        "seeds": seeds,
        "energy_checkpoint_sha256": _sha256(args.energy_checkpoint),
        "normalized_margin": "-u^T grad(V) / (||u|| ||grad(V)||)",
        "violation_event": "u^T grad(V) >= 0 on active pairs",
        "supervision_distribution": "held-out draws from training augmentation and minibatch OT",
    }
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        config,
        repository_root=Path(__file__).resolve().parents[2],
    )

    pooled_dots: list[torch.Tensor] = []
    pooled_margins: list[torch.Tensor] = []
    pooled_active: list[torch.Tensor] = []
    per_seed: list[dict[str, object]] = []
    for seed in seeds:
        seed_everything(seed)
        loader_generator = torch.Generator().manual_seed(70_000_000 + seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            generator=loader_generator,
        )
        seed_dots: list[torch.Tensor] = []
        seed_margins: list[torch.Tensor] = []
        seed_active: list[torch.Tensor] = []
        for batch_index, (real, _) in enumerate(loader):
            if batch_index >= args.batches_per_seed:
                break
            real = real.to(device, non_blocking=True)
            noise = torch.randn_like(real)
            _, x_t, target_velocity = matcher.sample_location_and_conditional_flow(
                noise, real
            )
            gradient = energy_gradient(energy, x_t)
            diagnostic = descent_condition_diagnostics(gradient, target_velocity)
            signed_dot = diagnostic["signed_dot"].detach().cpu()
            active = diagnostic["active"].detach().cpu()
            normalized_margin = -diagnostic["cosine"].detach().cpu()
            seed_dots.append(signed_dot)
            seed_margins.append(normalized_margin)
            seed_active.append(active)
        if len(seed_dots) != args.batches_per_seed:
            raise RuntimeError(
                f"dataset exhausted after {len(seed_dots)} batches; "
                f"expected {args.batches_per_seed}"
            )
        summary = summarize_batches(seed_dots, seed_margins, seed_active)
        per_seed.append({"seed": seed, **summary})
        artifacts.log(seed, summary, phase="held-out-margin", seed=seed)
        pooled_dots.extend(seed_dots)
        pooled_margins.extend(seed_margins)
        pooled_active.extend(seed_active)

    result = {
        "status": "complete",
        "failure_reason": None,
        "dataset": args.dataset,
        "per_seed": per_seed,
        "pooled": summarize_batches(pooled_dots, pooled_margins, pooled_active),
    }
    (artifacts.path / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
