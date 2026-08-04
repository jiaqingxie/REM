"""CIFAR-10 FID/KID/precision-recall versus wall-clock REM sweep."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rem.artifacts import RunArtifacts, seed_everything
from rem.image_metrics import ImageMetricSuite
from rem.geometry import mobility_diagnostics
from rem.networks import build_mobility
from rem.sampling import (
    linear_temperature,
    profile_langevin_chain,
    profile_step_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", default="10,20,50,100,200")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--temperature-warmup-fraction", type=float, default=0.8)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--component-profile-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--divergence-correction", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--divergence-samples", type=int, default=1)
    parser.add_argument(
        "--precision-recall", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--variant-label", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_models(path: str, device: torch.device, use_ema: bool):
    from experiments.cifar10.train_rem import build_energy

    state = torch.load(path, map_location="cpu")
    training_args = argparse.Namespace(**state["config"])
    energy = build_energy(training_args)
    mobility_kind = (
        "identity"
        if training_args.mode in {"baseline", "em-large"}
        else training_args.mobility
    )
    mobility = build_mobility(
        mobility_kind,
        (3, 32, 32),
        hidden_dim=training_args.mobility_hidden,
        depth=training_args.mobility_depth,
        log_bound=training_args.mobility_log_bound,
        rank=training_args.mobility_rank,
    )
    energy_key = "energy_ema" if use_ema and state.get("energy_ema") else "energy"
    mobility_key = "mobility_ema" if use_ema and state.get("mobility_ema") else "mobility"
    energy.load_state_dict(state[energy_key])
    mobility.load_state_dict(state[mobility_key])
    energy.to(device).eval()
    mobility.to(device).eval()
    for module in (energy, mobility):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return energy, mobility, state


def real_loader(args):
    from torchvision import datasets, transforms

    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_step_count(
    args,
    artifacts: RunArtifacts,
    energy,
    mobility,
    step_count: int,
) -> dict[str, float]:
    device = torch.device(args.device)
    suite = ImageMetricSuite(
        device=device,
        collect_precision_recall=args.precision_recall,
        kid_subset_size=min(1000, max(2, args.samples // 10)),
    )
    real_seen = 0
    for images, _ in real_loader(args):
        remaining = args.samples - real_seen
        if remaining <= 0:
            break
        suite.update(images[:remaining], real=True)
        real_seen += min(images.shape[0], remaining)

    generator = torch.Generator(device=device.type).manual_seed(args.seed)
    profile_initial = torch.randn(
        args.batch_size, 3, 32, 32, device=device, generator=generator
    )
    _, _, cold_profile = profile_langevin_chain(
        energy,
        mobility,
        profile_initial,
        steps=step_count,
        dt=args.dt,
        temperature=linear_temperature(
            args.temperature,
            warmup_fraction=args.temperature_warmup_fraction,
        ),
        include_divergence_correction=args.divergence_correction,
        divergence_samples=args.divergence_samples,
        clamp=(-1.0, 1.0),
        generator=generator,
    )
    for _ in range(args.warmup_batches):
        warmup_initial = torch.randn(
            args.batch_size, 3, 32, 32, device=device, generator=generator
        )
        profile_langevin_chain(
            energy,
            mobility,
            warmup_initial,
            steps=step_count,
            dt=args.dt,
            temperature=linear_temperature(
                args.temperature,
                warmup_fraction=args.temperature_warmup_fraction,
            ),
            include_divergence_correction=args.divergence_correction,
            divergence_samples=args.divergence_samples,
            clamp=(-1.0, 1.0),
            generator=generator,
        )
    component_profile = profile_step_components(
        energy,
        mobility,
        profile_initial,
        divergence_samples=args.divergence_samples,
        include_divergence_correction=args.divergence_correction,
        repeats=args.component_profile_repeats,
    )
    fake_seen = 0
    wall_seconds = 0.0
    peak_memory = 0
    divergence_jvps = 0
    mobility_summary: dict[str, float] = {}
    while fake_seen < args.samples:
        count = min(args.batch_size, args.samples - fake_seen)
        initial = torch.randn(
            count, 3, 32, 32, device=device, generator=generator
        )
        generated, _, profile = profile_langevin_chain(
            energy,
            mobility,
            initial,
            steps=step_count,
            dt=args.dt,
            temperature=linear_temperature(
                args.temperature,
                warmup_fraction=args.temperature_warmup_fraction,
            ),
            include_divergence_correction=args.divergence_correction,
            divergence_samples=args.divergence_samples,
            clamp=(-1.0, 1.0),
            generator=generator,
        )
        suite.update(generated, real=False)
        wall_seconds += profile.wall_seconds
        peak_memory = max(peak_memory, profile.peak_memory_bytes)
        divergence_jvps += profile.divergence_probes
        if fake_seen == 0:
            with torch.no_grad():
                diagnostics = mobility_diagnostics(mobility, generated)
                mobility_summary = {
                    "sample_mean_abs_logdet": float(
                        diagnostics["logdet"].abs().mean()
                    ),
                    "sample_mean_condition_number": float(
                        diagnostics["condition_number"].mean()
                    ),
                    "sample_max_condition_number": float(
                        diagnostics["condition_number"].max()
                    ),
                    "sample_mean_log_distance_from_identity": float(
                        diagnostics["log_distance_from_identity"].mean()
                    ),
                }
            from torchvision.utils import save_image

            save_image(
                (generated[:64].cpu() + 1) / 2,
                artifacts.path / "samples" / f"grid_steps{step_count}.png",
                nrow=8,
            )
        fake_seen += count

    metrics = suite.compute()
    metrics.update(
        {
            "steps": float(step_count),
            "nfe": float(step_count),
            "divergence_jvps": float(divergence_jvps),
            "divergence_jvps_per_image": float(divergence_jvps / args.samples),
            "cold_start_wall_seconds": cold_profile.wall_seconds,
            "sampling_wall_seconds": wall_seconds,
            "steady_state_wall_seconds": wall_seconds,
            "images_per_second": args.samples / max(wall_seconds, 1e-12),
            "peak_memory_bytes": float(peak_memory),
            **component_profile,
            **mobility_summary,
        }
    )
    artifacts.log(
        step_count,
        metrics,
        split="test",
        seed=args.seed,
        variant=args.variant_label or checkpoint_variant(args),
    )
    return metrics


def checkpoint_variant(args: argparse.Namespace) -> str:
    return Path(args.checkpoint).parent.parent.name


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    energy, mobility, checkpoint = load_models(args.checkpoint, device, args.use_ema)
    step_counts = [int(value) for value in args.steps.split(",") if value.strip()]
    experiment_id = args.experiment_id or (
        f"cifar10_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {
            **vars(args),
            "step_counts": step_counts,
            "training_config": checkpoint["config"],
        },
        repository_root=Path(__file__).resolve().parents[2],
    )
    results = {
        step_count: evaluate_step_count(
            args, artifacts, energy, mobility, step_count
        )
        for step_count in step_counts
    }
    summary_values = {
        f"steps={step_count}/{name}": [value]
        for step_count, metrics in results.items()
        for name, value in metrics.items()
    }
    summary = artifacts.summarize(summary_values, status="complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
