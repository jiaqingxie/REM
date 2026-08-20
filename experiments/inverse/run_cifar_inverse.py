"""Evaluate transfer of a frozen CIFAR REM mobility to posterior sampling."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.cifar10.evaluate_rem import load_models
from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import IdentityMobility, TemperedDiagonalMobility
from rem.inverse import (
    DownsampleOperator,
    GaussianBlurOperator,
    MaskOperator,
    center_mask,
    measurement_error,
    posterior_diversity,
    posterior_chain,
    psnr,
    random_mask,
)
from rem.metrics import effective_sample_size, split_rhat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=["center-inpaint", "random-inpaint", "superres", "deblur"], default="center-inpaint")
    parser.add_argument("--variants", default="euclidean,rem")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--posterior-draws", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--observation-sigma", type=float, default=0.05)
    parser.add_argument("--divergence-samples", type=int, default=1)
    parser.add_argument(
        "--mobility-strength",
        type=float,
        default=1.0,
        help="Geodesic REM/identity blend; 0 is exactly Euclidean and 1 is full REM",
    )
    parser.add_argument(
        "--diagnostic-thin",
        type=int,
        default=20,
        help="Save every N posterior steps for low-dimensional ESS/R-hat diagnostics",
    )
    parser.add_argument(
        "--divergence-correction", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--perceptual", action="store_true")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def dataset_loader(args):
    from torchvision import datasets, transforms

    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=False,
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
    )


def make_observation(args, target: torch.Tensor):
    device = target.device
    if args.task == "center-inpaint":
        operator = MaskOperator(center_mask((3, 32, 32), device=device)).to(device)
        observation = operator(target)
        initial = observation + (1 - operator.mask) * torch.randn_like(target)
    elif args.task == "random-inpaint":
        operator = MaskOperator(
            random_mask((3, 32, 32), observed_fraction=0.5, seed=args.seed, device=device)
        ).to(device)
        observation = operator(target)
        initial = observation + (1 - operator.mask) * torch.randn_like(target)
    elif args.task == "superres":
        operator = DownsampleOperator(4).to(device)
        observation = operator(target)
        initial = F.interpolate(observation, size=(32, 32), mode="bilinear", align_corners=False)
    else:
        operator = GaussianBlurOperator().to(device)
        observation = operator(target)
        initial = observation.clone()
    if args.observation_sigma > 0:
        observation = observation + args.observation_sigma * torch.randn_like(observation)
    return operator, observation, initial.clamp(-1, 1)


def optional_perceptual_metrics(reconstruction, target) -> dict[str, float]:
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as error:
        raise RuntimeError("perceptual metrics require torchmetrics image extras") from error
    ssim = structural_similarity_index_measure(
        reconstruction, target, data_range=2.0, reduction="elementwise_mean"
    )
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=False).to(target.device)
    return {"ssim": float(ssim), "lpips": float(lpips(reconstruction, target))}


def evaluate_variant(args, artifacts, energy, mobility, variant: str):
    device = torch.device(args.device)
    if variant == "euclidean":
        evaluation_mobility = IdentityMobility().to(device)
    elif variant == "rem":
        evaluation_mobility = TemperedDiagonalMobility(
            mobility, args.mobility_strength
        )
    else:
        raise ValueError(f"unknown inverse variant: {variant}")

    metric_values: dict[str, list[float]] = {
        "sample_psnr": [],
        "mean_psnr": [],
        "measurement_error": [],
        "diversity": [],
        "coverage_90": [],
        "posterior_mean_ess": [],
        "posterior_min_ess": [],
        "posterior_max_rhat": [],
    }
    seen = 0
    total_wall = 0.0
    first_saved = False
    for target, _ in dataset_loader(args):
        if seen >= args.samples:
            break
        target = target[: args.samples - seen].to(device)
        operator, observation, initial = make_observation(args, target)
        draws = args.posterior_draws
        expanded_target = target[:, None].expand(-1, draws, -1, -1, -1)
        expanded_observation = observation[:, None].expand(
            -1, draws, *observation.shape[1:]
        )
        expanded_initial = initial[:, None].expand(-1, draws, -1, -1, -1)
        flat_observation = expanded_observation.reshape(
            target.shape[0] * draws, *observation.shape[1:]
        )
        flat_initial = expanded_initial.reshape(target.shape[0] * draws, 3, 32, 32)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        reconstruction, trajectory = posterior_chain(
            energy,
            evaluation_mobility,
            operator,
            flat_observation,
            flat_initial,
            observation_sigma=args.observation_sigma,
            steps=args.steps,
            dt=args.dt,
            temperature=args.temperature,
            divergence_samples=args.divergence_samples,
            include_divergence_correction=args.divergence_correction,
            save_every=args.diagnostic_thin,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_wall += time.perf_counter() - start
        reconstruction = reconstruction.reshape(target.shape[0], draws, 3, 32, 32)
        posterior_mean = reconstruction.mean(dim=1)
        metric_values["sample_psnr"].extend(
            psnr(reconstruction.reshape(-1, 3, 32, 32), expanded_target.reshape(-1, 3, 32, 32)).cpu().tolist()
        )
        metric_values["mean_psnr"].extend(psnr(posterior_mean, target).cpu().tolist())
        metric_values["measurement_error"].extend(
            measurement_error(
                reconstruction.reshape(-1, 3, 32, 32),
                flat_observation,
                operator,
            ).cpu().tolist()
        )
        metric_values["diversity"].extend(posterior_diversity(reconstruction).cpu().tolist())
        lower = torch.quantile(reconstruction, 0.05, dim=1)
        upper = torch.quantile(reconstruction, 0.95, dim=1)
        coverage = ((target >= lower) & (target <= upper)).float().flatten(start_dim=1).mean(dim=1)
        metric_values["coverage_90"].extend(coverage.cpu().tolist())

        if args.diagnostic_thin > 0 and len(trajectory) >= 5:
            states = torch.stack(trajectory[1:], dim=1)
            channel_mean = states.mean(dim=(3, 4))
            channel_std = states.std(dim=(3, 4), unbiased=False)
            features = torch.cat([channel_mean, channel_std], dim=2).reshape(
                target.shape[0], draws, states.shape[1], -1
            )
            for observation_features in features:
                ess = effective_sample_size(observation_features)
                rhat = split_rhat(observation_features)
                metric_values["posterior_mean_ess"].append(float(ess.mean()))
                metric_values["posterior_min_ess"].append(float(ess.min()))
                metric_values["posterior_max_rhat"].append(float(rhat.max()))

        if args.perceptual:
            perceptual = optional_perceptual_metrics(posterior_mean, target)
            for name, value in perceptual.items():
                metric_values.setdefault(name, []).append(value)
        if not first_saved:
            from torchvision.utils import save_image

            comparison = torch.cat(
                [target[:8], initial[:8], posterior_mean[:8]], dim=0
            )
            save_image(
                (comparison.cpu() + 1) / 2,
                artifacts.path / "samples" / f"{variant}_{args.task}.png",
                nrow=min(8, target.shape[0]),
            )
            first_saved = True
        seen += target.shape[0]

    metrics = {
        name: float(torch.tensor(values).mean())
        for name, values in metric_values.items()
        if values
    }
    metrics.update(
        {
            "wall_seconds": total_wall,
            "posterior_samples_per_second": (
                seen * args.posterior_draws / max(total_wall, 1e-12)
            ),
        }
    )
    artifacts.log(-1, metrics, variant=variant, task=args.task, split="test")
    return metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    energy, mobility, checkpoint, state_selection = load_models(
        args.checkpoint, device, args.use_ema
    )
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    experiment_id = args.experiment_id or (
        f"inverse_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {
            **vars(args),
            "variants": variants,
            "training_config": checkpoint["config"],
            **state_selection,
        },
        repository_root=Path(__file__).resolve().parents[2],
    )
    results = {
        variant: evaluate_variant(args, artifacts, energy, mobility, variant)
        for variant in variants
    }
    summary_values = {
        f"{variant}/{name}": [value]
        for variant, metrics in results.items()
        for name, value in metrics.items()
    }
    print(json.dumps(artifacts.summarize(summary_values, status="complete"), indent=2))


if __name__ == "__main__":
    main()
