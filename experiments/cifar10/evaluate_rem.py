"""CIFAR-10 FID/KID/precision-recall versus wall-clock REM sweep."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rem.artifacts import RunArtifacts, seed_everything
from rem.image_metrics import ImageMetricSuite
from rem.geometry import TemperedDiagonalMobility, mobility_diagnostics
from rem.networks import build_mobility
from rem.sampling import (
    constant_temperature,
    energy_matching_temperature,
    linear_temperature,
    phase_gated_mobility,
    profile_langevin_chain,
    profile_step_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["cifar10", "imagenet32"], default="cifar10"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", default="10,20,50,100,200,325")
    parser.add_argument(
        "--t-end",
        type=float,
        default=3.25,
        help="Fixed physical terminal time for the solver sweep",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.0,
        help="Positive value overrides fixed-T dt=t_end/steps (legacy mode)",
    )
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--integrator", choices=["euler", "heun"], default="euler")
    parser.add_argument("--temperature-warmup-fraction", type=float, default=0.8)
    parser.add_argument("--time-cutoff", type=float, default=1.0)
    parser.add_argument("--temperature-ramp-end", type=float, default=1.0)
    parser.add_argument(
        "--temperature-schedule",
        choices=["official", "fraction", "constant"],
        default="official",
    )
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument(
        "--real-samples",
        type=int,
        default=0,
        help=(
            "Number of real images used for uncached metrics; zero preserves "
            "the historical behavior of matching --samples."
        ),
    )
    parser.add_argument(
        "--fid-real-stats",
        default="",
        help=(
            "Cached full-dataset FID real statistics. When supplied, FID does "
            "not update on the uncached real subset used by KID/PR."
        ),
    )
    parser.add_argument(
        "--expected-fid-real-samples",
        type=int,
        default=0,
        help="Fail unless cached/direct FID real statistics contain this count.",
    )
    parser.add_argument(
        "--kid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute KID; disable for full-real-statistics FID-only paper runs.",
    )
    parser.add_argument(
        "--image-quantization",
        choices=["round", "floor"],
        default="round",
        help=(
            "uint8 conversion used by Inception metrics; floor matches the "
            "official Energy Matching evaluator."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--component-profile-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mobility-training-seed",
        type=int,
        default=-1,
        help=(
            "Seed used to train the mobility checkpoint. This is provenance "
            "only and is deliberately separate from --seed, which controls "
            "the sampling stream."
        ),
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mobility-strength",
        type=float,
        default=1.0,
        help=(
            "Evaluation-only log-eigenvalue strength for diagonal mobility; "
            "0 is identity and 1 is the trained geometry."
        ),
    )
    parser.add_argument(
        "--mobility-active-until",
        type=float,
        default=1.0,
        help=(
            "Physical time before which learned mobility is active. The 1.0 "
            "default matches the trained conditional-flow interval and uses "
            "identity afterwards; a negative value enables the legacy "
            "full-chain REM ablation."
        ),
    )
    parser.add_argument(
        "--use-energy-ema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the energy state selection. By default frozen REM uses the "
            "exact frozen energy while trainable-energy runs follow --use-ema."
        ),
    )
    parser.add_argument(
        "--use-mobility-ema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the mobility state selection; defaults to --use-ema.",
    )
    parser.add_argument(
        "--divergence-correction", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--divergence-samples", type=int, default=1)
    parser.add_argument(
        "--precision-recall", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--clamp-every-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Legacy diagnostic only. The paper protocol matches upstream and "
            "clamps once at the terminal state."
        ),
    )
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--variant-label", default="")
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=100,
        help="Print machine-readable sampling progress every N batches; 0 disables.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_models(
    path: str,
    device: torch.device,
    use_ema: bool,
    use_energy_ema: bool | None = None,
    use_mobility_ema: bool | None = None,
    mobility_strength: float = 1.0,
):
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
    if use_energy_ema is None:
        # A frozen energy must remain exactly equal to the imported official
        # checkpoint. Repeated EMA arithmetic on an unchanged FP32 state can
        # introduce small cumulative rounding drift, so frozen runs use the
        # raw (unchanged) state unless explicitly overridden for diagnostics.
        use_energy_ema = use_ema and training_args.mode != "frozen"
    if use_mobility_ema is None:
        use_mobility_ema = use_ema
    energy_key = (
        "energy_ema" if use_energy_ema and state.get("energy_ema") else "energy"
    )
    mobility_key = (
        "mobility_ema"
        if use_mobility_ema and state.get("mobility_ema")
        else "mobility"
    )
    energy.load_state_dict(state[energy_key])
    mobility.load_state_dict(state[mobility_key])
    if mobility_strength != 1.0:
        if training_args.mobility not in {
            "identity",
            "constant",
            "scalar",
            "diagonal",
            "unfixed-diagonal",
        }:
            raise ValueError("--mobility-strength supports diagonal mobilities only")
        mobility = TemperedDiagonalMobility(mobility, mobility_strength)
    energy.to(device).eval()
    mobility.to(device).eval()
    for module in (energy, mobility):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return energy, mobility, state, {
        "energy_state_key": energy_key,
        "mobility_state_key": mobility_key,
        "mobility_strength": mobility_strength,
    }


def real_loader(args):
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(
            root=args.data_root,
            train=True,
            download=False,
            transform=transform,
        )
    else:
        from experiments.imagenet.dataset_imagenet32 import ImageNet32Dataset

        dataset = ImageNet32Dataset(
            split="train", root=args.data_root, transform=transform
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
    step_dt = args.dt if args.dt > 0 else args.t_end / step_count
    effective_t_end = step_count * step_dt
    if args.temperature_schedule == "official":
        temperature = energy_matching_temperature(
            args.temperature,
            dt=step_dt,
            time_cutoff=args.time_cutoff,
            ramp_end=args.temperature_ramp_end,
        )
    elif args.temperature_schedule == "fraction":
        temperature = linear_temperature(
            args.temperature,
            warmup_fraction=args.temperature_warmup_fraction,
        )
    else:
        temperature = constant_temperature(args.temperature)
    mobility_schedule = (
        phase_gated_mobility(
            mobility,
            dt=step_dt,
            active_until=args.mobility_active_until,
        )
        if args.mobility_active_until >= 0
        else None
    )
    suite = ImageMetricSuite(
        device=device,
        collect_precision_recall=args.precision_recall,
        kid_subset_size=min(1000, max(2, args.samples // 10)),
        compute_kid=args.kid,
        quantization=args.image_quantization,
    )
    cached_real_metadata = None
    if args.fid_real_stats:
        cached_real_metadata = suite.load_fid_real_statistics(args.fid_real_stats)
        cached_quantization = cached_real_metadata.get("quantization")
        if cached_quantization != args.image_quantization:
            raise ValueError(
                "cached FID quantization does not match --image-quantization: "
                f"{cached_quantization!r} != {args.image_quantization!r}"
            )
        cached_dataset = cached_real_metadata.get("dataset")
        if cached_dataset != args.dataset:
            raise ValueError(
                f"cached FID dataset {cached_dataset!r} does not match {args.dataset!r}"
            )
    real_limit = args.real_samples if args.real_samples > 0 else args.samples
    real_seen = 0
    need_real_subset = args.kid or args.precision_recall or not args.fid_real_stats
    if need_real_subset:
        for images, _ in real_loader(args):
            remaining = real_limit - real_seen
            if remaining <= 0:
                break
            count = min(images.shape[0], remaining)
            suite.update(
                images[:count],
                real=True,
                update_fid=not bool(args.fid_real_stats),
                update_kid=args.kid,
            )
            real_seen += count
    fid_real_samples = suite.fid_real_samples
    if (
        args.expected_fid_real_samples > 0
        and fid_real_samples != args.expected_fid_real_samples
    ):
        raise ValueError(
            "FID real sample count mismatch: "
            f"expected {args.expected_fid_real_samples}, got {fid_real_samples}"
        )

    profile_generator = torch.Generator(device=device.type).manual_seed(
        args.seed + 1_000_003 * step_count
    )
    profile_initial = torch.randn(
        args.batch_size, 3, 32, 32, device=device, generator=profile_generator
    )
    _, _, cold_profile = profile_langevin_chain(
        energy,
        mobility,
        profile_initial,
        steps=step_count,
        dt=step_dt,
        temperature=temperature,
        include_divergence_correction=args.divergence_correction,
        divergence_samples=args.divergence_samples,
        clamp=(-1.0, 1.0) if args.clamp_every_step else None,
        generator=profile_generator,
        integrator=args.integrator,
        mobility_schedule=mobility_schedule,
    )
    for _ in range(args.warmup_batches):
        warmup_initial = torch.randn(
            args.batch_size, 3, 32, 32, device=device, generator=profile_generator
        )
        profile_langevin_chain(
            energy,
            mobility,
            warmup_initial,
            steps=step_count,
            dt=step_dt,
            temperature=temperature,
            include_divergence_correction=args.divergence_correction,
            divergence_samples=args.divergence_samples,
            clamp=(-1.0, 1.0) if args.clamp_every_step else None,
            generator=profile_generator,
            integrator=args.integrator,
            mobility_schedule=mobility_schedule,
        )
    component_profile = profile_step_components(
        energy,
        mobility,
        profile_initial,
        divergence_samples=args.divergence_samples,
        include_divergence_correction=(
            args.divergence_correction and cold_profile.divergence_probes > 0
        ),
        repeats=args.component_profile_repeats,
    )
    fake_seen = 0
    wall_seconds = 0.0
    evaluation_start = time.perf_counter()
    peak_memory = 0
    energy_gradient_batched_calls = 0
    divergence_batched_jvp_calls = 0
    mobility_summary: dict[str, float] = {}
    sample_generator = torch.Generator(device=device.type).manual_seed(args.seed)
    batch_index = 0
    while fake_seen < args.samples:
        count = min(args.batch_size, args.samples - fake_seen)
        initial = torch.randn(
            count, 3, 32, 32, device=device, generator=sample_generator
        )
        generated, _, profile = profile_langevin_chain(
            energy,
            mobility,
            initial,
            steps=step_count,
            dt=step_dt,
            temperature=temperature,
            include_divergence_correction=args.divergence_correction,
            divergence_samples=args.divergence_samples,
            clamp=(-1.0, 1.0) if args.clamp_every_step else None,
            generator=sample_generator,
            integrator=args.integrator,
            mobility_schedule=mobility_schedule,
        )
        generated = generated.clamp(-1.0, 1.0)
        suite.update(generated, real=False)
        wall_seconds += profile.wall_seconds
        peak_memory = max(peak_memory, profile.peak_memory_bytes)
        energy_gradient_batched_calls += profile.energy_grad_evaluations
        divergence_batched_jvp_calls += profile.divergence_probes
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
        batch_index += 1
        if args.progress_every_batches > 0 and (
            batch_index % args.progress_every_batches == 0
            or fake_seen == args.samples
        ):
            print(
                json.dumps(
                    {
                        "event": "sampling_progress",
                        "steps": step_count,
                        "samples_complete": fake_seen,
                        "samples_total": args.samples,
                        "mobility_active_until": args.mobility_active_until,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    metrics = suite.compute()
    evaluation_wall_seconds = time.perf_counter() - evaluation_start
    metrics.update(
        {
            "steps": float(step_count),
            "dt": float(step_dt),
            "t_end": float(effective_t_end),
            "nfe": float(step_count * (2 if args.integrator == "heun" else 1)),
            "energy_gradient_evaluations_per_image": float(
                cold_profile.energy_grad_evaluations
            ),
            "energy_gradient_batched_calls": float(energy_gradient_batched_calls),
            "divergence_jvps_per_image": float(cold_profile.divergence_probes),
            "divergence_batched_jvp_calls": float(divergence_batched_jvp_calls),
            "cold_start_wall_seconds": cold_profile.wall_seconds,
            "sampling_wall_seconds": wall_seconds,
            "steady_state_wall_seconds": wall_seconds,
            "evaluation_wall_seconds": evaluation_wall_seconds,
            "images_per_second": args.samples / max(wall_seconds, 1e-12),
            "peak_memory_bytes": float(peak_memory),
            "mobility_active_until": float(args.mobility_active_until),
            "fake_samples": float(args.samples),
            "fid_real_samples": float(fid_real_samples),
            "kid_real_samples": float(real_seen if args.kid else 0),
            "fid_real_stats_cached": float(bool(args.fid_real_stats)),
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
    if args.t_end <= 0 or args.dt < 0:
        raise ValueError("t_end must be positive and dt must be nonnegative")
    if args.mobility_strength < 0:
        raise ValueError("mobility strength must be nonnegative")
    if args.mobility_active_until > args.t_end:
        raise ValueError("mobility active interval cannot exceed t_end")
    if args.progress_every_batches < 0:
        raise ValueError("progress interval must be nonnegative")
    if args.real_samples < 0 or args.expected_fid_real_samples < 0:
        raise ValueError("real sample counts must be nonnegative")
    if args.precision_recall and args.fid_real_stats and args.real_samples == 0:
        # This is valid and intentionally retains the historical 50k PR
        # subset, but make the split explicit in the resolved configuration.
        args.real_samples = args.samples
    seed_everything(args.seed)
    device = torch.device(args.device)
    energy, mobility, checkpoint, state_selection = load_models(
        args.checkpoint,
        device,
        args.use_ema,
        args.use_energy_ema,
        args.use_mobility_ema,
        args.mobility_strength,
    )
    step_counts = [int(value) for value in args.steps.split(",") if value.strip()]
    experiment_id = args.experiment_id or (
        f"{args.dataset}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        {
            **vars(args),
            "step_counts": step_counts,
            "training_config": checkpoint["config"],
            **state_selection,
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
