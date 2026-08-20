"""Paper-protocol evaluation of the official Energy Matching CIFAR checkpoint.

This keeps the upstream Stratonovich Heun sampler (including its temperature
schedule) while logging the same FID/KID/precision/recall and systems metrics
used by the REM evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rem.artifacts import RunArtifacts, seed_everything
from rem.image_metrics import ImageMetricSuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--t-end", type=float, default=3.25)
    parser.add_argument(
        "--steps",
        default="10,20,50,100,200,325",
        help="Fixed-terminal-time Heun interval counts",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.0,
        help="Legacy single-run mode; used only when --steps is empty",
    )
    parser.add_argument("--epsilon-max", type=float, default=0.01)
    parser.add_argument("--time-cutoff", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision-recall", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="outputs/official_eval")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epsilon(t: float, *, epsilon_max: float, time_cutoff: float) -> float:
    if t < time_cutoff:
        return 0.0
    if t < 1.0:
        return epsilon_max * (t - time_cutoff) / (1.0 - time_cutoff)
    return epsilon_max


def build_model(device: torch.device):
    # Import from the vendored upstream checkout, whose implementation is the
    # exact architecture used to produce the public checkpoint.
    upstream = Path(
        os.environ.get(
            "ENERGY_MATCHING_ROOT",
            str(Path(__file__).resolve().parents[6] / "external" / "EnergyMatching"),
        )
    )
    sys.path.insert(0, str(upstream / "experiments" / "cifar10"))
    from network_transformer_vit import EBViTModelWrapper

    model = EBViTModelWrapper(
        dim=(3, 32, 32),
        num_channels=128,
        num_res_blocks=2,
        channel_mult=[1, 2, 2, 2],
        attention_resolutions="16",
        num_heads=4,
        num_head_channels=64,
        dropout=0.1,
        output_scale=1000.0,
        energy_clamp=None,
        patch_size=4,
        embed_dim=384,
        transformer_nheads=4,
        transformer_nlayers=8,
    ).to(device).eval()
    return model


def solve_heun(
    model,
    x: torch.Tensor,
    *,
    t_end: float,
    steps: int,
    epsilon_max: float,
    time_cutoff: float,
) -> tuple[torch.Tensor, int, int]:
    import torchsde

    if steps < 1:
        raise ValueError("Heun interval count must be positive")
    batch = x.shape[0]
    original_shape = x.shape
    flattened = x.reshape(batch, -1)
    drift_calls = 0
    diffusion_calls = 0

    class FlattenSDE(torchsde.SDEStratonovich):
        def __init__(self):
            super().__init__(noise_type="diagonal")

        def f(self, t, y):
            nonlocal drift_calls
            drift_calls += 1
            image = y.reshape(original_shape)
            time_batch = t.expand(batch).to(device=y.device, dtype=image.dtype)
            return model(time_batch, image).reshape(batch, -1)

        def g(self, t, y):
            nonlocal diffusion_calls
            diffusion_calls += 1
            value = epsilon(float(t), epsilon_max=epsilon_max, time_cutoff=time_cutoff)
            if value <= 0:
                return torch.zeros_like(y)
            return torch.full_like(y, (2.0 * value) ** 0.5)

    dt = t_end / steps
    ts = torch.linspace(0.0, t_end, steps + 1, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        solution = torchsde.sdeint(FlattenSDE(), flattened, ts, method="heun", dt=dt)
    generated = solution[-1].reshape(original_shape).clamp(-1.0, 1.0)
    return generated, drift_calls, diffusion_calls


def real_loader(args: argparse.Namespace):
    from torchvision import datasets, transforms

    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True)


def evaluate_step_count(
    args: argparse.Namespace,
    artifacts: RunArtifacts,
    model,
    step_count: int,
) -> dict[str, float]:
    device = torch.device(args.device)
    suite = ImageMetricSuite(device=device, collect_precision_recall=args.precision_recall,
                             kid_subset_size=min(1000, max(2, args.samples // 10)))
    real_seen = 0
    for images, _ in real_loader(args):
        if real_seen >= args.samples:
            break
        count = min(images.shape[0], args.samples - real_seen)
        suite.update(images[:count].to(device, non_blocking=True), real=True)
        real_seen += count

    generator = torch.Generator(device=device.type).manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    fake_seen = 0
    first = None
    drift_batched_calls = 0
    diffusion_batched_calls = 0
    batch_count = 0
    while fake_seen < args.samples:
        count = min(args.batch_size, args.samples - fake_seen)
        initial = torch.randn(
            count, 3, 32, 32, device=device, generator=generator
        )
        generated, drift_calls, diffusion_calls = solve_heun(
            model,
            initial,
            t_end=args.t_end,
            steps=step_count,
            epsilon_max=args.epsilon_max,
            time_cutoff=args.time_cutoff,
        )
        drift_batched_calls += drift_calls
        diffusion_batched_calls += diffusion_calls
        batch_count += 1
        if first is None:
            first = generated[:64].detach().cpu()
        suite.update(generated, real=False)
        fake_seen += count
        if fake_seen % max(args.batch_size * 100, 1) == 0 or fake_seen == args.samples:
            print(
                json.dumps(
                    {
                        "steps": step_count,
                        "fake_seen": fake_seen,
                        "samples": args.samples,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory = 0
    wall = time.perf_counter() - start
    metrics = suite.compute()
    metrics.update({
        "samples": float(fake_seen),
        "steps": float(step_count),
        "t_end": args.t_end,
        "dt": args.t_end / step_count,
        "nfe_per_image": float(drift_batched_calls / max(batch_count, 1)),
        "drift_batched_calls": float(drift_batched_calls),
        "diffusion_batched_calls": float(diffusion_batched_calls),
        "sampling_wall_seconds": wall,
        "images_per_second": fake_seen / max(wall, 1e-12),
        "peak_memory_bytes": float(peak_memory),
    })
    artifacts.log(
        step_count,
        metrics,
        split="test",
        seed=args.seed,
        variant="official_em_heun",
    )
    if first is not None:
        from torchvision.utils import save_image

        save_image(
            (first + 1) / 2,
            artifacts.path / "samples" / f"grid_steps{step_count}.png",
            nrow=8,
        )
    return metrics


def main() -> None:
    args = parse_args()
    if args.t_end <= 0 or args.dt < 0:
        raise ValueError("t_end must be positive and dt must be nonnegative")
    step_counts = [int(value) for value in args.steps.split(",") if value.strip()]
    if not step_counts:
        if args.dt <= 0:
            raise ValueError("provide --steps or a positive legacy --dt")
        step_counts = [max(1, round(args.t_end / args.dt))]
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema_model" not in checkpoint or "net_model" not in checkpoint:
        raise KeyError(f"official checkpoint keys are {sorted(checkpoint)}")
    model = build_model(device)
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    model.eval()
    experiment_id = args.experiment_id or f"official_em_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifacts = RunArtifacts(args.output, experiment_id,
                             {**vars(args), "step_counts": step_counts,
                              "checkpoint_sha256": sha256(checkpoint_path),
                              "checkpoint_keys": sorted(checkpoint)},
                             repository_root=Path(__file__).resolve().parents[2])
    results = {
        step_count: evaluate_step_count(args, artifacts, model, step_count)
        for step_count in step_counts
    }
    summary_values = {
        f"steps={step_count}/{name}": [value]
        for step_count, metrics in results.items()
        for name, value in metrics.items()
    }
    summary = artifacts.summarize(summary_values, status="complete")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
