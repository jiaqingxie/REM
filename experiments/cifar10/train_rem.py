"""Distributed CIFAR-10 training for baseline, frozen, and joint REM.

Launch with ``torchrun`` even for a single GPU so multi-GPU semantics remain
identical:

    torchrun --standalone --nproc_per_node=2 -m experiments.cifar10.train_rem \
        --mode frozen --energy-checkpoint /path/to/em.pt --mobility diagonal
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import (
    descent_condition_diagnostics,
    minimal_distortion_regularizer,
    mobility_diagnostics,
)
from rem.networks import REMModel, build_mobility
from rem.sampling import langevin_chain
from rem.training import (
    ema_update,
    energy_gradient,
    load_energy_state,
    load_rem_checkpoint,
    save_rem_checkpoint,
    set_trainable,
    trainable_parameter_count,
    trimmed_mean,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "imagenet32"],
        default="cifar10",
        help="32x32 image dataset used for frozen-energy mobility training",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "continue-energy", "frozen", "joint", "em-large"],
        default="frozen",
    )
    parser.add_argument(
        "--mobility",
        choices=[
            "identity",
            "constant",
            "scalar",
            "diagonal",
            "unfixed-diagonal",
            "low-rank",
            "structured",
        ],
        default="diagonal",
    )
    parser.add_argument("--energy-checkpoint", default="")
    parser.add_argument(
        "--rem-init-checkpoint",
        default="",
        help="Frozen REM checkpoint used to initialize staged joint refinement",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--use-ema-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--data-root", default=os.environ.get("CIFAR10_PATH", "./data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=150_000)
    parser.add_argument(
        "--max-training-seconds",
        type=float,
        default=0.0,
        help="Positive wall-clock budget for compute-matched training",
    )
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-energy", type=float, default=1.2e-3)
    parser.add_argument("--lr-mobility", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--transport-weight", type=float, default=1.0)
    parser.add_argument("--geometry-weight", type=float, default=1e-3)
    parser.add_argument("--trust-region-weight", type=float, default=1e-3)
    parser.add_argument("--metric-weighted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--negative-steps", type=int, default=0)
    parser.add_argument("--negative-dt", type=float, default=0.01)
    parser.add_argument("--negative-temperature", type=float, default=0.01)
    parser.add_argument("--negative-trim-fraction", type=float, default=0.1)
    parser.add_argument("--divergence-samples", type=int, default=1)
    parser.add_argument(
        "--divergence-correction", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--mobility-hidden", type=int, default=32)
    parser.add_argument("--mobility-depth", type=int, default=3)
    parser.add_argument("--mobility-rank", type=int, default=4)
    parser.add_argument("--mobility-log-bound", type=float, default=1.5)
    parser.add_argument("--energy-width-multiplier", type=float, default=1.0)
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def distributed_context() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def configure_training_attention_backend(device: torch.device) -> None:
    """Force a CUDA attention backend with parameter-gradient support.

    The 25.06 image ships a PyTorch build where the efficient SDPA backward
    kernel is not implemented for the Transformer blocks used by the official
    CIFAR energy.  Math SDPA is slower but fully differentiable and is required
    for reproducible REM training.
    """

    if device.type != "cuda":
        return
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return
    if hasattr(cuda_backends, "enable_flash_sdp"):
        # Flash SDPA has a differentiable backward in the 25.06 image and is
        # substantially faster than the math fallback for this ViT.
        cuda_backends.enable_flash_sdp(True)
    if hasattr(cuda_backends, "enable_mem_efficient_sdp"):
        cuda_backends.enable_mem_efficient_sdp(False)
    if hasattr(cuda_backends, "enable_math_sdp"):
        cuda_backends.enable_math_sdp(True)


def build_energy(args: argparse.Namespace) -> nn.Module:
    try:
        from experiments.cifar10.network_transformer_vit import EBViTModelWrapper
    except ImportError as error:
        raise RuntimeError("CIFAR REM requires torchcfm and torchvision dependencies") from error
    multiplier = args.energy_width_multiplier
    if args.mode == "em-large" and multiplier <= 1:
        multiplier = 1.1
    # torchcfm's upstream UNet uses GroupNorm(32, channels), so every
    # resolution's base width must be divisible by 32.  Aligning only to 8
    # makes EM-Large capacity searches fail for otherwise valid multipliers.
    channels = max(32, int(round(args.num_channels * multiplier / 32)) * 32)
    embed_dim = max(
        args.transformer_heads,
        int(round(args.embed_dim * multiplier / args.transformer_heads))
        * args.transformer_heads,
    )
    return EBViTModelWrapper(
        dim=(3, 32, 32),
        num_channels=channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=[int(value) for value in args.channel_mult.split(",")],
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        dropout=args.dropout,
        output_scale=args.output_scale,
        energy_clamp=args.energy_clamp,
        patch_size=4,
        embed_dim=embed_dim,
        transformer_nheads=args.transformer_heads,
        transformer_nlayers=args.transformer_layers,
    )


def build_dataset(args, distributed: bool, rank: int):
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if distributed and rank != 0:
        dist.barrier()
    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(
            root=args.data_root,
            train=True,
            download=(not distributed or rank == 0),
            transform=transform,
        )
    else:
        from experiments.imagenet.dataset_imagenet32 import ImageNet32Dataset

        dataset = ImageNet32Dataset(
            split="train",
            root=args.data_root,
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                    ),
                ]
            ),
        )
    if distributed and rank == 0:
        dist.barrier()
    return dataset


def batch_stream(loader: DataLoader, sampler: DistributedSampler | None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for images, _ in loader:
            yield images
        epoch += 1


def flow_loss(
    prediction: Tensor,
    target: Tensor,
    mobility: nn.Module,
    x: Tensor,
    *,
    metric_weighted: bool,
) -> Tensor:
    residual = prediction - target
    if not metric_weighted:
        return residual.square().mean()
    inverse = mobility.inverse_apply(x, residual)
    return (
        residual.reshape(residual.shape[0], -1)
        * inverse.reshape(inverse.shape[0], -1)
    ).mean()


def configure_energy_training_mode(energy: nn.Module, *, frozen: bool) -> None:
    """Configure both gradients and stochastic-layer mode for the energy.

    Freezing parameters alone does not disable Dropout.  Frozen REM must see
    exactly the same deterministic official energy field during mobility
    training that is used at evaluation time.
    """

    set_trainable(energy, not frozen)
    if frozen:
        energy.eval()
    else:
        energy.train()


def main() -> None:
    args = parse_args()
    if args.max_training_seconds < 0:
        raise ValueError("max training seconds must be nonnegative")
    if args.mode == "frozen" and not args.energy_checkpoint and not args.resume:
        raise ValueError("frozen mode requires --energy-checkpoint or --resume")
    if (
        args.mode == "continue-energy"
        and not args.energy_checkpoint
        and not args.resume
    ):
        raise ValueError(
            "continue-energy mode requires the same pretrained checkpoint as REM"
        )
    if args.mode == "joint" and not args.rem_init_checkpoint and not args.resume:
        raise ValueError(
            "joint mode requires --rem-init-checkpoint from the frozen stage or --resume"
        )
    distributed, rank, local_rank, world_size, device = distributed_context()
    configure_training_attention_backend(device)
    if args.global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    local_batch = args.global_batch_size // world_size
    seed_everything(args.seed + rank)

    experiment_id = args.experiment_id or (
        f"{args.dataset}_{args.mode}_{args.mobility}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = None
    if rank == 0:
        artifacts = RunArtifacts(
            args.output,
            experiment_id,
            {**vars(args), "world_size": world_size, "local_batch_size": local_batch},
            repository_root=Path(__file__).resolve().parents[2],
            resume_existing=bool(args.resume),
        )

    dataset = build_dataset(args, distributed, rank)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    batches = batch_stream(loader, sampler)

    energy = build_energy(args).to(device)
    upstream_training_state = None
    if args.energy_checkpoint:
        upstream_training_state = load_energy_state(
            energy,
            args.energy_checkpoint,
            # True continuation restores the raw trainable network together
            # with its optimizer and scheduler; evaluation still uses EMA.
            use_ema=(
                args.use_ema_checkpoint and args.mode != "continue-energy"
            ),
            strict=args.strict_checkpoint,
        )
    mobility_kind = (
        "identity"
        if args.mode in {"baseline", "continue-energy", "em-large"}
        else args.mobility
    )
    mobility = build_mobility(
        mobility_kind,
        (3, 32, 32),
        hidden_dim=args.mobility_hidden,
        depth=args.mobility_depth,
        log_bound=args.mobility_log_bound,
        rank=args.mobility_rank,
    ).to(device)
    if args.rem_init_checkpoint:
        load_rem_checkpoint(
            args.rem_init_checkpoint,
            energy=energy,
            mobility=mobility,
            strict=args.strict_checkpoint,
        )
    configure_energy_training_mode(energy, frozen=args.mode == "frozen")
    set_trainable(mobility, mobility_kind != "identity")

    anchor_energy = copy.deepcopy(energy).eval()
    set_trainable(anchor_energy, False)
    ema_energy = copy.deepcopy(energy).eval()
    ema_mobility = copy.deepcopy(mobility).eval()
    composite = REMModel(energy, mobility).to(device)
    trainable = [parameter for parameter in composite.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("selected mode has no trainable parameters")
    optimizer_groups = []
    energy_parameters = [parameter for parameter in energy.parameters() if parameter.requires_grad]
    mobility_parameters = [parameter for parameter in mobility.parameters() if parameter.requires_grad]
    if energy_parameters:
        optimizer_groups.append({"params": energy_parameters, "lr": args.lr_energy})
    if mobility_parameters:
        optimizer_groups.append({"params": mobility_parameters, "lr": args.lr_mobility})
    optimizer = torch.optim.Adam(optimizer_groups, betas=(0.9, 0.95))

    def warmup(step: int) -> float:
        return min(1.0, (step + 1) / max(args.warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup)
    source_energy_step = -1
    if args.mode == "continue-energy" and not args.resume:
        if not isinstance(upstream_training_state, dict):
            raise ValueError("continued EM requires a full upstream training checkpoint")
        missing = [
            key
            for key in ("optim", "sched", "ema_model", "step")
            if key not in upstream_training_state
        ]
        if missing:
            raise ValueError(
                "continued EM checkpoint is missing training state: "
                + ", ".join(missing)
            )
        ema_energy.load_state_dict(
            upstream_training_state["ema_model"], strict=args.strict_checkpoint
        )
        optimizer.load_state_dict(upstream_training_state["optim"])
        scheduler.load_state_dict(upstream_training_state["sched"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, Tensor):
                    optimizer_state[key] = value.to(device)
        source_energy_step = int(upstream_training_state["step"])
    start_step = 0
    if args.resume:
        state = load_rem_checkpoint(
            args.resume,
            energy=energy,
            mobility=mobility,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_energy=ema_energy,
            ema_mobility=ema_mobility,
        )
        start_step = int(state["step"]) + 1
        if args.mode == "frozen":
            # Frozen REM promises to preserve the imported energy exactly.
            # Older checkpoints updated an unchanged FP32 energy EMA every
            # step, allowing rounding drift to accumulate. Repair that state
            # on resume; the mobility EMA remains a genuine learned EMA.
            ema_energy.load_state_dict(energy.state_dict())

    if distributed:
        composite_ddp: nn.Module = DDP(
            composite,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.mode == "frozen" or args.contrastive_weight == 0,
        )
    else:
        composite_ddp = composite

    try:
        from torchcfm.conditional_flow_matching import (
            ExactOptimalTransportConditionalFlowMatcher,
        )
    except ImportError as error:
        raise RuntimeError("CIFAR REM requires torchcfm") from error
    matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    if rank == 0:
        assert artifacts is not None
        artifacts.log(
            start_step,
            {
                "energy_parameters": trainable_parameter_count(energy),
                "mobility_parameters": trainable_parameter_count(mobility),
                "total_trainable_parameters": trainable_parameter_count(composite),
                "energy_eval_mode": float(not energy.training),
                "source_energy_step": float(source_energy_step),
            },
            split="setup",
            mode=args.mode,
        )

    training_start = time.perf_counter()
    last_time = training_start
    last_logged_step = start_step - 1
    last_step = start_step - 1

    def reduced(value, *, maximum: bool = False) -> float:
        tensor = torch.as_tensor(value, device=device, dtype=torch.float64).detach()
        if distributed:
            operation = dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM
            dist.all_reduce(tensor, op=operation)
            if not maximum:
                tensor = tensor / world_size
        return float(tensor.cpu())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step, args.total_steps):
        last_step = step
        real_transport = next(batches).to(device, non_blocking=True)
        real_cd = next(batches).to(device, non_blocking=True)
        noise = torch.randn_like(real_transport)
        t, x_t, target_velocity = matcher.sample_location_and_conditional_flow(
            noise, real_transport
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = composite_ddp(t, x_t)
        transport = flow_loss(
            prediction,
            target_velocity,
            mobility,
            x_t,
            metric_weighted=args.metric_weighted and mobility_kind != "identity",
        )
        geometry = minimal_distortion_regularizer(mobility, x_t)
        contrastive = transport.new_zeros(())
        positive_mean = transport.new_zeros(())
        negative_mean = transport.new_zeros(())

        if args.contrastive_weight > 0:
            if args.negative_steps < 1:
                raise ValueError("contrastive training requires --negative-steps > 0")
            half = real_cd.shape[0] // 2
            negative_init = torch.cat(
                [real_cd[:half], torch.randn_like(real_cd[half:])], dim=0
            )
            negative, _ = langevin_chain(
                energy,
                mobility,
                negative_init,
                steps=args.negative_steps,
                dt=args.negative_dt,
                temperature=args.negative_temperature,
                include_divergence_correction=args.divergence_correction,
                divergence_samples=args.divergence_samples,
                clamp=(-1.0, 1.0),
            )
            ones = torch.ones(real_cd.shape[0], device=device, dtype=real_cd.dtype)
            positive_values = composite_ddp(ones, real_cd, return_potential=True)
            negative_values = composite_ddp(ones, negative, return_potential=True)
            positive_mean = positive_values.mean()
            negative_mean = trimmed_mean(negative_values, args.negative_trim_fraction)
            contrastive = positive_mean - negative_mean

        trust_region = transport.new_zeros(())
        if args.mode == "joint" and args.trust_region_weight > 0:
            current = composite_ddp(t, x_t, return_potential=True)
            with torch.no_grad():
                anchor = anchor_energy.potential(x_t, t)
            trust_region = (current - anchor).square().mean()

        total = (
            args.transport_weight * transport
            + args.geometry_weight * geometry
            + args.contrastive_weight * contrastive
            + args.trust_region_weight * trust_region
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        scheduler.step()
        if energy_parameters:
            ema_update(energy, ema_energy, args.ema_decay)
        ema_update(mobility, ema_mobility, args.ema_decay)

        if step % args.log_every == 0 or step + 1 == args.total_steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            elapsed = reduced(now - last_time, maximum=True)
            logged_steps = step - last_logged_step
            last_time = now
            last_logged_step = step
            with torch.enable_grad():
                gradient = energy_gradient(energy, x_t)
                violation = descent_condition_diagnostics(
                    gradient, target_velocity
                )["violation_rate"]
            with torch.no_grad():
                mobility_stats = mobility_diagnostics(mobility, x_t.detach())
            metrics = {
                "loss": reduced(total),
                "transport_loss": reduced(transport),
                "geometry_penalty": reduced(geometry),
                "contrastive_loss": reduced(contrastive),
                "trust_region": reduced(trust_region),
                "positive_energy": reduced(positive_mean),
                "negative_energy": reduced(negative_mean),
                "descent_violation_rate": reduced(violation),
                "mean_abs_logdet": reduced(
                    mobility_stats["logdet"].abs().mean()
                ),
                "mean_condition_number": reduced(
                    mobility_stats["condition_number"].mean()
                ),
                "max_condition_number": reduced(
                    mobility_stats["condition_number"].max(), maximum=True
                ),
                "mean_log_distance_from_identity": reduced(
                    mobility_stats["log_distance_from_identity"].mean()
                ),
                "steps_per_second": logged_steps / max(elapsed, 1e-12),
                "global_examples_per_second": (
                    logged_steps * args.global_batch_size / max(elapsed, 1e-12)
                ),
                "peak_memory_bytes": reduced(
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0,
                    maximum=True,
                ),
                "training_wall_seconds": reduced(
                    now - training_start, maximum=True
                ),
                "examples_seen": (step + 1) * args.global_batch_size,
                "lr_energy": (
                    optimizer.param_groups[0]["lr"] if energy_parameters else 0.0
                ),
                "lr_mobility": (
                    optimizer.param_groups[-1]["lr"] if mobility_parameters else 0.0
                ),
            }
            if rank == 0:
                assert artifacts is not None
                artifacts.log(step, metrics, split="train", mode=args.mode)
                print(json.dumps({"step": step, **metrics}, sort_keys=True), flush=True)

        if rank == 0 and args.save_every > 0 and (step + 1) % args.save_every == 0:
            assert artifacts is not None
            save_rem_checkpoint(
                artifacts.path / "checkpoints" / f"checkpoint_{step + 1}.pt",
                energy=energy,
                mobility=mobility,
                optimizer=optimizer,
                scheduler=scheduler,
                ema_energy=ema_energy,
                ema_mobility=ema_mobility,
                step=step,
                config={**vars(args), "world_size": world_size},
                extra={"source_energy_step": source_energy_step},
            )

        if args.debug and step >= start_step + 2:
            break

        stop_for_budget = torch.zeros(1, device=device, dtype=torch.int32)
        if rank == 0 and args.max_training_seconds > 0:
            stop_for_budget[0] = int(
                time.perf_counter() - training_start >= args.max_training_seconds
            )
        if distributed:
            dist.broadcast(stop_for_budget, src=0)
        if bool(stop_for_budget.item()):
            break

    training_wall_seconds = reduced(
        time.perf_counter() - training_start, maximum=True
    )
    if rank == 0:
        assert artifacts is not None
        save_rem_checkpoint(
            artifacts.path / "checkpoints" / "latest.pt",
            energy=energy,
            mobility=mobility,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_energy=ema_energy,
            ema_mobility=ema_mobility,
            step=last_step,
            config={**vars(args), "world_size": world_size},
            extra={"source_energy_step": source_energy_step},
        )
        completed_steps = max(0, last_step - start_step + 1)
        artifacts.summarize(
            {
                "training_wall_seconds": [training_wall_seconds],
                "training_gpu_hours": [training_wall_seconds * world_size / 3600],
                "completed_steps": [float(completed_steps)],
                "examples_seen": [float(completed_steps * args.global_batch_size)],
            },
            status="complete",
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
