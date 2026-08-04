"""Training objectives and checkpoint helpers for REM experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .geometry import (
    descent_condition_diagnostics,
    minimal_distortion_regularizer,
    riemannian_transport_loss,
)
from .sampling import langevin_chain


@dataclass
class LossBreakdown:
    total: Tensor
    transport: Tensor
    contrastive: Tensor
    geometry: Tensor
    positive_energy: Tensor
    negative_energy: Tensor
    descent_violation_rate: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "transport_loss": float(self.transport.detach()),
            "contrastive_loss": float(self.contrastive.detach()),
            "geometry_penalty": float(self.geometry.detach()),
            "positive_energy": float(self.positive_energy.detach()),
            "negative_energy": float(self.negative_energy.detach()),
            "descent_violation_rate": float(self.descent_violation_rate.detach()),
        }


def energy_value(energy: nn.Module, x: Tensor, t: Tensor | None = None) -> Tensor:
    if hasattr(energy, "potential"):
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return energy.potential(x, t)
    return energy(x)


def energy_gradient(energy: nn.Module, x: Tensor, t: Tensor | None = None) -> Tensor:
    x_grad = x.detach().requires_grad_(True)
    value = energy_value(energy, x_grad, t)
    return torch.autograd.grad(value.sum(), x_grad, create_graph=False)[0]


def trimmed_mean(values: Tensor, trim_high_fraction: float = 0.0) -> Tensor:
    if not 0 <= trim_high_fraction < 1:
        raise ValueError("trim fraction must be in [0,1)")
    if trim_high_fraction == 0:
        return values.mean()
    keep = max(1, values.numel() - int(trim_high_fraction * values.numel()))
    return values.flatten().sort().values[:keep].mean()


def rem_objective(
    energy: nn.Module,
    mobility: nn.Module,
    x_transport: Tensor,
    target_velocity: Tensor,
    *,
    x_positive: Tensor | None = None,
    x_negative_init: Tensor | None = None,
    transport_weight: float = 1.0,
    contrastive_weight: float = 0.0,
    geometry_weight: float = 1e-3,
    metric_weighted: bool = True,
    negative_steps: int = 0,
    negative_dt: float = 0.01,
    negative_temperature: float = 0.01,
    divergence_samples: int = 1,
    include_divergence_correction: bool = True,
    negative_clamp: tuple[float, float] | None = None,
    trim_high_fraction: float = 0.0,
) -> tuple[LossBreakdown, Tensor | None]:
    transport = riemannian_transport_loss(
        energy,
        mobility,
        x_transport,
        target_velocity,
        metric_weighted=metric_weighted,
    )
    geometry = minimal_distortion_regularizer(mobility, x_transport)
    positive_energy = x_transport.new_zeros(())
    negative_energy = x_transport.new_zeros(())
    contrastive = x_transport.new_zeros(())
    x_negative = None

    if contrastive_weight > 0:
        if x_positive is None or x_negative_init is None or negative_steps < 1:
            raise ValueError("contrastive loss requires positive/negative samples and steps")
        x_negative, _ = langevin_chain(
            energy,
            mobility,
            x_negative_init,
            steps=negative_steps,
            dt=negative_dt,
            temperature=negative_temperature,
            include_divergence_correction=include_divergence_correction,
            divergence_samples=divergence_samples,
            clamp=negative_clamp,
        )
        positive_values = energy_value(energy, x_positive)
        negative_values = energy_value(energy, x_negative)
        positive_energy = positive_values.mean()
        negative_energy = trimmed_mean(negative_values, trim_high_fraction)
        contrastive = positive_energy - negative_energy

    gradient = energy_gradient(energy, x_transport)
    descent = descent_condition_diagnostics(gradient, target_velocity)
    total = (
        transport_weight * transport
        + contrastive_weight * contrastive
        + geometry_weight * geometry
    )
    return LossBreakdown(
        total=total,
        transport=transport,
        contrastive=contrastive,
        geometry=geometry,
        positive_energy=positive_energy,
        negative_energy=negative_energy,
        descent_violation_rate=descent["violation_rate"],
    ), x_negative


def set_trainable(module: nn.Module, trainable: bool) -> None:
    module.train(trainable)
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


@torch.no_grad()
def ema_update(source: nn.Module, target: nn.Module, decay: float) -> None:
    if not 0 <= decay <= 1:
        raise ValueError("EMA decay must be in [0,1]")
    source_state = source.state_dict()
    target_state = target.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("EMA modules have different state dictionaries")
    for key, source_value in source_state.items():
        target_value = target_state[key]
        if torch.is_floating_point(target_value):
            target_value.mul_(decay).add_(source_value, alpha=1 - decay)
        else:
            target_value.copy_(source_value)


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def load_energy_state(
    energy: nn.Module,
    checkpoint_path: str | Path,
    *,
    use_ema: bool = True,
    strict: bool = True,
) -> Mapping[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        energy.load_state_dict(checkpoint, strict=strict)
        return {"raw_state_dict": True}
    keys = (
        ("ema_model", "energy_ema", "energy")
        if use_ema
        else ("net_model", "energy", "ema_model")
    )
    state = next((checkpoint[key] for key in keys if key in checkpoint), checkpoint)
    energy.load_state_dict(state, strict=strict)
    return checkpoint


def save_rem_checkpoint(
    path: str | Path,
    *,
    energy: nn.Module,
    mobility: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any = None,
    ema_energy: nn.Module | None = None,
    ema_mobility: nn.Module | None = None,
    step: int,
    config: Mapping[str, Any] | Any,
    extra: Mapping[str, Any] | None = None,
) -> None:
    config_value = asdict(config) if is_dataclass(config) else dict(config)
    state = {
        "format_version": 1,
        "step": step,
        "config": config_value,
        "energy": energy.state_dict(),
        "mobility": mobility.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "energy_ema": ema_energy.state_dict() if ema_energy is not None else None,
        "mobility_ema": ema_mobility.state_dict() if ema_mobility is not None else None,
        "extra": dict(extra or {}),
    }
    torch.save(state, path)


def load_rem_checkpoint(
    path: str | Path,
    *,
    energy: nn.Module,
    mobility: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema_energy: nn.Module | None = None,
    ema_mobility: nn.Module | None = None,
    strict: bool = True,
) -> Mapping[str, Any]:
    state = torch.load(path, map_location="cpu")
    energy.load_state_dict(state["energy"], strict=strict)
    mobility.load_state_dict(state["mobility"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if ema_energy is not None and state.get("energy_ema") is not None:
        ema_energy.load_state_dict(state["energy_ema"], strict=strict)
    if ema_mobility is not None and state.get("mobility_ema") is not None:
        ema_mobility.load_state_dict(state["mobility_ema"], strict=strict)
    return state
