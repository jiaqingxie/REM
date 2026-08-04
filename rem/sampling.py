"""Sampling utilities shared by toy, image, and inverse REM experiments."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from .geometry import riemannian_langevin_step


TemperatureSchedule = Callable[[int, int, Tensor], float | Tensor]


@dataclass
class SamplingProfile:
    steps: int
    batch_size: int
    wall_seconds: float
    samples_per_second: float
    energy_grad_evaluations: int
    divergence_probes: int
    peak_memory_bytes: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def constant_temperature(value: float) -> TemperatureSchedule:
    if value < 0:
        raise ValueError("temperature must be nonnegative")

    def schedule(step: int, total_steps: int, x: Tensor) -> float:
        del step, total_steps, x
        return value

    return schedule


def linear_temperature(
    epsilon_max: float,
    *,
    warmup_fraction: float = 0.8,
) -> TemperatureSchedule:
    """Piecewise-linear temperature schedule used by Energy Matching."""

    if epsilon_max < 0 or not 0 <= warmup_fraction < 1:
        raise ValueError("invalid temperature schedule")

    def schedule(step: int, total_steps: int, x: Tensor) -> float:
        del x
        progress = step / max(total_steps - 1, 1)
        if progress < warmup_fraction:
            return 0.0
        return epsilon_max * (progress - warmup_fraction) / (1 - warmup_fraction)

    return schedule


def _potential(model: nn.Module, x: Tensor, t: Optional[Tensor]) -> Tensor:
    if hasattr(model, "potential"):
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return model.potential(x, t)
    return model(x)


def langevin_chain(
    energy: nn.Module,
    mobility: nn.Module,
    x_init: Tensor,
    *,
    steps: int,
    dt: float,
    temperature: float | TemperatureSchedule,
    include_divergence_correction: bool = True,
    divergence_samples: int = 1,
    clamp: Optional[tuple[float, float]] = None,
    save_every: int = 0,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, list[Tensor]]:
    """Run an equilibrium-correct REM chain.

    ``temperature`` may be constant or a function of the step. Saved states
    are detached CPU tensors so long trajectories do not retain GPU graphs.
    """

    if steps < 0 or dt <= 0 or divergence_samples < 1:
        raise ValueError("invalid sampler configuration")
    x = x_init.detach()
    trajectory: list[Tensor] = []
    if save_every > 0:
        trajectory.append(x.cpu())

    for step in range(steps):
        epsilon = (
            temperature(step, steps, x)
            if callable(temperature)
            else temperature
        )
        t = torch.full(
            (x.shape[0],),
            step * dt,
            device=x.device,
            dtype=x.dtype,
        )
        noise = torch.randn(
            x.shape,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        x = riemannian_langevin_step(
            lambda value: _potential(energy, value, t),
            mobility,
            x,
            dt=dt,
            epsilon=epsilon,
            include_divergence_correction=include_divergence_correction,
            divergence_samples=divergence_samples,
            noise=noise,
        )
        if clamp is not None:
            x = x.clamp(*clamp)
        if save_every > 0 and ((step + 1) % save_every == 0 or step + 1 == steps):
            trajectory.append(x.cpu())
    return x, trajectory


def profile_langevin_chain(
    energy: nn.Module,
    mobility: nn.Module,
    x_init: Tensor,
    **kwargs,
) -> tuple[Tensor, list[Tensor], SamplingProfile]:
    """Run and profile a chain with synchronized CUDA timing."""

    if x_init.is_cuda:
        torch.cuda.synchronize(x_init.device)
        torch.cuda.reset_peak_memory_stats(x_init.device)
    start = time.perf_counter()
    final, trajectory = langevin_chain(energy, mobility, x_init, **kwargs)
    if x_init.is_cuda:
        torch.cuda.synchronize(x_init.device)
        peak_memory = torch.cuda.max_memory_allocated(x_init.device)
    else:
        peak_memory = 0
    elapsed = time.perf_counter() - start
    steps = int(kwargs["steps"])
    probes = int(kwargs.get("divergence_samples", 1))
    corrected = bool(kwargs.get("include_divergence_correction", True))
    probe_cost_value = getattr(mobility, "divergence_probe_cost", probes)
    probe_cost = int(
        probe_cost_value(probes) if callable(probe_cost_value) else probe_cost_value
    )
    profile = SamplingProfile(
        steps=steps,
        batch_size=x_init.shape[0],
        wall_seconds=elapsed,
        samples_per_second=x_init.shape[0] / max(elapsed, 1e-12),
        energy_grad_evaluations=steps,
        divergence_probes=steps * probe_cost if corrected else 0,
        peak_memory_bytes=peak_memory,
    )
    return final, trajectory, profile


def profile_step_components(
    energy: nn.Module,
    mobility: nn.Module,
    x: Tensor,
    *,
    divergence_samples: int = 1,
    include_divergence_correction: bool = True,
    repeats: int = 5,
) -> dict[str, float]:
    """Microprofile one sampler step without changing main-chain timing.

    CUDA is synchronized around each component, so these numbers diagnose
    where time is spent but must not be summed and substituted for the
    end-to-end steady-state latency.
    """

    if repeats < 1 or divergence_samples < 1:
        raise ValueError("component profile counts must be positive")

    totals = {
        "energy_gradient_seconds_per_step": 0.0,
        "mobility_drift_seconds_per_step": 0.0,
        "divergence_seconds_per_step": 0.0,
        "diffusion_seconds_per_step": 0.0,
    }

    def synchronize() -> None:
        if x.is_cuda:
            torch.cuda.synchronize(x.device)

    for _ in range(repeats):
        x_grad = x.detach().requires_grad_(True)
        synchronize()
        start = time.perf_counter()
        potential = _potential(energy, x_grad, None)
        gradient = torch.autograd.grad(potential.sum(), x_grad)[0]
        synchronize()
        totals["energy_gradient_seconds_per_step"] += time.perf_counter() - start

        synchronize()
        start = time.perf_counter()
        mobility.apply(x_grad, gradient)
        synchronize()
        totals["mobility_drift_seconds_per_step"] += time.perf_counter() - start

        if include_divergence_correction:
            synchronize()
            start = time.perf_counter()
            mobility.divergence(
                x_grad,
                n_samples=divergence_samples,
                create_graph=False,
            )
            synchronize()
            totals["divergence_seconds_per_step"] += time.perf_counter() - start

        noise = torch.randn_like(x_grad)
        synchronize()
        start = time.perf_counter()
        if hasattr(mobility, "sample_sqrt_noise"):
            mobility.sample_sqrt_noise(x_grad, noise=noise)
        else:
            mobility.sqrt_apply(x_grad, noise)
        synchronize()
        totals["diffusion_seconds_per_step"] += time.perf_counter() - start

    return {name: value / repeats for name, value in totals.items()}
