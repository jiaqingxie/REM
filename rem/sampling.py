"""Sampling utilities shared by toy, image, and inverse REM experiments."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from .geometry import (
    IdentityMobility,
    riemannian_langevin_heun_step,
    riemannian_langevin_step,
)


TemperatureSchedule = Callable[[int, int, Tensor], float | Tensor]
MobilitySchedule = Callable[[int, int, Tensor], nn.Module]


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


def energy_matching_temperature(
    epsilon_max: float,
    *,
    dt: float,
    time_cutoff: float = 1.0,
    ramp_end: float = 1.0,
) -> TemperatureSchedule:
    """Absolute-time noise schedule used by the official Energy Matching sampler.

    The schedule is zero before ``time_cutoff``, ramps linearly to
    ``epsilon_max`` by ``ramp_end``, and stays constant afterwards.  The
    public CIFAR-10 configuration uses ``time_cutoff == ramp_end == 1``,
    which gives the intended step from zero to ``epsilon_max`` at ``t=1``.

    Unlike :func:`linear_temperature`, this schedule is invariant to the
    number of integration steps when the physical terminal time is fixed.
    That property is required for a valid FID-versus-NFE solver sweep.
    """

    if epsilon_max < 0 or dt <= 0:
        raise ValueError("epsilon_max must be nonnegative and dt must be positive")
    if time_cutoff < 0 or ramp_end < time_cutoff:
        raise ValueError("temperature times must satisfy 0 <= cutoff <= ramp_end")

    def schedule(step: int, total_steps: int, x: Tensor) -> float:
        del total_steps, x
        time_value = step * dt
        if time_value < time_cutoff:
            return 0.0
        if ramp_end > time_cutoff and time_value < ramp_end:
            fraction = (time_value - time_cutoff) / (ramp_end - time_cutoff)
            return epsilon_max * fraction
        return epsilon_max

    return schedule


def phase_gated_mobility(
    mobility: nn.Module,
    *,
    dt: float,
    active_until: float,
) -> MobilitySchedule:
    """Use learned mobility only on the physical-time interval it trained on.

    REM's conditional-flow objective is trained with ``t`` in ``[0, 1]``.
    The sampler can continue beyond that interval for Langevin refinement, but
    those endpoints must use the original Energy Matching identity mobility.
    At the boundary, Heun naturally evaluates the predictor with the learned
    mobility and the corrector with identity mobility.
    """

    if dt <= 0 or active_until < 0:
        raise ValueError("dt must be positive and active_until nonnegative")
    identity = IdentityMobility()

    def schedule(step: int, total_steps: int, x: Tensor) -> nn.Module:
        del total_steps, x
        return mobility if step * dt < active_until else identity

    return schedule


def _scheduled_mobility(
    mobility: nn.Module,
    schedule: Optional[MobilitySchedule],
    step: int,
    total_steps: int,
    x: Tensor,
) -> nn.Module:
    return mobility if schedule is None else schedule(step, total_steps, x)


def _temperature_at(
    temperature: float | TemperatureSchedule,
    step: int,
    total_steps: int,
    x: Tensor,
) -> float | Tensor:
    return temperature(step, total_steps, x) if callable(temperature) else temperature


def _temperature_has_work(value: float | Tensor) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.any(value != 0).item())
    return float(value) != 0.0


def _divergence_probe_cost(mobility: nn.Module, requested_probes: int) -> int:
    value = getattr(mobility, "divergence_probe_cost", requested_probes)
    return int(value(requested_probes) if callable(value) else value)


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
    divergence_method: str = "hutchinson",
    clamp: Optional[tuple[float, float]] = None,
    save_every: int = 0,
    generator: Optional[torch.Generator] = None,
    integrator: str = "euler",
    mobility_schedule: Optional[MobilitySchedule] = None,
) -> tuple[Tensor, list[Tensor]]:
    """Run an equilibrium-correct REM chain.

    ``temperature`` may be constant or a function of the step. Saved states
    are detached CPU tensors so long trajectories do not retain GPU graphs.
    With Heun, the correction flag toggles the explicit Stratonovich drift
    term; Euler ablates the entire Ito divergence drift when it is disabled.
    """

    if steps < 0 or dt <= 0 or divergence_samples < 1:
        raise ValueError("invalid sampler configuration")
    if integrator not in {"euler", "heun"}:
        raise ValueError("integrator must be 'euler' or 'heun'")
    x = x_init.detach()
    trajectory: list[Tensor] = []
    if save_every > 0:
        trajectory.append(x.cpu())

    for step in range(steps):
        epsilon = _temperature_at(temperature, step, steps, x)
        current_mobility = _scheduled_mobility(
            mobility, mobility_schedule, step, steps, x
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
        if integrator == "euler":
            x = riemannian_langevin_step(
                lambda value: _potential(energy, value, t),
                current_mobility,
                x,
                dt=dt,
                epsilon=epsilon,
                include_divergence_correction=include_divergence_correction,
                divergence_samples=divergence_samples,
                divergence_method=divergence_method,
                noise=noise,
            )
        else:
            # Heun evaluates coefficients at both ends of the interval.  The
            # final corrector therefore lives at ``step == steps`` rather than
            # being clamped back onto the final predictor time.
            next_step = step + 1
            next_epsilon = _temperature_at(temperature, next_step, steps, x)
            next_mobility = _scheduled_mobility(
                mobility, mobility_schedule, next_step, steps, x
            )
            next_t = torch.full(
                (x.shape[0],),
                (step + 1) * dt,
                device=x.device,
                dtype=x.dtype,
            )
            x = riemannian_langevin_heun_step(
                lambda value: _potential(energy, value, t),
                lambda value: _potential(energy, value, next_t),
                current_mobility,
                x,
                dt=dt,
                epsilon=epsilon,
                next_epsilon=next_epsilon,
                next_mobility=next_mobility,
                include_divergence_correction=include_divergence_correction,
                divergence_samples=divergence_samples,
                divergence_method=divergence_method,
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
    requested_probes = int(kwargs.get("divergence_samples", 1))
    corrected = bool(kwargs.get("include_divergence_correction", True))
    evaluations_per_step = 2 if kwargs.get("integrator", "euler") == "heun" else 1
    divergence_probes = 0
    if corrected:
        temperature = kwargs["temperature"]
        mobility_schedule = kwargs.get("mobility_schedule")
        for step in range(steps):
            epsilon = _temperature_at(temperature, step, steps, x_init)
            if _temperature_has_work(epsilon):
                current_mobility = _scheduled_mobility(
                    mobility, mobility_schedule, step, steps, x_init
                )
                divergence_probes += _divergence_probe_cost(
                    current_mobility, requested_probes
                )
            if evaluations_per_step == 2:
                next_epsilon = _temperature_at(
                    temperature, step + 1, steps, x_init
                )
                if _temperature_has_work(next_epsilon):
                    next_mobility = _scheduled_mobility(
                        mobility, mobility_schedule, step + 1, steps, x_init
                    )
                    divergence_probes += _divergence_probe_cost(
                        next_mobility, requested_probes
                    )
    profile = SamplingProfile(
        steps=steps,
        batch_size=x_init.shape[0],
        wall_seconds=elapsed,
        samples_per_second=x_init.shape[0] / max(elapsed, 1e-12),
        energy_grad_evaluations=steps * evaluations_per_step,
        divergence_probes=divergence_probes,
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
