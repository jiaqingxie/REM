"""Dependency-light metrics for REM mechanism and stationarity studies."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor


def bootstrap_confidence_interval(
    values: Iterable[float] | Tensor,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    tensor = torch.as_tensor(list(values) if not isinstance(values, Tensor) else values)
    tensor = tensor.detach().double().flatten().cpu()
    if tensor.numel() == 0:
        raise ValueError("bootstrap values must be nonempty")
    if tensor.numel() < 2:
        value = float(tensor.mean())
        return value, value
    if not 0 < confidence < 1 or n_resamples < 1:
        raise ValueError("invalid bootstrap configuration")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        tensor.numel(),
        (n_resamples, tensor.numel()),
        generator=generator,
    )
    means = tensor[indices].mean(dim=1)
    alpha = (1 - confidence) / 2
    quantiles = torch.tensor([alpha, 1 - alpha], dtype=means.dtype)
    lower, upper = torch.quantile(means, quantiles)
    return float(lower), float(upper)


def rbf_mmd(
    samples_a: Tensor,
    samples_b: Tensor,
    *,
    bandwidth: float | None = None,
) -> Tensor:
    """Biased RBF MMD, stable for the moderate synthetic sample sizes."""

    a = samples_a.reshape(samples_a.shape[0], -1).double()
    b = samples_b.reshape(samples_b.shape[0], -1).double()
    if bandwidth is None:
        combined = torch.cat([a, b], dim=0)
        distances = torch.pdist(combined).square()
        positive = distances[distances > 0]
        bandwidth_tensor = positive.median() if positive.numel() else a.new_tensor(1.0)
    else:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        bandwidth_tensor = a.new_tensor(bandwidth)

    def kernel(x: Tensor, y: Tensor) -> Tensor:
        squared = torch.cdist(x, y).square()
        return torch.exp(-squared / (2 * bandwidth_tensor.clamp_min(1e-12)))

    return kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean()


def sliced_wasserstein(
    samples_a: Tensor,
    samples_b: Tensor,
    *,
    n_projections: int = 128,
    seed: int = 0,
) -> Tensor:
    a = samples_a.reshape(samples_a.shape[0], -1).double()
    b = samples_b.reshape(samples_b.shape[0], -1).double()
    count = min(a.shape[0], b.shape[0])
    if count < 1 or n_projections < 1:
        raise ValueError("samples and projections must be nonempty")
    a = a[:count]
    b = b[:count]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projections = torch.randn(
        a.shape[1], n_projections, generator=generator, dtype=a.dtype
    ).to(a.device)
    projections = projections / projections.norm(dim=0, keepdim=True).clamp_min(1e-12)
    projected_a = (a @ projections).sort(dim=0).values
    projected_b = (b @ projections).sort(dim=0).values
    return (projected_a - projected_b).square().mean().sqrt()


def effective_sample_size(chains: Tensor, *, max_lag: int | None = None) -> Tensor:
    """Estimate ESS per flattened event dimension.

    Input has shape ``(n_chains, n_draws, *event_shape)``. The estimator uses
    the initial-positive sequence of averaged autocorrelations.
    """

    if chains.ndim < 3:
        raise ValueError("chains must have shape (chains, draws, event...)")
    chain_count, draws = chains.shape[:2]
    values = chains.reshape(chain_count, draws, -1).double()
    values = values - values.mean(dim=1, keepdim=True)
    variance = values.square().mean(dim=(0, 1)).clamp_min(1e-12)
    if max_lag is None:
        max_lag = min(draws - 1, int(10 * math.sqrt(draws)))
    correlation_sum = torch.zeros_like(variance)
    for lag in range(1, max_lag + 1):
        covariance = (values[:, :-lag] * values[:, lag:]).mean(dim=(0, 1))
        correlation = covariance / variance
        if torch.all(correlation <= 0):
            break
        correlation_sum = correlation_sum + correlation.clamp_min(0)
    return chain_count * draws / (1 + 2 * correlation_sum)


def split_rhat(chains: Tensor) -> Tensor:
    """Classical split-Rhat per flattened event dimension."""

    if chains.ndim < 3 or chains.shape[1] < 4:
        raise ValueError("Rhat requires (chains, draws>=4, event...)")
    chain_count, draws = chains.shape[:2]
    half = draws // 2
    first = chains[:, :half]
    second = chains[:, -half:]
    split = torch.cat([first, second], dim=0).reshape(2 * chain_count, half, -1).double()
    within_variances = split.var(dim=1, unbiased=True)
    within = within_variances.mean(dim=0)
    means = split.mean(dim=1)
    between = half * means.var(dim=0, unbiased=True)
    variance = ((half - 1) / half) * within + between / half
    return torch.sqrt(variance / within.clamp_min(1e-12))


def histogram_kl_2d(
    samples: Tensor,
    reference_log_density,
    *,
    bounds: tuple[float, float] = (-4.0, 4.0),
    bins: int = 80,
) -> Tensor:
    """Grid KL between a sample histogram and a normalized reference density."""

    points = samples.reshape(samples.shape[0], -1)
    if points.shape[1] != 2:
        raise ValueError("histogram_kl_2d requires two-dimensional samples")
    low, high = bounds
    histogram = torch.histogramdd(
        points.detach().cpu(),
        bins=(bins, bins),
        range=(low, high, low, high),
    ).hist.double()
    empirical = histogram / histogram.sum().clamp_min(1)
    centers = torch.linspace(low, high, bins + 1, dtype=torch.double)
    centers = 0.5 * (centers[:-1] + centers[1:])
    grid_x, grid_y = torch.meshgrid(centers, centers, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    log_density = reference_log_density(grid).reshape(bins, bins).double()
    reference = torch.softmax(log_density.flatten(), dim=0).reshape(bins, bins)
    mask = empirical > 0
    return (
        empirical[mask]
        * (empirical[mask].log() - reference[mask].clamp_min(1e-300).log())
    ).sum()
