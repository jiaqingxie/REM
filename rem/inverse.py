"""Posterior potentials and operators for REM inverse-problem transfer."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .sampling import langevin_chain


class ForwardOperator(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...


class MaskOperator(nn.Module):
    def __init__(self, mask: Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.mask


class DownsampleOperator(nn.Module):
    def __init__(self, scale: int = 4) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError("scale must be positive")
        self.scale = int(scale)

    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, kernel_size=self.scale, stride=self.scale)


class GaussianBlurOperator(nn.Module):
    def __init__(self, channels: int = 3, kernel_size: int = 9, sigma: float = 2.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or sigma <= 0:
            raise ValueError("blur kernel must be odd and sigma positive")
        coordinates = torch.arange(kernel_size) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coordinates.float() / sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
        self.register_buffer("kernel", kernel)
        self.channels = channels
        self.padding = kernel_size // 2

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)


def center_mask(
    shape: tuple[int, int, int],
    *,
    missing_fraction: float = 0.5,
    device: torch.device | str = "cpu",
) -> Tensor:
    channels, height, width = shape
    if not 0 < missing_fraction < 1:
        raise ValueError("missing_fraction must be in (0,1)")
    side_fraction = missing_fraction ** 0.5
    missing_height = max(1, round(height * side_fraction))
    missing_width = max(1, round(width * side_fraction))
    top = (height - missing_height) // 2
    left = (width - missing_width) // 2
    mask = torch.ones(1, channels, height, width, device=device)
    mask[:, :, top : top + missing_height, left : left + missing_width] = 0
    return mask


def random_mask(
    shape: tuple[int, int, int],
    *,
    observed_fraction: float = 0.5,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> Tensor:
    if not 0 < observed_fraction < 1:
        raise ValueError("observed_fraction must be in (0,1)")
    generator = torch.Generator(device=torch.device(device).type).manual_seed(seed)
    return (
        torch.rand((1, *shape), device=device, generator=generator)
        < observed_fraction
    ).float()


class PosteriorEnergy(nn.Module):
    """Combine a frozen prior energy with a differentiable likelihood."""

    def __init__(
        self,
        prior_energy: nn.Module,
        operator: nn.Module,
        observation: Tensor,
        *,
        observation_sigma: float,
    ) -> None:
        super().__init__()
        if observation_sigma <= 0:
            raise ValueError("observation_sigma must be positive")
        self.prior_energy = prior_energy
        self.operator = operator
        self.register_buffer("observation", observation)
        self.observation_sigma = float(observation_sigma)

    def _observation_for(self, x: Tensor) -> Tensor:
        if self.observation.shape[0] == x.shape[0]:
            return self.observation
        if self.observation.shape[0] == 1:
            return self.observation.expand(x.shape[0], *self.observation.shape[1:])
        raise ValueError("observation batch must equal x batch or be one")

    def potential(self, x: Tensor, t: Tensor | None = None) -> Tensor:
        if hasattr(self.prior_energy, "potential"):
            if t is None:
                t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            prior = self.prior_energy.potential(x, t)
        else:
            prior = self.prior_energy(x)
        residual = self.operator(x) - self._observation_for(x)
        likelihood = 0.5 * residual.flatten(start_dim=1).square().sum(dim=1)
        return prior + likelihood / (self.observation_sigma ** 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.potential(x)


def posterior_sample(
    prior_energy: nn.Module,
    mobility: nn.Module,
    operator: nn.Module,
    observation: Tensor,
    x_init: Tensor,
    *,
    observation_sigma: float,
    steps: int,
    dt: float,
    temperature: float,
    divergence_samples: int = 1,
    include_divergence_correction: bool = True,
) -> Tensor:
    result, _ = posterior_chain(
        prior_energy,
        mobility,
        operator,
        observation,
        x_init,
        observation_sigma=observation_sigma,
        steps=steps,
        dt=dt,
        temperature=temperature,
        divergence_samples=divergence_samples,
        include_divergence_correction=include_divergence_correction,
        save_every=0,
    )
    return result


def posterior_chain(
    prior_energy: nn.Module,
    mobility: nn.Module,
    operator: nn.Module,
    observation: Tensor,
    x_init: Tensor,
    *,
    observation_sigma: float,
    steps: int,
    dt: float,
    temperature: float,
    divergence_samples: int = 1,
    include_divergence_correction: bool = True,
    save_every: int = 0,
) -> tuple[Tensor, list[Tensor]]:
    """Run a posterior chain and optionally retain thinned CPU states."""

    posterior = PosteriorEnergy(
        prior_energy,
        operator,
        observation,
        observation_sigma=observation_sigma,
    )
    return langevin_chain(
        posterior,
        mobility,
        x_init,
        steps=steps,
        dt=dt,
        temperature=temperature,
        divergence_samples=divergence_samples,
        include_divergence_correction=include_divergence_correction,
        clamp=(-1.0, 1.0),
        save_every=save_every,
    )


def psnr(reconstruction: Tensor, target: Tensor, *, data_range: float = 2.0) -> Tensor:
    mse = (reconstruction - target).flatten(start_dim=1).square().mean(dim=1)
    return 20 * torch.log10(
        torch.as_tensor(data_range, device=mse.device, dtype=mse.dtype)
    ) - 10 * torch.log10(mse.clamp_min(1e-12))


def measurement_error(
    reconstruction: Tensor,
    observation: Tensor,
    operator: nn.Module,
) -> Tensor:
    return (
        (operator(reconstruction) - observation)
        .flatten(start_dim=1)
        .square()
        .mean(dim=1)
        .sqrt()
    )


def posterior_diversity(samples: Tensor) -> Tensor:
    """Mean per-pixel variance for shape ``(observations, draws, C, H, W)``."""

    if samples.ndim != 5 or samples.shape[1] < 2:
        raise ValueError("posterior samples must have shape (N,draws,C,H,W)")
    return samples.var(dim=1, unbiased=True).flatten(start_dim=1).mean(dim=1)
