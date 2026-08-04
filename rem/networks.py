"""Neural parameterizations used by REM experiments.

The final layer of every mobility network is initialized to zero so the
gauge-fixed models start exactly at the Euclidean mobility ``G = I``.
"""

from __future__ import annotations

from math import prod
from typing import Sequence

import torch
from torch import Tensor, nn

from .geometry import (
    BoundedScalarMobility,
    ConstantDiagonalMobility,
    DiagonalPlusLowRankMobility,
    GaugeFixedDiagonalMobility,
    GaugeFixedFullMobility,
    IdentityMobility,
    UnfixedLogDiagonalMobility,
    riemannian_velocity,
)


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class MLP(nn.Module):
    """Small SiLU MLP with a zero-initialized output layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 128,
        depth: int = 3,
        zero_output: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1 or hidden_dim < 1 or depth < 1:
            raise ValueError("MLP dimensions and depth must be positive")
        layers: list[nn.Module] = []
        previous = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(previous, hidden_dim), nn.SiLU()])
            previous = hidden_dim
        output = nn.Linear(previous, output_dim)
        if zero_output:
            _zero_module(output)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x.reshape(x.shape[0], -1))


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ImageDiagonalMobilityNetwork(nn.Module):
    """Convolutional same-shape log-mobility network for images."""

    def __init__(self, channels: int = 3, hidden_channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, hidden_channels, 3, padding=1)]
        layers.extend(ResBlock(hidden_channels) for _ in range(depth))
        output = nn.Conv2d(hidden_channels, channels, 3, padding=1)
        _zero_module(output)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("image mobility expects BCHW input")
        return self.network(x)


class ImageScalarMobilityNetwork(nn.Module):
    """Convolutional network returning one scalar logit per image."""

    def __init__(self, channels: int = 3, hidden_channels: int = 32, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, hidden_channels, 3, padding=1)]
        layers.extend(ResBlock(hidden_channels) for _ in range(depth))
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_channels, 1)
        _zero_module(self.output)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x).mean(dim=(2, 3))
        return self.output(features).squeeze(1)


class ImageLowRankMobilityNetwork(nn.Module):
    """Produces diagonal logits and low-rank factors without dense matrices."""

    def __init__(
        self,
        channels: int = 3,
        hidden_channels: int = 32,
        depth: int = 3,
        rank: int = 4,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.channels = channels
        self.rank = rank
        layers: list[nn.Module] = [nn.Conv2d(channels, hidden_channels, 3, padding=1)]
        layers.extend(ResBlock(hidden_channels) for _ in range(depth))
        self.features = nn.Sequential(*layers)
        self.output = nn.Conv2d(hidden_channels, channels * (rank + 1), 3, padding=1)
        _zero_module(self.output)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        output = self.output(self.features(x))
        batch, _, height, width = output.shape
        diagonal = output[:, : self.channels]
        factors = output[:, self.channels :].reshape(
            batch, self.rank, self.channels, height, width
        )
        factors = factors.permute(0, 2, 3, 4, 1).reshape(batch, -1, self.rank)
        return diagonal, factors


def build_mobility(
    kind: str,
    sample_shape: Sequence[int],
    *,
    hidden_dim: int = 128,
    depth: int = 3,
    log_bound: float = 1.5,
    rank: int = 4,
    min_scalar: float = 0.25,
    max_scalar: float = 4.0,
) -> nn.Module:
    """Build a mobility with a stable, serializable variant name."""

    shape = tuple(int(value) for value in sample_shape)
    dimension = prod(shape)
    normalized = kind.lower().replace("_", "-")

    if normalized in {"identity", "em"}:
        return IdentityMobility()
    if normalized in {"constant", "em-const"}:
        return ConstantDiagonalMobility(dimension, log_bound=log_bound)

    is_image = len(shape) == 3
    if normalized in {"scalar", "rem-scalar"}:
        network: nn.Module
        if is_image:
            network = ImageScalarMobilityNetwork(shape[0], hidden_dim, depth)
        else:
            network = MLP(dimension, 1, hidden_dim=hidden_dim, depth=depth)
        return BoundedScalarMobility(
            network,
            min_mobility=min_scalar,
            max_mobility=max_scalar,
        )
    if normalized in {
        "diagonal",
        "rem-diag",
        "unfixed-diagonal",
        "rem-unfixed",
    }:
        if is_image:
            network = ImageDiagonalMobilityNetwork(shape[0], hidden_dim, depth)
        else:
            base = MLP(dimension, dimension, hidden_dim=hidden_dim, depth=depth)

            class ReshapeNetwork(nn.Module):
                def __init__(self, wrapped: nn.Module, output_shape: tuple[int, ...]) -> None:
                    super().__init__()
                    self.wrapped = wrapped
                    self.output_shape = output_shape

                def forward(self, x: Tensor) -> Tensor:
                    return self.wrapped(x).reshape(x.shape[0], *self.output_shape)

            network = ReshapeNetwork(base, shape)
        if normalized in {"unfixed-diagonal", "rem-unfixed"}:
            return UnfixedLogDiagonalMobility(network, log_bound=log_bound)
        return GaugeFixedDiagonalMobility(network, log_bound=log_bound)
    if normalized in {"full", "rem-full"}:
        if dimension > 64:
            raise ValueError("full mobility is restricted to dimension <= 64")
        packed = dimension * (dimension + 1) // 2
        network = MLP(dimension, packed, hidden_dim=hidden_dim, depth=depth)
        return GaugeFixedFullMobility(
            network,
            dimension,
            log_eigenvalue_bound=log_bound,
        )
    if normalized in {"low-rank", "lowrank", "rem-lr"}:
        if is_image:
            network = ImageLowRankMobilityNetwork(
                shape[0], hidden_dim, depth, rank=rank
            )
        else:
            network = MLP(
                dimension,
                dimension * (rank + 1),
                hidden_dim=hidden_dim,
                depth=depth,
            )
        return DiagonalPlusLowRankMobility(
            network,
            rank,
            log_bound=log_bound,
        )
    raise ValueError(f"unknown mobility kind: {kind}")


class REMModel(nn.Module):
    """Composite energy/mobility module suitable for DDP and checkpoints."""

    def __init__(self, energy: nn.Module, mobility: nn.Module) -> None:
        super().__init__()
        self.energy = energy
        self.mobility = mobility

    def potential(self, x: Tensor, t: Tensor | None = None) -> Tensor:
        if hasattr(self.energy, "potential"):
            if t is None:
                t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            return self.energy.potential(x, t)
        return self.energy(x)

    def velocity(
        self,
        x: Tensor,
        t: Tensor | None = None,
        *,
        create_graph: bool = True,
    ) -> Tensor:
        return riemannian_velocity(
            lambda value: self.potential(value, t),
            self.mobility,
            x,
            create_graph=create_graph,
        )

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        return_potential: bool = False,
    ) -> Tensor:
        if return_potential:
            return self.potential(x, t)
        return self.velocity(x, t, create_graph=self.training)

    def freeze_energy(self) -> None:
        self.energy.eval()
        for parameter in self.energy.parameters():
            parameter.requires_grad_(False)

    def unfreeze_energy(self) -> None:
        for parameter in self.energy.parameters():
            parameter.requires_grad_(True)
