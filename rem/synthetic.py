"""Controlled synthetic problems for REM representability and stationarity."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import Tensor, nn

from .geometry import hutchinson_divergence


class QuadraticPotential(nn.Module):
    def __init__(self, precision: Tensor) -> None:
        super().__init__()
        if precision.ndim != 2 or precision.shape[0] != precision.shape[1]:
            raise ValueError("precision must be square")
        self.register_buffer("precision", precision)

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(x.shape[0], -1)
        return 0.5 * torch.einsum("bi,ij,bj->b", flat, self.precision, flat)


class BananaPotential(nn.Module):
    def __init__(self, curvature: float = 0.25) -> None:
        super().__init__()
        self.curvature = float(curvature)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 2:
            raise ValueError("banana potential is two-dimensional")
        first = x[:, 0]
        second = x[:, 1] - self.curvature * first.square()
        return 0.5 * (first.square() + second.square())


class GaussianMixturePotential(nn.Module):
    """Equal isotropic Gaussian mixture with a known tempered density."""

    def __init__(
        self,
        means: Tensor,
        *,
        component_variance: float = 0.16,
        equilibrium_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if means.ndim != 2 or component_variance <= 0 or equilibrium_temperature <= 0:
            raise ValueError("invalid Gaussian mixture configuration")
        self.register_buffer("means", means)
        self.component_variance = float(component_variance)
        self.equilibrium_temperature = float(equilibrium_temperature)

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(x.shape[0], -1)
        squared = (flat[:, None, :] - self.means[None, :, :]).square().sum(dim=2)
        scores = -squared / (
            2 * self.component_variance * self.equilibrium_temperature
        )
        return -self.equilibrium_temperature * (
            torch.logsumexp(scores, dim=1) - math.log(self.means.shape[0])
        )


class WarpedGaussianMixturePotential(nn.Module):
    """Area-preserving nonlinear warp of an equal Gaussian mixture.

    The latent-to-observed map is

    ``x_1 = z_1`` and ``x_2 = z_2 + curvature * z_1**2``.

    Its Jacobian determinant is one, so the observed-space potential is the
    latent mixture potential evaluated at the analytic inverse map.  This
    gives an exactly sampleable target whose equilibrium is unchanged across
    mobility ablations while its coordinate geometry can be made harder.
    """

    def __init__(
        self,
        means: Tensor,
        *,
        curvature: float,
        component_variance: float = 0.25,
        equilibrium_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.curvature = float(curvature)
        self.latent_potential = GaussianMixturePotential(
            means,
            component_variance=component_variance,
            equilibrium_temperature=equilibrium_temperature,
        )

    @property
    def means(self) -> Tensor:
        return self.latent_potential.means

    @property
    def component_variance(self) -> float:
        return self.latent_potential.component_variance

    def inverse_warp(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 2:
            raise ValueError("warped mixture is two-dimensional")
        return torch.stack(
            [x[:, 0], x[:, 1] - self.curvature * x[:, 0].square()], dim=1
        )

    def warp(self, z: Tensor) -> Tensor:
        if z.shape[-1] != 2:
            raise ValueError("warped mixture is two-dimensional")
        return torch.stack(
            [z[:, 0], z[:, 1] + self.curvature * z[:, 0].square()], dim=1
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.latent_potential(self.inverse_warp(x))

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        target_device = torch.device(device) if device is not None else self.means.device
        target_dtype = dtype if dtype is not None else self.means.dtype
        means = self.means.to(device=target_device, dtype=target_dtype)
        indices = torch.randint(
            means.shape[0], (count,), device=target_device, generator=generator
        )
        noise = torch.randn(
            count, 2, device=target_device, dtype=target_dtype, generator=generator
        )
        latent = means[indices] + self.component_variance**0.5 * noise
        return self.warp(latent)


class AnalyticWarpMobility(nn.Module):
    """Pull-forward mobility for :class:`WarpedGaussianMixturePotential`.

    If ``J`` is the warp Jacobian, this class uses ``G = J J^T``.  It has
    determinant one and the exact divergence ``(0, 2 * curvature)``.  With
    the Itô correction, its Langevin diffusion is precisely the nonlinear
    coordinate transform of Euclidean Langevin in latent space.
    """

    divergence_probe_cost = 0
    stochastic_heun_compatible = False

    def __init__(self, curvature: float) -> None:
        super().__init__()
        self.curvature = float(curvature)

    def _shear(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 2:
            raise ValueError("warp mobility is two-dimensional")
        return 2.0 * self.curvature * x[:, 0]

    def matrix(self, x: Tensor) -> Tensor:
        shear = self._shear(x)
        one = torch.ones_like(shear)
        return torch.stack(
            [one, shear, shear, one + shear.square()], dim=1
        ).reshape(-1, 2, 2)

    def diagonal(self, x: Tensor) -> Tensor:
        return torch.diagonal(self.matrix(x), dim1=1, dim2=2)

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return (self.matrix(x) @ vector.unsqueeze(-1)).squeeze(-1)

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        shear = self._shear(x)
        inverse = torch.stack(
            [1 + shear.square(), -shear, -shear, torch.ones_like(shear)], dim=1
        ).reshape(-1, 2, 2)
        return (inverse @ vector.unsqueeze(-1)).squeeze(-1)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        # The diffusion factor need not be the symmetric principal square
        # root; J is cheaper and satisfies J J^T = G exactly.
        shear = self._shear(x)
        return torch.stack(
            [noise[:, 0], shear * noise[:, 0] + noise[:, 1]], dim=1
        )

    def logdet(self, x: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def log_eigenvalues(self, x: Tensor) -> Tensor:
        return torch.linalg.eigvalsh(self.matrix(x)).clamp_min(1e-12).log()

    def divergence(self, x: Tensor, *, n_samples=1, create_graph=False) -> Tensor:
        del n_samples
        result = torch.stack(
            [torch.zeros_like(x[:, 0]), torch.full_like(x[:, 1], 2 * self.curvature)],
            dim=1,
        )
        return result if create_graph else result.detach()


class ScalarPotentialMLP(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = dimension
        for _ in range(depth):
            layers.extend([nn.Linear(previous, hidden_dim), nn.SiLU()])
            previous = hidden_dim
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x.reshape(x.shape[0], -1)).squeeze(1)


class FixedFullMobility(nn.Module):
    divergence_probe_cost = 0

    def __init__(self, matrix: Tensor) -> None:
        super().__init__()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        torch.linalg.cholesky(matrix)
        self.register_buffer("fixed_matrix", matrix)

    def matrix(self, x: Tensor) -> Tensor:
        return self.fixed_matrix.unsqueeze(0).expand(x.shape[0], -1, -1)

    def log_eigenvalues(self, x: Tensor) -> Tensor:
        values = torch.linalg.eigvalsh(self.fixed_matrix).log()
        return values.unsqueeze(0).expand(x.shape[0], -1)

    def diagonal(self, x: Tensor) -> Tensor:
        values = torch.diagonal(self.fixed_matrix)
        return values.unsqueeze(0).expand_as(x)

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return torch.einsum("ij,bj->bi", self.fixed_matrix, vector)

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        del x
        return torch.linalg.solve(self.fixed_matrix, vector.unsqueeze(-1)).squeeze(-1)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        del x
        root = torch.linalg.cholesky(self.fixed_matrix)
        return torch.einsum("ij,bj->bi", root, noise)

    def logdet(self, x: Tensor) -> Tensor:
        value = torch.linalg.slogdet(self.fixed_matrix).logabsdet
        return value.expand(x.shape[0])

    def divergence(self, x: Tensor, *, n_samples=1, create_graph=False) -> Tensor:
        del n_samples, create_graph
        return torch.zeros_like(x)


class ConvexRidgeMirrorPotential(nn.Module):
    """Strongly convex learned mirror potential with an analytic Hessian.

    The potential is a positive weighted sum of softplus ridge functions plus
    an isotropic quadratic term.  It is intentionally small and exact: in the
    low-dimensional inductive-bias experiment, its inverse Hessian is the
    mobility induced by a learned mirror map.  This is a controlled
    frozen-energy adaptation, not an implementation of any paper's full
    image-scale training pipeline.
    """

    def __init__(
        self,
        dimension: int,
        width: int = 128,
        *,
        log_base_bound: float = 2.0,
    ) -> None:
        super().__init__()
        if dimension < 1 or width < 1 or log_base_bound <= 0:
            raise ValueError("invalid mirror-potential configuration")
        self.dimension = int(dimension)
        self.width = int(width)
        self.log_base_bound = float(log_base_bound)
        self.directions = nn.Parameter(
            torch.randn(width, dimension) / math.sqrt(dimension)
        )
        self.bias = nn.Parameter(torch.zeros(width))
        # Near-zero positive ridge weights make the initial mirror close to
        # the Euclidean quadratic while preserving strict convexity.
        self.raw_weights = nn.Parameter(torch.full((width,), -5.0))
        self.raw_log_base = nn.Parameter(torch.zeros(()))

    def base_scale(self) -> Tensor:
        return torch.exp(
            self.log_base_bound * torch.tanh(self.raw_log_base)
        )

    def positive_weights(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_weights) / self.width

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(x.shape[0], -1)
        if flat.shape[1] != self.dimension:
            raise ValueError("input dimension does not match mirror potential")
        ridge = torch.nn.functional.softplus(
            flat @ self.directions.transpose(0, 1) + self.bias
        )
        quadratic = 0.5 * self.base_scale() * flat.square().sum(dim=1)
        return quadratic + ridge @ self.positive_weights()

    def gradient(self, x: Tensor) -> Tensor:
        flat = x.reshape(x.shape[0], -1)
        logits = flat @ self.directions.transpose(0, 1) + self.bias
        coefficients = torch.sigmoid(logits) * self.positive_weights()
        result = self.base_scale() * flat + coefficients @ self.directions
        return result.reshape_as(x)

    def hessian(self, x: Tensor) -> Tensor:
        flat = x.reshape(x.shape[0], -1)
        logits = flat @ self.directions.transpose(0, 1) + self.bias
        sigmoid = torch.sigmoid(logits)
        curvature = sigmoid * (1.0 - sigmoid) * self.positive_weights()
        identity = torch.eye(
            self.dimension, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        ridge_hessian = torch.einsum(
            "bk,ki,kj->bij", curvature, self.directions, self.directions
        )
        return self.base_scale() * identity + ridge_hessian


class InverseHessianMobility(nn.Module):
    """Mobility ``G=(nabla^2 Phi)^{-1}`` induced by a convex mirror map."""

    stochastic_heun_compatible = False

    def __init__(self, mirror_potential: ConvexRidgeMirrorPotential) -> None:
        super().__init__()
        self.mirror_potential = mirror_potential

    def matrix(self, x: Tensor) -> Tensor:
        return torch.linalg.inv(self.mirror_potential.hessian(x))

    def diagonal(self, x: Tensor) -> Tensor:
        return torch.diagonal(self.matrix(x), dim1=-2, dim2=-1).reshape_as(x)

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        flat = vector.reshape(vector.shape[0], -1)
        result = torch.linalg.solve(
            self.mirror_potential.hessian(x), flat.unsqueeze(-1)
        ).squeeze(-1)
        return result.reshape_as(vector)

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        flat = vector.reshape(vector.shape[0], -1)
        result = self.mirror_potential.hessian(x) @ flat.unsqueeze(-1)
        return result.squeeze(-1).reshape_as(vector)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        root = torch.linalg.cholesky(self.matrix(x))
        flat = noise.reshape(noise.shape[0], -1)
        return (root @ flat.unsqueeze(-1)).squeeze(-1).reshape_as(noise)

    def logdet(self, x: Tensor) -> Tensor:
        return -torch.linalg.slogdet(self.mirror_potential.hessian(x)).logabsdet

    def log_eigenvalues(self, x: Tensor) -> Tensor:
        return -torch.linalg.eigvalsh(
            self.mirror_potential.hessian(x)
        ).clamp_min(1e-12).log()

    def divergence(self, x: Tensor, *, n_samples=1, create_graph=False) -> Tensor:
        return hutchinson_divergence(
            self, x, n_samples=n_samples, create_graph=create_graph
        )


class AdditiveResidualSampler(nn.Module):
    """Adapter for an unrestricted additive residual drift.

    ``apply(x, grad V)`` returns ``grad V - r(x)``, so the shared sampler uses
    drift ``-grad V + r(x)`` and identity diffusion.  This is deliberately not
    an SPD mobility and has no equilibrium correction; it exists to measure
    the inductive-bias trade-off of a PEM-style residual field.
    """

    divergence_probe_cost = 0

    def __init__(self, residual_field: nn.Module) -> None:
        super().__init__()
        self.residual_field = residual_field

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return vector - self.residual_field(x).reshape_as(vector)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        del x
        return noise

    def divergence(self, x: Tensor, *, n_samples=1, create_graph=False) -> Tensor:
        del n_samples, create_graph
        return torch.zeros_like(x)


class AnalyticRotatingMobility(nn.Module):
    """Two-dimensional determinant-one mobility with rotating eigendirections."""

    stochastic_heun_compatible = False

    def __init__(self, anisotropy: float = 0.8, rotation_rate: float = 0.7) -> None:
        super().__init__()
        self.anisotropy = float(anisotropy)
        self.rotation_rate = float(rotation_rate)

    def _decomposition(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.shape[-1] != 2:
            raise ValueError("rotating mobility is two-dimensional")
        angle = self.rotation_rate * x[:, 0]
        cosine, sine = angle.cos(), angle.sin()
        vectors = torch.stack(
            [
                torch.stack([cosine, -sine], dim=1),
                torch.stack([sine, cosine], dim=1),
            ],
            dim=1,
        )
        log_scale = self.anisotropy * torch.tanh(x[:, 1])
        logs = torch.stack([log_scale, -log_scale], dim=1)
        return logs, vectors

    def log_eigenvalues(self, x: Tensor) -> Tensor:
        return self._decomposition(x)[0]

    def matrix(self, x: Tensor) -> Tensor:
        logs, vectors = self._decomposition(x)
        return (vectors * logs.exp().unsqueeze(1)) @ vectors.transpose(1, 2)

    def diagonal(self, x: Tensor) -> Tensor:
        return torch.diagonal(self.matrix(x), dim1=1, dim2=2)

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return (self.matrix(x) @ vector.unsqueeze(-1)).squeeze(-1)

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        logs, vectors = self._decomposition(x)
        inverse = (vectors * (-logs).exp().unsqueeze(1)) @ vectors.transpose(1, 2)
        return (inverse @ vector.unsqueeze(-1)).squeeze(-1)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        logs, vectors = self._decomposition(x)
        root = (vectors * (0.5 * logs).exp().unsqueeze(1)) @ vectors.transpose(1, 2)
        return (root @ noise.unsqueeze(-1)).squeeze(-1)

    def logdet(self, x: Tensor) -> Tensor:
        return self.log_eigenvalues(x).sum(dim=1)

    def divergence(self, x: Tensor, *, n_samples=1, create_graph=False) -> Tensor:
        return hutchinson_divergence(
            self, x, n_samples=n_samples, create_graph=create_graph
        )


class AnalyticDiagonalMobility(nn.Module):
    """Two-dimensional state-dependent mobility with exact divergence."""

    divergence_probe_cost = 0

    def __init__(self, strength: float = 0.8) -> None:
        super().__init__()
        self.strength = float(strength)

    def log_diagonal(self, x: Tensor) -> Tensor:
        scale = self.strength * torch.tanh(x[:, 0])
        return torch.stack([scale, -scale], dim=1)

    def diagonal(self, x: Tensor) -> Tensor:
        return self.log_diagonal(x).exp()

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return self.diagonal(x) * vector

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return vector / self.diagonal(x)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        return (0.5 * self.log_diagonal(x)).exp() * noise

    def logdet(self, x: Tensor) -> Tensor:
        return self.log_diagonal(x).sum(dim=1)

    def divergence(self, x: Tensor, *, n_samples=1, create_graph=False) -> Tensor:
        del n_samples
        first = self.diagonal(x)[:, 0]
        derivative = self.strength * (1 - torch.tanh(x[:, 0]).square()) * first
        result = torch.stack([derivative, torch.zeros_like(derivative)], dim=1)
        if not create_graph:
            result = result.detach()
        return result


class HutchinsonDivergenceWrapper(nn.Module):
    """Use stochastic divergence while delegating all other mobility methods."""

    @staticmethod
    def divergence_probe_cost(requested_probes: int) -> int:
        return requested_probes

    def __init__(self, mobility: nn.Module) -> None:
        super().__init__()
        self.mobility = mobility

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.mobility, name)

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return self.mobility.apply(x, vector)

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return self.mobility.inverse_apply(x, vector)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        return self.mobility.sqrt_apply(x, noise)

    def diagonal(self, x: Tensor) -> Tensor:
        return self.mobility.diagonal(x)

    def logdet(self, x: Tensor) -> Tensor:
        return self.mobility.logdet(x)

    def divergence(self, x: Tensor, *, n_samples=1, create_graph=False) -> Tensor:
        return hutchinson_divergence(
            self.mobility, x, n_samples=n_samples, create_graph=create_graph
        )


@dataclass
class SyntheticProblem:
    name: str
    dimension: int
    potential: nn.Module
    mobility: nn.Module
    representable: bool = True
    point_scale: float = 2.0

    def sample_points(
        self,
        count: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return self.point_scale * torch.randn(
            count,
            self.dimension,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def energy_gradient(self, x: Tensor, *, create_graph: bool = False) -> Tensor:
        x_grad = x if x.requires_grad else x.detach().requires_grad_(True)
        value = self.potential(x_grad)
        return torch.autograd.grad(
            value.sum(), x_grad, create_graph=create_graph
        )[0]

    def target_velocity(self, x: Tensor, *, create_graph: bool = False) -> Tensor:
        gradient = self.energy_gradient(x, create_graph=create_graph)
        velocity = -self.mobility.apply(x, gradient)
        return velocity if self.representable else -velocity

    def target_matrix(self, x: Tensor) -> Tensor:
        if hasattr(self.mobility, "matrix"):
            return self.mobility.matrix(x)
        return torch.diag_embed(self.mobility.diagonal(x))


def make_synthetic_problem(name: str) -> SyntheticProblem:
    normalized = name.lower().replace("_", "-")
    identity_precision = torch.eye(2)
    if normalized in {"constant", "anisotropic-gaussian"}:
        matrix = torch.tensor([[1.8, 0.55], [0.55, 0.75]])
        matrix = matrix / torch.linalg.det(matrix).sqrt()
        return SyntheticProblem(
            "constant",
            2,
            QuadraticPotential(identity_precision),
            FixedFullMobility(matrix),
        )
    if normalized in {"rotating", "rotating-anisotropy"}:
        return SyntheticProblem(
            "rotating",
            2,
            QuadraticPotential(identity_precision),
            AnalyticRotatingMobility(),
        )
    if normalized == "banana":
        return SyntheticProblem(
            "banana",
            2,
            BananaPotential(),
            AnalyticDiagonalMobility(),
        )
    if normalized in {"nonrepresentable", "negative-control"}:
        return SyntheticProblem(
            "nonrepresentable",
            2,
            QuadraticPotential(identity_precision),
            AnalyticRotatingMobility(),
            representable=False,
        )
    raise ValueError(f"unknown synthetic problem: {name}")


def dense_mobility_matrix(mobility: nn.Module, x: Tensor) -> Tensor:
    if hasattr(mobility, "matrix"):
        return mobility.matrix(x)
    return torch.diag_embed(mobility.diagonal(x).reshape(x.shape[0], -1))
