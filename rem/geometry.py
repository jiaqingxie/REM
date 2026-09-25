"""Riemannian geometry primitives for Energy Matching.

The convention in this module is that ``G(x)`` is a symmetric positive-
definite mobility. The corresponding Riemannian metric is ``G(x)^{-1}``.
For a potential ``V`` and temperature ``epsilon``, the Itô diffusion

    dX = [-G grad(V) + epsilon div(G)] dt + sqrt(2 epsilon G) dW

has stationary density proportional to ``exp(-V / epsilon)`` under the usual
regularity and integrability assumptions.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor, nn


PotentialFn = Callable[[Tensor], Tensor]


def _flatten_batch(value: Tensor) -> Tensor:
    if value.ndim < 2:
        raise ValueError("REM tensors must include a batch dimension")
    return value.reshape(value.shape[0], -1)


def _restore_batch(value: Tensor, reference: Tensor) -> Tensor:
    return value.reshape_as(reference)


def _batch_scalar_like(value: Tensor, reference: Tensor) -> Tensor:
    return value.reshape(value.shape[0], *([1] * (reference.ndim - 1)))


class IdentityMobility(nn.Module):
    """Euclidean mobility ``G(x) = I`` used by standard Energy Matching."""

    divergence_probe_cost = 0

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        del x
        return vector

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        del x
        return noise

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        del x
        return vector

    def diagonal(self, x: Tensor) -> Tensor:
        return torch.ones_like(x)

    def logdet(self, x: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        del n_samples, create_graph
        return torch.zeros_like(x)


class AdditiveResidualTransport(nn.Module):
    """Capacity-matched non-geometric transport control.

    The sampler forms its deterministic drift as ``-apply(x, grad V)``.  This
    adapter therefore returns ``grad V - r_psi(x)``, giving the additive field
    ``-grad V + r_psi(x)`` while retaining identity diffusion.  It is
    deliberately *not* an SPD mobility and has no same-energy equilibrium
    guarantee if left active at positive temperature; image experiments gate
    it off before the Langevin refinement phase.
    """

    divergence_probe_cost = 0

    def __init__(self, residual_network: nn.Module) -> None:
        super().__init__()
        self.residual_network = residual_network

    def residual(self, x: Tensor) -> Tensor:
        value = self.residual_network(x)
        if value.shape != x.shape:
            raise ValueError(
                "residual_network must preserve the input shape; "
                f"received input {tuple(x.shape)} and output {tuple(value.shape)}"
            )
        return value

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return vector - self.residual(x)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        del x
        return noise

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        # This method exists only so the shared training/evaluation plumbing can
        # score the control in Euclidean norm.  It is not a metric inverse.
        del x
        return vector

    def diagonal(self, x: Tensor) -> Tensor:
        return torch.ones_like(x)

    def log_diagonal(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def logdet(self, x: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        del n_samples, create_graph
        return torch.zeros_like(x)


class DiagonalMobility(nn.Module):
    """Bounded state-dependent diagonal mobility.

    ``logit_network`` must return a tensor with the same shape as its input.
    A sigmoid maps its output into ``[min_mobility, max_mobility]``, ensuring
    uniform positive definiteness while avoiding an unconstrained scale.
    """

    def __init__(
        self,
        logit_network: nn.Module,
        *,
        min_mobility: float = 0.1,
        max_mobility: float = 10.0,
    ) -> None:
        super().__init__()
        if min_mobility <= 0:
            raise ValueError("min_mobility must be positive")
        if max_mobility <= min_mobility:
            raise ValueError("max_mobility must exceed min_mobility")

        self.logit_network = logit_network
        self.min_mobility = float(min_mobility)
        self.max_mobility = float(max_mobility)

    def diagonal(self, x: Tensor) -> Tensor:
        logits = self.logit_network(x)
        if logits.shape != x.shape:
            raise ValueError(
                "logit_network must preserve the input shape; "
                f"received input {tuple(x.shape)} and output {tuple(logits.shape)}"
            )
        scale = self.max_mobility - self.min_mobility
        return self.min_mobility + scale * torch.sigmoid(logits)

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return self.diagonal(x) * vector

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        return self.diagonal(x).sqrt() * noise

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return vector / self.diagonal(x)

    def log_diagonal(self, x: Tensor) -> Tensor:
        return self.diagonal(x).log()

    def logdet(self, x: Tensor) -> Tensor:
        return _flatten_batch(self.log_diagonal(x)).sum(dim=1)

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        return hutchinson_divergence(
            self,
            x,
            n_samples=n_samples,
            create_graph=create_graph,
        )


class GaugeFixedDiagonalMobility(nn.Module):
    """State-dependent diagonal mobility with an exact determinant-one gauge.

    The network produces unconstrained logits with the same shape as ``x``.
    Log-mobilities are smoothly bounded and centered over event dimensions,
    making ``log det G(x) = 0`` for every sample. If ``log_bound`` is ``a``,
    the centered log-eigenvalues lie in ``[-2a, 2a]``.
    """

    def __init__(self, logit_network: nn.Module, *, log_bound: float = 1.5) -> None:
        super().__init__()
        if log_bound <= 0:
            raise ValueError("log_bound must be positive")
        self.logit_network = logit_network
        self.log_bound = float(log_bound)

    def log_diagonal(self, x: Tensor) -> Tensor:
        logits = self.logit_network(x)
        if logits.shape != x.shape:
            raise ValueError(
                "logit_network must preserve the input shape; "
                f"received input {tuple(x.shape)} and output {tuple(logits.shape)}"
            )
        bounded = self.log_bound * torch.tanh(logits)
        flat = _flatten_batch(bounded)
        centered = flat - flat.mean(dim=1, keepdim=True)
        return _restore_batch(centered, x)

    def diagonal(self, x: Tensor) -> Tensor:
        return self.log_diagonal(x).exp()

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return self.diagonal(x) * vector

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return vector / self.diagonal(x)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        return (0.5 * self.log_diagonal(x)).exp() * noise

    def logdet(self, x: Tensor) -> Tensor:
        return _flatten_batch(self.log_diagonal(x)).sum(dim=1)

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        return hutchinson_divergence(
            self,
            x,
            n_samples=n_samples,
            create_graph=create_graph,
        )


class UnfixedLogDiagonalMobility(nn.Module):
    """Bounded log-diagonal mobility without determinant gauge fixing.

    This module is intentionally provided only for the ``REM-Unfixed``
    identifiability ablation. Zero network output still gives ``G = I``, but
    the state-dependent scalar degree of freedom is left unconstrained so the
    experiment can directly measure scale drift and seed instability.
    """

    def __init__(self, logit_network: nn.Module, *, log_bound: float = 1.5) -> None:
        super().__init__()
        if log_bound <= 0:
            raise ValueError("log_bound must be positive")
        self.logit_network = logit_network
        self.log_bound = float(log_bound)

    def log_diagonal(self, x: Tensor) -> Tensor:
        logits = self.logit_network(x)
        if logits.shape != x.shape:
            raise ValueError(
                "logit_network must preserve the input shape; "
                f"received input {tuple(x.shape)} and output {tuple(logits.shape)}"
            )
        return self.log_bound * torch.tanh(logits)

    def diagonal(self, x: Tensor) -> Tensor:
        return self.log_diagonal(x).exp()

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return self.diagonal(x) * vector

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return vector / self.diagonal(x)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        return (0.5 * self.log_diagonal(x)).exp() * noise

    def logdet(self, x: Tensor) -> Tensor:
        return _flatten_batch(self.log_diagonal(x)).sum(dim=1)

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        return hutchinson_divergence(
            self,
            x,
            n_samples=n_samples,
            create_graph=create_graph,
        )


class BoundedScalarMobility(nn.Module):
    """State-dependent isotropic mobility used as a step-size control.

    A nontrivial scalar mobility cannot simultaneously have pointwise
    determinant one, so this ablation is anchored by an explicit scale
    regularizer instead of the REM gauge.
    """

    def __init__(
        self,
        logit_network: nn.Module,
        *,
        min_mobility: float = 0.25,
        max_mobility: float = 4.0,
    ) -> None:
        super().__init__()
        if (
            min_mobility <= 0
            or max_mobility <= min_mobility
            or not min_mobility < 1.0 < max_mobility
        ):
            raise ValueError("scalar bounds must satisfy 0 < min < 1 < max")
        self.logit_network = logit_network
        self.min_mobility = float(min_mobility)
        self.max_mobility = float(max_mobility)
        log_min = math.log(self.min_mobility)
        log_max = math.log(self.max_mobility)
        unit_position = -log_min / (log_max - log_min)
        self.logit_offset = math.log(unit_position / (1 - unit_position))

    def scale(self, x: Tensor) -> Tensor:
        logits = self.logit_network(x)
        if logits.shape == x.shape:
            logits = _flatten_batch(logits).mean(dim=1)
        elif logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        elif logits.ndim != 1 or logits.shape[0] != x.shape[0]:
            raise ValueError("scalar logit network must return (B,), (B,1), or x.shape")
        log_min = math.log(self.min_mobility)
        log_range = math.log(self.max_mobility) - log_min
        log_scale = log_min + log_range * torch.sigmoid(logits + self.logit_offset)
        return log_scale.exp()

    def diagonal(self, x: Tensor) -> Tensor:
        return torch.ones_like(x) * _batch_scalar_like(self.scale(x), x)

    def log_diagonal(self, x: Tensor) -> Tensor:
        return self.diagonal(x).log()

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return _batch_scalar_like(self.scale(x), x) * vector

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return vector / _batch_scalar_like(self.scale(x), x)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        return _batch_scalar_like(self.scale(x).sqrt(), x) * noise

    def logdet(self, x: Tensor) -> Tensor:
        dimension = _flatten_batch(x).shape[1]
        return dimension * self.scale(x).log()

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        return hutchinson_divergence(
            self,
            x,
            n_samples=n_samples,
            create_graph=create_graph,
        )


class ConstantDiagonalMobility(nn.Module):
    """Learnable constant diagonal preconditioner with optional gauge fixing."""

    divergence_probe_cost = 0

    def __init__(
        self,
        dimension: int,
        *,
        log_bound: float = 1.5,
        gauge_fixed: bool = True,
    ) -> None:
        super().__init__()
        if dimension < 1 or log_bound <= 0:
            raise ValueError("dimension and log_bound must be positive")
        self.dimension = int(dimension)
        self.log_bound = float(log_bound)
        self.gauge_fixed = bool(gauge_fixed)
        self.raw_log_diagonal = nn.Parameter(torch.zeros(dimension))

    def _logs(self, x: Tensor) -> Tensor:
        if _flatten_batch(x).shape[1] != self.dimension:
            raise ValueError("input dimension does not match constant mobility")
        logs = self.log_bound * torch.tanh(self.raw_log_diagonal)
        if self.gauge_fixed:
            logs = logs - logs.mean()
        return logs.unsqueeze(0).expand(x.shape[0], -1)

    def log_diagonal(self, x: Tensor) -> Tensor:
        return _restore_batch(self._logs(x), x)

    def diagonal(self, x: Tensor) -> Tensor:
        return self.log_diagonal(x).exp()

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return self.diagonal(x) * vector

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return vector / self.diagonal(x)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        return (0.5 * self.log_diagonal(x)).exp() * noise

    def logdet(self, x: Tensor) -> Tensor:
        return self._logs(x).sum(dim=1)

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        del n_samples, create_graph
        return torch.zeros_like(x)


class TemperedDiagonalMobility(nn.Module):
    """Geodesically blend a diagonal mobility with the identity.

    ``strength=0`` is exactly Euclidean, ``strength=1`` is the wrapped
    mobility, and intermediate values scale its log-eigenvalues.  This is an
    evaluation-time diagnostic for determining whether a learned geometry is
    useful but over-applied along a long discretized trajectory.
    """

    def __init__(self, mobility: nn.Module, strength: float) -> None:
        super().__init__()
        if strength < 0:
            raise ValueError("mobility strength must be nonnegative")
        if not hasattr(mobility, "diagonal"):
            raise TypeError("tempering requires a diagonal mobility interface")
        self.mobility = mobility
        self.strength = float(strength)

    def divergence_probe_cost(self, requested_probes: int) -> int:
        return 0 if self.strength == 0 else int(requested_probes)

    def log_diagonal(self, x: Tensor) -> Tensor:
        if self.strength == 0:
            return torch.zeros_like(x)
        diagonal = self.mobility.diagonal(x).clamp_min(1e-12)
        return self.strength * diagonal.log()

    def diagonal(self, x: Tensor) -> Tensor:
        return self.log_diagonal(x).exp()

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return self.diagonal(x) * vector

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return vector / self.diagonal(x)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        return (0.5 * self.log_diagonal(x)).exp() * noise

    def logdet(self, x: Tensor) -> Tensor:
        return _flatten_batch(self.log_diagonal(x)).sum(dim=1)

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        if self.strength == 0:
            return torch.zeros_like(x)
        return hutchinson_divergence(
            self,
            x,
            n_samples=n_samples,
            create_graph=create_graph,
        )


class TemperedFullMobility(nn.Module):
    """Geodesically blend a full SPD mobility with the identity.

    The wrapped mobility must expose its symmetric matrix logarithm. Scaling
    that logarithm preserves eigenvectors, positive definiteness, and the
    determinant-one gauge while retaining off-diagonal geometry.
    """

    stochastic_heun_compatible = False

    def __init__(self, mobility: nn.Module, strength: float) -> None:
        super().__init__()
        if strength < 0:
            raise ValueError("mobility strength must be nonnegative")
        if not hasattr(mobility, "log_matrix"):
            raise TypeError("full tempering requires a log_matrix interface")
        self.mobility = mobility
        self.strength = float(strength)

    def divergence_probe_cost(self, requested_probes: int) -> int:
        return 0 if self.strength == 0 else int(requested_probes)

    def log_matrix(self, x: Tensor) -> Tensor:
        if self.strength == 0:
            dimension = _flatten_batch(x).shape[1]
            return torch.zeros(
                x.shape[0], dimension, dimension, device=x.device, dtype=x.dtype
            )
        return self.strength * self.mobility.log_matrix(x)

    def matrix(self, x: Tensor) -> Tensor:
        return torch.matrix_exp(self.log_matrix(x))

    def log_eigenvalues(self, x: Tensor) -> Tensor:
        return torch.linalg.eigvalsh(self.log_matrix(x))

    def diagonal(self, x: Tensor) -> Tensor:
        diagonal = torch.diagonal(self.matrix(x), dim1=-2, dim2=-1)
        return _restore_batch(diagonal, x)

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        result = self.matrix(x) @ _flatten_batch(vector).unsqueeze(-1)
        return _restore_batch(result.squeeze(-1), vector)

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        matrix_inverse = torch.matrix_exp(-self.log_matrix(x))
        result = matrix_inverse @ _flatten_batch(vector).unsqueeze(-1)
        return _restore_batch(result.squeeze(-1), vector)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        matrix_sqrt = torch.matrix_exp(0.5 * self.log_matrix(x))
        result = matrix_sqrt @ _flatten_batch(noise).unsqueeze(-1)
        return _restore_batch(result.squeeze(-1), noise)

    def logdet(self, x: Tensor) -> Tensor:
        return torch.diagonal(self.log_matrix(x), dim1=-2, dim2=-1).sum(dim=1)

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        if self.strength == 0:
            return torch.zeros_like(x)
        return hutchinson_divergence(
            self,
            x,
            n_samples=n_samples,
            create_graph=create_graph,
        )


def temper_mobility(mobility: nn.Module, strength: float) -> nn.Module:
    """Return the geometry-preserving identity interpolation for ``mobility``."""

    if strength < 0:
        raise ValueError("mobility strength must be nonnegative")
    if strength == 0:
        return IdentityMobility()
    if strength == 1:
        return mobility
    if hasattr(mobility, "log_matrix"):
        return TemperedFullMobility(mobility, strength)
    if hasattr(mobility, "log_diagonal") or hasattr(mobility, "diagonal"):
        return TemperedDiagonalMobility(mobility, strength)
    raise TypeError("mobility does not expose a supported tempering interface")


class GaugeFixedFullMobility(nn.Module):
    """Full SPD mobility for low-dimensional controlled experiments.

    The network may return ``(B,d,d)``, ``(B,d*d)``, or packed lower-triangle
    values ``(B,d*(d+1)/2)``. Its symmetric output is spectrally squashed,
    made traceless, and exponentiated. Consequently the mobility is SPD and
    has determinant one by construction.
    """

    stochastic_heun_compatible = False

    def __init__(
        self,
        matrix_network: nn.Module,
        dimension: int,
        *,
        log_eigenvalue_bound: float = 1.5,
    ) -> None:
        super().__init__()
        if dimension < 1 or log_eigenvalue_bound <= 0:
            raise ValueError("dimension and log_eigenvalue_bound must be positive")
        self.matrix_network = matrix_network
        self.dimension = int(dimension)
        self.log_eigenvalue_bound = float(log_eigenvalue_bound)

    def _raw_matrix(self, x: Tensor) -> Tensor:
        raw = self.matrix_network(x)
        batch = x.shape[0]
        dimension = self.dimension
        if raw.shape == (batch, dimension, dimension):
            return raw
        if raw.shape == (batch, dimension * dimension):
            return raw.reshape(batch, dimension, dimension)
        packed_size = dimension * (dimension + 1) // 2
        if raw.shape == (batch, packed_size):
            matrix = raw.new_zeros(batch, dimension, dimension)
            rows, cols = torch.tril_indices(dimension, dimension, device=raw.device)
            matrix[:, rows, cols] = raw
            return matrix
        raise ValueError(
            "full mobility network must return (B,d,d), (B,d*d), or "
            "(B,d*(d+1)/2)"
        )

    def log_matrix(self, x: Tensor) -> Tensor:
        if _flatten_batch(x).shape[1] != self.dimension:
            raise ValueError("input dimension does not match full mobility")
        raw = self._raw_matrix(x)
        symmetric = 0.5 * (raw + raw.transpose(-1, -2))
        trace = torch.diagonal(symmetric, dim1=-2, dim2=-1).sum(dim=1)
        identity = torch.eye(
            self.dimension, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        traceless = symmetric - (trace / self.dimension).view(-1, 1, 1) * identity
        squared_frobenius = traceless.square().sum(dim=(1, 2), keepdim=True)
        denominator = (
            self.log_eigenvalue_bound ** 2 + squared_frobenius
        ).sqrt()
        return self.log_eigenvalue_bound * traceless / denominator

    def log_eigendecomposition(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return torch.linalg.eigh(self.log_matrix(x))

    def matrix(self, x: Tensor) -> Tensor:
        return torch.matrix_exp(self.log_matrix(x))

    def log_eigenvalues(self, x: Tensor) -> Tensor:
        return self.log_eigendecomposition(x)[0]

    def diagonal(self, x: Tensor) -> Tensor:
        diagonal = torch.diagonal(self.matrix(x), dim1=-2, dim2=-1)
        return _restore_batch(diagonal, x)

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        flat = _flatten_batch(vector).unsqueeze(-1)
        result = self.matrix(x) @ flat
        return _restore_batch(result.squeeze(-1), vector)

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        matrix_inverse = torch.matrix_exp(-self.log_matrix(x))
        result = matrix_inverse @ _flatten_batch(vector).unsqueeze(-1)
        return _restore_batch(result.squeeze(-1), vector)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        matrix_sqrt = torch.matrix_exp(0.5 * self.log_matrix(x))
        result = matrix_sqrt @ _flatten_batch(noise).unsqueeze(-1)
        return _restore_batch(result.squeeze(-1), noise)

    def logdet(self, x: Tensor) -> Tensor:
        return torch.diagonal(self.log_matrix(x), dim1=-2, dim2=-1).sum(dim=1)

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        return hutchinson_divergence(
            self,
            x,
            n_samples=n_samples,
            create_graph=create_graph,
        )


def orthonormal_cosine_basis(
    dimension: int,
    rank: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return deterministic dense orthonormal directions in ``R^dimension``.

    The nonconstant DCT-II modes avoid privileging individual coordinates and
    make structured-mobility runs reproducible without storing a random basis.
    """

    if dimension < 2 or rank < 1 or rank >= dimension:
        raise ValueError("require dimension >= 2 and 1 <= rank < dimension")
    coordinates = torch.arange(dimension, dtype=dtype).add_(0.5).unsqueeze(1)
    frequencies = torch.arange(1, rank + 1, dtype=dtype).unsqueeze(0)
    return (2.0 / dimension) ** 0.5 * torch.cos(
        math.pi * coordinates * frequencies / dimension
    )


class LowRankCongruenceMobility(nn.Module):
    """Scalable determinant-one structured mobility with first-order mixing.

    Let ``D`` be a learned determinant-one diagonal matrix, ``B`` a fixed
    orthonormal ``d x r`` basis, and ``C(x)`` a learned ``d x r`` correction.
    This class represents

        G(x) = s(x) D(x)^{1/2} Q(x) Q(x)^T D(x)^{1/2},
        Q(x) = I + B C(x)^T,

    where the scalar ``s`` enforces ``det G = 1`` exactly.  Unlike the common
    ``D + U U^T`` parameterization, ``G`` has a nonzero first derivative with
    respect to ``C`` at ``C=0``.  A zero-output network therefore starts at
    identity while the off-diagonal branch can learn immediately.

    All drift, inverse, log-determinant, and noise-factor operations cost
    ``O(d r^2 + r^3)``, including small-matrix construction, and never
    materialize a dense ``d x d`` image matrix.
    """

    stochastic_heun_compatible = False

    def __init__(
        self,
        parameter_network: nn.Module,
        dimension: int,
        rank: int,
        *,
        log_bound: float = 1.5,
        correction_scale: float = 0.75,
    ) -> None:
        super().__init__()
        if dimension < 2 or rank < 1 or rank >= dimension:
            raise ValueError("require dimension >= 2 and 1 <= rank < dimension")
        if log_bound <= 0 or not 0 < correction_scale < 1:
            raise ValueError("log_bound must be positive and correction_scale in (0,1)")
        self.parameter_network = parameter_network
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.log_bound = float(log_bound)
        self.correction_scale = float(correction_scale)
        self.register_buffer(
            "basis",
            orthonormal_cosine_basis(self.dimension, self.rank),
            persistent=True,
        )

    def _raw_components(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if _flatten_batch(x).shape[1] != self.dimension:
            raise ValueError("input event dimension does not match structured mobility")
        output = self.parameter_network(x)
        if isinstance(output, tuple):
            diagonal_logits, corrections = output
            diagonal_logits = _flatten_batch(diagonal_logits)
            if corrections.shape != (x.shape[0], self.dimension, self.rank):
                raise ValueError("structured corrections must have shape (B,d,rank)")
        else:
            expected = self.dimension * (self.rank + 1)
            if output.shape != (x.shape[0], expected):
                raise ValueError(
                    f"parameter network must return (B,{expected}) for this input"
                )
            diagonal_logits = output[:, : self.dimension]
            corrections = output[:, self.dimension :].reshape(
                x.shape[0], self.dimension, self.rank
            )
        return diagonal_logits, corrections

    def components(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        diagonal_logits, raw_corrections = self._raw_components(x)
        bounded = self.log_bound * torch.tanh(diagonal_logits)
        diagonal_logs = bounded - bounded.mean(dim=1, keepdim=True)
        half_diagonal = torch.exp(0.5 * diagonal_logs)
        corrections = (
            self.correction_scale
            * torch.tanh(raw_corrections)
            / math.sqrt(self.dimension * self.rank)
        )
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        small = torch.eye(
            self.rank, device=x.device, dtype=x.dtype
        ).unsqueeze(0) + corrections.transpose(1, 2) @ basis
        sign, logabsdet = torch.linalg.slogdet(small)
        if bool(torch.any(sign <= 0).item()):
            raise RuntimeError("structured mobility lost orientation")
        log_scale = -(
            diagonal_logs.sum(dim=1) + 2.0 * logabsdet
        ) / self.dimension
        return half_diagonal, corrections, small, log_scale

    def _q_apply(self, vector: Tensor, corrections: Tensor, basis: Tensor) -> Tensor:
        return vector + torch.einsum(
            "dr,br->bd",
            basis,
            torch.einsum("bdr,bd->br", corrections, vector),
        )

    def _qt_apply(self, vector: Tensor, corrections: Tensor, basis: Tensor) -> Tensor:
        return vector + torch.einsum(
            "bdr,br->bd",
            corrections,
            torch.einsum("dr,bd->br", basis, vector),
        )

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        half, corrections, _, log_scale = self.components(x)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        middle = half * _flatten_batch(vector)
        middle = self._qt_apply(middle, corrections, basis)
        middle = self._q_apply(middle, corrections, basis)
        result = log_scale.exp().unsqueeze(1) * half * middle
        return _restore_batch(result, vector)

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        half, corrections, small, log_scale = self.components(x)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        middle = _flatten_batch(vector) / half
        # Q^{-1} y = y - B (I + C^T B)^{-1} C^T y.
        coefficient = torch.linalg.solve(
            small,
            torch.einsum("bdr,bd->br", corrections, middle).unsqueeze(-1),
        ).squeeze(-1)
        middle = middle - torch.einsum("dr,br->bd", basis, coefficient)
        # Q^{-T} y = y - C (I + B^T C)^{-1} B^T y.
        coefficient = torch.linalg.solve(
            small.transpose(1, 2),
            torch.einsum("dr,bd->br", basis, middle).unsqueeze(-1),
        ).squeeze(-1)
        middle = middle - torch.einsum("bdr,br->bd", corrections, coefficient)
        result = torch.exp(-log_scale).unsqueeze(1) * middle / half
        return _restore_batch(result, vector)

    def diagonal(self, x: Tensor) -> Tensor:
        half, corrections, _, log_scale = self.components(x)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        cross = 2.0 * torch.einsum("dr,bdr->bd", basis, corrections)
        correction_gram = corrections.transpose(1, 2) @ corrections
        quadratic = torch.einsum("dr,brs,ds->bd", basis, correction_gram, basis)
        q_diagonal = 1.0 + cross + quadratic
        result = log_scale.exp().unsqueeze(1) * half.square() * q_diagonal
        return _restore_batch(result, x)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        half, corrections, _, log_scale = self.components(x)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        transformed = self._q_apply(_flatten_batch(noise), corrections, basis)
        result = torch.exp(0.5 * log_scale).unsqueeze(1) * half * transformed
        return _restore_batch(result, noise)

    def sample_sqrt_noise(
        self,
        x: Tensor,
        *,
        noise: Optional[Tensor] = None,
        rank_noise: Optional[Tensor] = None,
    ) -> Tensor:
        del rank_noise
        if noise is None:
            noise = torch.randn_like(x)
        return self.sqrt_apply(x, noise)

    def logdet(self, x: Tensor) -> Tensor:
        half, _, small, log_scale = self.components(x)
        _, logabsdet = torch.linalg.slogdet(small)
        return (
            self.dimension * log_scale
            + 2.0 * half.log().sum(dim=1)
            + 2.0 * logabsdet
        )

    def distortion_regularizer(self, x: Tensor) -> Tensor:
        half, corrections, _, log_scale = self.components(x)
        centered_logs = 2.0 * half.log() + log_scale.unsqueeze(1)
        return centered_logs.square().mean() + 2.0 * corrections.square().sum(
            dim=(1, 2)
        ).mean() / self.dimension

    def dense_matrix(self, x: Tensor) -> Tensor:
        """Materialize ``G`` for diagnostics in moderate dimensions."""

        half, corrections, _, log_scale = self.components(x)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        identity = torch.eye(
            self.dimension, device=x.device, dtype=x.dtype
        ).expand(x.shape[0], -1, -1)
        q = identity + basis.unsqueeze(0) @ corrections.transpose(1, 2)
        factor = half.unsqueeze(2) * q
        return log_scale.exp().view(-1, 1, 1) * factor @ factor.transpose(1, 2)

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        return hutchinson_divergence(
            self,
            x,
            n_samples=n_samples,
            create_graph=create_graph,
        )


class DiagonalPlusLowRankMobility(nn.Module):
    """Scalable determinant-one mobility ``G = D + U U^T``.

    ``parameter_network`` returns ``(B, d * (rank + 1))`` or a tuple
    ``(diagonal_logits, factors)`` with shapes matching ``x`` and
    ``(B,d,rank)``. The determinant is normalized using the matrix
    determinant lemma, without materializing a dense ``d x d`` matrix.
    """

    stochastic_heun_compatible = False

    def __init__(
        self,
        parameter_network: nn.Module,
        rank: int,
        *,
        log_bound: float = 1.5,
        factor_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if rank < 1 or log_bound <= 0 or factor_scale <= 0:
            raise ValueError("rank, log_bound, and factor_scale must be positive")
        self.parameter_network = parameter_network
        self.rank = int(rank)
        self.log_bound = float(log_bound)
        self.factor_scale = float(factor_scale)

    def _raw_components(self, x: Tensor) -> tuple[Tensor, Tensor]:
        output = self.parameter_network(x)
        dimension = _flatten_batch(x).shape[1]
        if isinstance(output, tuple):
            diagonal_logits, factors = output
            diagonal_logits = _flatten_batch(diagonal_logits)
            if factors.shape != (x.shape[0], dimension, self.rank):
                raise ValueError("low-rank factors must have shape (B,d,rank)")
        else:
            expected = dimension * (self.rank + 1)
            if output.shape != (x.shape[0], expected):
                raise ValueError(
                    f"low-rank network must return (B,{expected}) for this input"
                )
            diagonal_logits = output[:, :dimension]
            factors = output[:, dimension:].reshape(x.shape[0], dimension, self.rank)
        return diagonal_logits, factors

    def components(self, x: Tensor) -> tuple[Tensor, Tensor]:
        diagonal_logits, raw_factors = self._raw_components(x)
        bounded = self.log_bound * torch.tanh(diagonal_logits)
        base_logs = bounded - bounded.mean(dim=1, keepdim=True)
        base_diagonal = base_logs.exp()
        factors = (
            self.factor_scale
            * torch.tanh(raw_factors)
            / (self.rank ** 0.5)
        )

        inverse_diagonal_factors = factors / base_diagonal.unsqueeze(-1)
        small = torch.eye(
            self.rank,
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0) + factors.transpose(1, 2) @ inverse_diagonal_factors
        total_logdet = base_logs.sum(dim=1) + torch.linalg.slogdet(small).logabsdet
        scale = torch.exp(-total_logdet / base_diagonal.shape[1])
        return scale.unsqueeze(1) * base_diagonal, scale.sqrt().view(-1, 1, 1) * factors

    def diagonal(self, x: Tensor) -> Tensor:
        diagonal, factors = self.components(x)
        result = diagonal + factors.square().sum(dim=-1)
        return _restore_batch(result, x)

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        diagonal, factors = self.components(x)
        flat = _flatten_batch(vector)
        result = diagonal * flat + torch.bmm(
            factors,
            torch.bmm(factors.transpose(1, 2), flat.unsqueeze(-1)),
        ).squeeze(-1)
        return _restore_batch(result, vector)

    def inverse_apply(self, x: Tensor, vector: Tensor) -> Tensor:
        diagonal, factors = self.components(x)
        flat = _flatten_batch(vector)
        diagonal_inverse = diagonal.reciprocal()
        dinv_vector = diagonal_inverse * flat
        dinv_factors = diagonal_inverse.unsqueeze(-1) * factors
        small = torch.eye(
            self.rank,
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0) + factors.transpose(1, 2) @ dinv_factors
        correction = torch.linalg.solve(
            small,
            torch.bmm(factors.transpose(1, 2), dinv_vector.unsqueeze(-1)),
        )
        result = dinv_vector - torch.bmm(dinv_factors, correction).squeeze(-1)
        return _restore_batch(result, vector)

    def sample_sqrt_noise(
        self,
        x: Tensor,
        *,
        noise: Optional[Tensor] = None,
        rank_noise: Optional[Tensor] = None,
    ) -> Tensor:
        diagonal, factors = self.components(x)
        if noise is None:
            noise = torch.randn_like(x)
        if noise.shape != x.shape:
            raise ValueError("noise must have the same shape as x")
        if rank_noise is None:
            rank_noise = torch.randn(
                x.shape[0], self.rank, device=x.device, dtype=x.dtype
            )
        if rank_noise.shape != (x.shape[0], self.rank):
            raise ValueError("rank_noise must have shape (B,rank)")
        flat = diagonal.sqrt() * _flatten_batch(noise)
        flat = flat + torch.bmm(factors, rank_noise.unsqueeze(-1)).squeeze(-1)
        return _restore_batch(flat, x)

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        dimension = _flatten_batch(x).shape[1]
        if dimension > 256:
            raise RuntimeError(
                "deterministic dense square-root application is limited to d<=256; "
                "use sample_sqrt_noise for scalable sampling"
            )
        diagonal, factors = self.components(x)
        dense = torch.diag_embed(diagonal) + factors @ factors.transpose(1, 2)
        eigenvalues, eigenvectors = torch.linalg.eigh(dense)
        square_root = (
            eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(-2)
        ) @ eigenvectors.transpose(1, 2)
        result = square_root @ _flatten_batch(noise).unsqueeze(-1)
        return _restore_batch(result.squeeze(-1), noise)

    def logdet(self, x: Tensor) -> Tensor:
        diagonal, factors = self.components(x)
        inverse_diagonal_factors = factors / diagonal.unsqueeze(-1)
        small = torch.eye(
            self.rank,
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0) + factors.transpose(1, 2) @ inverse_diagonal_factors
        return diagonal.log().sum(dim=1) + torch.linalg.slogdet(small).logabsdet

    def distortion_regularizer(self, x: Tensor) -> Tensor:
        """Scalable near-identity penalty including off-diagonal factors."""

        diagonal, factors = self.components(x)
        diagonal_penalty = diagonal.log().square().mean()
        factor_penalty = factors.square().sum(dim=2).mean()
        return diagonal_penalty + factor_penalty

    def divergence(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        create_graph: bool = False,
    ) -> Tensor:
        return hutchinson_divergence(
            self,
            x,
            n_samples=n_samples,
            create_graph=create_graph,
        )


def _energy_gradient(
    potential_fn: PotentialFn,
    x: Tensor,
    *,
    create_graph: bool,
) -> tuple[Tensor, Tensor]:
    if x.requires_grad:
        x_for_grad = x
    else:
        x_for_grad = x.detach().requires_grad_(True)

    potential = potential_fn(x_for_grad)
    if potential.ndim > 1:
        potential = potential.reshape(potential.shape[0], -1).sum(dim=1)

    gradient = torch.autograd.grad(
        potential.sum(),
        x_for_grad,
        create_graph=create_graph,
    )[0]
    return x_for_grad, gradient


def riemannian_velocity(
    potential_fn: PotentialFn,
    mobility: nn.Module,
    x: Tensor,
    *,
    create_graph: bool = True,
) -> Tensor:
    """Return the deterministic REM velocity ``-G(x) grad V(x)``."""

    x_for_grad, gradient = _energy_gradient(
        potential_fn,
        x,
        create_graph=create_graph,
    )
    return -mobility.apply(x_for_grad, gradient)


def riemannian_transport_loss(
    potential_fn: PotentialFn,
    mobility: nn.Module,
    x_t: Tensor,
    target_velocity: Tensor,
    *,
    metric_weighted: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """Transport loss for Riemannian Energy Matching.

    If ``metric_weighted`` is true, each per-sample residual is measured in
    the inverse-mobility norm. Gauge fixing is required when this option is
    used with a learned mobility; otherwise the metric could rescale the loss.
    """

    prediction = riemannian_velocity(
        potential_fn,
        mobility,
        x_t,
        create_graph=True,
    )
    residual = prediction - target_velocity

    if metric_weighted:
        if not hasattr(mobility, "inverse_apply"):
            raise TypeError("metric-weighted loss requires mobility.inverse_apply")
        inverse_residual = mobility.inverse_apply(x_t, residual)
        per_sample = (
            _flatten_batch(residual) * _flatten_batch(inverse_residual)
        ).mean(dim=1)
        if reduction == "mean":
            return per_sample.mean()
        if reduction == "sum":
            return per_sample.sum()
        if reduction == "none":
            return per_sample
        raise ValueError(f"unsupported reduction: {reduction}")

    error = residual.square()

    if reduction == "mean":
        return error.mean()
    if reduction == "sum":
        return error.sum()
    if reduction == "none":
        return error
    raise ValueError(f"unsupported reduction: {reduction}")


def hutchinson_divergence(
    mobility: nn.Module,
    x: Tensor,
    *,
    n_samples: int = 1,
    create_graph: bool = False,
    probes: Optional[Tensor] = None,
) -> Tensor:
    """Estimate ``div G`` with Hutchinson probes and Jacobian-vector products.

    For a Rademacher probe ``xi``, the estimator is

        J_x [G(x) xi] xi.

    Its expectation is the vector whose i-th component is
    ``sum_j partial_j G_ij``. ``probes`` can be supplied for deterministic
    tests and must have shape ``(n_samples, *x.shape)``.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be at least one")
    if probes is not None and probes.shape != (n_samples, *x.shape):
        raise ValueError(
            f"probes must have shape {(n_samples, *x.shape)}, "
            f"received {tuple(probes.shape)}"
        )

    estimates = []
    for sample_index in range(n_samples):
        if probes is None:
            probe = torch.empty_like(x).bernoulli_(0.5).mul_(2).sub_(1)
        else:
            probe = probes[sample_index].to(device=x.device, dtype=x.dtype)

        def probe_field(value: Tensor) -> Tensor:
            return mobility.apply(value, probe)

        _, directional_derivative = torch.autograd.functional.jvp(
            probe_field,
            x,
            probe,
            create_graph=create_graph,
            strict=False,
        )
        estimates.append(directional_derivative)

    return torch.stack(estimates, dim=0).mean(dim=0)


def exact_mobility_divergence(
    mobility: nn.Module,
    x: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Compute ``div G`` exactly by summing coordinate JVPs.

    This costs one JVP per state dimension, so it is intended for small
    latent spaces such as the 16-dimensional AAV experiments. Known constant
    mobilities can expose ``divergence_probe_cost = 0`` to skip the JVPs.
    """

    dimension = _flatten_batch(x).shape[1]
    probe_cost = getattr(mobility, "divergence_probe_cost", None)
    if probe_cost == 0:
        return mobility.divergence(x, n_samples=1, create_graph=create_graph)

    divergence = torch.zeros_like(x)
    for coordinate in range(dimension):
        basis = torch.zeros_like(x)
        basis.reshape(x.shape[0], -1)[:, coordinate] = 1

        def column_field(value: Tensor) -> Tensor:
            return mobility.apply(value, basis)

        _, directional_derivative = torch.autograd.functional.jvp(
            column_field,
            x,
            basis,
            create_graph=create_graph,
            strict=False,
        )
        divergence = divergence + directional_derivative
    return divergence


def _mobility_divergence(
    mobility: nn.Module,
    x: Tensor,
    *,
    method: str,
    n_samples: int,
    create_graph: bool,
) -> Tensor:
    normalized = method.lower().replace("_", "-")
    if normalized == "hutchinson":
        return mobility.divergence(
            x,
            n_samples=n_samples,
            create_graph=create_graph,
        )
    if normalized == "exact":
        return exact_mobility_divergence(
            mobility,
            x,
            create_graph=create_graph,
        )
    raise ValueError(f"unknown divergence method: {method}")


def descent_condition_diagnostics(
    energy_gradient: Tensor,
    target_velocity: Tensor,
    *,
    tolerance: float = 0.0,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Measure the necessary descent condition for an SPD representation.

    For ``u = -G grad(V)`` with ``G`` SPD, ``u^T grad(V)`` must be negative
    away from zeros. Returned tensors retain one entry per sample except for
    ``violation_rate``.
    """

    if energy_gradient.shape != target_velocity.shape:
        raise ValueError("energy_gradient and target_velocity must have equal shapes")
    gradient_flat = _flatten_batch(energy_gradient)
    velocity_flat = _flatten_batch(target_velocity)
    signed_dot = (gradient_flat * velocity_flat).sum(dim=1)
    gradient_norm = gradient_flat.norm(dim=1)
    velocity_norm = velocity_flat.norm(dim=1)
    active = (gradient_norm > eps) & (velocity_norm > eps)
    violation = active & (signed_dot >= -tolerance)
    denominator = (gradient_norm * velocity_norm).clamp_min(eps)
    cosine = signed_dot / denominator
    if active.any():
        violation_rate = violation[active].to(gradient_flat.dtype).mean()
    else:
        violation_rate = gradient_flat.new_tensor(float("nan"))
    return {
        "signed_dot": signed_dot,
        "cosine": cosine,
        "active": active,
        "violation": violation,
        "violation_rate": violation_rate,
    }


def construct_spd_mapping(
    energy_gradient: Tensor,
    target_velocity: Tensor,
    *,
    perpendicular_regularization: float = 1.0,
    descent_margin: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Construct an SPD matrix satisfying ``G grad(V) = -u`` pointwise.

    The construction is valid when ``-u`` has a strictly positive inner
    product with ``grad(V)``. Invalid samples receive the identity matrix and
    are marked false in the returned validity mask. Under a uniform descent
    margin and bounded vector norms, the construction is uniformly bounded.
    """

    if energy_gradient.shape != target_velocity.shape:
        raise ValueError("energy_gradient and target_velocity must have equal shapes")
    if perpendicular_regularization <= 0 or descent_margin < 0:
        raise ValueError("regularization must be positive and margin nonnegative")

    gradient = _flatten_batch(energy_gradient)
    mapped = -_flatten_batch(target_velocity)
    batch, dimension = gradient.shape
    norm = gradient.norm(dim=1)
    inner = (gradient * mapped).sum(dim=1)
    valid = (norm > descent_margin) & (inner > descent_margin)
    safe_norm = norm.clamp_min(max(descent_margin, torch.finfo(norm.dtype).eps))
    unit = gradient / safe_norm.unsqueeze(1)
    coefficient = inner / safe_norm.square()
    safe_coefficient = coefficient.clamp_min(
        max(descent_margin, torch.finfo(coefficient.dtype).eps)
    )
    perpendicular = mapped / safe_norm.unsqueeze(1) - coefficient.unsqueeze(1) * unit

    identity = torch.eye(dimension, device=gradient.device, dtype=gradient.dtype)
    identity = identity.unsqueeze(0).expand(batch, -1, -1)
    unit_outer = unit.unsqueeze(2) * unit.unsqueeze(1)
    cross = unit.unsqueeze(2) * perpendicular.unsqueeze(1)
    cross = cross + perpendicular.unsqueeze(2) * unit.unsqueeze(1)
    perpendicular_scale = (
        perpendicular.square().sum(dim=1) / safe_coefficient
        + perpendicular_regularization
    )
    matrix = (
        safe_coefficient.view(-1, 1, 1) * unit_outer
        + cross
        + perpendicular_scale.view(-1, 1, 1) * (identity - unit_outer)
    )
    matrix = torch.where(valid.view(-1, 1, 1), matrix, identity)
    return matrix, valid


def minimal_distortion_regularizer(mobility: nn.Module, x: Tensor) -> Tensor:
    """Squared log-eigenvalue distance from the Euclidean mobility."""

    if hasattr(mobility, "distortion_regularizer"):
        return mobility.distortion_regularizer(x)
    if hasattr(mobility, "log_eigenvalues"):
        logs = mobility.log_eigenvalues(x)
    elif hasattr(mobility, "log_matrix"):
        logs = mobility.log_matrix(x)
    elif hasattr(mobility, "log_diagonal"):
        logs = _flatten_batch(mobility.log_diagonal(x))
    else:
        logs = _flatten_batch(mobility.diagonal(x).clamp_min(1e-12).log())
    return logs.square().mean()


def mobility_diagnostics(mobility: nn.Module, x: Tensor) -> dict[str, Tensor]:
    """Return per-sample gauge and conditioning diagnostics."""

    if hasattr(mobility, "log_eigenvalues"):
        log_eigenvalues = mobility.log_eigenvalues(x)
        eigenvalues = log_eigenvalues.exp()
    else:
        eigenvalues = _flatten_batch(mobility.diagonal(x))
        log_eigenvalues = eigenvalues.clamp_min(1e-12).log()
    minimum = eigenvalues.min(dim=1).values
    maximum = eigenvalues.max(dim=1).values
    if hasattr(mobility, "logdet"):
        logdet = mobility.logdet(x)
    else:
        logdet = log_eigenvalues.sum(dim=1)
    return {
        "logdet": logdet,
        "min_eigenvalue": minimum,
        "max_eigenvalue": maximum,
        "condition_number": maximum / minimum.clamp_min(1e-12),
        "log_distance_from_identity": log_eigenvalues.square().mean(dim=1).sqrt(),
    }


def _as_batch_scalar(value: float | Tensor, reference: Tensor) -> Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim != 1 or tensor.shape[0] != reference.shape[0]:
        raise ValueError("a non-scalar temperature must have shape (batch_size,)")
    return tensor.reshape(tensor.shape[0], *([1] * (reference.ndim - 1)))


def _temperature_is_nonzero(value: float | Tensor) -> bool:
    """Return whether a scalar or per-sample temperature contains work."""

    if isinstance(value, Tensor):
        return bool(torch.any(value != 0).item())
    return float(value) != 0.0


def riemannian_langevin_step(
    potential_fn: PotentialFn,
    mobility: nn.Module,
    x: Tensor,
    *,
    dt: float,
    epsilon: float | Tensor,
    include_divergence_correction: bool = True,
    divergence_samples: int = 1,
    divergence_method: str = "hutchinson",
    noise: Optional[Tensor] = None,
) -> Tensor:
    """Perform one Euler-Maruyama step of the equilibrium-correct REM SDE.

    Sampling is intentionally detached from the parameter graph, matching the
    stop-gradient negative chains used by the imported Energy Matching code.
    """

    if dt <= 0:
        raise ValueError("dt must be positive")

    with torch.enable_grad():
        x_for_grad, gradient = _energy_gradient(
            potential_fn,
            x.detach(),
            create_graph=False,
        )
        drift = -mobility.apply(x_for_grad, gradient)

        epsilon_tensor = _as_batch_scalar(epsilon, x_for_grad)
        has_temperature = _temperature_is_nonzero(epsilon)
        if include_divergence_correction and has_temperature:
            divergence = _mobility_divergence(
                mobility,
                x_for_grad,
                method=divergence_method,
                n_samples=divergence_samples,
                create_graph=False,
            )
            drift = drift + epsilon_tensor * divergence

        if not has_temperature:
            diffusion = torch.zeros_like(x_for_grad)
        elif hasattr(mobility, "sample_sqrt_noise"):
            diffusion = mobility.sample_sqrt_noise(x_for_grad, noise=noise)
        else:
            if noise is None:
                noise = torch.randn_like(x_for_grad)
            elif noise.shape != x_for_grad.shape:
                raise ValueError("noise must have the same shape as x")
            diffusion = mobility.sqrt_apply(x_for_grad, noise)
        noise_scale = torch.sqrt(2.0 * epsilon_tensor * dt)
        updated = x_for_grad + dt * drift + noise_scale * diffusion

    return updated.detach()


def riemannian_langevin_heun_step(
    potential_fn: PotentialFn,
    next_potential_fn: PotentialFn,
    mobility: nn.Module,
    x: Tensor,
    *,
    dt: float,
    epsilon: float | Tensor,
    next_epsilon: float | Tensor,
    next_mobility: Optional[nn.Module] = None,
    include_divergence_correction: bool = True,
    divergence_samples: int = 1,
    divergence_method: str = "hutchinson",
    noise: Optional[Tensor] = None,
) -> Tensor:
    """One stochastic Heun step for diagonal Riemannian mobility.

    The equilibrium-correct Itô drift contains ``epsilon * div G``.  For a
    diagonal diffusion square root, the equivalent Stratonovich drift used by
    Heun contains ``0.5 * epsilon * div G``.  The same Brownian increment is
    reused in the predictor and corrector.

    Disabling the flag removes this explicit Stratonovich term. Multiplicative
    noise still induces ``0.5 * epsilon * div G`` in the equivalent Ito drift.
    Use Euler-Maruyama to ablate the entire Ito divergence correction.
    State-dependent non-diagonal factors require their own conversion and
    must declare ``stochastic_heun_compatible = False``.
    """

    if dt <= 0:
        raise ValueError("dt must be positive")
    if next_mobility is None:
        next_mobility = mobility
    current_requires_full_sqrt = _temperature_is_nonzero(epsilon) and (
        not getattr(mobility, "stochastic_heun_compatible", True)
        or hasattr(mobility, "sample_sqrt_noise")
    )
    next_requires_full_sqrt = _temperature_is_nonzero(next_epsilon) and (
        not getattr(next_mobility, "stochastic_heun_compatible", True)
        or hasattr(next_mobility, "sample_sqrt_noise")
    )
    if current_requires_full_sqrt or next_requires_full_sqrt:
        raise TypeError(
            "stochastic Heun currently supports diagonal mobility square roots only"
        )
    if noise is None:
        noise = torch.randn_like(x)
    elif noise.shape != x.shape:
        raise ValueError("noise must have the same shape as x")

    def coefficients(
        value: Tensor,
        potential: PotentialFn,
        temperature: float | Tensor,
        local_mobility: nn.Module,
    ) -> tuple[Tensor, Tensor]:
        value_for_grad, gradient = _energy_gradient(
            potential,
            value.detach(),
            create_graph=False,
        )
        temperature_tensor = _as_batch_scalar(temperature, value_for_grad)
        has_temperature = _temperature_is_nonzero(temperature)
        drift = -local_mobility.apply(value_for_grad, gradient)
        if include_divergence_correction and has_temperature:
            divergence = _mobility_divergence(
                local_mobility,
                value_for_grad,
                method=divergence_method,
                n_samples=divergence_samples,
                create_graph=False,
            )
            drift = drift + 0.5 * temperature_tensor * divergence
        if has_temperature:
            diffusion = local_mobility.sqrt_apply(value_for_grad, noise)
            increment = torch.sqrt(2.0 * temperature_tensor * dt) * diffusion
        else:
            increment = torch.zeros_like(value_for_grad)
        return drift, increment

    with torch.enable_grad():
        drift, increment = coefficients(x, potential_fn, epsilon, mobility)
        predictor = x.detach() + dt * drift + increment
        next_drift, next_increment = coefficients(
            predictor,
            next_potential_fn,
            next_epsilon,
            next_mobility,
        )
        updated = x.detach() + 0.5 * dt * (drift + next_drift)
        updated = updated + 0.5 * (increment + next_increment)
    return updated.detach()


def mobility_scale_regularizer(
    mobility: nn.Module,
    x: Tensor,
    *,
    target_mean: float = 1.0,
    log_variance_weight: float = 0.01,
) -> Tensor:
    """Regularize diagonal mobility scale without forcing it to be constant."""

    diagonal = mobility.diagonal(x)
    mean_penalty = (diagonal.mean() - target_mean).square()
    log_diagonal = diagonal.log()
    spread_penalty = log_diagonal.var(unbiased=False)
    return mean_penalty + log_variance_weight * spread_penalty
