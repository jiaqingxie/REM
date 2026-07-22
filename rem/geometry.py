"""Riemannian geometry primitives for Energy Matching.

The convention in this module is that ``G(x)`` is a symmetric positive-
definite mobility. The corresponding Riemannian metric is ``G(x)^{-1}``.
For a potential ``V`` and temperature ``epsilon``, the Itô diffusion

    dX = [-G grad(V) + epsilon div(G)] dt + sqrt(2 epsilon G) dW

has stationary density proportional to ``exp(-V / epsilon)`` under the usual
regularity and integrability assumptions.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor, nn


PotentialFn = Callable[[Tensor], Tensor]


class IdentityMobility(nn.Module):
    """Euclidean mobility ``G(x) = I`` used by standard Energy Matching."""

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        del x
        return vector

    def sqrt_apply(self, x: Tensor, noise: Tensor) -> Tensor:
        del x
        return noise

    def diagonal(self, x: Tensor) -> Tensor:
        return torch.ones_like(x)

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
    reduction: str = "mean",
) -> Tensor:
    """Squared-error transport loss for Riemannian Energy Matching."""

    prediction = riemannian_velocity(
        potential_fn,
        mobility,
        x_t,
        create_graph=True,
    )
    error = (prediction - target_velocity).square()

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


def _as_batch_scalar(value: float | Tensor, reference: Tensor) -> Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim != 1 or tensor.shape[0] != reference.shape[0]:
        raise ValueError("a non-scalar temperature must have shape (batch_size,)")
    return tensor.reshape(tensor.shape[0], *([1] * (reference.ndim - 1)))


def riemannian_langevin_step(
    potential_fn: PotentialFn,
    mobility: nn.Module,
    x: Tensor,
    *,
    dt: float,
    epsilon: float | Tensor,
    include_divergence_correction: bool = True,
    divergence_samples: int = 1,
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
        if include_divergence_correction:
            divergence = mobility.divergence(
                x_for_grad,
                n_samples=divergence_samples,
                create_graph=False,
            )
            drift = drift + epsilon_tensor * divergence

        if noise is None:
            noise = torch.randn_like(x_for_grad)
        elif noise.shape != x_for_grad.shape:
            raise ValueError("noise must have the same shape as x")

        diffusion = mobility.sqrt_apply(x_for_grad, noise)
        noise_scale = torch.sqrt(2.0 * epsilon_tensor * dt)
        updated = x_for_grad + dt * drift + noise_scale * diffusion

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
