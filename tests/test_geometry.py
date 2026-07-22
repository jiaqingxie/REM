import unittest

import torch
from torch import nn

from rem.geometry import (
    DiagonalMobility,
    IdentityMobility,
    hutchinson_divergence,
    riemannian_langevin_step,
    riemannian_transport_loss,
    riemannian_velocity,
)


class ZeroLogits(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class QuadraticDiagonalMobility(nn.Module):
    """G_ii(x) = 1 + x_i^2 / 2, whose divergence is exactly x."""

    def diagonal(self, x):
        return 1.0 + 0.5 * x.square()

    def apply(self, x, vector):
        return self.diagonal(x) * vector

    def sqrt_apply(self, x, noise):
        return self.diagonal(x).sqrt() * noise

    def divergence(self, x, *, n_samples=1, create_graph=False):
        return hutchinson_divergence(
            self,
            x,
            n_samples=n_samples,
            create_graph=create_graph,
        )


def quadratic_potential(x):
    return 0.5 * x.flatten(start_dim=1).square().sum(dim=1)


class GeometryTests(unittest.TestCase):
    def test_identity_recovers_euclidean_velocity(self):
        x = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        velocity = riemannian_velocity(
            quadratic_potential,
            IdentityMobility(),
            x,
            create_graph=False,
        )
        torch.testing.assert_close(velocity, -x)

    def test_bounded_diagonal_mobility(self):
        mobility = DiagonalMobility(
            ZeroLogits(),
            min_mobility=0.5,
            max_mobility=1.5,
        )
        x = torch.randn(4, 3)
        torch.testing.assert_close(mobility.diagonal(x), torch.ones_like(x))

        velocity = riemannian_velocity(
            quadratic_potential,
            mobility,
            x,
            create_graph=False,
        )
        torch.testing.assert_close(velocity, -x)

    def test_hutchinson_divergence_is_exact_for_coordinatewise_diagonal(self):
        x = torch.tensor([[0.2, -0.4], [1.0, 2.0]], requires_grad=True)
        probes = torch.ones((1, *x.shape))
        divergence = hutchinson_divergence(
            QuadraticDiagonalMobility(),
            x,
            probes=probes,
        )
        torch.testing.assert_close(divergence, x)

    def test_transport_loss_zero_for_matching_quadratic_velocity(self):
        x = torch.randn(8, 2)
        loss = riemannian_transport_loss(
            quadratic_potential,
            IdentityMobility(),
            x,
            -x,
        )
        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_deterministic_identity_step(self):
        x = torch.tensor([[1.0, -2.0]])
        updated = riemannian_langevin_step(
            quadratic_potential,
            IdentityMobility(),
            x,
            dt=0.1,
            epsilon=0.0,
            noise=torch.zeros_like(x),
        )
        torch.testing.assert_close(updated, 0.9 * x)


if __name__ == "__main__":
    unittest.main()
