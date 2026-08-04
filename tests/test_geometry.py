import unittest

import torch
from torch import nn

from rem.geometry import (
    DiagonalPlusLowRankMobility,
    DiagonalMobility,
    GaugeFixedDiagonalMobility,
    GaugeFixedFullMobility,
    IdentityMobility,
    UnfixedLogDiagonalMobility,
    construct_spd_mapping,
    descent_condition_diagnostics,
    hutchinson_divergence,
    minimal_distortion_regularizer,
    mobility_diagnostics,
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


class CoordinateLogits(nn.Module):
    def forward(self, x):
        return x


class ZeroPackedMatrix(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.output_dimension = dimension * (dimension + 1) // 2

    def forward(self, x):
        return torch.zeros(
            x.shape[0], self.output_dimension, device=x.device, dtype=x.dtype
        )


class FixedLowRankParameters(nn.Module):
    def __init__(self, rank=1):
        super().__init__()
        self.rank = rank

    def forward(self, x):
        flat = x.reshape(x.shape[0], -1)
        diagonal = torch.zeros_like(x)
        factors = torch.zeros(
            x.shape[0], flat.shape[1], self.rank, device=x.device, dtype=x.dtype
        )
        factors[:, 0, 0] = 1.0
        return diagonal, factors


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

    def test_gauge_fixed_diagonal_has_unit_determinant(self):
        mobility = GaugeFixedDiagonalMobility(CoordinateLogits(), log_bound=1.0)
        x = torch.tensor([[0.2, -0.7, 1.1], [2.0, 0.5, -1.0]])
        torch.testing.assert_close(
            mobility.logdet(x),
            torch.zeros(x.shape[0]),
            atol=1e-6,
            rtol=0,
        )
        diagnostics = mobility_diagnostics(mobility, x)
        self.assertTrue(torch.all(diagnostics["min_eigenvalue"] > 0))

    def test_unfixed_diagonal_exposes_scale_gauge(self):
        mobility = UnfixedLogDiagonalMobility(CoordinateLogits(), log_bound=1.0)
        x = torch.tensor([[0.2, -0.7, 1.1], [2.0, 0.5, -1.0]])
        self.assertTrue(torch.all(mobility.diagonal(x) > 0))
        self.assertGreater(float(mobility.logdet(x).abs().max()), 1e-3)

    def test_full_mobility_starts_at_identity(self):
        mobility = GaugeFixedFullMobility(ZeroPackedMatrix(2), 2)
        x = torch.randn(4, 2)
        vector = torch.randn_like(x)
        torch.testing.assert_close(mobility.apply(x, vector), vector)
        torch.testing.assert_close(mobility.inverse_apply(x, vector), vector)
        torch.testing.assert_close(
            mobility.logdet(x), torch.zeros(4), atol=1e-6, rtol=0
        )

    def test_constructive_spd_mapping(self):
        gradient = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        target_velocity = torch.tensor([[-2.0, -1.0], [-1.0, -3.0]])
        matrix, valid = construct_spd_mapping(gradient, target_velocity)
        self.assertTrue(bool(valid.all()))
        mapped = torch.bmm(matrix, gradient.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(mapped, -target_velocity)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(matrix) > 0))

    def test_descent_condition_flags_impossible_sample(self):
        gradient = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        velocity = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        diagnostics = descent_condition_diagnostics(gradient, velocity)
        torch.testing.assert_close(
            diagnostics["violation_rate"], torch.tensor(0.5)
        )
        _, valid = construct_spd_mapping(gradient, velocity)
        torch.testing.assert_close(valid, torch.tensor([True, False]))

    def test_low_rank_mobility_is_gauge_fixed_and_invertible(self):
        mobility = DiagonalPlusLowRankMobility(
            FixedLowRankParameters(), rank=1, factor_scale=0.25
        )
        x = torch.randn(3, 4)
        vector = torch.randn_like(x)
        transformed = mobility.apply(x, vector)
        recovered = mobility.inverse_apply(x, transformed)
        torch.testing.assert_close(recovered, vector, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            mobility.logdet(x), torch.zeros(3), atol=1e-5, rtol=0
        )

    def test_minimal_distortion_is_zero_at_identity(self):
        x = torch.randn(4, 3)
        penalty = minimal_distortion_regularizer(IdentityMobility(), x)
        torch.testing.assert_close(penalty, torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
