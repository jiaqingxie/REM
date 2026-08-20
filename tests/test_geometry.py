import unittest

import torch
from torch import nn

from rem.geometry import (
    DiagonalPlusLowRankMobility,
    DiagonalMobility,
    GaugeFixedDiagonalMobility,
    GaugeFixedFullMobility,
    IdentityMobility,
    TemperedFullMobility,
    UnfixedLogDiagonalMobility,
    construct_spd_mapping,
    descent_condition_diagnostics,
    exact_mobility_divergence,
    hutchinson_divergence,
    minimal_distortion_regularizer,
    mobility_diagnostics,
    TemperedDiagonalMobility,
    temper_mobility,
    riemannian_langevin_step,
    riemannian_langevin_heun_step,
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


class ExactQuadraticDiagonalMobility(QuadraticDiagonalMobility):
    def divergence(self, x, *, n_samples=1, create_graph=False):
        del n_samples, create_graph
        return x


class ConstantScaleMobility(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def apply(self, x, vector):
        return self.scale * vector

    def sqrt_apply(self, x, noise):
        return self.scale**0.5 * noise

    def divergence(self, x, *, n_samples=1, create_graph=False):
        del n_samples, create_graph
        return torch.zeros_like(x)


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


class FixedPackedMatrix(nn.Module):
    def forward(self, x):
        # Lower-triangle packing for [[0, 0], [2, 0]], symmetrized downstream.
        return x.new_tensor([0.0, 2.0, 0.0]).expand(x.shape[0], -1)


class OuterProductMobility(nn.Module):
    """G(x) = I + 0.5 x x^T, with div(G) = 1.5 x in two dimensions."""

    def apply(self, x, vector):
        flat_x = x.reshape(x.shape[0], -1)
        flat_vector = vector.reshape(vector.shape[0], -1)
        projection = (flat_x * flat_vector).sum(dim=1, keepdim=True)
        result = flat_vector + 0.5 * flat_x * projection
        return result.reshape_as(vector)


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
    def test_tempered_diagonal_mobility_interpolates_to_identity(self):
        base = QuadraticDiagonalMobility()
        x = torch.tensor([[0.5, -1.0]])
        vector = torch.tensor([[2.0, 3.0]])

        identity = TemperedDiagonalMobility(base, 0.0)
        torch.testing.assert_close(identity.apply(x, vector), vector)
        torch.testing.assert_close(identity.divergence(x), torch.zeros_like(x))

        half = TemperedDiagonalMobility(base, 0.5)
        torch.testing.assert_close(
            half.diagonal(x).square(), base.diagonal(x), rtol=1e-6, atol=1e-6
        )

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

    def test_exact_divergence_recovers_full_matrix_cross_terms(self):
        x = torch.tensor([[0.2, -0.4], [1.0, 2.0]], requires_grad=True)
        divergence = exact_mobility_divergence(OuterProductMobility(), x)
        torch.testing.assert_close(divergence, 1.5 * x)

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

    def test_langevin_step_can_use_exact_divergence(self):
        x = torch.tensor([[0.4, -0.7]])
        dt = 0.05
        epsilon = 0.2
        mobility = QuadraticDiagonalMobility()
        diagonal = mobility.diagonal(x)
        expected = x + dt * (-diagonal * x + epsilon * x)
        actual = riemannian_langevin_step(
            quadratic_potential,
            mobility,
            x,
            dt=dt,
            epsilon=epsilon,
            divergence_method="exact",
            noise=torch.zeros_like(x),
        )
        torch.testing.assert_close(actual, expected)

    def test_deterministic_heun_has_second_order_quadratic_update(self):
        x = torch.tensor([[1.0, -2.0]])
        updated = riemannian_langevin_heun_step(
            quadratic_potential,
            quadratic_potential,
            IdentityMobility(),
            x,
            dt=0.1,
            epsilon=0.0,
            next_epsilon=0.0,
            noise=torch.zeros_like(x),
        )
        torch.testing.assert_close(updated, (1.0 - 0.1 + 0.5 * 0.1**2) * x)

    def test_heun_can_switch_mobility_at_phase_boundary(self):
        x = torch.tensor([[1.0, -2.0]])
        updated = riemannian_langevin_heun_step(
            quadratic_potential,
            quadratic_potential,
            ConstantScaleMobility(2.0),
            x,
            dt=0.1,
            epsilon=0.0,
            next_epsilon=0.0,
            next_mobility=IdentityMobility(),
            noise=torch.zeros_like(x),
        )
        torch.testing.assert_close(updated, 0.86 * x)

    def test_stochastic_heun_uses_shared_noise_and_stratonovich_correction(self):
        x = torch.tensor([[0.4, -0.7]])
        noise = torch.tensor([[0.25, -1.5]])
        dt = 0.05
        epsilon = 0.2
        mobility = ExactQuadraticDiagonalMobility()

        def coefficients(value):
            diagonal = mobility.diagonal(value)
            drift = -diagonal * value + 0.5 * epsilon * value
            increment = (2.0 * epsilon * dt) ** 0.5 * diagonal.sqrt() * noise
            return drift, increment

        drift, increment = coefficients(x)
        predictor = x + dt * drift + increment
        next_drift, next_increment = coefficients(predictor)
        expected = x + 0.5 * dt * (drift + next_drift)
        expected = expected + 0.5 * (increment + next_increment)

        actual = riemannian_langevin_heun_step(
            quadratic_potential,
            quadratic_potential,
            mobility,
            x,
            dt=dt,
            epsilon=epsilon,
            next_epsilon=epsilon,
            noise=noise,
        )
        torch.testing.assert_close(actual, expected)

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

    def test_full_mobility_tempering_retains_off_diagonal_geometry(self):
        base = GaugeFixedFullMobility(FixedPackedMatrix(), 2)
        mobility = temper_mobility(base, 0.5)
        self.assertIsInstance(mobility, TemperedFullMobility)
        x = torch.randn(3, 2)
        expected = torch.matrix_exp(0.5 * base.log_matrix(x))
        torch.testing.assert_close(mobility.matrix(x), expected)
        self.assertGreater(float(mobility.matrix(x)[:, 0, 1].abs().min()), 1e-3)
        torch.testing.assert_close(
            mobility.logdet(x), torch.zeros(3), atol=1e-6, rtol=0
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
