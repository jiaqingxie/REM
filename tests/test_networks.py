import unittest

import torch
from torch import nn

from rem.geometry import (
    DiagonalPlusLowRankMobility,
    GaugeFixedDiagonalMobility,
    GaugeFixedFullMobility,
    IdentityMobility,
    LowRankCongruenceMobility,
    UnfixedLogDiagonalMobility,
)
from rem.networks import REMModel, build_mobility


class QuadraticEnergy(nn.Module):
    def forward(self, x):
        return 0.5 * x.flatten(start_dim=1).square().sum(dim=1)


class NetworkTests(unittest.TestCase):
    def test_builders_start_at_identity(self):
        x = torch.randn(3, 4)
        vector = torch.randn_like(x)
        expected_types = {
            "identity": IdentityMobility,
            "diagonal": GaugeFixedDiagonalMobility,
            "unfixed-diagonal": UnfixedLogDiagonalMobility,
            "full": GaugeFixedFullMobility,
            "low-rank": DiagonalPlusLowRankMobility,
            "structured": LowRankCongruenceMobility,
        }
        for name, expected_type in expected_types.items():
            with self.subTest(name=name):
                mobility = build_mobility(name, (4,), hidden_dim=8, depth=1, rank=2)
                self.assertIsInstance(mobility, expected_type)
                torch.testing.assert_close(
                    mobility.apply(x, vector), vector, atol=1e-6, rtol=1e-6
                )

    def test_structured_branch_has_gradient_at_identity(self):
        mobility = build_mobility(
            "structured", (8,), hidden_dim=12, depth=1, rank=2
        )
        x = torch.randn(5, 8)
        vector = torch.randn_like(x)
        target = torch.roll(vector, shifts=1, dims=1)
        loss = (mobility.apply(x, vector) - target).square().mean()
        loss.backward()
        output = mobility.parameter_network.network[-1]
        factor_gradient = output.bias.grad[8:]
        self.assertGreater(float(factor_gradient.abs().sum()), 0.0)

    def test_image_diagonal_preserves_shape_and_gauge(self):
        x = torch.randn(2, 3, 8, 8)
        mobility = build_mobility("diagonal", x.shape[1:], hidden_dim=8, depth=1)
        self.assertEqual(mobility.diagonal(x).shape, x.shape)
        torch.testing.assert_close(
            mobility.logdet(x), torch.zeros(2), atol=1e-6, rtol=0
        )

    def test_rem_model_quadratic_velocity(self):
        model = REMModel(QuadraticEnergy(), IdentityMobility())
        x = torch.randn(5, 3)
        t = torch.rand(5)
        torch.testing.assert_close(model(t, x), -x)
        torch.testing.assert_close(model(t, x, return_potential=True), QuadraticEnergy()(x))


if __name__ == "__main__":
    unittest.main()
