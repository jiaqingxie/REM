import unittest

import torch

from rem.synthetic import AnalyticWarpMobility, WarpedGaussianMixturePotential


class GeometryStressTests(unittest.TestCase):
    def setUp(self):
        self.means = torch.tensor(
            [[-1.5, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.5, 0.0]]
        )

    def test_warp_has_exact_inverse_and_preserves_energy(self):
        target = WarpedGaussianMixturePotential(self.means, curvature=0.6)
        latent = torch.randn(32, 2)
        observed = target.warp(latent)
        torch.testing.assert_close(target.inverse_warp(observed), latent)
        torch.testing.assert_close(target(observed), target.latent_potential(latent))

    def test_analytic_warp_mobility_is_spd_and_determinant_one(self):
        mobility = AnalyticWarpMobility(0.6)
        points = torch.randn(32, 2)
        matrix = mobility.matrix(points)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(matrix) > 0))
        torch.testing.assert_close(
            torch.linalg.det(matrix), torch.ones(points.shape[0]), atol=1e-5, rtol=1e-5
        )

    def test_analytic_divergence_matches_closed_form(self):
        mobility = AnalyticWarpMobility(0.6)
        points = torch.randn(16, 2)
        analytic = mobility.divergence(points)
        expected = torch.tensor([0.0, 1.2]).expand_as(points)
        torch.testing.assert_close(analytic, expected)


if __name__ == "__main__":
    unittest.main()
