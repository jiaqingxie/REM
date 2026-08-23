import unittest

import torch

from experiments.synthetic.run_highdim_structured_ot import (
    CorrelatedGaussian,
    TrainableQuadraticEnergy,
    capacity_matched_diagonal_width,
    make_chords,
    mlp_parameter_count,
)


class HighDimStructuredOTTests(unittest.TestCase):
    def test_target_transport_and_precision_are_consistent(self):
        target = CorrelatedGaussian.build(12, 3, 0.72, 0.18)
        identity = target.precision @ target.covariance
        torch.testing.assert_close(
            identity, torch.eye(12, dtype=torch.float64), atol=1e-9, rtol=1e-9
        )
        self.assertTrue(torch.all(torch.linalg.eigvalsh(target.covariance) > 0))

    def test_population_chords_use_observed_space_interpolation(self):
        target = CorrelatedGaussian.build(10, 2, 0.72, 0.18)
        points, velocity = make_chords(
            target, count=32, seed=7, coupling="population"
        )
        self.assertEqual(points.shape, (32, 10))
        self.assertEqual(velocity.shape, points.shape)
        gradient = target.gradient(points)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_continued_quadratic_starts_at_exact_energy(self):
        target = CorrelatedGaussian.build(8, 2, 0.72, 0.18)
        model = TrainableQuadraticEnergy(target.precision.float())
        torch.testing.assert_close(
            model.precision(), target.precision.float(), atol=1e-5, rtol=1e-5
        )

    def test_capacity_matched_diagonal_is_near_structured_count(self):
        width = capacity_matched_diagonal_width(32, 4, 64, 2)
        structured = mlp_parameter_count(32, 32 * 5, 64, 2)
        matched = mlp_parameter_count(32, 32, width, 2)
        self.assertLess(abs(matched - structured) / structured, 0.02)


if __name__ == "__main__":
    unittest.main()
