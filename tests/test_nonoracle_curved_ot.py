import unittest

import torch

from experiments.synthetic.run_nonoracle_curved_ot import (
    CurvedMixtureTarget,
    make_chords,
)


class NonOracleCurvedOTTests(unittest.TestCase):
    def test_target_samples_and_energy_are_finite(self):
        target = CurvedMixtureTarget(8, 2)
        samples = target.sample(32, generator=torch.Generator().manual_seed(3))
        self.assertEqual(samples.shape, (32, 8))
        self.assertTrue(torch.isfinite(target(samples)).all())
        self.assertTrue(torch.isfinite(target.gradient(samples.float())).all())

    def test_population_chords_are_observed_space_lines(self):
        target = CurvedMixtureTarget(8, 2)
        points, velocity = make_chords(
            target, count=16, seed=5, coupling="population"
        )
        self.assertEqual(points.shape, (16, 8))
        self.assertEqual(velocity.shape, points.shape)


if __name__ == "__main__":
    unittest.main()
