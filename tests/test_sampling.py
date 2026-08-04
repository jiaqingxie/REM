import unittest

import torch
from torch import nn

from rem.geometry import IdentityMobility
from rem.sampling import (
    constant_temperature,
    langevin_chain,
    linear_temperature,
    profile_langevin_chain,
    profile_step_components,
)
from rem.synthetic import AnalyticDiagonalMobility, HutchinsonDivergenceWrapper


class QuadraticEnergy(nn.Module):
    def forward(self, x):
        return 0.5 * x.flatten(start_dim=1).square().sum(dim=1)


class SamplingTests(unittest.TestCase):
    def test_zero_temperature_chain_is_deterministic(self):
        initial = torch.tensor([[1.0, -2.0]])
        final, trajectory = langevin_chain(
            QuadraticEnergy(),
            IdentityMobility(),
            initial,
            steps=3,
            dt=0.1,
            temperature=0.0,
            save_every=1,
        )
        torch.testing.assert_close(final, initial * 0.9**3)
        self.assertEqual(len(trajectory), 4)

    def test_profile_counts_gradient_and_divergence_work(self):
        initial = torch.randn(2, 3)
        _, _, profile = profile_langevin_chain(
            QuadraticEnergy(),
            IdentityMobility(),
            initial,
            steps=2,
            dt=0.1,
            temperature=0.0,
            divergence_samples=4,
        )
        self.assertEqual(profile.energy_grad_evaluations, 2)
        self.assertEqual(profile.divergence_probes, 0)
        self.assertGreaterEqual(profile.wall_seconds, 0)

        _, _, hutch_profile = profile_langevin_chain(
            QuadraticEnergy(),
            HutchinsonDivergenceWrapper(AnalyticDiagonalMobility()),
            torch.randn(2, 2),
            steps=2,
            dt=0.1,
            temperature=0.0,
            divergence_samples=4,
        )
        self.assertEqual(hutch_profile.divergence_probes, 8)

    def test_temperature_schedules(self):
        x = torch.zeros(1, 1)
        self.assertEqual(constant_temperature(0.2)(3, 10, x), 0.2)
        schedule = linear_temperature(1.0, warmup_fraction=0.5)
        self.assertEqual(schedule(0, 5, x), 0.0)
        self.assertEqual(schedule(4, 5, x), 1.0)

    def test_component_profile_has_nonnegative_timings(self):
        timings = profile_step_components(
            QuadraticEnergy(),
            IdentityMobility(),
            torch.randn(2, 2),
            repeats=1,
        )
        self.assertEqual(len(timings), 4)
        self.assertTrue(all(value >= 0 for value in timings.values()))


if __name__ == "__main__":
    unittest.main()
