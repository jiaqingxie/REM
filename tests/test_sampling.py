import unittest

import torch
from torch import nn

from rem.geometry import IdentityMobility
from rem.sampling import (
    constant_temperature,
    energy_matching_temperature,
    langevin_chain,
    linear_temperature,
    phase_gated_mobility,
    profile_langevin_chain,
    profile_step_components,
)
from rem.synthetic import AnalyticDiagonalMobility, HutchinsonDivergenceWrapper


class QuadraticEnergy(nn.Module):
    def forward(self, x):
        return 0.5 * x.flatten(start_dim=1).square().sum(dim=1)


class ConstantScaleMobility(nn.Module):
    divergence_probe_cost = 0

    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)
        self.divergence_calls = 0

    def apply(self, x, vector):
        return self.scale * vector

    def sqrt_apply(self, x, noise):
        return self.scale**0.5 * noise

    def divergence(self, x, *, n_samples=1, create_graph=False):
        del n_samples, create_graph
        self.divergence_calls += 1
        return torch.zeros_like(x)


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
            temperature=0.1,
            divergence_samples=4,
        )
        self.assertEqual(hutch_profile.divergence_probes, 8)

        _, _, heun_profile = profile_langevin_chain(
            QuadraticEnergy(),
            IdentityMobility(),
            initial,
            steps=2,
            dt=0.1,
            temperature=0.0,
            integrator="heun",
        )
        self.assertEqual(heun_profile.energy_grad_evaluations, 4)

    def test_zero_temperature_skips_divergence_computation(self):
        mobility = ConstantScaleMobility(2.0)
        langevin_chain(
            QuadraticEnergy(),
            mobility,
            torch.ones(2, 2),
            steps=3,
            dt=0.1,
            temperature=0.0,
            integrator="heun",
        )
        self.assertEqual(mobility.divergence_calls, 0)

    def test_phase_gate_uses_learned_mobility_then_identity(self):
        mobility = ConstantScaleMobility(2.0)
        schedule = phase_gated_mobility(mobility, dt=0.1, active_until=0.1)
        x = torch.ones(1, 1)
        self.assertIs(schedule(0, 2, x), mobility)
        self.assertIsInstance(schedule(1, 2, x), IdentityMobility)

        final, _ = langevin_chain(
            QuadraticEnergy(),
            mobility,
            x,
            steps=2,
            dt=0.1,
            temperature=0.0,
            mobility_schedule=schedule,
        )
        torch.testing.assert_close(final, 0.72 * x)

    def test_phase_gated_heun_has_no_langevin_divergence_jvps(self):
        mobility = HutchinsonDivergenceWrapper(AnalyticDiagonalMobility())
        initial = torch.randn(2, 2)
        schedule = phase_gated_mobility(mobility, dt=0.1, active_until=0.1)
        temperature = energy_matching_temperature(
            0.1, dt=0.1, time_cutoff=0.1, ramp_end=0.1
        )
        _, _, profile = profile_langevin_chain(
            QuadraticEnergy(),
            mobility,
            initial,
            steps=2,
            dt=0.1,
            temperature=temperature,
            integrator="heun",
            mobility_schedule=schedule,
        )
        self.assertEqual(profile.divergence_probes, 0)

    def test_temperature_schedules(self):
        x = torch.zeros(1, 1)
        self.assertEqual(constant_temperature(0.2)(3, 10, x), 0.2)
        schedule = linear_temperature(1.0, warmup_fraction=0.5)
        self.assertEqual(schedule(0, 5, x), 0.0)
        self.assertEqual(schedule(4, 5, x), 1.0)

        official = energy_matching_temperature(
            0.2,
            dt=0.25,
            time_cutoff=0.5,
            ramp_end=1.0,
        )
        self.assertEqual(official(1, 5, x), 0.0)
        self.assertEqual(official(2, 5, x), 0.0)
        self.assertAlmostEqual(official(3, 5, x), 0.1)
        self.assertAlmostEqual(official(4, 5, x), 0.2)

        cifar = energy_matching_temperature(
            0.01,
            dt=0.01,
            time_cutoff=1.0,
            ramp_end=1.0,
        )
        self.assertEqual(cifar(99, 325, x), 0.0)
        self.assertEqual(cifar(100, 325, x), 0.01)

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
