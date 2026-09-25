import unittest

import torch
from torch import nn

from rem.geometry import (
    AdditiveResidualTransport,
    IdentityMobility,
    TemperedFullMobility,
    riemannian_langevin_heun_step,
)
from rem.networks import build_mobility
from rem.sampling import (
    constant_temperature,
    energy_matching_temperature,
    langevin_chain,
    linear_temperature,
    phase_gated_mobility,
    profile_langevin_chain,
    profile_step_components,
)
from rem.synthetic import (
    AnalyticDiagonalMobility,
    AnalyticRotatingMobility,
    AnalyticWarpMobility,
    HutchinsonDivergenceWrapper,
)


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


class FullSquareRootMobility(ConstantScaleMobility):
    """Marker for a mobility whose stochastic square root is non-diagonal."""

    def sample_sqrt_noise(self, x, *, noise=None, generator=None):
        del generator
        if noise is None:
            noise = torch.randn_like(x)
        return self.scale**0.5 * noise


class SamplingTests(unittest.TestCase):
    def test_additive_residual_transport_matches_requested_drift(self):
        class ConstantResidual(nn.Module):
            def forward(self, x):
                return torch.full_like(x, 0.5)

        transport = AdditiveResidualTransport(ConstantResidual())
        initial = torch.ones(1, 1)
        final, _ = langevin_chain(
            QuadraticEnergy(),
            transport,
            initial,
            steps=1,
            dt=0.1,
            temperature=0.0,
        )
        # -grad V + r = -1 + 0.5 at x=1.
        torch.testing.assert_close(final, torch.full_like(initial, 0.95))

    def test_phase_gate_turns_additive_residual_off_before_noise(self):
        class ConstantResidual(nn.Module):
            def forward(self, x):
                return torch.full_like(x, 0.5)

        transport = AdditiveResidualTransport(ConstantResidual())
        schedule = phase_gated_mobility(transport, dt=0.1, active_until=0.1)
        final, _ = langevin_chain(
            QuadraticEnergy(),
            transport,
            torch.ones(1, 1),
            steps=2,
            dt=0.1,
            temperature=0.0,
            mobility_schedule=schedule,
        )
        # First step uses -x + .5; second step uses the identity drift -x.
        torch.testing.assert_close(final, torch.full((1, 1), 0.855))

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

    def test_phase_gated_heun_allows_full_mobility_only_at_zero_temperature(self):
        mobility = FullSquareRootMobility(2.0)
        initial = torch.ones(1, 1)
        schedule = phase_gated_mobility(mobility, dt=0.1, active_until=0.1)
        temperature = energy_matching_temperature(
            0.1, dt=0.1, time_cutoff=0.1, ramp_end=0.1
        )

        final, _ = langevin_chain(
            QuadraticEnergy(),
            mobility,
            initial,
            steps=2,
            dt=0.1,
            temperature=temperature,
            integrator="heun",
            mobility_schedule=schedule,
            generator=torch.Generator().manual_seed(0),
        )

        self.assertEqual(final.shape, initial.shape)

    def test_heun_rejects_full_mobility_at_positive_temperature(self):
        with self.assertRaisesRegex(TypeError, "diagonal mobility square roots"):
            langevin_chain(
                QuadraticEnergy(),
                FullSquareRootMobility(2.0),
                torch.ones(1, 1),
                steps=1,
                dt=0.1,
                temperature=0.1,
                integrator="heun",
            )

    def test_heun_rejects_non_diagonal_sqrt_apply_models_at_both_endpoints(self):
        full = build_mobility("full", (2,), hidden_dim=4, depth=2).double()
        variants = [
            full,
            TemperedFullMobility(full, 0.5),
            AnalyticWarpMobility(1.2).double(),
            AnalyticRotatingMobility().double(),
            HutchinsonDivergenceWrapper(full),
        ]
        initial = torch.tensor([[0.4, -0.7]], dtype=torch.float64)
        energy = QuadraticEnergy()
        for mobility in variants:
            for at_next_endpoint in (False, True):
                with self.subTest(mobility=type(mobility).__name__, next=at_next_endpoint):
                    with self.assertRaisesRegex(TypeError, "diagonal mobility square roots"):
                        riemannian_langevin_heun_step(
                            energy, energy,
                            IdentityMobility() if at_next_endpoint else mobility,
                            initial, dt=0.01,
                            epsilon=0.0 if at_next_endpoint else 0.1,
                            next_epsilon=0.1,
                            next_mobility=mobility if at_next_endpoint else IdentityMobility(),
                        )
        # Full mobility remains valid in the deterministic phase of the paper schedule.
        final, _ = langevin_chain(
            energy, full, initial, steps=2, dt=0.1, integrator="heun",
            temperature=energy_matching_temperature(0.1, dt=0.1, time_cutoff=0.1, ramp_end=0.1),
            mobility_schedule=phase_gated_mobility(full, dt=0.1, active_until=0.1),
            generator=torch.Generator().manual_seed(0),
        )
        self.assertTrue(torch.isfinite(final).all())

    def test_heun_disabled_explicit_term_retains_noise_induced_ito_drift(self):
        class ExponentialDiagonal(nn.Module):
            def diagonal(self, x):
                log_first = 0.3 * x[:, 0]
                return torch.stack((log_first.exp(), (-log_first).exp()), dim=1)

            def apply(self, x, vector):
                return self.diagonal(x) * vector

            def sqrt_apply(self, x, noise):
                return self.diagonal(x).sqrt() * noise

            def divergence(self, x, **kwargs):
                return torch.stack((0.3 * (0.3 * x[:, 0]).exp(), torch.zeros_like(x[:, 0])), dim=1)

        mobility = ExponentialDiagonal()
        point = torch.tensor([[0.4, -0.7]], dtype=torch.float64)
        # These four noises integrate the first two Gaussian moments exactly.
        noise = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=torch.float64)
        initial = point.expand(4, -1)
        dt, epsilon = 1e-6, 0.2
        for enabled, ito_fraction in ((True, 1.0), (False, 0.5)):
            with self.subTest(explicit_correction=enabled):
                final = riemannian_langevin_heun_step(
                    QuadraticEnergy(), QuadraticEnergy(), mobility, initial,
                    dt=dt, epsilon=epsilon, next_epsilon=epsilon, noise=noise,
                    include_divergence_correction=enabled, divergence_method="exact",
                )
                observed = (final - initial).mean(dim=0) / dt
                expected = (-mobility.apply(point, point)
                            + ito_fraction * epsilon * mobility.divergence(point))[0]
                torch.testing.assert_close(observed, expected, atol=2e-6, rtol=0.0)

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
