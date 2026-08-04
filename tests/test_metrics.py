import unittest

import torch

from rem.metrics import (
    bootstrap_confidence_interval,
    effective_sample_size,
    rbf_mmd,
    sliced_wasserstein,
    split_rhat,
)


class MetricTests(unittest.TestCase):
    def test_identical_samples_have_zero_distances(self):
        samples = torch.randn(32, 3)
        torch.testing.assert_close(rbf_mmd(samples, samples), torch.tensor(0.0, dtype=torch.double))
        torch.testing.assert_close(
            sliced_wasserstein(samples, samples, n_projections=8),
            torch.tensor(0.0, dtype=torch.double),
        )

    def test_bootstrap_is_reproducible(self):
        first = bootstrap_confidence_interval([1.0, 2.0, 3.0], n_resamples=100)
        second = bootstrap_confidence_interval([1.0, 2.0, 3.0], n_resamples=100)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 2.0)
        self.assertGreaterEqual(first[1], 2.0)

    def test_chain_diagnostics_are_finite(self):
        generator = torch.Generator().manual_seed(7)
        chains = torch.randn(4, 100, 2, generator=generator)
        ess = effective_sample_size(chains)
        rhat = split_rhat(chains)
        self.assertEqual(tuple(ess.shape), (2,))
        self.assertEqual(tuple(rhat.shape), (2,))
        self.assertTrue(bool(torch.isfinite(ess).all()))
        self.assertTrue(bool(torch.isfinite(rhat).all()))


if __name__ == "__main__":
    unittest.main()
