import unittest

import torch

from experiments.synthetic.run_highdim_corrected_sampling import (
    _sampling_metrics,
    global_covariance_mobility,
)
from experiments.synthetic.run_highdim_structured_ot import CorrelatedGaussian
from rem.sampling import SamplingProfile


class HighDimCorrectedSamplingTests(unittest.TestCase):
    def test_sampling_metrics_are_finite(self):
        target = CorrelatedGaussian.build(4, 2, 0.72, 0.18)
        generator = torch.Generator().manual_seed(3)
        draws = target.target_sample(32, generator=generator).float().reshape(4, 8, 4)
        reference = target.target_sample(
            32, generator=torch.Generator().manual_seed(4)
        )
        profile = SamplingProfile(10, 4, 1.0, 4.0, 10, 0, 0)
        metrics = _sampling_metrics(draws, reference, target, profile)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertGreater(metrics["mean_ess"], 0)

    def test_global_covariance_mobility_is_spd_and_gauge_fixed(self):
        target = CorrelatedGaussian.build(8, 2, 0.72, 0.18)
        mobility = global_covariance_mobility(target, torch.device("cpu"))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(mobility.matrix) > 0))
        self.assertAlmostEqual(float(torch.linalg.det(mobility.matrix)), 1.0, places=4)
        points = torch.randn(3, 8)
        vectors = torch.randn(3, 8)
        recovered = mobility.inverse_apply(points, mobility.apply(points, vectors))
        torch.testing.assert_close(recovered, vectors, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(mobility.divergence(points), torch.zeros_like(points))


if __name__ == "__main__":
    unittest.main()
