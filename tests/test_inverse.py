import unittest

import torch
from torch import nn

from rem.inverse import (
    DownsampleOperator,
    MaskOperator,
    PosteriorEnergy,
    center_mask,
    measurement_error,
    posterior_diversity,
    posterior_chain,
    psnr,
    random_mask,
)


class ZeroEnergy(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


class InverseTests(unittest.TestCase):
    def test_masks_have_expected_shape_and_are_reproducible(self):
        centered = center_mask((3, 8, 8), missing_fraction=0.25)
        first = random_mask((3, 8, 8), observed_fraction=0.5, seed=3)
        second = random_mask((3, 8, 8), observed_fraction=0.5, seed=3)
        self.assertEqual(tuple(centered.shape), (1, 3, 8, 8))
        torch.testing.assert_close(first, second)
        self.assertTrue(bool((centered == 0).any()))

    def test_posterior_energy_is_likelihood_for_zero_prior(self):
        mask = torch.ones(1, 1, 2, 2)
        operator = MaskOperator(mask)
        observation = torch.zeros(1, 1, 2, 2)
        posterior = PosteriorEnergy(
            ZeroEnergy(), operator, observation, observation_sigma=0.5
        )
        x = torch.ones(2, 1, 2, 2)
        torch.testing.assert_close(posterior(x), torch.full((2,), 8.0))

    def test_inverse_metrics_and_downsampling(self):
        target = torch.zeros(2, 1, 4, 4)
        reconstruction = target.clone()
        downsample = DownsampleOperator(2)
        self.assertEqual(tuple(downsample(target).shape), (2, 1, 2, 2))
        self.assertTrue(bool(torch.isfinite(psnr(reconstruction, target)).all()))
        torch.testing.assert_close(
            measurement_error(reconstruction, downsample(target), downsample),
            torch.zeros(2),
        )
        draws = torch.stack([target, target + 1], dim=1)
        self.assertTrue(bool((posterior_diversity(draws) > 0).all()))

    def test_posterior_chain_retains_thinned_states(self):
        from rem.geometry import IdentityMobility

        x = torch.ones(2, 1, 2, 2)
        operator = MaskOperator(torch.ones(1, 1, 2, 2))
        final, trajectory = posterior_chain(
            ZeroEnergy(),
            IdentityMobility(),
            operator,
            torch.zeros_like(x),
            x,
            observation_sigma=1.0,
            steps=4,
            dt=0.1,
            temperature=0.0,
            save_every=2,
        )
        self.assertEqual(len(trajectory), 3)
        self.assertEqual(final.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
