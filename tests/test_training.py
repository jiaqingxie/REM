import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from rem.geometry import IdentityMobility
from rem.training import (
    ema_update,
    load_rem_checkpoint,
    save_rem_checkpoint,
    trimmed_mean,
)


class LinearEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.weight * x.flatten(start_dim=1).sum(dim=1)


class TrainingTests(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        energy = LinearEnergy()
        mobility = IdentityMobility()
        optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_rem_checkpoint(
                path,
                energy=energy,
                mobility=mobility,
                optimizer=optimizer,
                step=7,
                config={"mode": "test"},
            )
            energy.weight.data.fill_(3.0)
            state = load_rem_checkpoint(
                path,
                energy=energy,
                mobility=mobility,
                optimizer=optimizer,
            )
            self.assertEqual(state["step"], 7)
            torch.testing.assert_close(energy.weight, torch.tensor(1.0))

    def test_ema_and_trimmed_mean(self):
        source = LinearEnergy()
        target = LinearEnergy()
        source.weight.data.fill_(3.0)
        target.weight.data.fill_(1.0)
        ema_update(source, target, 0.5)
        torch.testing.assert_close(target.weight, torch.tensor(2.0))
        torch.testing.assert_close(
            trimmed_mean(torch.tensor([1.0, 2.0, 100.0]), 1 / 3),
            torch.tensor(1.5),
        )


if __name__ == "__main__":
    unittest.main()
