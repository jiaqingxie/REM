import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from experiments.cifar10.train_rem import configure_energy_training_mode

from rem.geometry import IdentityMobility
from rem.training import (
    ema_update,
    load_rem_checkpoint,
    rem_objective,
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
    def test_rem_objective_uses_transport_time(self):
        class TimeConditionedQuadratic(nn.Module):
            def potential(self, x, t=None):
                if t is None:
                    t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
                squared_norm = x.flatten(start_dim=1).square().sum(dim=1)
                return 0.5 * (1.0 + t) * squared_norm

            def forward(self, x):
                return self.potential(x)

        x = torch.tensor([[1.0], [2.0]])
        time = torch.tensor([0.25, 0.75])
        target = -(1.0 + time).unsqueeze(1) * x
        timed, _ = rem_objective(
            TimeConditionedQuadratic(),
            IdentityMobility(),
            x,
            target,
            time=time,
            geometry_weight=0.0,
            metric_weighted=False,
        )
        untimed, _ = rem_objective(
            TimeConditionedQuadratic(),
            IdentityMobility(),
            x,
            target,
            geometry_weight=0.0,
            metric_weighted=False,
        )
        torch.testing.assert_close(timed.transport, torch.tensor(0.0))
        self.assertGreater(float(untimed.transport), 0.0)

    def test_frozen_energy_disables_dropout(self):
        energy = nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.5), nn.Linear(4, 1))
        configure_energy_training_mode(energy, frozen=True)
        self.assertFalse(energy.training)
        self.assertFalse(any(parameter.requires_grad for parameter in energy.parameters()))

        configure_energy_training_mode(energy, frozen=False)
        self.assertTrue(energy.training)
        self.assertTrue(all(parameter.requires_grad for parameter in energy.parameters()))

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
