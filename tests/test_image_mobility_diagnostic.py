import torch

from experiments.cifar10.diagnose_image_mobility import (
    _mean_std,
    _quantiles,
    _sample_correlation,
)


def test_sample_correlation_recovers_agreement_and_sign() -> None:
    left = torch.tensor([[[[1.0, 2.0, 4.0]]], [[[2.0, 3.0, 5.0]]]])
    assert torch.allclose(_sample_correlation(left, left), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(_sample_correlation(left, -left), -torch.ones(2, dtype=torch.float64))


def test_quantiles_and_nested_summary_are_serializable() -> None:
    quantiles = _quantiles(torch.tensor([1.0, 2.0, 3.0]))
    assert quantiles["q00"] == 1.0
    assert quantiles["q50"] == 2.0
    assert quantiles["q100"] == 3.0
    summary = _mean_std([1.0, 3.0, 5.0])
    assert summary["mean"] == 3.0
    assert summary["std"] == 2.0
