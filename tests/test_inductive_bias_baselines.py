import unittest

import torch

from experiments.synthetic.run_pem_helmholtz_control import (
    exact_divergence_penalty,
)
from rem.networks import MLP
from rem.synthetic import (
    AdditiveResidualSampler,
    ConvexRidgeMirrorPotential,
    InverseHessianMobility,
)


class InductiveBiasBaselineTests(unittest.TestCase):
    def test_mirror_gradient_and_hessian_match_autograd(self):
        torch.manual_seed(3)
        mirror = ConvexRidgeMirrorPotential(2, width=8)
        points = torch.randn(4, 2, requires_grad=True)
        gradient = torch.autograd.grad(
            mirror(points).sum(), points, create_graph=True
        )[0]
        torch.testing.assert_close(
            mirror.gradient(points), gradient, atol=1e-6, rtol=1e-5
        )
        rows = []
        for coordinate in range(2):
            rows.append(
                torch.autograd.grad(
                    gradient[:, coordinate].sum(),
                    points,
                    retain_graph=True,
                )[0]
            )
        autograd_hessian = torch.stack(rows, dim=1)
        torch.testing.assert_close(
            mirror.hessian(points), autograd_hessian, atol=1e-6, rtol=1e-5
        )

    def test_inverse_hessian_mobility_is_spd(self):
        mirror = ConvexRidgeMirrorPotential(2, width=8)
        mobility = InverseHessianMobility(mirror)
        points = torch.randn(16, 2)
        matrix = mobility.matrix(points)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(matrix) > 0))
        vectors = torch.randn_like(points)
        torch.testing.assert_close(
            mobility.inverse_apply(points, mobility.apply(points, vectors)),
            vectors,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_residual_adapter_has_additive_drift_semantics(self):
        field = MLP(2, 2, hidden_dim=4, depth=1)
        with torch.no_grad():
            field.network[-1].bias.copy_(torch.tensor([0.25, -0.5]))
        sampler = AdditiveResidualSampler(field)
        points = torch.randn(3, 2)
        gradient = torch.randn_like(points)
        expected_drift = -gradient + field(points)
        torch.testing.assert_close(
            -sampler.apply(points, gradient), expected_drift
        )

    def test_exact_pem_divergence_penalty_matches_linear_trace(self):
        field = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            field.weight.copy_(torch.tensor([[2.0, 3.0], [-4.0, -0.5]]))
        penalty, divergence = exact_divergence_penalty(
            field, torch.randn(7, 2), create_graph=True
        )
        torch.testing.assert_close(divergence, torch.full((7,), 1.5))
        torch.testing.assert_close(penalty, torch.tensor((1.5 / 2) ** 2))
        penalty.backward()
        self.assertIsNotNone(field.weight.grad)


if __name__ == "__main__":
    unittest.main()
