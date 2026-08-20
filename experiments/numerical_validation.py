"""Paper-level numerical validation for the REM geometry and sampler.

This is intentionally larger than the unit-test suite.  It records finite-
difference divergence checks in dimensions 2/4/8, a fixed-state 10k-probe
Hutchinson audit with variance versus probe count, exact identity reduction,
SPD/gauge diagnostics, and an Euler weak-convergence trend.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from rem.artifacts import RunArtifacts, seed_everything
from rem.geometry import (
    GaugeFixedDiagonalMobility,
    GaugeFixedFullMobility,
    IdentityMobility,
    hutchinson_divergence,
    mobility_diagnostics,
    riemannian_langevin_step,
    riemannian_velocity,
)
from rem.networks import MLP
from rem.sampling import constant_temperature, langevin_chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--hutchinson-replicates", type=int, default=2500)
    parser.add_argument("--weak-chains", type=int, default=65536)
    parser.add_argument("--output", default="outputs/paper_results/numerical")
    parser.add_argument("--experiment-id", default="")
    return parser.parse_args()


class QuadraticPotential(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x.square().reshape(x.shape[0], -1).sum(dim=1)


class AnalyticRankOneMobility(nn.Module):
    """Smooth SPD field with a closed-form non-diagonal divergence."""

    def __init__(self, dimension: int, alpha: float = 0.2) -> None:
        super().__init__()
        self.dimension = dimension
        self.alpha = float(alpha)
        u = torch.arange(1, dimension + 1, dtype=torch.float64)
        u = u / u.norm()
        w = torch.arange(dimension, 0, -1, dtype=torch.float64)
        w = w / w.norm()
        self.register_buffer("u", u)
        self.register_buffer("w", w)

    def scalar(self, x: Tensor) -> Tensor:
        return torch.tanh(x @ self.w)

    def matrix(self, x: Tensor) -> Tensor:
        identity = torch.eye(self.dimension, device=x.device, dtype=x.dtype)
        outer = self.u[:, None] * self.u[None, :]
        return identity + self.alpha * self.scalar(x)[:, None, None] * outer

    def apply(self, x: Tensor, vector: Tensor) -> Tensor:
        return (self.matrix(x) @ vector.unsqueeze(-1)).squeeze(-1)

    def exact_divergence(self, x: Tensor) -> Tensor:
        derivative = 1.0 - self.scalar(x).square()
        coefficient = self.alpha * torch.dot(self.u, self.w)
        return derivative[:, None] * coefficient * self.u[None, :]


def finite_difference_divergence(
    mobility: AnalyticRankOneMobility,
    x: Tensor,
    step: float,
) -> Tensor:
    batch, dimension = x.shape
    result = torch.zeros_like(x)
    for coordinate in range(dimension):
        offset = torch.zeros_like(x)
        offset[:, coordinate] = step
        plus = mobility.matrix(x + offset)
        minus = mobility.matrix(x - offset)
        derivative = (plus - minus) / (2 * step)
        result += derivative[:, :, coordinate]
    return result


def identity_audit(device: torch.device) -> dict[str, float]:
    dtype = torch.float64
    generator = torch.Generator(device=device.type).manual_seed(11)
    x = torch.randn(64, 8, device=device, dtype=dtype, generator=generator)
    noise = torch.randn(x.shape, device=device, dtype=dtype, generator=generator)
    potential = QuadraticPotential().to(device)
    identity = IdentityMobility().to(device)
    velocity = riemannian_velocity(potential, identity, x, create_graph=False)
    expected_velocity = -x
    updated = riemannian_langevin_step(
        potential,
        identity,
        x,
        dt=0.01,
        epsilon=0.3,
        noise=noise,
    )
    expected_update = x - 0.01 * x + (2 * 0.3 * 0.01) ** 0.5 * noise
    return {
        "identity_velocity_max_abs_error": float((velocity - expected_velocity).abs().max()),
        "identity_step_max_abs_error": float((updated - expected_update).abs().max()),
    }


def spd_and_gauge_audit(device: torch.device) -> dict[str, float]:
    dtype = torch.float64
    generator = torch.Generator(device=device.type).manual_seed(17)
    x = torch.randn(256, 8, device=device, dtype=dtype, generator=generator)
    diagonal_network = MLP(8, 8, hidden_dim=32, depth=2, zero_output=False).to(
        device=device, dtype=dtype
    )
    diagonal = GaugeFixedDiagonalMobility(diagonal_network, log_bound=1.5).to(device)
    diagonal_stats = mobility_diagnostics(diagonal, x)

    packed = 8 * 9 // 2
    full_network = MLP(8, packed, hidden_dim=32, depth=2, zero_output=False).to(
        device=device, dtype=dtype
    )
    full = GaugeFixedFullMobility(full_network, 8, log_eigenvalue_bound=1.5).to(device)
    full_matrix = full.matrix(x)
    full_eigenvalues = torch.linalg.eigvalsh(full_matrix)
    return {
        "diagonal_max_abs_logdet": float(diagonal_stats["logdet"].abs().max()),
        "diagonal_min_eigenvalue": float(diagonal_stats["min_eigenvalue"].min()),
        "diagonal_max_eigenvalue": float(diagonal_stats["max_eigenvalue"].max()),
        "full_max_abs_logdet": float(full.logdet(x).abs().max()),
        "full_min_eigenvalue": float(full_eigenvalues.min()),
        "full_max_eigenvalue": float(full_eigenvalues.max()),
        "full_max_asymmetry": float(
            (full_matrix - full_matrix.transpose(1, 2)).abs().max()
        ),
    }


def divergence_audit(
    device: torch.device,
    replicates: int,
) -> dict[str, float]:
    dtype = torch.float64
    results: dict[str, float] = {}
    for dimension in (2, 4, 8):
        mobility = AnalyticRankOneMobility(dimension).to(device)
        generator = torch.Generator(device=device.type).manual_seed(100 + dimension)
        x = torch.randn(16, dimension, device=device, dtype=dtype, generator=generator)
        exact = mobility.exact_divergence(x)
        finite = finite_difference_divergence(mobility, x, step=1e-5)
        results[f"finite_difference_dim{dimension}_max_abs_error"] = float(
            (finite - exact).abs().max()
        )

    dimension = 8
    mobility = AnalyticRankOneMobility(dimension).to(device)
    fixed = torch.linspace(-0.7, 0.9, dimension, device=device, dtype=dtype)
    fixed = fixed.unsqueeze(0).expand(replicates, -1).clone().requires_grad_(True)
    exact = mobility.exact_divergence(fixed[:1])[0].detach()
    for probes_per_estimate in (1, 2, 4, 8):
        generator = torch.Generator(device=device.type).manual_seed(
            1000 + probes_per_estimate
        )
        probes = torch.empty(
            probes_per_estimate,
            replicates,
            dimension,
            device=device,
            dtype=dtype,
        )
        probes.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        estimates = hutchinson_divergence(
            mobility,
            fixed,
            n_samples=probes_per_estimate,
            probes=probes,
        ).detach()
        mean = estimates.mean(dim=0)
        squared_error = (mean - exact).square().mean()
        variance = estimates.var(dim=0, unbiased=True).mean()
        results[f"hutchinson_{probes_per_estimate}_mean_mse"] = float(squared_error)
        results[f"hutchinson_{probes_per_estimate}_mean_component_variance"] = float(
            variance
        )
        results[f"hutchinson_{probes_per_estimate}_total_probes"] = float(
            probes_per_estimate * replicates
        )
    return results


def weak_convergence_audit(
    device: torch.device,
    chain_count: int,
) -> dict[str, float]:
    dtype = torch.float64
    terminal_time = 2.0
    exact_second_moment = 1.0 - torch.exp(torch.tensor(-2 * terminal_time)).item()
    results: dict[str, float] = {}
    empirical_errors = []
    theory_biases = []
    step_sizes = []
    for index, dt in enumerate((0.04, 0.02, 0.01, 0.005)):
        steps = round(terminal_time / dt)
        generator = torch.Generator(device=device.type).manual_seed(50_000 + index)
        initial = torch.zeros(chain_count, 1, device=device, dtype=dtype)
        final, _ = langevin_chain(
            QuadraticPotential().to(device),
            IdentityMobility().to(device),
            initial,
            steps=steps,
            dt=dt,
            temperature=constant_temperature(1.0),
            generator=generator,
        )
        second_moment = float(final.square().mean())
        error = abs(second_moment - exact_second_moment)
        decay = 1.0 - dt
        discrete_second_moment = (
            2.0
            * dt
            * (1.0 - decay ** (2 * steps))
            / (1.0 - decay**2)
        )
        theory_bias = abs(discrete_second_moment - exact_second_moment)
        empirical_vs_theory = abs(second_moment - discrete_second_moment)
        second_moment_standard_error = (2.0 / chain_count) ** 0.5 * discrete_second_moment
        empirical_errors.append(error)
        theory_biases.append(theory_bias)
        step_sizes.append(dt)
        key = str(dt).replace(".", "p")
        results[f"weak_dt_{key}_second_moment"] = second_moment
        results[f"weak_dt_{key}_absolute_error"] = error
        results[f"weak_dt_{key}_discrete_theory_second_moment"] = discrete_second_moment
        results[f"weak_dt_{key}_theory_discretization_bias"] = theory_bias
        results[f"weak_dt_{key}_empirical_vs_theory_error"] = empirical_vs_theory
        results[f"weak_dt_{key}_second_moment_standard_error"] = (
            second_moment_standard_error
        )
        results[f"weak_dt_{key}_empirical_theory_z_score"] = (
            empirical_vs_theory / max(second_moment_standard_error, 1e-15)
        )
    log_dt = torch.tensor(step_sizes, dtype=torch.float64).log()
    log_bias = torch.tensor(theory_biases, dtype=torch.float64).log()
    centered_dt = log_dt - log_dt.mean()
    slope = ((centered_dt * (log_bias - log_bias.mean())).sum() / centered_dt.square().sum())
    results["weak_empirical_error_first_to_last_ratio"] = empirical_errors[0] / max(
        empirical_errors[-1], 1e-15
    )
    results["weak_finest_empirical_error_below_coarsest"] = float(
        empirical_errors[-1] < empirical_errors[0]
    )
    results["weak_theory_bias_monotone"] = float(
        all(later < earlier for earlier, later in zip(theory_biases, theory_biases[1:]))
    )
    results["weak_theory_loglog_slope"] = float(slope)
    return results


def main() -> None:
    args = parse_args()
    if args.hutchinson_replicates < 1250:
        raise ValueError("at least 1250 replicates are required for >=10k 8-probe audit")
    if args.weak_chains < 4096:
        raise ValueError("weak-convergence audit requires at least 4096 chains")
    seed_everything(args.seed)
    device = torch.device(args.device)
    experiment_id = args.experiment_id or (
        f"numerical_validation_{device.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    artifacts = RunArtifacts(
        args.output,
        experiment_id,
        vars(args),
        repository_root=Path(__file__).resolve().parents[1],
    )
    metrics = {
        **identity_audit(device),
        **spd_and_gauge_audit(device),
        **divergence_audit(device, args.hutchinson_replicates),
        **weak_convergence_audit(device, args.weak_chains),
    }
    artifacts.log(0, metrics, split="test", seed=args.seed, device=str(device))
    summary = artifacts.summarize(
        {name: [value] for name, value in metrics.items()},
        status="complete",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
