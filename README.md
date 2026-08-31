# Riemannian Energy Matching

**Fixed scalar density → supervised transport class → equilibrium-compatible lifting**

Riemannian Energy Matching (REM) keeps a pretrained Energy Matching potential fixed and learns a bounded, determinant-one SPD mobility from external OT velocity evidence. It is a preconditioner---but a supervised and gauge-fixed one paired with a corrected sampler: the scalar density is held fixed, supervision selects a representable transport field, and corrected dynamics preserve the same equilibrium law.

![REM overview](media/rem_method_overview.png)

## Method

Given a frozen potential \(V_\theta\) and an OT-supervised velocity \(u_t\), REM learns \(G_\phi(x) \succ 0\) through

$$
u_t \approx -G_\phi(x_t)\nabla V_\theta(x_t),
\qquad
\det G_\phi(x_t)=1.
$$

The learned mobility changes transport without changing the scalar energy. At positive temperature, the corrected diffusion is

$$
dX_s =
\left[-G_\phi\nabla V_\theta
+\varepsilon\,\mathrm{div}\,G_\phi\right]ds
+\sqrt{2\varepsilon G_\phi}\,dW_s,
$$

which preserves the original Lebesgue-density Boltzmann law

$$
\rho_\infty(x)\propto \exp[-V_\theta(x)/\varepsilon].
$$

The repository supports two residual choices under the same mobility parameterization:

$$
\mathcal L_{\mathrm{REM}}(\phi;W)
=\mathbb E\!\left[r_t^\top W_\phi(x_t)r_t\right]
+\lambda_{\mathrm{geo}}\mathbb E\!\left[\|\log G_\phi(x_t)\|_F^2\right],
\qquad
r_t=u_t+G_\phi(x_t)\nabla V_\theta(x_t).
$$

- `W = I` is the Euclidean residual used for controlled geometry recovery.
- `W = G_phi^{-1}` is the intrinsic residual used in the image experiments.

## What is implemented

- Pointwise SPD representability constructions and descent-margin diagnostics.
- Identity, constant, scalar, determinant-one diagonal, full-SPD, diagonal-plus-low-rank, and first-order structured congruence mobilities.
- Exact and Hutchinson divergence estimators for corrected Riemannian Langevin dynamics.
- Controlled warped-mixture recovery, first-passage, and stationary-bias experiments.
- Frozen-energy CIFAR-10 and ImageNet32 mobility training and official-protocol evaluation.
- Independent mobility-training and sampling-seed aggregation.
- PEM-inspired residual and learned-mirror representation controls.
- A parameter-matched additive-residual control that separates geometric inductive bias from generic extra transport capacity.
- Non-oracle curved 16D and correlated-Gaussian 16D/64D corrected-sampling diagnostics, including an oracle global-covariance calibration.
- Image mobility diagnostics, including held-out OT residuals, descent margins, log-eigenvalue statistics, and cross-fit consistency.
- Inverse-problem and protein experiment entry points.

## Main results

All image comparisons use the same frozen energy checkpoint and paired sampling streams.

| Dataset | Frozen EM | REM | Change |
|---|---:|---:|---:|
| CIFAR-10 FID | 3.462 ± 0.030 | **3.297 ± 0.009** | −0.165 |
| ImageNet32 FID | 6.742 ± 0.117 | **6.555 ± 0.064** | −0.187 |

Across independent mobility fits, REM reduces held-out Euclidean OT velocity residual by 8.3% on CIFAR-10 and 11.2% on ImageNet32. The learned diagonal fields are also consistent across fits, with cross-fit correlations of 0.895 and 0.904, respectively.

In the controlled nonlinear-warp diagnostic at curvature \(c=1.2\), full REM reaches relative velocity MSE \(0.00108\pm0.00025\) and first passage in \(111\pm14\) steps, compared with \(196\pm75\) steps for identity mobility. Removing the divergence correction produces persistent stationary bias under step-size refinement.

With ordinary minibatch-OT chords in 64 dimensions, rank-four structured REM reaches held-out relative velocity MSE \(3.286\pm0.035\), versus \(11.055\pm0.143\) for a capacity-matched diagonal model, while preserving the exact target density. Continuing the exact scalar energy on the same evidence lowers the residual to \(0.924\pm0.005\) but changes equilibrium by \(18.301\pm0.067\) KL.

On CIFAR-10, a parameter-matched, phase-gated additive residual trained with the same OT chords reaches 3.686 FID, compared with 3.462 for frozen EM and 3.297 for REM. This control tests whether the gain comes merely from adding another state-dependent transport network.

## Repository layout

```text
rem/
  geometry.py       mobility parameterizations and divergence estimators
  sampling.py       deterministic and corrected stochastic samplers
  training.py       OT-supervised mobility objectives
  synthetic.py      analytic warped targets and diagnostics
experiments/
  synthetic/        recovery, first-passage, stationarity, and baselines
  cifar10/           frozen-mobility training and image evaluation
  imagenet/          ImageNet32 dataset and FID-stat utilities
  inverse/           inverse-problem transfer
  proteins/          AAV inference-time experiments
tests/               unit and smoke tests
```

## Installation

The codebase targets Python 3.10 and PyTorch with CUDA.

```bash
conda create -n rem python=3.10 -y
conda activate rem
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Run the test suite from the repository root:

```bash
PYTHONPATH=. pytest -q
```

## Controlled geometry experiment

Run the warped-mixture recovery and equilibrium-validity suite:

```bash
python -m experiments.synthetic.run_geometry_stress \
  --curvatures 0,0.4,0.8,1.2 \
  --seeds 0,1,2,3,4 \
  --output outputs/geometry_stress
```

The controlled recovery phase directly supervises the analytic pushforward field, allowing the learned tensor to be compared with the known oracle mobility. The validity phase separately tests corrected and uncorrected state-dependent Langevin dynamics.

## High-dimensional OT-chord lifting

Run the non-oracle structured-mobility and energy-continuation comparison:

```bash
python -m experiments.synthetic.run_highdim_structured_ot \
  --dimensions 64 \
  --rank 4 \
  --seeds 0,1,2,3,4 \
  --train-steps 1200 \
  --output outputs/highdim_structured_ot
```

The experiment uses independent minibatches, minibatch OT, and observed-space straight chords exactly as in image REM. `diagonal-matched` controls for network capacity; `continue-energy` starts from the exact target potential and reports its analytic equilibrium KL after fitting the same velocities.

Run the non-oracle curved target and corrected positive-temperature diagnostics:

```bash
python -m experiments.synthetic.run_nonoracle_curved_ot \
  --dimension 16 --rank 4 --seeds 0,1,2,3,4 \
  --output outputs/nonoracle_curved_ot

python -m experiments.synthetic.run_highdim_corrected_sampling \
  --dimensions 16,64 --rank 4 --seeds 0,1,2,3,4 \
  --variants identity,global-covariance,diagonal,structured,structured-no-div \
  --output outputs/highdim_corrected_sampling
```

The global-covariance row uses the known Gaussian covariance and is therefore an oracle-style classical calibration, not a learned baseline.

## Frozen-energy image mobility

Train a determinant-one diagonal mobility around an existing Energy Matching checkpoint:

```bash
python -m experiments.cifar10.train_rem \
  --mode frozen \
  --mobility diagonal \
  --energy-checkpoint /path/to/energy_checkpoint.pt \
  --data-root /path/to/cifar10 \
  --seed 0 \
  --output outputs/cifar10_rem
```

Continue the same Energy Matching checkpoint instead of freezing it:

```bash
python -m experiments.cifar10.train_rem \
  --mode continue-energy \
  --mobility identity \
  --energy-checkpoint /path/to/full_energy_training_checkpoint.pt \
  --data-root /path/to/cifar10 \
  --total-steps 150000 \
  --output outputs/cifar10_continue_energy
```

This mode restores the raw trainable energy together with its optimizer, scheduler, and EMA;
evaluation uses the continued EMA checkpoint.

Evaluate a trained checkpoint with the matched phase-gated sampler:

```bash
python -m experiments.cifar10.evaluate_rem \
  --dataset cifar10 \
  --checkpoint /path/to/rem_checkpoint.pt \
  --data-root /path/to/cifar10 \
  --steps 325 \
  --t-end 3.25 \
  --time-cutoff 1.0 \
  --temperature-schedule official \
  --samples 50000 \
  --output outputs/cifar10_eval
```

Use `--no-metric-weighted` during training for the Euclidean-residual ablation. The evaluator also exposes solver, cutoff, mobility-strength, divergence, and sampling-seed controls through `--help`.

## Scope

The full-SPD corrected stochastic sampler is evaluated in the controlled low-dimensional setting. Structured off-diagonal mobility is tested with non-oracle OT supervision in 64 dimensions. Image experiments currently use determinant-one diagonal mobility during deterministic transport and identity mobility during positive-temperature refinement.

## Upstream attribution

This repository builds on the official [Energy Matching](https://github.com/m1balcerak/EnergyMatching) implementation. Imported code and media retain the upstream MIT license.

```bibtex
@inproceedings{balcerak2025energy,
  title={Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling},
  author={Balcerak, Michal and Amiranashvili, Tamaz and Terpin, Antonio and Shit, Suprosanna and Bogensperger, Lea and Kaltenbach, Sebastian and Koumoutsakos, Petros and Menze, Bjoern},
  booktitle={Advances in Neural Information Processing Systems},
  volume={38},
  pages={8583--8609},
  year={2025}
}
```
