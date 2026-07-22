# Riemannian Energy Matching (REM)

REM is a research project that extends [Energy Matching](https://arxiv.org/abs/2504.10612) with a learned state-dependent geometry. The goal is to improve transport alignment and Langevin mixing without giving up the time-independent scalar energy or its Boltzmann equilibrium.

> **Status:** research prototype (`v0.1`). This first revision imports the official Energy Matching implementation, records the research plan, and adds reusable Riemannian geometry primitives. Results have not yet been validated.

## Motivation

Energy Matching uses the Euclidean gradient field

```math
v_\theta(x)=-\nabla V_\theta(x)
```

both to transport noise toward data and to define the equilibrium density

```math
\pi_\theta(x)\propto \exp\!\left(-V_\theta(x)/\varepsilon\right).
```

A Euclidean gradient can be poorly conditioned and too restrictive for curved or anisotropic transport. REM learns a positive-definite mobility `G_phi(x)` and uses the Riemannian gradient

```math
v_{\theta,\phi}(x)=-G_\phi(x)\nabla V_\theta(x).
```

Here `G_phi` is the mobility (the inverse Riemannian metric). The Euclidean Energy Matching model is recovered by setting `G_phi(x) = I`.

## Equilibrium-correct dynamics

REM samples with the Itô SDE

```math
dX_t = \left[-G_\phi(X_t)\nabla V_\theta(X_t)
+\varepsilon\,\nabla\!\cdot G_\phi(X_t)\right]dt
+\sqrt{2\varepsilon G_\phi(X_t)}\,dW_t.
```

The divergence term is essential. With it, the Fokker-Planck equation becomes

```math
\partial_t\rho
=\nabla\!\cdot\left[G_\phi\left(\rho\nabla V_\theta
+\varepsilon\nabla\rho\right)\right],
```

so the stationary density remains

```math
\rho_\infty(x)\propto\exp\!\left(-V_\theta(x)/\varepsilon\right).
```

This distinction is central to REM: the geometry changes the dynamics and mixing, not the target density.

## Training objective

For an OT-coupled source/data pair, let

```math
x_t=(1-t)x_0+t x_1, \qquad u_t=x_1-x_0.
```

The proposed transport objective is

```math
\mathcal L_{\mathrm{ROT}}
=\mathbb E\left[\left\|-G_\phi(x_t)\nabla V_\theta(x_t)-u_t\right\|^2\right].
```

The full Phase 2 objective is

```math
\mathcal L
=\mathcal L_{\mathrm{ROT}}
+\lambda_{\mathrm{CD}}\mathcal L_{\mathrm{CD}}
+\lambda_G\mathcal R_G.
```

`R_G` prevents scale and condition-number collapse. The initial implementation will bound the diagonal mobility between `g_min` and `g_max` and regularize its mean toward one.

## What is implemented

The [`rem/geometry.py`](rem/geometry.py) module currently provides:

- identity and bounded diagonal mobility modules;
- Riemannian velocity and transport loss;
- Hutchinson/JVP estimation of `div G`;
- an Itô-correct Riemannian Langevin step;
- metric scale regularization.

The module is intentionally independent of the CIFAR/ImageNet model definitions. Existing energy models can be wrapped with a one-argument potential function:

```python
potential_fn = lambda x: energy_model.potential(x, time_tensor)
velocity = riemannian_velocity(potential_fn, mobility, x, create_graph=True)
```

Run the lightweight unit tests with:

```bash
python -m unittest discover -s tests -v
```

## Integration map

The imported Energy Matching baseline has three main insertion points:

1. **Toy transport:** replace `velocity_training` and `velocity_inference` in `experiments/toy2d/utils_2D.py` with `riemannian_velocity`.
2. **Negative sampling:** replace the Euclidean update in `gibbs_sampler` and `utils_cifar_imagenet.gibbs_sampling_time_sweep` with `riemannian_langevin_step`.
3. **CIFAR/ImageNet transport loss:** instantiate a mobility network next to `EBViTModelWrapper`, include its parameters in DDP/Adam/EMA/checkpoints, and compute `L_ROT` in `forward_all`.

The first experimental implementation will stay diagonal. A diagonal-plus-low-rank mobility

```math
G_\phi(x)=D_\phi(x)+U_\phi(x)U_\phi(x)^\top
```

is deferred until the diagonal pilot verifies the hypothesis.

## Research roadmap

### Phase 0 - Baseline and geometry scaffold

- [x] Import the official Energy Matching repository.
- [x] Preserve the upstream MIT license and attribution.
- [x] Add bounded diagonal mobility primitives.
- [x] Add equilibrium-correct Riemannian Langevin dynamics.
- [x] Add unit tests for the Euclidean limit and divergence correction.

### Phase 1 - Controlled 2D study

- [ ] Integrate REM into the Eight-Gaussians-to-Two-Moons experiment.
- [ ] Implement a full `2 x 2` SPD mobility using a Cholesky parameterization.
- [ ] Compare `G=I`, constant diagonal, state-dependent diagonal, and full SPD variants.
- [ ] Verify equilibrium against known densities with KL, MMD, and long-run chains.
- [ ] Ablate exact, Hutchinson, and omitted divergence corrections.
- [ ] Track transport residual, eigenvalues, and metric condition number.

### Phase 2 - CIFAR-10 frozen-energy pilot

- [ ] Load the official Energy Matching checkpoint and freeze `V_theta`.
- [ ] Train only a small diagonal mobility network from OT transport supervision.
- [ ] Measure whether the learned geometry reduces transport error.
- [ ] Compare FID/KID, precision/recall, wall-clock time, and NFE.
- [ ] Stop if transport error does not improve by at least 20% or the metric collapses.

### Phase 3 - Joint REM training

- [ ] Jointly train `V_theta` and `G_phi` in Phase 1 warm-up.
- [ ] Use Riemannian negatives during contrastive Phase 2.
- [ ] Add mobility parameters to EMA and checkpoint handling.
- [ ] Evaluate long-run stability from both noise and data initialization.
- [ ] Compare diagonal and diagonal-plus-low-rank mobilities.

### Phase 4 - Theory and broader evaluation

- [ ] Formalize the generalized/weighted Wasserstein JKO interpretation.
- [ ] Prove invariance of the Boltzmann density.
- [ ] Characterize descent vector fields representable as `-G grad V`.
- [ ] Study convergence under uniform ellipticity bounds.
- [ ] Evaluate ImageNet32 and at least one inverse problem.

## Primary success criteria

The project should demonstrate at least one of:

- at least 30% fewer sampling steps at matched FID;
- at least 0.2 CIFAR-10 FID improvement at matched compute;
- at least 20% lower transport regression error with stable metric conditioning;
- materially better effective sample size or mode coverage at the same stationary density.

## Setup

The baseline follows the upstream CUDA setup:

```bash
conda create -n rem python=3.10 -y
conda activate rem
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The original training entry points remain available under `experiments/`. Until the REM integration flags land, they reproduce standard Energy Matching rather than REM.

## Prior and adjacent work

- [Energy Matching](https://arxiv.org/abs/2504.10612) provides the static scalar-energy baseline.
- [Projected Energy Matching](https://arxiv.org/abs/2607.07749) studies conservative-field mismatch and negative caching.
- [Convex Potential Mirror Langevin Algorithm](https://openreview.net/forum?id=oiDvwOhvjq) applies learned mirror geometry to EBM sampling. REM differs by learning geometry directly from Energy Matching transport supervision and using the same geometry during training and equilibrium sampling.
- [Energy Guided Geometric Flow Matching](https://openreview.net/forum?id=5NpCMdGy1A) uses learned geometry to guide flow trajectories, but does not retain REM's single static Boltzmann energy objective.

## Upstream attribution

This repository is based on the official [`m1balcerak/EnergyMatching`](https://github.com/m1balcerak/EnergyMatching) repository at commit `18176e4222a32dc9d2b323e92e8ff96e686ef2b2`. The imported code and media remain under the upstream MIT license. REM-specific changes are documented through this repository's Git history.

If you use the baseline, please cite the original work:

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
