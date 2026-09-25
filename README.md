# Riemannian Energy Matching

Minimal reproduction code for **Riemannian Energy Matching for Fixed-Density
Transport**. REM freezes an energy and learns a positive-definite, determinant-one
mobility from transport supervision. The code includes the mobility models,
training, corrected sampling, diagnostics, experiment drivers, and tests used by
the paper.

This repository contains source and protocol settings. It does not bundle datasets,
pretrained or fitted checkpoints, experiment outputs, plots, or the manuscript.
Synthetic experiments generate their own data. Image experiments use the public
Energy Matching checkpoints and datasets documented in [data.md](data.md).

## Install

Use Python 3.10 or newer. A CPU environment is sufficient for the synthetic
experiments and tests. The release checks use Python 3.12, PyTorch 2.4.0,
NumPy 1.26.4, and SciPy 1.12.0.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest -q
```

For images, install a matching PyTorch/torchvision pair for your CUDA environment
before installing `requirements-images.txt`. A compatible example is PyTorch
2.4.0 with torchvision 0.19.0; the original large image runs used the NVIDIA
PyTorch 25.02 container (Python 3.12.3, PyTorch 2.7.0a0, CUDA 12.8) on one H100
80GB GPU. These environments need not produce bit-identical numerical results.

```bash
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-images.txt
```

## Check the complete synthetic pipeline

Run from the repository root:

```bash
python scripts/reproduce.py warp-ot --smoke
```

This trains small full and diagonal mobilities, saves and reloads checkpoints,
samples identity/diagonal/full/full-no-divergence conditions across five seeds,
and verifies checkpoint identity while aggregating results. Its reduced budgets
only check the software path. They do not reproduce the paper's measurements.
Smoke outputs are separated under `outputs/reproduction/smoke/`.

Run the full main 2D experiment with the paper's settings:

```bash
python scripts/reproduce.py warp-ot
```

All commands support `--dry-run`. The runner prints the underlying experiment
commands, stops on a failed command, and writes newly generated artifacts under
`outputs/reproduction/`. `configs/paper_protocol.json` contains the explicit
hyperparameters. See [docs/REPRODUCING.md](docs/REPRODUCING.md) for the remaining
experiments, image commands, and ablations.

## Layout

```text
rem/                      geometry, objectives, samplers, metrics, run recording
experiments/synthetic/    analytic and ordinary-OT targets, controls, aggregation
experiments/cifar10/      shared CIFAR-10/ImageNet32 training and evaluation
experiments/imagenet/     ImageNet32 loader and full-dataset FID statistics
configs/paper_protocol.json
scripts/reproduce.py      explicit paper protocols and a short integration check
tests/                    numerical and protocol regression tests
data.md                   dataset and pretrained-checkpoint acquisition
docs/REPRODUCING.md        experiment mapping, commands and interpretation
```

## Sampling conventions

The continuous-time Itô diffusion uses drift
`-G grad V + epsilon div G` and noise covariance `2 epsilon G`. Under the
regularity and boundary assumptions in the paper, it preserves the fixed
Boltzmann density. Finite unadjusted steps introduce discretization error.

The 2D full/no-divergence comparison uses Euler–Maruyama and the identical fitted
mobility. Image generation uses learned mobility during deterministic transport
and identity mobility during stochastic refinement. For an ungated diagonal
stochastic Heun run, disabling the explicit Stratonovich correction still leaves
half of the divergence drift in equivalent Itô form. It is a different ablation
from removing the complete Itô drift term. State-dependent full-matrix stochastic
Heun is intentionally rejected; corrected full-matrix Euler–Maruyama is supported.

## Attribution

The image energy architecture is adapted from the official
[Energy Matching implementation](https://github.com/m1balcerak/EnergyMatching).
The upstream MIT license and copyright notice are retained in [LICENSE](LICENSE).
Pretrained weights are obtained separately from the upstream authors.
