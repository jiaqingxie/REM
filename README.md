# Riemannian Energy Matching (REM)

**Equilibrium-preserving mobility learning for static energy models**

REM is a research project built on [Energy Matching](https://arxiv.org/abs/2504.10612). Its aim is not merely to attach a metric network to an energy-based model. The paper-level hypothesis is that a learned, gauge-fixed positive-definite mobility can resolve the conservative-field mismatch of Energy Matching while retaining one time-independent energy and exactly the same Boltzmann equilibrium.

> **Status: runnable experimental prototype; scientific claims not yet established.** The REM model, controlled experiments, CIFAR training/evaluation, inverse-problem transfer, artifact aggregation, and claim-gate checks are implemented. Full GPU runs have not yet been completed, so none of the target improvements below should be quoted as results.

## One-sentence claim

REM learns the smallest state-dependent deformation of Euclidean geometry under which an OT transport field is representable by the gradient of a single static energy, and reuses that geometry in an equilibrium-correct Langevin sampler without changing the modeled density.

## Why this is a research problem

Energy Matching regresses an OT velocity target `u(x)` with a Euclidean gradient field

```math
u(x) \approx -\nabla V_\theta(x).
```

This is restrictive: a Euclidean gradient is curl-free, whereas finite-sample, distilled, or otherwise approximate transport targets generally contain non-conservative components. [Projected Energy Matching](https://arxiv.org/abs/2607.07749) identifies this as a structural conflict and introduces an auxiliary residual field.

REM studies a different resolution. It learns an SPD mobility `G_phi(x)` and represents the transport as

```math
u(x) \approx -G_\phi(x)\nabla V_\theta(x),
\qquad G_\phi(x) \succ 0.
```

Although `-G grad V` need not be a Euclidean gradient, it is a gradient under the inverse metric `g = G^{-1}`. This can enlarge the representable velocity class without abandoning the scalar energy used by the EBM.

## Precise scope and terminology

The first paper studies data in an ambient Euclidean space with Lebesgue reference measure `dx`. In this setting, `G(x)` is a state-dependent mobility, equivalently the inverse of a Riemannian metric. The induced probability-space dynamics can be interpreted through a mobility-weighted Wasserstein geometry.

This distinction matters. REM does **not** initially claim to model data intrinsically supported on a known manifold, and it does not define the target density relative to the Riemannian volume form. Extensions to sphere-, Lie-group-, or mesh-valued data are outside the first-paper scope.

## Theoretical core

### 1. Representability

Let `g(x) = grad V(x)`. If a nonzero target velocity satisfies

```math
u(x)=-G(x)g(x), \qquad G(x)\succ0,
```

then necessarily

```math
u(x)^\top g(x)<0.
```

Pointwise, this descent condition is also sufficient for the existence of an SPD matrix mapping `g(x)` to `-u(x)`. The intended first theorem will formalize this statement, handle zero sets, and give a constructive bounded-SPD solution. This yields a falsifiable diagnostic: for a frozen Energy Matching potential, we can measure how often the OT target violates the descent condition before training any mobility.

The paper will not claim that every arbitrary vector field is globally a Riemannian gradient. Global smoothness, topology, and the existence of a common Lyapunov potential remain explicit assumptions.

### 2. Gauge fixing and minimal distortion

The factorization `G grad V` is not identifiable without constraints. A transport loss alone permits scale exchange and other local degeneracies between the energy and mobility. REM therefore treats gauge fixing as part of the method rather than an implementation detail.

The target constrained objective is

```math
\min_{\theta,\phi}\;
\mathbb E\!\left[
\left\|u+G_\phi\nabla V_\theta\right\|_{G_\phi^{-1}}^2
\right]
+\lambda_{\mathrm{geom}}\,
\mathbb E\!\left[\|\log G_\phi\|_F^2\right]
+\lambda_{\mathrm{CD}}\mathcal L_{\mathrm{CD}},
```

subject to

```math
g_{\min}I\preceq G_\phi(x)\preceq g_{\max}I,
\qquad \log\det G_\phi(x)=0.
```

`log det G = 0` removes the pointwise scalar gauge, uniform ellipticity prevents singular shortcuts, and the log-metric penalty selects the least deformation from Euclidean geometry. The Euclidean model remains the exact `G = I` special case.

For a diagonal pilot, the determinant constraint can be enforced by centering predicted log-mobilities:

```math
r_\phi(x)=\alpha\tanh h_\phi(x),\qquad
\log G_{ii}(x)=r_i(x)-\frac{1}{d}\sum_j r_j(x).
```

### 3. Equilibrium preservation

REM uses the Itô diffusion

```math
dX_t=
\left[-G_\phi(X_t)\nabla V_\theta(X_t)
+\varepsilon\,\nabla\!\cdot G_\phi(X_t)\right]dt
+\sqrt{2\varepsilon G_\phi(X_t)}\,dW_t.
```

Its Fokker-Planck equation is

```math
\partial_t\rho
=\nabla\!\cdot\left[
G_\phi\left(\rho\nabla V_\theta+\varepsilon\nabla\rho\right)
\right],
```

so, under standard regularity and integrability assumptions, the stationary density with respect to `dx` is

```math
\rho_\infty(x)\propto\exp\!\left[-V_\theta(x)/\varepsilon\right].
```

The divergence correction is therefore mandatory. Continuous-time invariance does not remove Euler-Maruyama discretization bias; the experiments separately quantify both sources of error.

## Target contributions

A main-conference submission must support all four contributions below.

1. **Representation:** characterize when a transport field can be expressed as a gradient under a learned SPD mobility, including a constructive result and a measurable violation criterion.
2. **Identifiable learning:** introduce a gauge-fixed, minimal-distortion objective that avoids trivial metric scaling and recovers Energy Matching at `G = I`.
3. **Equilibrium-correct generation:** use the same learned geometry for transport and Langevin mixing while provably preserving the static energy's Boltzmann density.
4. **Mechanistic evidence:** show that gains come from closing the conservative projection gap and improving mixing, not merely from adding parameters or compute.

If the work only produces a small FID improvement from an extra network, it does not satisfy the intended contribution.

## Training strategy

REM deliberately avoids joint training from scratch in the first experiments.

1. **Energy Matching baseline:** train or load the official `V_theta` and reproduce its reported sampling behavior.
2. **Frozen-energy diagnostic:** freeze `V_theta`, measure the descent-condition violation rate, and train only `G_phi`. This is the clean causal test of whether geometry helps a fixed energy.
3. **Gauge-fixed joint refinement:** jointly update `V_theta` and `G_phi` with a trust-region penalty around the frozen solution.
4. **Contrastive phase:** use the equilibrium-correct mobility sampler for negative generation and include both modules in EMA and checkpoints.

The frozen-energy result is a required experiment, not an optional ablation. It separates a better sampler from a better/larger model.

## Claim gates

The project advances only when the preceding gate passes.

### Gate A: mathematical validity

- prove continuous-time invariance relative to the declared base measure;
- prove the pointwise SPD representability result and state its global assumptions;
- specify a gauge-fixed parameterization with explicit eigenvalue bounds;
- recover Energy Matching exactly when `G = I`.

### Gate B: controlled synthetic evidence

- recover a constructed non-Euclidean field with known `V_star` and `G_star`;
- reduce held-out velocity error by at least 25% over parameter-matched Euclidean Energy Matching on a field with verified projection mismatch;
- match the known stationary density without statistically detectable degradation from the Euclidean chain;
- demonstrate that omitting `div G` creates the predicted stationary bias.

### Gate C: frozen-energy CIFAR-10

- improve the FID-versus-NFE or FID-versus-wall-clock Pareto frontier using the exact same energy checkpoint;
- achieve at least 25--30% lower sampling wall-clock/NFE at matched FID, or at least 0.3 lower FID at matched sampling compute;
- report peak memory and divergence-estimation overhead;
- show no material precision, recall, or mode-coverage regression.

### Gate D: full paper

- reproduce the main result across at least three training seeds with confidence intervals;
- beat constant, scalar, parameter-matched, and sampling-only geometry baselines;
- demonstrate one posterior/inverse-problem setting in which the learned mobility transfers without retraining the energy;
- report negative results and regimes where geometry does not help.

ImageNet-32 is a scaling experiment after Gate C, not a substitute for mechanism or rigor.

## Experimental program

The full preregistered protocol is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). The primary studies are:

1. analytic vector-field recovery and representability diagnostics;
2. exact-density and long-chain equilibrium tests;
3. 2D joint training and geometry visualization;
4. CIFAR-10 frozen-energy sampling;
5. CIFAR-10 joint REM training;
6. posterior sampling for an inverse problem;
7. optional ImageNet-32 scaling after the central claims pass.

The code-by-code completion state, GPU validation boundary, external baselines, and remaining TODOs are tracked in [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md). This status file is authoritative when the broader protocol mentions an experiment that may not yet have an implementation.

Primary endpoints are selected before runs. Results are reported over fixed seeds with uncertainty, parameter-matched and compute-matched controls, and complete wall-clock/VRAM accounting.

## Positioning against adjacent work

| Method | Single static energy | Learned geometry | OT/transport supervision | Geometry preserves that energy's equilibrium |
|---|---:|---:|---:|---:|
| [Energy Matching](https://arxiv.org/abs/2504.10612) | Yes | No | Yes | Yes |
| [Projected Energy Matching](https://arxiv.org/abs/2607.07749) | Yes | No; auxiliary residual | Yes | Energy remains available |
| [Riemannian Flow Matching](https://arxiv.org/abs/2302.03660) | No | Generally prescribed/premetric | Yes | Not its objective |
| [Energy Guided Geometric Flow Matching](https://arxiv.org/abs/2509.25230) | Energy guides geometry | Yes | Yes | No static Boltzmann generator |
| [CPMLA](https://papers.neurips.cc/paper_files/paper/2025/hash/7f64034009f4a5fa417a57e1a987c5cd-Abstract-Conference.html) | EBM energy | Yes, mirror map | No OT supervision | Sampling objective |
| **REM (target)** | **Yes** | **Yes, gauge-fixed mobility** | **Yes** | **Yes, with explicit correction** |

[Riemannian Score-Based Generative Modelling](https://arxiv.org/abs/2202.02763) and recent Riemannian metric-learning work address intrinsic manifolds or geometric recovery. REM instead studies whether learned ambient mobility can reconcile transport expressivity with a static Euclidean-density EBM. The paper must preserve this distinction in its title, abstract, and experiments.

## What is currently implemented

The code now covers the preregistered core experiment paths:

- [`rem/geometry.py`](rem/geometry.py): identity, constant, scalar, determinant-one diagonal, unfixed diagonal, full SPD, and diagonal-plus-low-rank mobilities; constructive SPD mapping; descent/conditioning diagnostics; exact-gauge regularization; Hutchinson divergence; and corrected Riemannian Langevin steps.
- [`rem/networks.py`](rem/networks.py): zero-at-identity MLP/image parameterizations and the composite `REMModel`.
- [`experiments/synthetic/run_representability.py`](experiments/synthetic/run_representability.py): known-metric recovery, Euclidean projection-gap diagnostics, negative representability controls, `EM/EM-Large/constant/scalar/diagonal/full` comparisons, seeds, checkpoints, and optional field plots.
- [`experiments/synthetic/run_stationarity.py`](experiments/synthetic/run_stationarity.py): Euclidean, exact-divergence, Hutchinson-probe, and no-divergence chains over step-size grids with KL, MMD, sliced Wasserstein, ESS, R-hat, expectation error, wall-clock, and JVP accounting.
- [`experiments/toy2d/train_rem_2d.py`](experiments/toy2d/train_rem_2d.py): Energy Matching, frozen-mobility, and joint-refinement stages.
- [`experiments/cifar10/train_rem.py`](experiments/cifar10/train_rem.py): single-/multi-GPU training for baseline, frozen, joint, `EM-Large`, constant/scalar/diagonal/unfixed/low-rank mobility, trust region, corrected contrastive negatives, EMA, and resumable checkpoints.
- [`experiments/cifar10/evaluate_rem.py`](experiments/cifar10/evaluate_rem.py): 50k FID/KID/precision/recall sweeps versus NFE, divergence JVPs, wall-clock, throughput, and peak memory.
- [`experiments/cifar10/match_capacity.py`](experiments/cifar10/match_capacity.py): parameter matching for `EM-Large`.
- [`experiments/inverse/run_cifar_inverse.py`](experiments/inverse/run_cifar_inverse.py): frozen-mobility transfer to inpainting, super-resolution, and deblurring with reconstruction, consistency, diversity, coverage, and runtime metrics.
- [`experiments/run_ablation_suite.py`](experiments/run_ablation_suite.py): reproducible three-seed ablation command generation/execution from [`configs/cifar10_ablation.json`](configs/cifar10_ablation.json).
- [`experiments/aggregate_results.py`](experiments/aggregate_results.py) and [`experiments/check_claim_gates.py`](experiments/check_claim_gates.py): seed aggregation, bootstrap confidence intervals, CSV/JSON export, and nonzero-exit claim threshold checks.

`REM-Full` is deliberately restricted to dimensions at most 64. A dense `3072 x 3072` CIFAR mobility would be an impractical and misleading baseline; CIFAR uses determinant-one diagonal and diagonal-plus-low-rank variants instead. CPMLA and PEM remain external comparison methods: they should only be reported after their official implementations are run under the same checkpoint and compute protocol, rather than approximated under those names here.

The code paths have CPU unit/smoke coverage, but CIFAR FID, multi-GPU equivalence, CUDA memory, and the paper thresholds require actual server runs. “Implemented” therefore means the experiment can be launched and audited, not that its scientific gate has passed.

## Run order

Run the inexpensive validity gates before committing to full CIFAR training:

```bash
python -m unittest discover -s tests -v
python -m experiments.synthetic.run_representability \
  --problem rotating --variants em,em-large,constant,scalar,diagonal,full \
  --seeds 0,1,2,3,4 --steps 5000 --learn-energy --plot
python -m experiments.synthetic.run_representability \
  --problem rotating --variants constant,scalar,diagonal,full \
  --seeds 0,1,2,3,4 --steps 5000 --no-learn-energy --plot
python -m experiments.synthetic.run_stationarity \
  --variants euclidean,rem-exact,rem-hutch-1,rem-hutch-4,rem-no-div \
  --seeds 0,1,2,3,4
```

After selecting a frozen upstream Energy Matching checkpoint:

```bash
python -m experiments.cifar10.match_capacity --mobility diagonal
python -m experiments.run_ablation_suite configs/cifar10_ablation.json \
  --gpus 2 --energy-checkpoint /path/to/em.pt --output outputs \
  --em-large-multiplier <reported_multiplier>
python -m experiments.run_ablation_suite configs/cifar10_ablation.json \
  --gpus 2 --energy-checkpoint /path/to/em.pt --output outputs \
  --em-large-multiplier <reported_multiplier> --execute
```

The manifest runs frozen `REM-Diag` before joint REM and automatically loads the same seed's frozen checkpoint into the joint stage. After measuring joint REM training time, run the separate compute-matched manifest with that preregistered per-run wall-clock budget:

```bash
python -m experiments.run_ablation_suite configs/cifar10_compute_matched.json \
  --gpus 2 --output outputs --compute-matched-seconds <seconds> --execute
```

Evaluate every selected checkpoint at the same preregistered sampler grid. `REM-NoDiv` deliberately reuses the exact `REM-Diag` checkpoint and only removes the sampler correction:

```bash
python -m experiments.run_evaluation_suite configs/cifar10_evaluation.json \
  --training-output outputs --output outputs/evaluation --execute
python -m experiments.aggregate_results outputs/evaluation \
  --group-by variant,steps \
  --metrics fid,kid_mean,precision,recall,sampling_wall_seconds,peak_memory_bytes \
  --json-out outputs/cifar_aggregate.json \
  --csv-out outputs/cifar_aggregate.csv
```

Copy [`configs/claim_gates.example.json`](configs/claim_gates.example.json), preregister the target FID/wall-clock values before inspecting final test results, and run:

```bash
python -m experiments.check_claim_gates \
  outputs/cifar_aggregate.json configs/claim_gates.json \
  --report outputs/claim_gate_report.json
```

## Reproducibility contract

Every reported run must record:

- Git commit and full resolved configuration;
- training and evaluation seeds;
- dataset version and preprocessing hash;
- global and per-device batch size;
- GPU model/count, peak VRAM, training GPU-hours, and sampling wall-clock;
- checkpoint-selection rule fixed before test evaluation;
- raw per-seed metrics, not only the best run.

All main tables will report uncertainty. Hyperparameters are tuned on validation criteria and frozen before final test evaluation.

## Setup

The baseline currently follows the upstream CUDA environment:

```bash
conda create -n rem python=3.10 -y
conda activate rem
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Upstream attribution

This repository is based on the official [`m1balcerak/EnergyMatching`](https://github.com/m1balcerak/EnergyMatching) repository at commit `18176e4222a32dc9d2b323e92e8ff96e686ef2b2`. Imported code and media retain the upstream MIT license.

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
