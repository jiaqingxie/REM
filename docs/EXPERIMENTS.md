# REM Experimental Protocol

This document defines the experiments required to evaluate Riemannian Energy Matching (REM). It is intentionally written before the main implementation and large-scale runs. Any deviation used in a paper must be recorded with a reason rather than silently replacing an unsuccessful protocol.

## 1. Claims and primary evidence

| ID | Claim | Primary evidence | Main failure criterion |
|---|---|---|---|
| C1 | Learned SPD mobility closes a conservative projection gap | Held-out velocity residual on constructed and OT fields | No improvement over a parameter-matched Euclidean energy model |
| C2 | Gauge fixing prevents trivial geometry/energy exchange | Scale, determinant, eigenvalue, and seed stability diagnostics | Equivalent velocity with unstable energy or collapsed eigenvalues |
| C3 | Corrected dynamics preserve the energy's target density | Exact-density long-chain tests | REM has larger stationary bias than Euclidean Langevin at matched step size |
| C4 | Geometry improves sampling rather than only model capacity | Frozen-energy CIFAR-10 Pareto curves | No gain at matched wall-clock or NFE |
| C5 | Joint REM improves the full generative model | Multi-seed CIFAR-10 evaluation | Gain disappears under parameter- or compute-matched controls |
| C6 | Mobility transfers to posterior sampling | Inverse problem without retraining `V` | No improvement in quality, mixing, or calibrated diversity |

The primary paper should make no claim whose corresponding row has not passed.

## 2. Experimental rules

### 2.1 Seeds and uncertainty

- Synthetic experiments: five fixed training seeds `{0, 1, 2, 3, 4}`.
- CIFAR-10 main results: at least three independent training seeds `{0, 1, 2}`.
- Evaluation randomness is separated from training randomness and stored in the run configuration.
- Report every seed, mean, standard deviation, and a 95% bootstrap confidence interval.
- Do not select the best training seed for a main table.

### 2.2 Model selection

- Define the checkpoint-selection metric before training.
- Select checkpoints using a validation split or a fixed training-step rule, never test FID.
- Evaluate the test set once after freezing the selected checkpoint and sampling configuration.
- Hyperparameter sweeps receive the same number of trials for REM and baselines.

### 2.3 Fair-comparison axes

Every central comparison is reported in three forms:

1. **architecture matched:** identical energy model and sampler budget;
2. **parameter matched:** enlarge the Euclidean baseline to match REM trainable parameters;
3. **compute matched:** match training GPU-hours or sampling wall-clock, including `div G` evaluation.

The REM DDP entry point interprets `--global-batch-size` globally, checks divisibility by world size, and derives the per-device batch size. Any imported upstream entry point used as a baseline must be checked separately rather than assumed to have the same semantics.

### 2.4 Compute accounting

Record for every run:

- GPU type and number;
- mixed-precision mode;
- global/per-device batch sizes;
- training steps, examples, GPU-hours, and peak allocated VRAM;
- sampling NFE, JVP/VJP count, wall-clock, throughput, and peak VRAM;
- parameter count for the energy, mobility, and auxiliary networks separately.

Compilation warm-up and data loading are excluded from sampling latency, but the exclusion procedure must be identical across methods. Both cold-start and steady-state latency are retained in artifacts.

## 3. Model variants

Use stable names in configurations, logs, and tables.

| Name | Energy | Mobility | Purpose |
|---|---|---|---|
| `EM` | learned | `I` | official Euclidean baseline |
| `EM-Large` | enlarged | `I` | parameter-matched control |
| `EM-Const` | learned/frozen | learned constant SPD | global preconditioning control |
| `REM-Scalar` | learned/frozen | `a(x) I`, normalized | state-dependent step-size control |
| `REM-Diag` | learned/frozen | determinant-one diagonal | scalable primary REM model |
| `REM-Full` | learned/frozen | full SPD | low-dimensional expressivity ceiling |
| `REM-LR` | learned/frozen | diagonal plus low rank | optional image-space structured model |
| `REM-NoDiv` | same as REM | correction omitted | stationarity ablation only |
| `REM-Unfixed` | same as REM | no gauge normalization | identifiability ablation only |
| `PEM-Residual` | energy plus residual field | `I` | conservative-mismatch control |
| `CPMLA` | EBM energy | learned mirror map | sampling-geometry baseline where reproducible |

`REM-NoDiv` and `REM-Unfixed` are diagnostic models and must not be presented as valid samplers.

## 4. Experiment E0: numerical correctness

### Goal

Verify the implementation before learning anything.

### Tests

1. `G = I` reproduces Energy Matching velocity and Langevin updates under identical noise.
2. Cholesky/full-SPD and diagonal parameterizations satisfy symmetry and eigenvalue bounds.
3. Determinant-normalized mobilities satisfy `abs(logdet(G)) < 1e-5` in float64 tests.
4. Exact `div G` agrees with finite differences in dimensions 2, 4, and 8.
5. Hutchinson estimates are unbiased over at least 10,000 fixed-state probes; report variance versus probe count `{1, 2, 4, 8}`.
6. The sampler is tested under decreasing step sizes to verify the expected weak-convergence trend.

### Pass condition

All deterministic unit tests pass on CPU and CUDA. Numerical error decreases with step size/probe count and no gradient path needed for training is accidentally detached.

## 5. Experiment E1: representability and metric recovery

### 5.1 Constructed ground-truth fields

Construct fields from known pairs `(V_star, G_star)`:

```math
u_\star(x)=-G_\star(x)\nabla V_\star(x).
```

Use four families:

1. anisotropic Gaussian with constant full SPD mobility;
2. banana/warped Gaussian with rotating eigenvectors;
3. multimodal potential with state-dependent diagonal mobility;
4. a field with a controlled descent-condition violation, used as a negative control.

Train `EM`, `EM-Large`, `EM-Const`, `REM-Scalar`, `REM-Diag`, and `REM-Full` on identical samples and held-out points.

### 5.2 OT fields

Use the existing Eight-Gaussians-to-Two-Moons task plus two harder pairs:

- Gaussian to Swiss roll;
- Gaussian mixture to a curved, strongly anisotropic target.

Freeze OT couplings for a comparison block so all models observe identical target velocities. Repeat with independently resampled couplings to verify robustness.

### Metrics

- relative velocity MSE and cosine error on held-out points;
- Euclidean curl/Jacobian antisymmetry norm of the target and fitted field;
- descent violation rate

```math
r_{\mathrm{viol}}=
\Pr\left[u(x)^\top\nabla V(x)\ge 0\right];
```

- metric error and eigenspace angle where `G_star` is known;
- `log det G`, minimum/maximum eigenvalue, condition number, and distance from identity;
- potential error up to an additive constant where `V_star` is known.

### Primary endpoint

Held-out relative velocity MSE on the constructed rotating-anisotropy field.

### Pass condition

`REM-Full` improves the primary endpoint by at least 25% relative to `EM-Large`, recovers the principal metric eigenspaces, and remains within the declared ellipticity bounds. On the negative-control field, the violation diagnostic must predict irreducible residual rather than hiding it through metric collapse.

## 6. Experiment E2: equilibrium and discretization

### Targets

Use normalized densities for which direct or high-accuracy reference sampling is available:

1. correlated Gaussian;
2. Gaussian mixture with separated modes;
3. banana/funnel density;
4. the learned 2D Energy Matching potential evaluated on a dense grid.

### Samplers

- Euclidean Langevin;
- REM with exact divergence;
- REM with Hutchinson divergence using `{1, 2, 4, 8}` probes;
- `REM-NoDiv`;
- constant preconditioned Langevin;
- a Metropolis-corrected reference chain where feasible.

Evaluate step sizes on a logarithmic grid and hold the number of energy-gradient evaluations fixed. Use at least 32 chains from both data-like and noise-like initialization.

### Metrics

- grid KL/JS divergence in 2D;
- sliced Wasserstein distance and MMD;
- effective sample size per gradient evaluation and per second;
- integrated autocorrelation time;
- split-Rhat and between-chain mode occupancy;
- round-trip time between separated modes;
- stationary expectation error for a preregistered set of test functions;
- bias versus step size and divergence-probe variance.

### Primary endpoints

1. stationary expectation error at the smallest three step sizes;
2. ESS per wall-clock second at matched stationary error.

### Pass condition

Corrected REM converges toward the same reference density as Euclidean Langevin as the step size decreases. `REM-NoDiv` must expose a measurable bias on at least one state-dependent mobility. Corrected REM must improve ESS per second without exceeding the baseline stationary-error tolerance.

## 7. Experiment E3: end-to-end 2D Energy Matching

### Protocol

Run three training regimes on identical OT minibatches:

1. Energy Matching from scratch;
2. Energy Matching followed by frozen-energy mobility learning;
3. frozen-energy initialization followed by gauge-fixed joint refinement.

Use five seeds and an equal hyperparameter-search budget. Log energy and mobility surfaces on a fixed grid throughout training.

### Metrics

- transport metrics from E1;
- density and chain metrics from E2;
- training stability and time-to-threshold;
- change in `V` between frozen and joint phases;
- geometry maps: eigenvalues, eigenvectors, determinant, condition number;
- correlation between learned anisotropy and target-field projection error.

### Key ablations

- Euclidean versus metric-weighted transport residual;
- determinant gauge on/off;
- minimal-distortion penalty strength;
- alternating versus simultaneous updates;
- exact versus Hutchinson divergence during sampling.

### Decision

Do not start a full CIFAR-10 joint run if frozen-energy REM fails to improve both held-out transport fit and mixing on the controlled tasks.

## 8. Experiment E4: CIFAR-10 frozen-energy pilot

This is the central causal experiment.

### Setup

1. Reproduce or load one official Energy Matching checkpoint.
2. Freeze all energy parameters and BatchNorm/statistics.
3. Generate and persist the OT supervision protocol and its random seeds.
4. Train only `EM-Const`, `REM-Scalar`, and `REM-Diag` with equal search budgets.
5. Keep the target energy, temperature schedule, data preprocessing, and initialization distribution identical.

Before mobility training, measure the descent violation rate and signed alignment distribution between the OT target and `grad V`. Report this result even if REM fails.

### Sampling sweep

Generate 50,000 samples per selected checkpoint at a preregistered step/NFE grid, initially `{10, 20, 50, 100, 200}`. If numerical stability requires a different grid, update it before looking at FID.

### Metrics

- FID-50k and KID;
- improved precision and recall;
- density and coverage on CIFAR-10 feature clusters;
- NFE/JVP count;
- images/second and wall-clock for 50,000 samples;
- peak VRAM and energy/mobility time breakdown;
- trajectory length, drift norm, and stiffness proxy;
- metric eigenvalue/diagonal distributions across time and samples.

### Baselines

- official `EM` sampler;
- tuned constant diagonal/full preconditioner;
- `REM-Scalar`;
- parameter-matched `EM-Large` is unnecessary in the frozen-energy block because `V` is identical, but is required later;
- CPMLA or another learned sampling-geometry method when it can be applied to the same checkpoint without changing the target.

### Primary endpoint

The Pareto frontier of FID versus measured sampling wall-clock, with FID versus NFE as a secondary mechanism plot.

### Pass condition

At the same frozen energy, REM must achieve either:

- at least 25--30% lower sampling wall-clock/NFE at matched FID, or
- at least 0.3 lower FID at matched wall-clock,

without a material precision/recall regression. The conclusion is negative if an NFE gain disappears after including divergence and mobility overhead.

## 9. Experiment E5: CIFAR-10 joint REM

### Training stages

1. initialize from the reproduced Energy Matching model;
2. train the gauge-fixed mobility with frozen energy;
3. unfreeze the energy with a trust-region penalty;
4. enable contrastive training using corrected REM negatives;
5. maintain separate EMA states and atomic checkpoints for `V` and `G`.

### Main comparisons

- `EM` at the original architecture and compute;
- `EM-Large` matched to REM parameter count;
- `EM` trained for matched GPU-hours;
- `PEM-Residual` or the closest reproducible projected-field baseline;
- `REM-Scalar`, `REM-Diag`, and optional `REM-LR`;
- `REM-Unfixed` as an identifiability diagnostic.

### Evaluation

Use three independent training seeds. Apply the same 50,000-sample evaluation and sampler sweep as E4. Report training curves against both optimization steps and GPU-hours.

### Primary endpoint

Compute-matched FID-50k averaged across training seeds. Sampling efficiency is co-primary if Gate C passed.

### Supporting endpoints

- KID and precision/recall;
- training stability/failure rate;
- energy calibration and negative-energy distributions;
- held-out transport residual;
- total training GPU-hours and inference cost;
- metric condition and gauge diagnostics.

### Pass condition

The improvement must survive both parameter-matched and compute-matched controls, and its confidence interval must exclude a practically negligible effect selected before the run. A single favorable seed is insufficient.

## 10. Experiment E6: inverse-problem transfer

### Motivation

Energy Matching is valuable because a static energy can be combined with observation likelihoods. REM should preserve this property and ideally improve posterior exploration without retraining the prior energy.

### Tasks

Use at least two standardized linear inverse problems on CIFAR-10:

- center/random-mask inpainting;
- `4x` super-resolution or Gaussian deblurring.

For observation `y` and forward operator `A`, define a posterior potential

```math
V_{\mathrm{post}}(x;y)=V_\theta(x)+
\frac{1}{2\sigma_y^2}\|Ax-y\|^2.
```

Apply the already learned mobility to the posterior drift and retain the appropriate divergence correction. Do not retrain `V` for a test operator. Any adaptation of `G` is a separate ablation.

### Metrics

- PSNR, SSIM, and LPIPS;
- measurement consistency;
- posterior sample diversity at matched reconstruction error;
- coverage/calibration on synthetic observations with known ground truth;
- ESS, between-mode movement, NFE, and wall-clock.

### Baselines

- Euclidean Energy Matching posterior sampler;
- tuned constant preconditioning;
- REM frozen prior mobility;
- task-adapted mobility as an upper-bound ablation;
- a standard diffusion/score inverse solver when evaluation code and compute permit a fair comparison.

### Pass condition

Frozen REM mobility improves mixing or calibrated diversity at matched reconstruction quality and runtime. Improvement from task-specific retraining alone does not establish transfer.

## 11. Experiment E7: optional ImageNet-32 scaling

Run ImageNet-32 only after E4 and E5 establish the mechanism on CIFAR-10. The purpose is scalability, not rescue of a weak core claim.

Start with frozen-energy diagonal mobility. Profile one GPU, then two, four, and eight GPUs before allocating a full run. Report global batch size and verify DDP equivalence numerically on a short deterministic window.

The minimum ImageNet result is a compute-matched comparison between the same energy checkpoint sampled with Euclidean and REM dynamics. Joint ImageNet training is optional for the first submission.

## 12. Required ablation matrix

The following rows must appear in either the main paper or appendix:

| Question | Required variants |
|---|---|
| Is geometry more than a step size? | `EM-Const`, `REM-Scalar`, `REM-Diag` |
| Is state dependence necessary? | constant versus state-dependent mobility |
| Is anisotropy necessary? | scalar versus diagonal/full/low-rank |
| Does gauge fixing matter? | `REM-Diag` versus `REM-Unfixed` |
| Does equilibrium correction matter? | exact correction, Hutchinson, no correction |
| Is the gain just capacity? | `EM-Large` and compute-matched `EM` |
| Is the gain just another solver? | identical solver/order and matched NFE |
| Does joint training help? | frozen energy versus joint refinement |
| Is geometry robust? | metric initialization and regularization sweep |

## 13. Figures and tables planned before results

### Main figures

1. **Mechanism figure:** one non-conservative target field, Euclidean projection residual, and learned-metric fit with metric ellipses.
2. **Validity figure:** stationary bias versus step size for exact, Hutchinson, and omitted divergence correction.
3. **Causal figure:** frozen-energy CIFAR FID versus wall-clock Pareto frontier.
4. **Full-model figure:** joint REM quality/coverage versus compute with confidence intervals.

### Main tables

1. synthetic representability and metric-recovery results;
2. frozen-energy CIFAR sampling results;
3. joint CIFAR parameter-/compute-matched results;
4. inverse-problem transfer and posterior-diversity results.

Negative or failed variants remain in the appendix with their configured budgets.

## 14. Artifact layout

The intended experiment output structure is:

```text
outputs/
  <experiment_id>/
    resolved_config.json
    environment.json
    metrics.jsonl
    summary.json
    checkpoints/
    samples/
    figures/
```

`environment.json` stores the Git commit and hardware/software environment; `resolved_config.json` stores seeds and the full configuration; `summary.json` stores status, failure reason, values, means, standard deviations, and confidence intervals. Final tables are regenerated with `experiments.aggregate_results`, and preregistered thresholds are evaluated with `experiments.check_claim_gates` rather than manual copying.

## 15. Go/no-go interpretation

- **Strong main-conference path:** C1-C6 pass, with a clean theorem, frozen-energy causal gain, corrected equilibrium, and multi-seed compute-matched CIFAR evidence.
- **Promising but incomplete:** synthetic mechanism passes but CIFAR efficiency is neutral; continue as a theory/geometry paper or move the method to latent space.
- **Negative result:** frozen-energy geometry does not close residual or improve mixing after overhead; do not spend an ImageNet-scale budget.
- **Reframe required:** gains only appear without divergence correction or gauge fixing; the current REM hypothesis is not supported.
