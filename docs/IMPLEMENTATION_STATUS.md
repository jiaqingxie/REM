# REM Implementation Status

Last updated: 2026-08-31

This file separates implemented experiment code from completed scientific evidence. A row marked **implemented** means the code path exists and can be inspected or launched; it does not mean the corresponding paper claim has passed.

## Status legend

| Status | Meaning |
|---|---|
| `Implemented + CPU tested` | Unit tests and/or a dependency-light smoke run passed locally |
| `Implemented; GPU unverified` | Code and configuration exist, but the full CUDA/data run has not been completed |
| `GPU completed` | The declared full GPU run finished with a complete machine-readable artifact |
| `GPU running` | The declared full GPU run is active and must not yet be used as final evidence |
| `External baseline` | Must use the authors' official implementation; REM does not claim a substitute implementation |
| `TODO` | Planned in the protocol but no complete experiment code path exists yet |
| `Optional` | Not required before the central CIFAR claim gates pass |

## Core checklist requested for the first submission

| Requirement | Status | Code / artifact |
|---|---|---|
| Bounded SPD, determinant-one gauge, unfixed gauge ablation | Implemented + CPU tested | `rem/geometry.py`, `tests/test_geometry.py` |
| Constant, scalar, diagonal, full-SPD, and diagonal-plus-low-rank mobility | Implemented + CPU tested | `rem/geometry.py`, `rem/networks.py` |
| Constructive pointwise SPD mapping and descent-condition violation diagnostic | Implemented + CPU tested | `construct_spd_mapping`, `descent_condition_diagnostics` |
| Minimal-distortion penalty and gauge/conditioning diagnostics | Implemented + CPU tested | `minimal_distortion_regularizer`, `mobility_diagnostics` |
| Synthetic recovery of a known metric | Implemented + CPU tested | `experiments/synthetic/run_representability.py --no-learn-energy` |
| Parameter-matched Euclidean projection-gap comparison | Implemented + CPU tested | learned-energy mode with automatic `EM-Large` width matching |
| Prediction of non-representable target fields | Implemented + CPU tested | `--problem nonrepresentable --no-learn-energy` |
| Exact, Hutchinson, and omitted-divergence stationarity comparison | Implemented + CPU tested | `experiments/synthetic/run_stationarity.py` |
| Correlated Gaussian, Gaussian-mixture, and banana stationary targets | Implemented + CPU tested | `--targets gaussian,mixture,banana` |
| Frozen-energy CIFAR REM training | GPU completed | mobility-training seeds 0/1/2 under `outputs/paper_results/rem_diagonal_seed*/checkpoints/checkpoint_150000.pt` |
| Joint REM initialized from the matching frozen seed | Implemented; GPU unverified | `--mode joint --rem-init-checkpoint ...` and the ablation manifest |
| Parameter-matched `EM-Large` | Implemented; GPU unverified | `experiments/cifar10/match_capacity.py` |
| Wall-clock compute-matched EM | Implemented; GPU unverified | `--max-training-seconds` and `configs/cifar10_compute_matched.json` |
| FID/KID versus NFE and wall-clock | GPU completed for the paper protocol | original seed-0 evaluation plus finalized nested three-mobility aggregate under `outputs/reviewer_expansion_20260818` |
| Cold-start, steady-state, VRAM, JVP, and sampler component timing | Implemented; GPU unverified | CIFAR evaluator and `rem/sampling.py` |
| Three mobility-training seeds with separate sampling seeds | GPU completed | exact 3x3 REM grid; `aggregate_reviewer_seed_expansion.py` enforces nested aggregation |
| `REM-NoDiv` reusing the exact same `REM-Diag` checkpoint | Implemented; GPU unverified | `configs/cifar10_evaluation.json` |
| `REM-Unfixed` gauge ablation with log-determinant diagnostics | Implemented; GPU unverified | CIFAR ablation/evaluation manifests |
| Frozen mobility transferred to posterior sampling | Implemented; GPU unverified | `experiments/inverse/run_cifar_inverse.py` |
| Inpainting, super-resolution, and deblurring | Implemented; GPU unverified | inverse runner task flags |
| Posterior PSNR, consistency, diversity, coverage, ESS, R-hat, and runtime | Implemented; GPU unverified | `rem/inverse.py`, inverse runner |
| Automatic 25--30% wall-clock / 0.3 FID / coverage claim checks | Implemented + CPU tested | `experiments/check_claim_gates.py`, `configs/claim_gates.example.json` |

## Validation and formal evidence completed

- `85/85` unit tests pass on CPU as of the second reviewer response, including duplicate-record
  rejection, matched-protocol drift detection, rejection of contradictory complete summaries
  carrying a failure reason, and an end-to-end synthetic check of the complete reviewer
  aggregate-to-figure path.
- Every new experiment module imports without starting optional runtime work.
- `python -m compileall -q rem experiments tests` passes.
- Learned/frozen synthetic representability smoke runs pass.
- Gaussian, mixture, and banana stationarity smoke runs pass.
- Multi-seed aggregation and claim-gate unit tests pass.
- CIFAR-10 and ImageNet32 formal sampling runs execute on H100/CUDA 12.8 Jobs; the shared local node is used only for artifact inspection and CPU tests.
- The original five-seed controlled inductive-bias comparison completed successfully with 30/30
  records under `outputs/inductive_bias_controlled_20260818`; a five-record PEM Helmholtz
  residual extension v2 completed after correcting loss normalization, and the final combined
  gate passes 35/35.
- The residual/REM and mirror/REM parameter ratios are 0.9962 and 0.9696, satisfying the preregistered 5% capacity check.
- The original practical claim gates remain negative: the available CIFAR gain does not reach the 0.3-FID or 30% wall-clock thresholds. The reviewer expansion strengthens uncertainty and mechanism evidence; it does not retroactively change those gates.

## Reviewer expansion completed

| Requirement | Live status | Completion evidence |
|---|---|---|
| ImageNet32 sampling seeds 7 and 42, identity and REM | GPU completed (4 Jobs) | three-stream identity/REM FID pair with seeds 0/7/42; 3/3 paired improvements |
| ImageNet32 mobility-training replication | GPU completed (2 composite Jobs) | three independent mobility fits, each evaluated on three paired 50k streams; nested REM FID `6.5547±0.0637`; 9/9 paired improvements |
| CIFAR mobility-training seeds 1 and 2 crossed with sampling seeds 0/1/2 | GPU completed (6 Jobs) | complete 3x3 nested REM grid; 9/9 paired improvements |
| CIFAR learned constant-SPD control on sampling seeds 0/1/2 | GPU completed (3 Jobs) | three paired FID/KID records |
| Free residual / learned mirror / constant preconditioner controlled comparison | GPU completed | original 30-record artifact under `outputs/inductive_bias_controlled_20260818` |
| PEM divergence-penalized Helmholtz residual | GPU completed | five seeds under `outputs/pem_helmholtz_controlled_v2_20260818`; final combined grid 35 records; first scale-mismatched run stopped and excluded |
| Synthetic/image descent-margin diagnostic | GPU/CPU completed | 69,632 held-out supervision pairs; zero violations; normalized-margin distributions recorded |
| Synthetic Euclidean/metric-weighted loss ablation | GPU completed | five matched fits per objective; exact config gate and generated appendix table complete |
| Held-out image mobility structure | GPU completed | 24,576 pairs and three final fits per dataset; external Euclidean residual, eigenvalues, phase bins, and cross-fit correlations recorded |

The one-shot live monitor is `tools/monitor_reviewer_jobs.py`. The hard-gated finalizer passed
with all 27 declared image records and all 35 controlled records, producing the final aggregate,
paper assets, and `finalization_manifest.json`.

## Final closure extensions

- The parameter-matched additive-residual CIFAR control is implemented in
  `rem.geometry.AdditiveResidualTransport` and the CIFAR training/evaluation entry points; its
  complete three-fit by three-stream evaluation is aggregated by
  `experiments/cifar10/aggregate_additive_residual.py`.
- `experiments/synthetic/run_nonoracle_curved_ot.py` trains diagonal and structured mobility
  from ordinary minibatch-OT chords on a curved multimodal 16D target.
- `experiments/synthetic/run_highdim_corrected_sampling.py` evaluates corrected 16D/64D
  positive-temperature sampling and includes an exact-covariance global preconditioner as an
  oracle-style classical calibration.

## Explicit remaining TODOs

These items appeared in the broader experimental protocol but are not part of the currently implemented core code.

| Item | Status | Completion condition |
|---|---|---|
| Controlled free residual representation | GPU completed | same frozen energy, supervised velocity, points, seeds, and matched capacity; an unrestricted upper bound |
| Controlled PEM Helmholtz residual representation | GPU completed | same frozen energy/evidence plus exact 2D divergence penalty corresponding to PEM Phase 2; explicitly not the full PEM pipeline |
| Controlled learned-mirror sampler | GPU completed | inverse-Hessian mobility with a capacity-matched strongly convex mirror potential |
| Official full Projected Energy Matching pipeline | External baseline | requires the authors' teacher/refinement setup; the paper does not claim an official reproduction |
| Official full CPMLA pipeline | External baseline | requires adapting the authors' joint EBM/mirror training; the paper reports a controlled representation-class adaptation |
| Standard diffusion/score inverse solver | External baseline | Select one reproducible official solver and match measurement/runtime settings |
| Gaussian-to-Swiss-roll OT field | TODO | Add deterministic OT-pair generation and the same representability metrics |
| Gaussian-mixture-to-curved-anisotropic OT field | TODO | Add dataset/coupling generator and frozen-coupling artifact |
| Metropolis-corrected low-dimensional reference chain | TODO | Add an accept/reject reference implementation and matched-gradient accounting |
| Round-trip time and explicit mode-occupancy diagnostics | GPU completed | five-seed geometry stress and inductive-bias records |
| Learned 2D Energy Matching potential evaluated as an exact grid density | TODO | Normalize the learned grid density and add it to the stationarity target interface |
| ImageNet-32 frozen REM scaling | GPU completed | three independent mobility fits crossed with three paired 50k streams under the strict full-real FID protocol |
| Intrinsic-manifold data support | Optional / out of first-paper scope | Define reference measure, manifold operators, and intrinsic SDE separately |

## Theory items that code cannot replace

The repository contains the constructive mapping and numerical diagnostics. The current paper
also includes formal proofs/derivations for the following, but these remain theory claims whose
assumptions must be audited independently of passing code:

- smooth bounded-SPD construction across zero sets under explicit uniform assumptions;
- the precise minimal-distortion optimization problem and any uniqueness statement;
- gauge identifiability after determinant normalization;
- the formal relation between Euclidean projection error, curl/Jacobian antisymmetry, and REM residual;
- continuous-time invariance and the assumptions under which boundary terms vanish.

These must not be treated as established merely because their numerical counterparts are implemented.

## Server execution order

1. Clone the private repository and install the CUDA/Python dependencies from `README.md`.
2. Run `PYTHONPATH="$PWD" pytest -q`.
3. Run the synthetic learned-energy, known-metric, non-representable, and stationarity suites.
4. Obtain or reproduce the frozen Energy Matching checkpoint.
5. Run `match_capacity.py` and record the selected `EM-Large` multiplier.
6. Generate the CIFAR ablation commands without `--execute` and audit paths/budgets.
7. Execute frozen runs before joint runs; the manifest resolves the same seed's frozen checkpoint.
8. Run the separate compute-matched manifest using the measured joint wall-clock budget.
9. Run `configs/cifar10_evaluation.json`, aggregate all three seeds, and evaluate the preregistered claim gates.
10. Run inverse transfer only with the selected frozen mobility checkpoint.

See `README.md` for exact command templates and `docs/EXPERIMENTS.md` for the preregistered protocol.
