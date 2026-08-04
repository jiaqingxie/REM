# REM Implementation Status

Last updated: 2026-08-04

This file separates implemented experiment code from completed scientific evidence. A row marked **implemented** means the code path exists and can be inspected or launched; it does not mean the corresponding paper claim has passed.

## Status legend

| Status | Meaning |
|---|---|
| `Implemented + CPU tested` | Unit tests and/or a dependency-light smoke run passed locally |
| `Implemented; GPU unverified` | Code and configuration exist, but the full CUDA/data run has not been completed |
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
| Frozen-energy CIFAR REM training | Implemented; GPU unverified | `experiments/cifar10/train_rem.py --mode frozen` |
| Joint REM initialized from the matching frozen seed | Implemented; GPU unverified | `--mode joint --rem-init-checkpoint ...` and the ablation manifest |
| Parameter-matched `EM-Large` | Implemented; GPU unverified | `experiments/cifar10/match_capacity.py` |
| Wall-clock compute-matched EM | Implemented; GPU unverified | `--max-training-seconds` and `configs/cifar10_compute_matched.json` |
| FID/KID/precision/recall versus NFE and wall-clock | Implemented; GPU unverified | `experiments/cifar10/evaluate_rem.py` |
| Cold-start, steady-state, VRAM, JVP, and sampler component timing | Implemented; GPU unverified | CIFAR evaluator and `rem/sampling.py` |
| Three training seeds and bootstrap confidence intervals | Implemented; GPU unverified | manifests, `rem/artifacts.py`, `experiments/aggregate_results.py` |
| `REM-NoDiv` reusing the exact same `REM-Diag` checkpoint | Implemented; GPU unverified | `configs/cifar10_evaluation.json` |
| `REM-Unfixed` gauge ablation with log-determinant diagnostics | Implemented; GPU unverified | CIFAR ablation/evaluation manifests |
| Frozen mobility transferred to posterior sampling | Implemented; GPU unverified | `experiments/inverse/run_cifar_inverse.py` |
| Inpainting, super-resolution, and deblurring | Implemented; GPU unverified | inverse runner task flags |
| Posterior PSNR, consistency, diversity, coverage, ESS, R-hat, and runtime | Implemented; GPU unverified | `rem/inverse.py`, inverse runner |
| Automatic 25--30% wall-clock / 0.3 FID / coverage claim checks | Implemented + CPU tested | `experiments/check_claim_gates.py`, `configs/claim_gates.example.json` |

## Validation completed locally

- `32/32` unit tests pass on CPU.
- Every new experiment module imports without starting optional runtime work.
- `python -m compileall -q rem experiments tests` passes.
- Learned/frozen synthetic representability smoke runs pass.
- Gaussian, mixture, and banana stationarity smoke runs pass.
- Multi-seed aggregation and claim-gate unit tests pass.
- CIFAR-10 FID, multi-GPU DDP equivalence, CUDA memory, and inverse-problem image runs have **not** been executed locally.

## Explicit remaining TODOs

These items appeared in the broader experimental protocol but are not part of the currently implemented core code.

| Item | Status | Completion condition |
|---|---|---|
| Official Projected Energy Matching residual baseline | External baseline | Run the authors' code under the same data, parameter, and compute protocol |
| Official CPMLA learned-mirror sampler | External baseline | Adapt the official sampler to the same frozen REM/EM energy checkpoint |
| Standard diffusion/score inverse solver | External baseline | Select one reproducible official solver and match measurement/runtime settings |
| Gaussian-to-Swiss-roll OT field | TODO | Add deterministic OT-pair generation and the same representability metrics |
| Gaussian-mixture-to-curved-anisotropic OT field | TODO | Add dataset/coupling generator and frozen-coupling artifact |
| Metropolis-corrected low-dimensional reference chain | TODO | Add an accept/reject reference implementation and matched-gradient accounting |
| Round-trip time and explicit mode-occupancy diagnostics | TODO | Add mode labels and transition/round-trip logging to stationarity runs |
| Learned 2D Energy Matching potential evaluated as an exact grid density | TODO | Normalize the learned grid density and add it to the stationarity target interface |
| ImageNet-32 frozen REM scaling | Optional | Start only after frozen and joint CIFAR claim gates pass |
| Intrinsic-manifold data support | Optional / out of first-paper scope | Define reference measure, manifold operators, and intrinsic SDE separately |

## Theory items that code cannot replace

The repository contains the constructive mapping and numerical diagnostics, but the paper still needs formal proofs for:

- smooth bounded-SPD construction across zero sets under explicit uniform assumptions;
- the precise minimal-distortion optimization problem and any uniqueness statement;
- gauge identifiability after determinant normalization;
- the formal relation between Euclidean projection error, curl/Jacobian antisymmetry, and REM residual;
- continuous-time invariance and the assumptions under which boundary terms vanish.

These must not be marked complete merely because their numerical counterparts are implemented.

## Server execution order

1. Clone the private repository and install the CUDA/Python dependencies from `README.md`.
2. Run `python -m unittest discover -s tests -v`.
3. Run the synthetic learned-energy, known-metric, non-representable, and stationarity suites.
4. Obtain or reproduce the frozen Energy Matching checkpoint.
5. Run `match_capacity.py` and record the selected `EM-Large` multiplier.
6. Generate the CIFAR ablation commands without `--execute` and audit paths/budgets.
7. Execute frozen runs before joint runs; the manifest resolves the same seed's frozen checkpoint.
8. Run the separate compute-matched manifest using the measured joint wall-clock budget.
9. Run `configs/cifar10_evaluation.json`, aggregate all three seeds, and evaluate the preregistered claim gates.
10. Run inverse transfer only with the selected frozen mobility checkpoint.

See `README.md` for exact command templates and `docs/EXPERIMENTS.md` for the preregistered protocol.
