# Reproducing the paper

This release supplies executable code and protocol settings, with no saved
results. All commands below run from the repository root. Use `--dry-run` to
inspect a runner command before launching it. The runner's `--output` changes
the location of newly generated runs; the default is `outputs/reproduction`.

## Experiment map

| Paper evidence | Entry point |
| --- | --- |
| Main 2D ordinary-OT field fitting, mode passage and same-checkpoint divergence ablation | `python scripts/reproduce.py warp-ot` |
| Analytic-field geometry recovery and step-size validity grid | `python scripts/reproduce.py geometry` |
| Constant, diagonal, full, mirror and additive representation controls | `python scripts/reproduce.py controls` |
| Frozen-energy PEM residual-class control | `python scripts/reproduce.py pem-control` |
| 64D structured OT field fitting and continued-energy control | `python scripts/reproduce.py highdim` |
| Curved non-oracle 16D target and corrected chains | `python scripts/reproduce.py curved` |
| 16D/64D corrected Gaussian chains | `python scripts/reproduce.py corrected-highdim` |
| CIFAR-10 and ImageNet32 training, nested evaluation and held-out geometry diagnostics | Image commands below |

Synthetic commands use CPU by default. The main ordinary-OT 2D run keeps the
paper's CPU protocol; other synthetic suites accept `--device cuda`. These are
full experiments, not short examples. `warp-ot --smoke` is the separate short
integration check.

The representation construction and stationary-law utilities also have direct
entry points in `experiments.synthetic.run_representability` and
`experiments.synthetic.run_stationarity`; use `--help` for their controls.
The PEM comparison isolates an additive residual representation over a frozen
energy. It does not reproduce the complete energy-changing PEM pipeline.

## Main ordinary-OT experiment

The runner uses 1,500 independent OT minibatches, batch size 256, width 128,
depth 3, learning rate 0.002, log bound 3 and regularization weight 0.0001.
Checkpoint selection uses eight fresh OT batches, with sixteen disjoint test
batches. The curved target uses five seeds; the zero-curvature field control
uses three, matching the paper. Each fitted diagonal/full model is evaluated
with 64 chains, step size 0.002, burn-in time 4 and draw time 12. The full-no-div
condition reloads the same full checkpoint and removes only the Itô divergence
term. The aggregator verifies checkpoint hashes before combining seeds.

Training and sampling may be launched separately with `--stage train` and
`--stage evaluate`, using the same `--output` for both commands.

Passage starts at the first retained state after burn-in, with nearest-mean
initial labels and radius-0.65 modal cores. It is conditioned on arrival and
averaged over per-seed medians; completion rates are recorded separately.
MMD means the biased squared Gaussian-kernel estimator. The 2D comparison uses
2,048 randomly selected retained states in latent coordinates. The 16D/64D
diagnostics use the first 1,024 flattened retained states in observed coordinates.
Bandwidth squared is the median positive pooled squared pairwise distance.
ESS uses modal indicators in 2D and observed coordinates in the higher-dimensional
sampling diagnostics. Seeds, not individual chains, are the independent units.

## Image training and paired evaluation

Acquire the datasets and energy checkpoints as described in [data.md](../data.md).
The following commands run three independent mobility fits, then identity and
nine paired REM evaluations, followed by held-out field diagnostics:

```bash
python scripts/reproduce.py cifar10 --device cuda \
  --data-root data/cifar10 \
  --energy-checkpoint data/checkpoints/cifar10_main_training_147000.pt

python scripts/reproduce.py imagenet32 --device cuda \
  --data-root data/imagenet32/Imagenet32_train \
  --energy-checkpoint data/checkpoints/imagenet32x32_main_training_641000.pt
```

Use `--stage train`, `--stage evaluate` or `--stage diagnose` to run only one
phase. Image training selects the available CUDA device; `--device` selects
the device for evaluation and diagnostics. Select the training GPU through
`CUDA_VISIBLE_DEVICES`. The shared training module also supports `torchrun`.
The paper uses global batch size 128 and one GPU per fit.

The runner explicitly fixes the settings that differ from generic CLI defaults:

| Setting | CIFAR-10 | ImageNet32 |
| --- | --- | --- |
| Mobility fitting steps | 150,000 | 150,000 |
| Training seeds | 0, 1, 2 | 0, 1, 2 |
| Sampling seeds | 0, 1, 2 | 0, 7, 42 |
| Samples per evaluation | 50,000 | 50,000 |
| Euler–Heun intervals | 325 | 250 |
| Terminal time | 3.25 | 2.50 |
| Batch size | 64 | 64 |
| Geometry/temperature cutoff | 1.0 | 1.0 |
| Maximum temperature | 0.01 | 0.01 |
| Pixel conversion | round | floor |
| KID | enabled | disabled |
| Real FID data | all 50,000 training images | all 1,281,167 training images |

Images are clamped only at the terminal state. The raw frozen energy and EMA
mobility are selected explicitly. Identity evaluations reuse fit 0's unchanged
energy with the mobility gate disabled. Fit/stream variations are aggregated
separately by `aggregate_reviewer_seed_expansion`; identity references are reused
for pairing rather than counted as additional independent training fits.

## Ablations

Use the printed commands from `--dry-run` as the base for direct module calls.
Keep the energy, seeds and other protocol settings fixed, and give every
condition its own `--experiment-id`.

- Image objective: pass `--no-metric-weighted` to `train_rem`.
- Image structure: pass `--mobility structured --mobility-rank 4` to `train_rem`.
- Image constant/additive controls: use `--mobility constant` or
  `--mobility additive-residual` with the same training protocol.
- Solver sweep: change `evaluate_rem --steps` to `50`, `100` or `200`, retaining
  terminal time 3.25. Use the matched one-stream-per-fit subset for comparison
  with the 325-interval row. Each Heun interval uses two energy-gradient calls.
- Cutoff screen: set `--samples 10000 --mobility-active-until` to each of
  `0.75`, `0.90`, `1.00`, `1.10` and `1.25`, with the same three sampling streams.
  The final cutoff is not retuned separately for each solver budget.
- Ungated correction screen: use fit 0, sampling seed 0, 10,000 samples,
  `--mobility-active-until -1` and either `--divergence-correction` or
  `--no-divergence-correction`. Compare with the same 10,000-sample gated run.
  In stochastic Heun, Off removes the explicit Stratonovich term; its equivalent
  Itô drift retains half the divergence term. No new FID is bundled here.
- Synthetic objective: rerun `run_geometry_stress` with
  `--phases main --curvatures 1.2 --variants rem-full --metric-weighted` and the
  same geometry-suite settings. `aggregate_loss_weighting_ablation` compares
  the Euclidean and intrinsic runs; its input paths are explicit CLI arguments.

The optional `aggregate_phase_gate` utility groups streams within a specified
fit. Do not mix several mobility fits through that single-fit utility. Use the
nested aggregator for cross-fit results. The additive-control aggregator expects
its documented run naming convention, available in `--help` and the source.

## Interpretation and reproducibility limits

The full corrected stochastic sampler is tested on low-dimensional targets.
The principal image method uses learned geometry only while temperature is
zero, avoiding high-dimensional divergence probes. The 64D analytic KL and
covariance values are properties of the energy, not estimates of chain accuracy.
A better field fit does not by itself establish faster mixing.

The paper reports finite-step gains, not a wall-clock speedup or image-scale
equilibrium convergence. Reruns depend on numerical library, hardware and random
number implementations. Run the tests and short pipeline before starting the
full experiments. New runs write their configurations, environment and metrics
locally; these outputs are not part of the source release.
