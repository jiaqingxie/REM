# Data and Checkpoint Reproduction

This document describes the public data path used by REM. Datasets, downloaded energy
checkpoints, generated samples, and experiment outputs are intentionally excluded from Git.
Commands below are run from the repository root.

## Expected layout

```text
data/
  cifar10/
    cifar-10-batches-py/
  imagenet32/
    Imagenet32_train/
      train_data_batch_1
      ...
      train_data_batch_10
  checkpoints/
    cifar10_main_training_147000.pt
    cifar10_warm_up_145000.pt              # optional robustness experiment
    imagenet32x32_main_training_641000.pt
  fid_stats/
    imagenet32_train_1281167_floor_fid_stats.pt
```

You may use different paths; pass them explicitly through `--data-root`,
`--energy-checkpoint`, and `--fid-real-stats`. The loaders never require the layout above when
those flags are supplied.

## 1. CIFAR-10

REM uses the 50,000-image CIFAR-10 training split. Mobility training applies random horizontal
flips, converts pixels to tensors, and maps `[0, 1]` to `[-1, 1]`. Metric evaluation uses the
same normalization without augmentation.

Download through torchvision:

```bash
python - <<'PY'
from torchvision.datasets import CIFAR10
CIFAR10(root="data/cifar10", train=True, download=True)
PY
```

Verify the layout and sample count:

```bash
test -s data/cifar10/cifar-10-batches-py/data_batch_1
python - <<'PY'
from torchvision.datasets import CIFAR10
dataset = CIFAR10(root="data/cifar10", train=True, download=False)
assert len(dataset) == 50_000
print({"dataset": "cifar10", "train_samples": len(dataset)})
PY
```

Download the upstream checkpoints from the
[Energy Matching CIFAR-10 repository](https://huggingface.co/m1balcerak/energy_matching_cifar10)
and place them under `data/checkpoints/`.

| Checkpoint | Purpose | SHA256 |
|---|---|---|
| `cifar10_main_training_147000.pt` | main frozen-energy experiments | `5708b966d7873534285353deb2a8af0417a43851e2b730a5fa0d41f254ded9fc` |
| `cifar10_warm_up_145000.pt` | checkpoint-robustness control | `edcada07d7ef0f5e64d56804fd143b5655a1942b61bd97e69316e09e88a63b38` |

```bash
sha256sum data/checkpoints/cifar10_*.pt
```

The default frozen REM run loads the checkpoint EMA state. The `continue-energy` control instead
restores the raw network together with optimizer, scheduler, EMA, and step state from the same
full checkpoint.

## 2. ImageNet32

Download the 32×32 downsampled ImageNet training archive from the
[official release](https://patrykchrabaszcz.github.io/Imagenet32/). The direct archive used by
the loader is `https://www.image-net.org/data/downsample/Imagenet32_train.zip`.

```bash
mkdir -p data/imagenet32/Imagenet32_train
curl -L --fail --retry 5 \
  https://www.image-net.org/data/downsample/Imagenet32_train.zip \
  -o data/imagenet32/Imagenet32_train.zip
unzip -q data/imagenet32/Imagenet32_train.zip \
  -d data/imagenet32/Imagenet32_train
```

The expected archive size is 3,590,350,354 bytes. The normalized directory must contain exactly
`train_data_batch_1` through `train_data_batch_10`. If an unzip tool creates an extra directory,
pass that directory directly to `--data-root` or move the ten batch files into
`data/imagenet32/Imagenet32_train/`.

Verify the loader:

```bash
python - <<'PY'
from experiments.imagenet.dataset_imagenet32 import ImageNet32Dataset
dataset = ImageNet32Dataset(root="data/imagenet32/Imagenet32_train")
assert len(dataset) == 1_281_167
print({"dataset": "imagenet32", "train_samples": len(dataset)})
PY
```

Download `imagenet32x32_main_training_641000.pt` from the upstream
[Energy Matching checkpoint collection](https://huggingface.co/m1balcerak/energy_matching).

```text
SHA256  e56e26da8c37f8af2ae60844ef5217227681887d3d401dcd3f8c9b9f7e6b8a34
```

The paper protocol computes ImageNet32 FID against all 1,281,167 training images, using floor
conversion to uint8. Cache the real-data sufficient statistics once:

```bash
python -m experiments.imagenet.cache_imagenet32_fid_stats \
  --data-root data/imagenet32/Imagenet32_train \
  --output data/fid_stats/imagenet32_train_1281167_floor_fid_stats.pt \
  --image-quantization floor \
  --expected-samples 1281167
```

Then reuse the cache during every identity/REM evaluation:

```bash
python -m experiments.cifar10.evaluate_rem \
  --dataset imagenet32 \
  --checkpoint /path/to/rem_checkpoint.pt \
  --data-root data/imagenet32/Imagenet32_train \
  --fid-real-stats data/fid_stats/imagenet32_train_1281167_floor_fid_stats.pt \
  --expected-fid-real-samples 1281167 \
  --image-quantization floor \
  --samples 50000 --steps 250 \
  --output outputs/imagenet32_eval
```

Do not mix `floor` and `round` statistics. The evaluator checks cached-statistic metadata and
rejects a quantization mismatch.

## 3. AAV assets

The exploratory AAV experiment uses the three CSV files from the upstream
[Energy Matching repository](https://github.com/m1balcerak/EnergyMatching/tree/main/experiments/proteins/data):

```text
experiments/proteins/data/aav_medium.csv
experiments/proteins/data/aav_hard.csv
experiments/proteins/data/aav_ground_truth.csv
```

The VAE, predictor, and oracle checkpoints are included under `experiments/proteins/`. The AAV
energy checkpoints are available from
[m1balcerak/energy_matching](https://huggingface.co/m1balcerak/energy_matching):

| Checkpoint | SHA256 |
|---|---|
| `aav_medium_main_training.pt` | `94a58b84137a9a5dcc62b49c7f9b21b2e18319dc87189a14e10ef85603a206a8` |
| `aav_hard_main_training.pt` | `fb2985a50e0af6bc6afdcfffc9bbf7f52c19ad2d93cf07820af690e696f8606b` |

AAV scores are predictor-level diagnostics, not biological validation.

## 4. Synthetic experiments

The 2D warp, curved 16D mixture, and correlated 16D/64D Gaussian experiments require no external
dataset. Targets, minibatch-OT pairs, train/held-out splits, and sampling chains are generated
from the declared seed. Keep the five training seeds and held-out batches unchanged when
reproducing the reported uncertainty.

## 5. Training data flow

For image mobility training, each iteration performs the following operations:

1. Draw a real image `x1` from the normalized training set.
2. Draw Gaussian noise `x0` with the same shape.
3. Use `ExactOptimalTransportConditionalFlowMatcher(sigma=0)` to form the minibatch OT coupling.
4. Sample `t` and construct a straight chord `xt = (1-t)x0 + t*x1` with conditional velocity
   `ut = x1-x0`.
5. Keep the imported scalar energy frozen and update only the mobility from the supervised
   residual `ut + G(xt) grad V(xt)`.
6. Save the energy state, mobility state, mobility EMA, optimizer, scheduler, configuration,
   environment, and source-checkpoint identity in the run directory.

The additive-residual control receives the same images, OT chords, training budget, phase cutoff,
and parameter budget. Only its representation class changes.

## 6. Seeds and aggregation

Mobility-training seeds and sampling seeds are different sources of variation. The image protocol
uses three independently trained mobilities and evaluates each on the same three paired sampling
streams:

```text
training seed 0 ─┬─ sampling stream 0
                 ├─ sampling stream 1
                 └─ sampling stream 2
training seed 1 ─┬─ sampling stream 0
                 ├─ sampling stream 1
                 └─ sampling stream 2
training seed 2 ─┬─ sampling stream 0
                 ├─ sampling stream 1
                 └─ sampling stream 2
```

Aggregate sampling streams within each mobility fit first, then report variation across the three
fit means. Do not treat generated images, or all nine crossed cells, as independent training
replicates. Identity and learned variants must share the same initialization stream and solver
configuration for paired comparisons.

## 7. Artifact checks

Every completed run should contain:

```text
environment.json
resolved_config.json
metrics.jsonl
summary.json
```

A run is eligible for aggregation only when `summary.json` contains `"status": "complete"` and
`"failure_reason": null`. Preserve the source checkpoint path and SHA256, resolved configuration,
code revision, training seed, sampling seed, quantization rule, sample count, solver, and phase
cutoff with each result.

Run the local validation suite before launching full GPU experiments:

```bash
python -m compileall -q rem experiments tests
PYTHONPATH=. pytest -q
```
