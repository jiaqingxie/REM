# Data and pretrained energy checkpoints

Run commands from the repository root. Data, weights, cached statistics and new
experiment outputs are excluded from this source release and from Git tracking.

## Synthetic targets

No download is needed. The 2D warped mixture, curved 16D mixture, and correlated
Gaussian targets generate data from their specified seeds. Training, checkpoint
selection, evaluation and chain seeds are separated by the experiment drivers.

## CIFAR-10

Download the 50,000-image training split using torchvision:

```bash
python - <<'PY'
from torchvision.datasets import CIFAR10
dataset = CIFAR10(root="data/cifar10", train=True, download=True)
assert len(dataset) == 50_000
PY
```

Mobility training applies random horizontal flips and normalizes pixels to
`[-1, 1]`. Evaluation uses the same normalization without augmentation. The
paper's CIFAR metrics use round-to-uint8 conversion and all 50,000 training images.

## ImageNet32

Obtain the training archive from the
[official downsampled ImageNet release](https://patrykchrabaszcz.github.io/Imagenet32/).
Point `--data-root` to the directory containing `train_data_batch_1` through
`train_data_batch_10`. Dataset access remains subject to the provider's terms.

```bash
mkdir -p data/imagenet32
curl -L --fail --retry 5 https://www.image-net.org/data/downsample/Imagenet32_train.zip -o data/imagenet32/train.zip
unzip -q data/imagenet32/train.zip -d data/imagenet32
```

If the archive adds a directory level, use that directory as `--data-root`.
Verify the normalized location:

```bash
python - <<'PY'
from experiments.imagenet.dataset_imagenet32 import ImageNet32Dataset
dataset = ImageNet32Dataset(root="data/imagenet32/Imagenet32_train")
assert len(dataset) == 1_281_167
PY
```

ImageNet32 FID uses all 1,281,167 training images and floor-to-uint8 conversion.
The reproduction runner caches their Inception sufficient statistics before
evaluation. To prepare the cache separately:

```bash
python -m experiments.imagenet.cache_imagenet32_fid_stats \
  --data-root data/imagenet32/Imagenet32_train \
  --output data/fid_stats/imagenet32_train_1281167_floor_fid_stats.pt \
  --expected-samples 1281167 --image-quantization floor --device cuda
```

Pass this file to the runner with `--fid-real-stats`. Cached and evaluation
quantization must agree; the evaluator checks that metadata.

## Frozen Energy Matching checkpoints

Both required files are published in the upstream
[Energy Matching checkpoint collection](https://huggingface.co/m1balcerak/energy_matching).

```bash
mkdir -p data/checkpoints
curl -L --fail --retry 5 https://huggingface.co/m1balcerak/energy_matching/resolve/main/cifar10_main_training_147000.pt -o data/checkpoints/cifar10_main_training_147000.pt
curl -L --fail --retry 5 https://huggingface.co/m1balcerak/energy_matching/resolve/main/imagenet32x32_main_training_641000.pt -o data/checkpoints/imagenet32x32_main_training_641000.pt
sha256sum data/checkpoints/*.pt
```

| File | SHA256 |
| --- | --- |
| `cifar10_main_training_147000.pt` | `5708b966d7873534285353deb2a8af0417a43851e2b730a5fa0d41f254ded9fc` |
| `imagenet32x32_main_training_641000.pt` | `e56e26da8c37f8af2ae60844ef5217227681887d3d401dcd3f8c9b9f7e6b8a34` |

Frozen REM imports the upstream EMA energy, holds it fixed, and trains only the
mobility. Evaluation explicitly selects the unchanged `energy` state and the
`mobility_ema` state from the resulting REM checkpoint. Do not substitute a
different pretrained energy and interpret its metrics as the same experiment.

TorchMetrics/torch-fidelity also obtain their public Inception weights on the
first FID/KID evaluation. Network access or a prepopulated local cache is needed
for that first use.
