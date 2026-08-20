"""Cache full-train ImageNet32 FID real statistics for paper evaluation.

The official Energy Matching evaluator uses all 1,281,167 ImageNet32 training
images as the real reference and 50,000 generated images. This command scans
the real set once and stores only the sufficient statistics needed by FID;
subsequent identity/REM evaluations can reuse the cache without retaining or
re-extracting 1.28M Inception features.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.imagenet.dataset_imagenet32 import ImageNet32Dataset
from rem.image_metrics import ImageMetricSuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--image-quantization",
        choices=["round", "floor"],
        default="floor",
        help="floor exactly matches the official Energy Matching evaluator",
    )
    parser.add_argument("--progress-every-batches", type=int, default=100)
    parser.add_argument("--expected-samples", type=int, default=1_281_167)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def dataset_manifest(root: Path) -> list[dict[str, int | str]]:
    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(root.glob("train_data_batch_*"))
    ]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    if args.progress_every_batches < 0 or args.expected_samples < 0:
        raise ValueError("progress interval and expected samples must be nonnegative")

    from torchvision import transforms
    import torchmetrics

    data_root = Path(args.data_root).expanduser().resolve()
    dataset = ImageNet32Dataset(
        split="train",
        root=data_root,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    if args.expected_samples > 0 and len(dataset) != args.expected_samples:
        raise ValueError(
            f"expected {args.expected_samples} ImageNet32 images, found {len(dataset)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    device = torch.device(args.device)
    suite = ImageMetricSuite(
        device=device,
        collect_precision_recall=False,
        compute_kid=False,
        quantization=args.image_quantization,
    )
    start = time.perf_counter()
    seen = 0
    for batch_index, (images, _) in enumerate(loader, start=1):
        suite.update(images, real=True, update_kid=False)
        seen += images.shape[0]
        if args.progress_every_batches > 0 and (
            batch_index % args.progress_every_batches == 0 or seen == len(dataset)
        ):
            print(
                json.dumps(
                    {
                        "event": "fid_real_statistics_progress",
                        "samples_complete": seen,
                        "samples_total": len(dataset),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if suite.fid_real_samples != len(dataset):
        raise RuntimeError(
            f"FID accumulated {suite.fid_real_samples} images, expected {len(dataset)}"
        )
    elapsed = time.perf_counter() - start
    data_folder = (
        data_root / "Imagenet32_train"
        if (data_root / "Imagenet32_train").is_dir()
        else data_root
    )
    metadata = {
        "dataset": "imagenet32",
        "split": "train",
        "samples": len(dataset),
        "quantization": args.image_quantization,
        "torch_version": torch.__version__,
        "torchmetrics_version": torchmetrics.__version__,
        "feature_extractor": "torchmetrics_fid_inception_v3_2048",
        "data_root": str(data_folder),
        "dataset_manifest": dataset_manifest(data_folder),
        "wall_seconds": elapsed,
    }
    output = Path(args.output).expanduser().resolve()
    suite.save_fid_real_statistics(output, metadata=metadata)
    print(
        json.dumps(
            {
                "event": "fid_real_statistics_complete",
                "output": str(output),
                **metadata,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
