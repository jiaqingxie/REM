"""Image quality, coverage, and feature-manifold metrics for REM."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FID_REAL_STAT_NAMES = (
    "real_features_sum",
    "real_features_cov_sum",
    "real_features_num_samples",
)


def to_uint8(images: Tensor, *, quantization: str = "round") -> Tensor:
    """Convert normalized images to uint8 using an explicit protocol.

    ``round`` is the historical REM behavior. ``floor`` matches the official
    Energy Matching evaluation, whose ``Tensor.byte()`` conversion truncates
    values after rescaling to ``[0, 255]``.
    """

    if images.dtype == torch.uint8:
        return images
    if quantization not in {"round", "floor"}:
        raise ValueError("quantization must be 'round' or 'floor'")
    scaled = (images.clamp(-1, 1) + 1) * 127.5
    if quantization == "round":
        scaled = scaled.round()
    return scaled.to(torch.uint8)


def export_fid_real_statistics(
    fid: nn.Module,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a portable snapshot of TorchMetrics' accumulated real stats."""

    missing = [name for name in FID_REAL_STAT_NAMES if not hasattr(fid, name)]
    if missing:
        raise RuntimeError(f"FID implementation lacks real-stat states: {missing}")
    count = int(getattr(fid, "real_features_num_samples").item())
    if count < 2:
        raise ValueError("at least two real samples are required for FID statistics")
    return {
        "format": "rem_fid_real_statistics",
        "format_version": 1,
        "feature_dimension": int(getattr(fid, "real_features_sum").numel()),
        "states": {
            name: getattr(fid, name).detach().cpu().clone()
            for name in FID_REAL_STAT_NAMES
        },
        "metadata": dict(metadata or {}),
    }


def import_fid_real_statistics(fid: nn.Module, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Restore cached real statistics into a compatible TorchMetrics FID."""

    if payload.get("format") != "rem_fid_real_statistics":
        raise ValueError("unrecognized FID real-statistics format")
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError("unsupported FID real-statistics format version")
    states = payload.get("states")
    if not isinstance(states, Mapping):
        raise ValueError("FID real-statistics payload has no states mapping")
    feature_dimension = int(payload.get("feature_dimension", -1))
    if feature_dimension != int(getattr(fid, "real_features_sum").numel()):
        raise ValueError("cached FID feature dimension is incompatible")
    for name in FID_REAL_STAT_NAMES:
        if name not in states or not hasattr(fid, name):
            raise ValueError(f"FID real-statistics payload is missing {name}")
        source = states[name]
        target = getattr(fid, name)
        if not isinstance(source, Tensor) or source.shape != target.shape:
            raise ValueError(
                f"incompatible FID state {name}: expected {tuple(target.shape)}"
            )
        target.copy_(source.to(device=target.device, dtype=target.dtype))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("FID real-statistics metadata must be a mapping")
    return dict(metadata)


def save_fid_real_statistics(
    path: str | Path,
    fid: nn.Module,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save accumulated real statistics without real features."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = export_fid_real_statistics(fid, metadata=metadata)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_fid_real_statistics(path: str | Path, fid: nn.Module) -> dict[str, Any]:
    """Load cached real statistics and return their provenance metadata."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("FID real-statistics file must contain a mapping")
    return import_fid_real_statistics(fid, payload)


class InceptionFeatures(nn.Module):
    """Torchvision Inception-v3 pool features for coverage diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import Inception_V3_Weights, inception_v3
        except ImportError as error:
            raise RuntimeError("image metrics require torchvision") from error
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=True)
        model.fc = nn.Identity()
        self.model = model.eval()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        images = images.float()
        if images.max() > 2:
            images = images / 255
        elif images.min() < 0:
            images = (images + 1) / 2
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = (images - self.mean) / self.std
        output = self.model(images)
        if hasattr(output, "logits"):
            output = output.logits
        return output.reshape(output.shape[0], -1)


def _knn_radius(features: Tensor, k: int, chunk_size: int) -> Tensor:
    if not 1 <= k < features.shape[0]:
        raise ValueError("k must be smaller than the feature count")
    radii = []
    for start in range(0, features.shape[0], chunk_size):
        query = features[start : start + chunk_size]
        distances = torch.cdist(query, features)
        local_indices = torch.arange(query.shape[0], device=features.device)
        global_indices = torch.arange(start, start + query.shape[0], device=features.device)
        distances[local_indices, global_indices] = float("inf")
        radii.append(distances.topk(k, largest=False).values[:, -1])
    return torch.cat(radii)


def _coverage(
    queries: Tensor,
    reference: Tensor,
    radii: Tensor,
    chunk_size: int,
) -> Tensor:
    covered = []
    for query_start in range(0, queries.shape[0], chunk_size):
        query = queries[query_start : query_start + chunk_size]
        query_covered = torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
        for reference_start in range(0, reference.shape[0], chunk_size):
            reference_chunk = reference[reference_start : reference_start + chunk_size]
            radius_chunk = radii[reference_start : reference_start + chunk_size]
            distance = torch.cdist(query, reference_chunk)
            query_covered |= (distance <= radius_chunk.unsqueeze(0)).any(dim=1)
        covered.append(query_covered)
    return torch.cat(covered).float().mean()


def improved_precision_recall(
    real_features: Tensor,
    fake_features: Tensor,
    *,
    k: int = 3,
    max_features: int = 10_000,
    chunk_size: int = 512,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Improved precision/recall using k-NN feature manifolds.

    A deterministic subset bounds the quadratic cost for 50k evaluations.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def subset(features: Tensor) -> Tensor:
        features = features.detach()
        if features.shape[0] <= max_features:
            return features
        indices = torch.randperm(features.shape[0], generator=generator)[:max_features]
        return features[indices.to(features.device)]

    real = subset(real_features).float()
    fake = subset(fake_features).float()
    real_radius = _knn_radius(real, k, chunk_size)
    fake_radius = _knn_radius(fake, k, chunk_size)
    precision = _coverage(fake, real, real_radius, chunk_size)
    recall = _coverage(real, fake, fake_radius, chunk_size)
    return precision, recall


class ImageMetricSuite:
    """Accumulate FID, KID, and optional precision/recall features."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        collect_precision_recall: bool = True,
        kid_subset_size: int = 1000,
        compute_kid: bool = True,
        quantization: str = "round",
    ) -> None:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance
        except ImportError as error:
            raise RuntimeError("FID/KID evaluation requires torchmetrics[image]") from error
        self.device = torch.device(device)
        if quantization not in {"round", "floor"}:
            raise ValueError("quantization must be 'round' or 'floor'")
        self.quantization = quantization
        self.fid = FrechetInceptionDistance(normalize=False).to(self.device)
        self.kid = (
            KernelInceptionDistance(
                subset_size=kid_subset_size, normalize=False
            ).to(self.device)
            if compute_kid
            else None
        )
        self.feature_model: Optional[nn.Module] = (
            InceptionFeatures().to(self.device).eval()
            if collect_precision_recall
            else None
        )
        self.real_features: list[Tensor] = []
        self.fake_features: list[Tensor] = []

    @torch.no_grad()
    def update(
        self,
        images: Tensor,
        *,
        real: bool,
        update_fid: bool = True,
        update_kid: bool = True,
    ) -> None:
        images = images.to(self.device)
        uint8 = to_uint8(images, quantization=self.quantization)
        if update_fid:
            self.fid.update(uint8, real=real)
        if update_kid and self.kid is not None:
            self.kid.update(uint8, real=real)
        if self.feature_model is not None:
            features = self.feature_model(images).cpu()
            (self.real_features if real else self.fake_features).append(features)

    @property
    def fid_real_samples(self) -> int:
        return int(self.fid.real_features_num_samples.item())

    def save_fid_real_statistics(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        save_fid_real_statistics(path, self.fid, metadata=metadata)

    def load_fid_real_statistics(self, path: str | Path) -> dict[str, Any]:
        return load_fid_real_statistics(path, self.fid)

    @torch.no_grad()
    def compute(self) -> dict[str, float]:
        metrics = {"fid": float(self.fid.compute())}
        if self.kid is not None:
            kid_mean, kid_std = self.kid.compute()
            metrics.update(
                {
                    "kid_mean": float(kid_mean),
                    "kid_std": float(kid_std),
                }
            )
        if self.feature_model is not None:
            real = torch.cat(self.real_features)
            fake = torch.cat(self.fake_features)
            precision, recall = improved_precision_recall(real, fake)
            metrics["precision"] = float(precision)
            metrics["recall"] = float(recall)
        return metrics
