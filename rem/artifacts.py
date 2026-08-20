"""Reproducible experiment artifact and metric logging."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .metrics import bootstrap_confidence_interval


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_identity(cwd: str | Path | None = None) -> dict[str, Any] | None:
    """Hash tracked changes and untracked source files used by dirty runs.

    A commit hash alone is insufficient when formal experiments run from a
    shared, modified checkout.  We hash the tracked binary diff and each
    reasonably sized untracked source/config file, while excluding outputs,
    dependency trees, and binary assets that would make provenance capture
    itself expensive.
    """

    if cwd is None:
        return None
    root = Path(cwd)
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "."],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                ".",
                ":(exclude).gpu-deps/**",
                ":(exclude)outputs/**",
            ],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None

    allowed_suffixes = {
        ".bib",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".tex",
        ".toml",
        ".yaml",
        ".yml",
    }
    ignored_roots = {".gpu-deps", "outputs"}
    untracked_sources: dict[str, str] = {}
    for raw_path in untracked.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        if relative.parts and relative.parts[0] in ignored_roots:
            continue
        path = root / relative
        try:
            if (
                not path.is_file()
                or path.suffix.lower() not in allowed_suffixes
                or path.stat().st_size > 10 * 1024 * 1024
            ):
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            untracked_sources[str(relative)] = digest.hexdigest()
        except OSError:
            continue
    return {
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_bytes": len(diff),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "dirty": bool(status),
        "untracked_source_sha256": untracked_sources,
    }


class RunArtifacts:
    """Writes one self-contained experiment directory."""

    def __init__(
        self,
        output_root: str | Path,
        experiment_id: str,
        config: Mapping[str, Any] | Any,
        *,
        repository_root: str | Path | None = None,
        resume_existing: bool = False,
    ) -> None:
        self.path = Path(output_root) / experiment_id
        path_existed = self.path.exists()
        self.path.mkdir(parents=True, exist_ok=resume_existing)
        for child in ("checkpoints", "samples", "figures"):
            (self.path / child).mkdir(exist_ok=resume_existing)
        config_value = asdict(config) if is_dataclass(config) else config
        environment = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(repository_root),
            "git_worktree_identity": git_worktree_identity(repository_root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "pid": os.getpid(),
        }
        if resume_existing and path_existed:
            self._append_jsonl(
                "resume_history.jsonl",
                {"config": config_value, "environment": environment},
            )
        else:
            self._write_json("resolved_config.json", config_value)
            self._write_json("environment.json", environment)
        self.metrics_path = self.path / "metrics.jsonl"

    def _write_json(self, name: str, value: Any) -> None:
        with (self.path / name).open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _append_jsonl(self, name: str, value: Any) -> None:
        with (self.path / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(value), sort_keys=True) + "\n")

    def log(self, step: int, metrics: Mapping[str, Any], **context: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            **context,
            "metrics": metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")

    def summarize(
        self,
        metrics: Mapping[str, list[float]],
        *,
        status: str,
        failure_reason: str | None = None,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "status": status,
            "failure_reason": failure_reason,
            "metrics": {},
        }
        for name, values in metrics.items():
            array = np.asarray(values, dtype=float)
            lower, upper = bootstrap_confidence_interval(
                array, confidence=confidence
            )
            summary["metrics"][name] = {
                "values": array.tolist(),
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "ci": [lower, upper],
            }
        self._write_json("summary.json", summary)
        return summary
