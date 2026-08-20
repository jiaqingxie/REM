"""Hard-gate the controlled Euclidean-versus-metric-weighted REM ablation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


CONFIG_KEYS = (
    "train_steps",
    "batch_size",
    "test_size",
    "lr",
    "hidden_dim",
    "depth",
    "log_bound",
    "geometry_weight",
    "step_size",
    "physical_burn",
    "physical_draw",
    "save_interval",
    "chains",
    "component_variance",
)
METRIC_TEST_KEYS = ("relative_velocity_mse", "relative_metric_mse")
CHAIN_KEYS = ("latent_mmd", "median_first_passage_steps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--euclidean-dir", required=True)
    parser.add_argument("--metric-weighted-dir", required=True)
    parser.add_argument("--expected-seeds", default="0,1,2,3,4")
    parser.add_argument("--curvature", type=float, default=1.2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latex-output", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _summary(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("metric values must be finite and nonempty")
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _load_run(
    directory: Path,
    *,
    curvature: float,
    expected_seeds: list[int],
) -> tuple[dict, dict[str, dict[str, object]]]:
    config = _read_json(directory / "resolved_config.json")
    metric_test: dict[int, dict] = {}
    with (directory / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record.get("phase") == "metric-test"
                and record.get("variant") == "rem-full"
                and float(record.get("curvature")) == curvature
                and int(record.get("seed")) in expected_seeds
            ):
                metric_test[int(record["seed"])] = record["metrics"]
    if sorted(metric_test) != expected_seeds:
        raise ValueError(
            f"{directory}: metric-test seeds {sorted(metric_test)} != {expected_seeds}"
        )

    chain: dict[int, dict] = {}
    for seed in expected_seeds:
        path = (
            directory
            / "records"
            / f"main_curvature{curvature:g}_rem-full_seed{seed}_dt{float(config['step_size']):g}.json"
        )
        record = _read_json(path)
        if (
            record.get("phase") != "main"
            or record.get("variant") != "rem-full"
            or int(record.get("seed")) != seed
            or float(record.get("curvature")) != curvature
        ):
            raise ValueError(f"record contract mismatch: {path}")
        chain[seed] = record["metrics"]

    metrics: dict[str, dict[str, object]] = {}
    for key in METRIC_TEST_KEYS:
        metrics[key] = _summary([float(metric_test[seed][key]) for seed in expected_seeds])
    for key in CHAIN_KEYS:
        metrics[key] = _summary([float(chain[seed][key]) for seed in expected_seeds])
    return config, metrics


def _fmt(metric: dict[str, object], *, scale: float = 1.0, digits: int = 4) -> str:
    mean = float(metric["mean"]) * scale
    std = float(metric["std"]) * scale
    return f"{mean:.{digits}f}\\pm{std:.{digits}f}"


def _latex(euclidean: dict, metric_weighted: dict) -> str:
    def row(label: str, metrics: dict) -> str:
        return (
            f"{label} & ${_fmt(metrics['relative_velocity_mse'], digits=5)}$ "
            f"& ${_fmt(metrics['relative_metric_mse'], digits=5)}$ "
            f"& ${_fmt(metrics['latent_mmd'], scale=1000, digits=2)}$ "
            f"& ${_fmt(metrics['median_first_passage_steps'], digits=0)}$ \\\\\n"
        )

    return """\\begin{table}[h]
\\centering
\\caption{Loss-weighting ablation at the strongest warp (five independent fits, mean $\\pm$ s.d.). All optimization, architecture, held-out, and chain settings are identical.}
\\label{tab:loss-weighting}
\\small
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{lrrrr}
\\toprule
Training residual & rel. velocity MSE & rel. metric MSE & latent MMD $\\times10^3$ & first passage \\\\
\\midrule
""" + row("Euclidean (diagnostic)", euclidean) + row(
        "Inverse-mobility (Eq.~\\eqref{eq:objective})", metric_weighted
    ) + """\\bottomrule
\\end{tabular}
}
\\end{table}
"""


def main() -> None:
    args = parse_args()
    expected_seeds = sorted(
        int(value) for value in args.expected_seeds.split(",") if value.strip()
    )
    euclidean_config, euclidean = _load_run(
        Path(args.euclidean_dir),
        curvature=args.curvature,
        expected_seeds=expected_seeds,
    )
    weighted_config, metric_weighted = _load_run(
        Path(args.metric_weighted_dir),
        curvature=args.curvature,
        expected_seeds=expected_seeds,
    )
    mismatches = {
        key: {"euclidean": euclidean_config.get(key), "metric_weighted": weighted_config.get(key)}
        for key in CONFIG_KEYS
        if euclidean_config.get(key) != weighted_config.get(key)
    }
    if mismatches:
        raise ValueError(f"matched-protocol config mismatch: {mismatches}")
    if bool(euclidean_config.get("metric_weighted", False)):
        raise ValueError("Euclidean run unexpectedly declares metric weighting")
    if not bool(weighted_config.get("metric_weighted", False)):
        raise ValueError("weighted run does not declare metric weighting")

    result = {
        "complete": True,
        "curvature": args.curvature,
        "expected_seeds": expected_seeds,
        "matched_config": {key: euclidean_config.get(key) for key in CONFIG_KEYS},
        "objectives": {
            "euclidean": euclidean,
            "metric_weighted": metric_weighted,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latex_output = Path(args.latex_output)
    latex_output.parent.mkdir(parents=True, exist_ok=True)
    latex_output.write_text(_latex(euclidean, metric_weighted), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
