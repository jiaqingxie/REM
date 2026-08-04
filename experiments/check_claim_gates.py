"""Evaluate preregistered REM claim gates from an aggregate JSON file.

The checker deliberately exits nonzero when any required comparison is
missing or fails. This prevents an incomplete experiment directory from
silently being interpreted as a successful result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate", help="JSON produced by aggregate_results.py")
    parser.add_argument("spec", help="Claim-gate JSON specification")
    parser.add_argument("--report", default="")
    return parser.parse_args()


def group_matches(group: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(str(group.get(key)) == str(value) for key, value in expected.items())


def select(groups: list[dict[str, Any]], expected: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in groups if group_matches(entry["group"], expected)]


def improvement(candidate: float, baseline: float, direction: str) -> float:
    if direction == "lower":
        return baseline - candidate
    if direction == "higher":
        return candidate - baseline
    raise ValueError("direction must be lower or higher")


def paired_check(check: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows = select(groups, check["candidate"])
    baseline_rows = select(groups, check["baseline"])
    match_on = check.get("match_on", [])
    metric = check["metric"]
    pairs = []
    for candidate in candidate_rows:
        for baseline in baseline_rows:
            if all(candidate["group"].get(key) == baseline["group"].get(key) for key in match_on):
                if metric not in candidate["metrics"] or metric not in baseline["metrics"]:
                    continue
                candidate_summary = candidate["metrics"][metric]
                baseline_summary = baseline["metrics"][metric]
                candidate_value = candidate_summary["mean"]
                baseline_value = baseline_summary["mean"]
                absolute = improvement(candidate_value, baseline_value, check["direction"])
                effect = (
                    absolute / max(abs(baseline_value), 1e-12)
                    if check["comparison"] == "relative_improvement"
                    else absolute
                )
                if check["direction"] == "lower":
                    conservative_absolute = (
                        baseline_summary.get("ci", [baseline_value, baseline_value])[0]
                        - candidate_summary.get("ci", [candidate_value, candidate_value])[1]
                    )
                else:
                    conservative_absolute = (
                        candidate_summary.get("ci", [candidate_value, candidate_value])[0]
                        - baseline_summary.get("ci", [baseline_value, baseline_value])[1]
                    )
                conservative_effect = (
                    conservative_absolute / max(abs(baseline_value), 1e-12)
                    if check["comparison"] == "relative_improvement"
                    else conservative_absolute
                )
                pairs.append(
                    {
                        "match": {key: candidate["group"].get(key) for key in match_on},
                        "candidate": candidate_value,
                        "baseline": baseline_value,
                        "effect": effect,
                        "conservative_ci_effect": conservative_effect,
                        "candidate_count": candidate_summary.get("count", 1),
                        "baseline_count": baseline_summary.get("count", 1),
                    }
                )
    threshold = float(check["threshold"])
    policy = check.get("policy", "all")
    require_ci = bool(check.get("require_ci", False))
    min_count = int(check.get("min_count", 1))
    passed_flags = [
        pair["candidate_count"] >= min_count
        and pair["baseline_count"] >= min_count
        and pair["conservative_ci_effect" if require_ci else "effect"] >= threshold
        for pair in pairs
    ]
    passed = bool(pairs) and (all(passed_flags) if policy == "all" else any(passed_flags))
    return {
        "name": check["name"],
        "passed": passed,
        "threshold": threshold,
        "require_ci": require_ci,
        "min_count": min_count,
        "pairs": pairs,
    }


def _curve(rows: list[dict[str, Any]], fid_metric: str, time_metric: str):
    points = []
    for row in rows:
        metrics = row["metrics"]
        if fid_metric in metrics and time_metric in metrics:
            points.append(
                (
                    metrics[fid_metric]["mean"],
                    metrics[time_metric]["mean"],
                    row["group"],
                    metrics[fid_metric],
                    metrics[time_metric],
                )
            )
    return points


def pareto_check(check: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = _curve(
        select(groups, check["candidate"]), check.get("fid_metric", "fid"), check.get("time_metric", "sampling_wall_seconds")
    )
    baseline = _curve(
        select(groups, check["baseline"]), check.get("fid_metric", "fid"), check.get("time_metric", "sampling_wall_seconds")
    )
    comparison = check["comparison"]
    if comparison == "matched_fid_wall_reduction":
        target = float(check["target_fid"])
        candidate_rows = [point for point in candidate if point[0] <= target]
        baseline_rows = [point for point in baseline if point[0] <= target]
        candidate_row = min(candidate_rows, key=lambda point: point[1]) if candidate_rows else None
        baseline_row = min(baseline_rows, key=lambda point: point[1]) if baseline_rows else None
        candidate_value = candidate_row[1] if candidate_row else None
        baseline_value = baseline_row[1] if baseline_row else None
        effect = (
            (baseline_value - candidate_value) / max(baseline_value, 1e-12)
            if candidate_value is not None and baseline_value is not None
            else None
        )
        conservative_effect = (
            (
                baseline_row[4].get("ci", [baseline_value, baseline_value])[0]
                - candidate_row[4].get("ci", [candidate_value, candidate_value])[1]
            )
            / max(baseline_value, 1e-12)
            if candidate_row is not None and baseline_row is not None
            else None
        )
    elif comparison == "matched_wall_fid_improvement":
        target = float(check["target_wall_seconds"])
        candidate_rows = [point for point in candidate if point[1] <= target]
        baseline_rows = [point for point in baseline if point[1] <= target]
        candidate_row = min(candidate_rows, key=lambda point: point[0]) if candidate_rows else None
        baseline_row = min(baseline_rows, key=lambda point: point[0]) if baseline_rows else None
        candidate_value = candidate_row[0] if candidate_row else None
        baseline_value = baseline_row[0] if baseline_row else None
        effect = (
            baseline_value - candidate_value
            if candidate_value is not None and baseline_value is not None
            else None
        )
        conservative_effect = (
            baseline_row[3].get("ci", [baseline_value, baseline_value])[0]
            - candidate_row[3].get("ci", [candidate_value, candidate_value])[1]
            if candidate_row is not None and baseline_row is not None
            else None
        )
    else:
        raise ValueError(f"unsupported Pareto comparison: {comparison}")
    threshold = float(check["threshold"])
    min_count = int(check.get("min_count", 1))
    require_ci = bool(check.get("require_ci", False))
    counts_pass = (
        candidate_row is not None
        and baseline_row is not None
        and candidate_row[3].get("count", 1) >= min_count
        and baseline_row[3].get("count", 1) >= min_count
        and candidate_row[4].get("count", 1) >= min_count
        and baseline_row[4].get("count", 1) >= min_count
    )
    return {
        "name": check["name"],
        "passed": counts_pass
        and (conservative_effect if require_ci else effect) is not None
        and (conservative_effect if require_ci else effect) >= threshold,
        "threshold": threshold,
        "require_ci": require_ci,
        "min_count": min_count,
        "candidate": candidate_value,
        "baseline": baseline_value,
        "effect": effect,
        "conservative_ci_effect": conservative_effect,
        "candidate_points": len(candidate),
        "baseline_points": len(baseline),
    }


def main() -> None:
    args = parse_args()
    aggregate = json.loads(Path(args.aggregate).read_text(encoding="utf-8"))
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    results = []
    for check in spec["checks"]:
        if check["comparison"].startswith("matched_"):
            results.append(pareto_check(check, aggregate["groups"]))
        else:
            results.append(paired_check(check, aggregate["groups"]))
    by_name = {result["name"]: result for result in results}
    requirements = spec.get(
        "requirements", [[result["name"]] for result in results]
    )
    requirement_results = [
        {
            "any_of": names,
            "passed": any(by_name.get(name, {}).get("passed", False) for name in names),
        }
        for names in requirements
    ]
    report = {
        "passed": all(result["passed"] for result in requirement_results),
        "checks": results,
        "requirements": requirement_results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
