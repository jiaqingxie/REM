import json
import tempfile
import unittest
from pathlib import Path

from experiments.aggregate_results import aggregate, discover, load_rows
from experiments.check_claim_gates import paired_check, pareto_check


class ReportingTests(unittest.TestCase):
    def test_load_and_aggregate_artifact_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, fid in ((0, 5.0), (1, 4.0), (2, 6.0)):
                run = root / f"run{seed}"
                run.mkdir()
                (run / "resolved_config.json").write_text(
                    json.dumps({"training_config": {"mobility": "diagonal"}})
                )
                record = {
                    "step": 10,
                    "split": "test",
                    "seed": seed,
                    "metrics": {"fid": fid, "sampling_wall_seconds": 10.0},
                }
                (run / "metrics.jsonl").write_text(json.dumps(record) + "\n")
            paths = discover([directory])
            rows = [row for row in load_rows(paths) if row["split"] == "test"]
            result = aggregate(
                rows,
                group_fields=["config.training_config.mobility", "step"],
                metric_allowlist={"fid"},
                confidence=0.95,
                n_resamples=100,
            )
            self.assertEqual(result[0]["metrics"]["fid"]["count"], 3)
            self.assertEqual(result[0]["metrics"]["fid"]["mean"], 5.0)

    def test_paired_and_pareto_gates(self):
        groups = [
            {
                "group": {"variant": "em", "steps": 10},
                "metrics": {
                    "error": {"mean": 1.0},
                    "fid": {"mean": 5.0},
                    "sampling_wall_seconds": {"mean": 10.0},
                },
            },
            {
                "group": {"variant": "rem", "steps": 10},
                "metrics": {
                    "error": {"mean": 0.7},
                    "fid": {"mean": 4.9},
                    "sampling_wall_seconds": {"mean": 7.0},
                },
            },
        ]
        paired = paired_check(
            {
                "name": "error",
                "comparison": "relative_improvement",
                "metric": "error",
                "direction": "lower",
                "candidate": {"variant": "rem"},
                "baseline": {"variant": "em"},
                "match_on": ["steps"],
                "threshold": 0.25,
            },
            groups,
        )
        self.assertTrue(paired["passed"])
        pareto = pareto_check(
            {
                "name": "wall",
                "comparison": "matched_fid_wall_reduction",
                "candidate": {"variant": "rem"},
                "baseline": {"variant": "em"},
                "target_fid": 5.0,
                "threshold": 0.25,
            },
            groups,
        )
        self.assertTrue(pareto["passed"])


if __name__ == "__main__":
    unittest.main()
