import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from experiments.cifar10.aggregate_phase_gate import main


class AggregatePhaseGateTest(unittest.TestCase):
    def test_fid_only_runs_are_aggregated_without_kid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in range(3):
                for name, cutoff, fid in (
                    ("identity", 0.0, 7.4 + 0.01 * seed),
                    ("gated", 1.0, 7.2 + 0.01 * seed),
                ):
                    run = root / f"{name}_{seed}"
                    run.mkdir()
                    (run / "resolved_config.json").write_text(
                        json.dumps(
                            {
                                "dataset": "imagenet32",
                                "samples": 50_000,
                                "seed": seed,
                                "mobility_active_until": cutoff,
                            }
                        )
                    )
                    (run / "summary.json").write_text(
                        json.dumps(
                            {
                                "status": "complete",
                                "metrics": {
                                    "steps=250/fid": {"mean": fid},
                                },
                            }
                        )
                    )
            output = root / "aggregate.json"
            with patch(
                "sys.argv",
                [
                    "aggregate_phase_gate",
                    "--roots",
                    str(root),
                    "--output",
                    str(output),
                ],
            ), redirect_stdout(StringIO()):
                main()

            aggregate = json.loads(output.read_text())
            self.assertEqual(len(aggregate["records"]), 6)
            self.assertEqual(len(aggregate["paired"]), 1)
            self.assertEqual(aggregate["paired"][0]["n"], 3)
            self.assertAlmostEqual(aggregate["paired"][0]["fid_delta_mean"], -0.2)
            self.assertIsNone(aggregate["paired"][0]["kid_delta_mean"])
            self.assertTrue(
                all(item["kid_mean"] is None for item in aggregate["aggregates"])
            )


if __name__ == "__main__":
    unittest.main()
