import json
import tempfile
import unittest
from pathlib import Path

from rem.artifacts import RunArtifacts


class ArtifactTests(unittest.TestCase):
    def test_run_artifacts_are_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(directory, "run", {"seed": 0})
            artifacts.log(1, {"loss": 2.0}, split="train")
            summary = artifacts.summarize({"score": [1.0, 2.0, 3.0]}, status="complete")
            root = Path(directory) / "run"
            self.assertTrue((root / "resolved_config.json").exists())
            self.assertTrue((root / "environment.json").exists())
            self.assertTrue((root / "metrics.jsonl").exists())
            loaded = json.loads((root / "summary.json").read_text())
            self.assertEqual(loaded["status"], "complete")
            self.assertEqual(summary["metrics"]["score"]["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
