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

    def test_repository_runs_record_dirty_worktree_identity(self):
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            RunArtifacts(
                directory,
                "run",
                {"seed": 0},
                repository_root=repository_root,
            )
            environment = json.loads(
                (Path(directory) / "run" / "environment.json").read_text()
            )
            identity = environment["git_worktree_identity"]
            self.assertEqual(len(identity["tracked_diff_sha256"]), 64)
            self.assertIsInstance(identity["untracked_source_sha256"], dict)

    def test_resume_flag_still_initializes_a_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            RunArtifacts(
                directory,
                "run",
                {"seed": 0},
                resume_existing=True,
            )
            root = Path(directory) / "run"
            self.assertTrue((root / "resolved_config.json").is_file())
            self.assertTrue((root / "environment.json").is_file())
            self.assertFalse((root / "resume_history.jsonl").exists())
            RunArtifacts(
                directory,
                "run",
                {"seed": 0},
                resume_existing=True,
            )
            self.assertTrue((root / "resume_history.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
