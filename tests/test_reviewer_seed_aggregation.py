import json
import tempfile
import unittest
from pathlib import Path

from experiments.cifar10.aggregate_reviewer_seed_expansion import (
    aggregate,
    collect_records,
    protocol_violations,
    reviewer_grid_completeness,
)


class ReviewerSeedAggregationTest(unittest.TestCase):
    def _run(
        self,
        root: Path,
        name: str,
        *,
        sampling_seed: int,
        training_seed: int,
        active_until: float,
        fid: float,
        variant_label: str = "",
        failure_reason: str | None = None,
    ) -> None:
        run = root / name
        run.mkdir()
        config = {
            "dataset": "cifar10",
            "samples": 50_000,
            "seed": sampling_seed,
            "mobility_training_seed": training_seed,
            "mobility_strength": 1.0,
            "mobility_active_until": active_until,
            "variant_label": variant_label,
        }
        summary = {
            "status": "complete",
            "failure_reason": failure_reason,
            "metrics": {
                "steps=325/fid": {"mean": fid},
                "steps=325/kid_mean": {"mean": fid / 1000},
            },
        }
        (run / "resolved_config.json").write_text(json.dumps(config))
        (run / "summary.json").write_text(json.dumps(summary))

    def test_training_and_sampling_seeds_remain_separate(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            for sampling_seed, identity_fid in enumerate((4.0, 5.0)):
                self._run(
                    root,
                    f"identity-{sampling_seed}",
                    sampling_seed=sampling_seed,
                    training_seed=0,
                    active_until=0.0,
                    fid=identity_fid,
                )
            for training_seed, offset in ((0, -0.2), (1, -0.1)):
                for sampling_seed, identity_fid in enumerate((4.0, 5.0)):
                    self._run(
                        root,
                        f"rem-{training_seed}-{sampling_seed}",
                        sampling_seed=sampling_seed,
                        training_seed=training_seed,
                        active_until=1.0,
                        fid=identity_fid + offset,
                    )

            payload = aggregate(collect_records([str(root)]))
            per_seed = payload["per_mobility_training_seed"]
            self.assertEqual([item["mobility_training_seed"] for item in per_seed], [0, 1])
            self.assertAlmostEqual(per_seed[0]["paired_fid_delta"]["mean"], -0.2)
            self.assertAlmostEqual(per_seed[1]["paired_fid_delta"]["mean"], -0.1)
            across = payload["across_mobility_training_seeds"][0]
            self.assertEqual(across["training_seed_count"], 2)
            self.assertAlmostEqual(
                across["mean_paired_fid_delta_across_training_seeds"]["mean"],
                -0.15,
            )

    def test_constant_preconditioner_is_not_misclassified_as_rem(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            self._run(
                root,
                "identity",
                sampling_seed=0,
                training_seed=0,
                active_until=0.0,
                fid=4.0,
            )
            self._run(
                root,
                "constant",
                sampling_seed=0,
                training_seed=0,
                active_until=1.0,
                fid=3.9,
                variant_label="cifar10_constant-gated_train0",
            )
            payload = aggregate(collect_records([str(root)]))
            self.assertEqual(payload["per_mobility_training_seed"], [])
            baseline = payload["baseline_summaries"][0]
            self.assertEqual(baseline["variant"], "learned_constant_gated")
            self.assertAlmostEqual(baseline["paired_fid_delta"]["mean"], -0.1)

    def test_duplicate_record_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            for name in ("first", "second"):
                self._run(
                    root,
                    name,
                    sampling_seed=0,
                    training_seed=0,
                    active_until=1.0,
                    fid=3.9,
                )
            with self.assertRaisesRegex(ValueError, "duplicate reviewer record key"):
                collect_records([str(root)])

    def test_complete_status_with_failure_reason_fails_execution_gate(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            self._run(
                root,
                "contradictory-summary",
                sampling_seed=0,
                training_seed=0,
                active_until=1.0,
                fid=3.9,
                failure_reason="late metric write failed",
            )
            payload = aggregate(collect_records([str(root)]))
            self.assertFalse(payload["execution_check"]["complete"])
            self.assertEqual(payload["execution_check"]["checked_record_count"], 1)
            self.assertEqual(
                payload["execution_check"]["failures"][0]["failure_reason"],
                "late metric write failed",
            )

    def test_matched_protocol_contract_detects_drift(self) -> None:
        common = {
            "batch_size": 64,
            "clamp_every_step": False,
            "divergence_correction": True,
            "energy_state_key": "energy",
            "integrator": "heun",
            "mobility_state_key": "mobility_ema",
            "mobility_strength": 1.0,
            "samples": 50_000,
            "temperature": 0.01,
            "temperature_ramp_end": 1.0,
            "temperature_schedule": "official",
            "time_cutoff": 1.0,
            "use_ema": True,
        }
        cifar = {
            **common,
            "dataset": "cifar10",
            "mobility_active_until": 1.0,
            "mobility_training_seed": 1,
            "step_counts": [325],
            "t_end": 3.25,
            "checkpoint": "/x/rem_diagonal_seed1/checkpoints/checkpoint_150000.pt",
            "training_config": {
                "energy_checkpoint": "/x/cifar10_main_training_147000.pt",
                "mode": "frozen",
                "mobility": "diagonal",
                "seed": 1,
                "total_steps": 150_000,
            },
        }
        self.assertEqual(protocol_violations(cifar, "rem_phase_gated"), [])
        cifar["step_counts"] = [324]
        self.assertRegex(
            "\n".join(protocol_violations(cifar, "rem_phase_gated")),
            "step_counts",
        )

        imagenet = {
            **common,
            "dataset": "imagenet32",
            "expected_fid_real_samples": 1_281_167,
            "fid_real_stats": "/x/imagenet32_train_1281167_floor_fid_stats.pt",
            "image_quantization": "floor",
            "kid": False,
            "mobility_active_until": 0.0,
            "step_counts": [250],
            "t_end": 2.5,
            "checkpoint": "/x/rem_diagonal_seed0/checkpoints/latest.pt",
            "training_config": {
                "energy_checkpoint": "/x/imagenet32x32_main_training_641000.pt",
                "mode": "frozen",
                "mobility": "diagonal",
                "seed": 0,
                "total_steps": 150_000,
            },
        }
        self.assertEqual(protocol_violations(imagenet, "identity"), [])
        imagenet_rem = {
            **imagenet,
            "mobility_active_until": 1.0,
            "mobility_training_seed": 1,
            "checkpoint": "/x/rem_diagonal_seed1/checkpoints/latest.pt",
            "training_config": {**imagenet["training_config"], "seed": 1},
        }
        self.assertEqual(
            protocol_violations(imagenet_rem, "rem_phase_gated"), []
        )

    def test_reviewer_grid_requires_every_declared_seed_level(self) -> None:
        empty = reviewer_grid_completeness([])
        self.assertFalse(empty["complete"])
        self.assertEqual(empty["expected_record_count"], 27)
        records = []
        for sampling_seed in (0, 1, 2):
            records.append(
                {
                    "dataset": "cifar10",
                    "variant": "identity",
                    "mobility_training_seed": None,
                    "sampling_seed": sampling_seed,
                }
            )
            records.append(
                {
                    "dataset": "cifar10",
                    "variant": "learned_constant_gated",
                    "mobility_training_seed": 0,
                    "sampling_seed": sampling_seed,
                }
            )
            for training_seed in (0, 1, 2):
                records.append(
                    {
                        "dataset": "cifar10",
                        "variant": "rem_phase_gated",
                        "mobility_training_seed": training_seed,
                        "sampling_seed": sampling_seed,
                    }
                )
        for sampling_seed in (0, 7, 42):
            records.append(
                {
                    "dataset": "imagenet32",
                    "variant": "identity",
                    "mobility_training_seed": None,
                    "sampling_seed": sampling_seed,
                }
            )
            for training_seed in (0, 1, 2):
                records.append(
                    {
                        "dataset": "imagenet32",
                        "variant": "rem_phase_gated",
                        "mobility_training_seed": training_seed,
                        "sampling_seed": sampling_seed,
                    }
                )
        self.assertTrue(reviewer_grid_completeness(records)["complete"])
        records.append(
            {
                "dataset": "cifar10",
                "variant": "rem_phase_gated",
                "mobility_training_seed": 3,
                "sampling_seed": 0,
            }
        )
        with_extra = reviewer_grid_completeness(records)
        self.assertFalse(with_extra["complete"])
        self.assertEqual(with_extra["unexpected_record_count"], 1)
        self.assertEqual(with_extra["unexpected"][0]["mobility_training_seed"], 3)


if __name__ == "__main__":
    unittest.main()
