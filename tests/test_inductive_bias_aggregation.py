import unittest

from experiments.synthetic.aggregate_inductive_bias_comparison import (
    EQUILIBRIUM_METRICS,
    REPRESENTATION_METRICS,
    VARIANTS,
    aggregate,
)


class InductiveBiasAggregationTests(unittest.TestCase):
    def _record(self, seed: int, variant: str) -> dict:
        parameters = {
            "identity": 0,
            "rem-full": 100,
            "pem-style-residual": 98,
            "pem-helmholtz-residual": 98,
            "learned-mirror": 102,
            "learned-preconditioner": 3,
            "oracle-full": 0,
        }[variant]
        representation = {metric: 0.1 + seed for metric in REPRESENTATION_METRICS}
        representation["trainable_parameters"] = parameters
        equilibrium = {metric: 0.2 + seed for metric in EQUILIBRIUM_METRICS}
        return {
            "seed": seed,
            "variant": variant,
            "representation": representation,
            "equilibrium": equilibrium,
            "equilibrium_correction": variant
            not in {"pem-style-residual", "pem-helmholtz-residual"},
        }

    def test_complete_grid_and_capacity_check(self) -> None:
        records = [self._record(seed, variant) for seed in (0, 1) for variant in VARIANTS]
        payload = aggregate(records, [0, 1])
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["record_count"], 14)
        self.assertTrue(payload["capacity_check"]["parameter_matched_within_5_percent"])

    def test_missing_record_is_rejected(self) -> None:
        records = [self._record(0, variant) for variant in VARIANTS[:-1]]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            aggregate(records, [0])


if __name__ == "__main__":
    unittest.main()
