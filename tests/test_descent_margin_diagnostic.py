import unittest

import torch

from experiments.cifar10.diagnose_descent_margin import summarize_batches


class DescentMarginDiagnosticTests(unittest.TestCase):
    def test_summary_uses_only_active_pairs_for_violation_rate(self):
        summary = summarize_batches(
            [torch.tensor([-2.0, 1.0, 0.0])],
            [torch.tensor([0.5, -0.25, 0.0])],
            [torch.tensor([True, True, False])],
        )
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["active_count"], 2)
        self.assertAlmostEqual(summary["inactive_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["descent_violation_rate"], 0.5)
        self.assertAlmostEqual(summary["normalized_margin_mean"], 0.125)


if __name__ == "__main__":
    unittest.main()
