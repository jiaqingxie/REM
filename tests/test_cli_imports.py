import importlib
import unittest


class CliImportTests(unittest.TestCase):
    def test_rem_experiment_modules_import_without_optional_runtime_work(self):
        modules = [
            "experiments.aggregate_results",
            "experiments.check_claim_gates",
            "experiments.run_ablation_suite",
            "experiments.run_evaluation_suite",
            "experiments.cifar10.evaluate_rem",
            "experiments.cifar10.match_capacity",
            "experiments.cifar10.train_rem",
            "experiments.inverse.run_cifar_inverse",
            "experiments.synthetic.run_representability",
            "experiments.synthetic.run_stationarity",
            "experiments.toy2d.train_rem_2d",
        ]
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
