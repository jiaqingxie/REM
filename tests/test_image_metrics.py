import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from rem.image_metrics import (
    export_fid_real_statistics,
    import_fid_real_statistics,
    load_fid_real_statistics,
    save_fid_real_statistics,
    to_uint8,
)


class DummyFID(nn.Module):
    def __init__(self, dimension: int = 4) -> None:
        super().__init__()
        self.register_buffer("real_features_sum", torch.zeros(dimension, dtype=torch.float64))
        self.register_buffer(
            "real_features_cov_sum",
            torch.zeros(dimension, dimension, dtype=torch.float64),
        )
        self.register_buffer("real_features_num_samples", torch.zeros((), dtype=torch.int64))


class ImageMetricProtocolTest(unittest.TestCase):
    def test_explicit_round_and_floor_quantization(self):
        images = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
        self.assertEqual(to_uint8(images, quantization="round").tolist(), [[[[0, 128, 255]]]])
        self.assertEqual(to_uint8(images, quantization="floor").tolist(), [[[[0, 127, 255]]]])
        with self.assertRaisesRegex(ValueError, "quantization"):
            to_uint8(images, quantization="unknown")

    def test_fid_real_statistics_round_trip_with_metadata(self):
        source = DummyFID()
        source.real_features_sum.copy_(torch.arange(4, dtype=torch.float64))
        source.real_features_cov_sum.copy_(
            torch.arange(16, dtype=torch.float64).reshape(4, 4)
        )
        source.real_features_num_samples.fill_(1_281_167)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real_stats.pt"
            save_fid_real_statistics(
                path,
                source,
                metadata={"dataset": "imagenet32", "quantization": "floor"},
            )
            target = DummyFID()
            metadata = load_fid_real_statistics(path, target)

        self.assertEqual(metadata["dataset"], "imagenet32")
        self.assertEqual(metadata["quantization"], "floor")
        self.assertTrue(torch.equal(target.real_features_sum, source.real_features_sum))
        self.assertTrue(
            torch.equal(target.real_features_cov_sum, source.real_features_cov_sum)
        )
        self.assertEqual(int(target.real_features_num_samples), 1_281_167)

    def test_export_rejects_empty_statistics(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            export_fid_real_statistics(DummyFID())

    def test_import_rejects_incompatible_feature_dimension(self):
        source = DummyFID(dimension=4)
        source.real_features_num_samples.fill_(2)
        payload = export_fid_real_statistics(source)
        target = DummyFID(dimension=3)
        with self.assertRaisesRegex(ValueError, "feature dimension"):
            import_fid_real_statistics(target, payload)


if __name__ == "__main__":
    unittest.main()
