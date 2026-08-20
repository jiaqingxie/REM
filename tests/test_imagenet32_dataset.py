import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.imagenet.dataset_imagenet32 import ImageNet32Dataset


class ImageNet32DatasetTest(unittest.TestCase):
    def test_uint8_storage_and_float_return_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.arange(3 * 32 * 32, dtype=np.uint16)
            pixels = (pixels % 256).astype(np.uint8)[None, :]
            for batch in range(1, 11):
                with (root / f"train_data_batch_{batch}").open("wb") as handle:
                    pickle.dump({"data": pixels, "labels": [batch]}, handle)

            dataset = ImageNet32Dataset(root=root)
            image, label = dataset[0]

            self.assertEqual(dataset.images.dtype, np.uint8)
            self.assertEqual(image.shape, (32, 32, 3))
            self.assertEqual(image.dtype, np.float32)
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)
            self.assertEqual(label, 0)


if __name__ == "__main__":
    unittest.main()
