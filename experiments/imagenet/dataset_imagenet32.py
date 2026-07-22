# dataset_imagenet32.py
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

def unpickle(file):
    """Helper to load a Python dict from the downsampled ImageNet32 files."""
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo)
    return data_dict

class ImageNet32Dataset(Dataset):
    """
    Loads the 10 'train_data_batch_#' files from ImageNet32 into
    a single in-memory array of shape (N, 3, 32, 32), storing pixel
    values in [0,1].
    
    If you need transforms (like random crops/flips or normalizing to [-1,1]),
    you can pass in a `transform` (typical PyTorch transforms) to apply in __getitem__.
    """
    def __init__(self, split="train", transform=None, root=None):
        super().__init__()

        if split != "train":
            raise ValueError(
                f"[ImageNet32Dataset] Only 'train' split is available. Received: '{split}'"
            )

        self.transform = transform

        # Resolve the directory that contains the ImageNet32 tar batches.
        # Priority: explicit root argument -> environment variable -> default repo path.
        env_root = os.environ.get("IMAGENET32_PATH")
        base_dir = Path(root or env_root or Path(__file__).resolve().parent / "data")
        base_dir = base_dir.expanduser().resolve()

        # Accept either a directory that already points at the train folder,
        # or a parent directory containing Imagenet32_train/.
        train_dir = base_dir / "Imagenet32_train"
        if train_dir.is_dir():
            data_folder = train_dir
        elif base_dir.is_dir():
            data_folder = base_dir
        else:
            raise FileNotFoundError(
                f"[ImageNet32Dataset] Could not locate ImageNet32 data directory. "
                f"Tried: '{base_dir}' and '{train_dir}'."
            )

        all_images = []
        all_labels = []

        # Load all 10 train_data_batch_{i}
        for i in range(1, 11):
            batch_file = data_folder / f"train_data_batch_{i}"
            if not batch_file.exists():
                raise FileNotFoundError(
                    f"[ImageNet32Dataset] Missing batch file: {batch_file}"
                )
            print(f"[ImageNet32Dataset] Loading: {batch_file}")
            d = unpickle(batch_file)

            x = d['data']          # shape: (N, 3072)
            labels = d['labels']   # list of length N

            # Convert to float and scale to [0..1]
            x = x.astype(np.float32) / 255.0

            # Reshape (N, 3072) -> (N, 3, 32, 32)
            N = x.shape[0]
            x = x.reshape(N, 3, 32, 32)

            # Shift labels from [1..1000] -> [0..999]
            labels = np.array([lab - 1 for lab in labels], dtype=np.int64)

            all_images.append(x)
            all_labels.append(labels)

        # Concatenate across all 10 files
        self.images = np.concatenate(all_images, axis=0)  # (N_total, 3, 32, 32)
        self.labels = np.concatenate(all_labels, axis=0)  # (N_total,)

        print(f"[ImageNet32Dataset] Loaded total of {len(self.images)} images.")

    def __getitem__(self, index):
        # Grab a single image (still a NumPy array, shape (3, 32, 32))
        img = self.images[index]
        label = self.labels[index]

        # Transpose to (32, 32, 3) so transforms like ToTensor() or RandomHorizontalFlip() will
        # behave as with standard PIL/ndarray input. If your transforms can handle channel-first
        # Tensors directly, you can skip this step. But most torchvision transforms expect HWC
        img = np.transpose(img, (1, 2, 0))  # shape (32, 32, 3)

        if self.transform:
            # e.g. transforms.ToTensor() -> returns a FloatTensor (C, H, W)
            # e.g. transforms.RandomHorizontalFlip() -> expects HWC or PIL
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)
