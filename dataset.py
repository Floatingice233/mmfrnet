"""MedMNIST dataset loading with multi-view augmentation."""

import numpy as np
import torch
from torch.utils.data import Dataset
import medmnist
from medmnist import INFO

from utils.transforms import generate_view_b


def get_num_classes(dataset_name):
    """Get the number of classes for a MedMNIST dataset."""
    info = INFO[dataset_name]
    return len(info["label"])


def get_dataset(dataset_name, data_dir, split, img_size=224):
    """Load a MedMNIST dataset.

    Args:
        dataset_name: MedMNIST dataset name (e.g. 'pneumoniamnist')
        data_dir: directory to download data
        split: 'train', 'val', or 'test'
        img_size: resize target (28, 64, 128, or 224)

    Returns:
        MedMNISTViewDataset instance
    """
    return MedMNISTViewDataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split=split,
        img_size=img_size,
    )


class MedMNISTViewDataset(Dataset):
    """MedMNIST dataset with multi-view augmentation.

    Returns view_a (original), view_b (center-cropped + CLAHE enhanced),
    and the target label.
    """

    def __init__(self, dataset_name, data_dir, split, img_size=224,
                 crop_ratio=0.75):
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.crop_ratio = crop_ratio

        # Map split names
        split_map = {"train": "train", "val": "val", "test": "test"}

        DataClass = getattr(medmnist, INFO[dataset_name]["python_class"])
        self.dataset = DataClass(
            root=data_dir,
            split=split_map[split],
            download=True,
            size=img_size,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Convert to numpy array (handles PIL, torch, numpy)
        try:
            image = np.array(image)
        except Exception:
            pass
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        # Ensure (H, W, C) format
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        elif image.ndim == 3 and image.shape[0] in (1, 3):
            # (C, H, W) → (H, W, C)
            image = image.transpose(1, 2, 0)

        # Normalize to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        # View A: original
        view_a = image
        if view_a.ndim == 2:
            view_a = np.stack([view_a] * 3, axis=-1)
        elif view_a.shape[-1] == 1:
            view_a = np.repeat(view_a, 3, axis=-1)
        view_a = torch.from_numpy(
            view_a.transpose(2, 0, 1).astype(np.float32)
        )

        # View B: center-crop + CLAHE enhanced
        view_b = generate_view_b(image, crop_ratio=self.crop_ratio)
        if view_b.shape[0] == 1:
            view_b = view_b.repeat(3, axis=0)
        elif view_b.shape[0] != 3:
            view_b = view_b[:3, :, :] if view_b.shape[0] > 3 else \
                view_b.repeat(3, axis=0)[:3]
        view_b = torch.from_numpy(view_b.astype(np.float32))

        if isinstance(label, np.ndarray):
            label = int(label.item()) if label.size == 1 else int(label.argmax())
        elif isinstance(label, torch.Tensor):
            label = int(label.item())
        elif isinstance(label, (np.integer,)):
            label = int(label)
        else:
            label = int(label)

        return view_a, view_b, label
