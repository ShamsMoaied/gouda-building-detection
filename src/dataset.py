"""PyTorch dataset that turns a chip + instance mask into the target format Mask R-CNN expects."""

import random
from pathlib import Path

import numpy as np
import rasterio
import torch


class BuildingChips(torch.utils.data.Dataset):
    def __init__(self, root, augment=False):
        self.images = sorted((Path(root) / "images").glob("*.tif"))
        self.mask_dir = Path(root) / "masks"
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        path = self.images[i]
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32) / 255.0          # (3, H, W), values 0..1
        with rasterio.open(self.mask_dir / path.name) as src:
            mask = src.read(1).astype(np.int64)                  # (H, W), 0 = background

        if self.augment:
            # Aerial images have no "up", so flips and 90 degree rotations are free extra data.
            k = random.randint(0, 3)
            img, mask = np.rot90(img, k, axes=(1, 2)), np.rot90(mask, k)
            if random.random() < 0.5:
                img, mask = img[:, :, ::-1], mask[:, ::-1]
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)

        ids = np.unique(mask)
        ids = ids[ids > 0]
        masks = (mask[None] == ids[:, None, None]).astype(np.uint8)   # one binary mask per building

        boxes = []
        for m in masks:
            ys, xs = np.where(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(ids), dtype=torch.int64),        # one class: building
            "masks": torch.from_numpy(masks),
        }
        return torch.from_numpy(img), target, path.stem


def collate(batch):
    return tuple(zip(*batch))
