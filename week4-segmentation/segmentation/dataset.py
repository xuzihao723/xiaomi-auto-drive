"""Dataset and paired augmentations for RGB/semantic-mask training."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CarlaSegmentationDataset(Dataset):
    def __init__(self, root, split, image_size=(640, 360), augment=False):
        self.root = Path(root)
        self.split = split
        self.width, self.height = image_size
        self.augment = augment
        image_dir = self.root / "images" / split
        mask_dir = self.root / "masks" / split
        self.samples = []
        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path.name}")
            self.samples.append((image_path, mask_path))
        if not self.samples:
            raise RuntimeError(f"No samples found for split={split} under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Failed to read {image_path} / {mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if self.augment:
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-12.0, 12.0)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.astype(np.int64))
        return image, mask, image_path.name
