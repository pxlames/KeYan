from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DrivePreparedDataset(Dataset):
    def __init__(self, root, split="train", image_size=256, return_id=False):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.return_id = return_id
        self.img_dir = self.root / split / "imgs"
        self.mask_dir = self.root / split / "masks"
        self.samples = []

        if not self.img_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Expected {self.img_dir} and {self.mask_dir} to exist")

        for image_path in sorted(self.img_dir.glob("*.png")):
            mask_path = self.mask_dir / image_path.name
            if mask_path.is_file():
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise ValueError(f"No DRIVE samples found in {self.img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.image_size is not None:
            size = (self.image_size, self.image_size)
            image = image.resize(size, Image.BILINEAR)
            mask = mask.resize(size, Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask)

        if self.return_id:
            return image, mask, image_path.stem
        return image, mask

