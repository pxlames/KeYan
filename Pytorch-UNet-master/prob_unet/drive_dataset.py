from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DrivePreparedDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        scale=1.0,
        crop_size=0,
        random_crop=None,
        return_id=False,
    ):
        self.root = Path(root)
        self.split = split
        assert 0 < scale <= 1.0, "scale must be in (0, 1]"
        self.scale = float(scale)
        self.crop_size = int(crop_size)
        self.random_crop = (split == "train") if random_crop is None else bool(random_crop)
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

    def _resize_pair(self, image: Image.Image, mask: Image.Image):
        if self.scale == 1.0:
            return image, mask

        width, height = image.size
        new_width = max(1, int(width * self.scale))
        new_height = max(1, int(height * self.scale))
        image = image.resize((new_width, new_height), resample=Image.BICUBIC)
        mask = mask.resize((new_width, new_height), resample=Image.NEAREST)
        return image, mask

    def _crop_pair(self, image: np.ndarray, mask: np.ndarray):
        if self.crop_size <= 0:
            return image, mask

        crop_h = min(self.crop_size, mask.shape[0])
        crop_w = min(self.crop_size, mask.shape[1])
        if crop_h == mask.shape[0] and crop_w == mask.shape[1]:
            return image, mask

        max_top = mask.shape[0] - crop_h
        max_left = mask.shape[1] - crop_w
        if self.random_crop:
            top = np.random.randint(0, max_top + 1) if max_top > 0 else 0
            left = np.random.randint(0, max_left + 1) if max_left > 0 else 0
        else:
            top = max_top // 2
            left = max_left // 2

        image = image[top:top + crop_h, left:left + crop_w]
        mask = mask[top:top + crop_h, left:left + crop_w]
        return image, mask

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        image, mask = self._resize_pair(image, mask)

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        image, mask = self._crop_pair(image, mask)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask)

        if self.return_id:
            return image, mask, image_path.stem
        return image, mask
