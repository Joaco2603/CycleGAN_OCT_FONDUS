from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset

Image.MAX_IMAGE_PIXELS = None


def _gather_images(root: Path) -> List[Path]:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"})
    if not paths:
        raise FileNotFoundError(f"No images under {root}")
    return paths


class CycleGANDataset(Dataset):
    def __init__(
        self,
        fundus_dir: Path,
        oct_dir: Path,
        transform: Callable[[Image.Image], object],
        sample_mode: str = "random",
    ):
        self.fundus = _gather_images(fundus_dir)
        self.oct = _gather_images(oct_dir)
        self.transform = transform
        self.sample_mode = sample_mode

    def __len__(self) -> int:
        return max(len(self.fundus), len(self.oct))

    def __getitem__(self, index: int) -> Dict[str, object]:
        fundus_img = Image.open(self.fundus[index % len(self.fundus)]).convert("RGB")
        if self.sample_mode == "sequential":
            oct_path = self.oct[index % len(self.oct)]
        else:
            oct_path = random.choice(self.oct)
        oct_img = Image.open(oct_path).convert("RGB")
        return {
            "fundus": self.transform(fundus_img),
            "oct": self.transform(oct_img),
        }


def build_dataloader(
    fundus_dir: Path,
    oct_dir: Path,
    transform: Callable[[Image.Image], object],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    sample_mode: str = "random",
) -> DataLoader:
    dataset = CycleGANDataset(fundus_dir, oct_dir, transform, sample_mode=sample_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
