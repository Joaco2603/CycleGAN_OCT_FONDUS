from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset

Image.MAX_IMAGE_PIXELS = None


def _gather_images(
    root: Path,
    quality_filter: Optional["CompositeFilter"] = None,
) -> Tuple[List[Path], int]:
    """
    Gather images from directory, optionally filtering by quality.

    Returns:
        (valid_paths, num_rejected)
    """
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"})
    if not paths:
        raise FileNotFoundError(f"No images under {root}")

    if quality_filter is None:
        return paths, 0

    valid = []
    rejected = 0
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            if quality_filter.check(img).passed:
                valid.append(p)
            else:
                rejected += 1
        except Exception:
            rejected += 1

    if not valid:
        raise FileNotFoundError(f"All {len(paths)} images rejected by quality filter in {root}")

    return valid, rejected


class CycleGANDataset(Dataset):
    def __init__(
        self,
        fundus_dir: Path,
        oct_dir: Path,
        transform: Callable[[Image.Image], object],
        sample_mode: str = "random",
        quality_filter_fundus: Optional["CompositeFilter"] = None,
        quality_filter_oct: Optional["CompositeFilter"] = None,
    ):
        self.fundus, rej_f = _gather_images(fundus_dir, quality_filter_fundus)
        self.oct, rej_o = _gather_images(oct_dir, quality_filter_oct)
        self.transform = transform
        self.sample_mode = sample_mode

        if rej_f > 0 or rej_o > 0:
            print(f"   Quality filter: rejected {rej_f} fundus, {rej_o} OCT images")
            print(f"   Using: {len(self.fundus)} fundus, {len(self.oct)} OCT images")

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
    quality_filter_fundus: Optional["CompositeFilter"] = None,
    quality_filter_oct: Optional["CompositeFilter"] = None,
) -> DataLoader:
    dataset = CycleGANDataset(
        fundus_dir,
        oct_dir,
        transform,
        sample_mode=sample_mode,
        quality_filter_fundus=quality_filter_fundus,
        quality_filter_oct=quality_filter_oct,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
