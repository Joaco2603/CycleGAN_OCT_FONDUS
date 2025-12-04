"""
Quality filters for fundus and OCT images.

Filters out low-quality images before training:
- Too dark / too bright
- Low contrast
- Mostly black (failed captures)
- Blurry images
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Type

import numpy as np
from PIL import Image


@dataclass
class QualityResult:
    """Result of quality check."""
    passed: bool
    reason: str
    metrics: Dict[str, float]


class BaseQualityFilter(ABC):
    """Abstract base for quality filters."""

    name: str = "base"

    @abstractmethod
    def check(self, img: Image.Image) -> QualityResult:
        """Check if image passes quality filter."""
        ...


class BrightnessFilter(BaseQualityFilter):
    """Filter images that are too dark or too bright."""

    name = "brightness"

    def __init__(self, min_mean: float = 15.0, max_mean: float = 240.0):
        self.min_mean = min_mean
        self.max_mean = max_mean

    def check(self, img: Image.Image) -> QualityResult:
        arr = np.array(img.convert("L"))
        mean_brightness = arr.mean()

        if mean_brightness < self.min_mean:
            return QualityResult(False, f"Too dark (mean={mean_brightness:.1f})", {"brightness": mean_brightness})
        if mean_brightness > self.max_mean:
            return QualityResult(False, f"Too bright (mean={mean_brightness:.1f})", {"brightness": mean_brightness})

        return QualityResult(True, "OK", {"brightness": mean_brightness})


class ContrastFilter(BaseQualityFilter):
    """Filter images with insufficient contrast."""

    name = "contrast"

    def __init__(self, min_std: float = 20.0):
        self.min_std = min_std

    def check(self, img: Image.Image) -> QualityResult:
        arr = np.array(img.convert("L"))
        std = arr.std()

        if std < self.min_std:
            return QualityResult(False, f"Low contrast (std={std:.1f})", {"contrast_std": std})

        return QualityResult(True, "OK", {"contrast_std": std})


class BlackRatioFilter(BaseQualityFilter):
    """Filter images with too much black area (failed captures, masks)."""

    name = "black_ratio"

    def __init__(self, max_black_ratio: float = 0.7, black_threshold: int = 10):
        self.max_black_ratio = max_black_ratio
        self.black_threshold = black_threshold

    def check(self, img: Image.Image) -> QualityResult:
        arr = np.array(img.convert("L"))
        black_pixels = (arr < self.black_threshold).sum()
        total_pixels = arr.size
        black_ratio = black_pixels / total_pixels

        if black_ratio > self.max_black_ratio:
            return QualityResult(
                False,
                f"Too much black area ({black_ratio:.1%})",
                {"black_ratio": black_ratio},
            )

        return QualityResult(True, "OK", {"black_ratio": black_ratio})


class ContentRatioFilter(BaseQualityFilter):
    """Filter based on useful content area (non-black circular region)."""

    name = "content_ratio"

    def __init__(self, min_content_brightness: float = 25.0, min_content_ratio: float = 0.15):
        self.min_content_brightness = min_content_brightness
        self.min_content_ratio = min_content_ratio

    def check(self, img: Image.Image) -> QualityResult:
        arr = np.array(img.convert("L"))
        # Consider "content" as pixels above threshold
        content_mask = arr > self.min_content_brightness
        content_ratio = content_mask.sum() / arr.size

        # Also check brightness of content area
        if content_mask.sum() > 0:
            content_brightness = arr[content_mask].mean()
        else:
            content_brightness = 0.0

        if content_ratio < self.min_content_ratio:
            return QualityResult(
                False,
                f"Insufficient content ({content_ratio:.1%} visible)",
                {"content_ratio": content_ratio, "content_brightness": content_brightness},
            )

        return QualityResult(
            True,
            "OK",
            {"content_ratio": content_ratio, "content_brightness": content_brightness},
        )


class LaplacianBlurFilter(BaseQualityFilter):
    """Filter blurry images using Laplacian variance."""

    name = "blur"

    def __init__(self, min_laplacian_var: float = 50.0):
        self.min_laplacian_var = min_laplacian_var

    def check(self, img: Image.Image) -> QualityResult:
        try:
            from scipy import ndimage
        except ImportError:
            # Skip if scipy not available
            return QualityResult(True, "scipy not available", {"laplacian_var": -1})

        arr = np.array(img.convert("L")).astype(np.float64)
        laplacian = ndimage.laplace(arr)
        var = laplacian.var()

        if var < self.min_laplacian_var:
            return QualityResult(False, f"Too blurry (var={var:.1f})", {"laplacian_var": var})

        return QualityResult(True, "OK", {"laplacian_var": var})


class CompositeFilter:
    """Combine multiple filters; image must pass all."""

    def __init__(self, filters: List[BaseQualityFilter]):
        self.filters = filters

    def check(self, img: Image.Image) -> QualityResult:
        all_metrics = {}
        for f in self.filters:
            result = f.check(img)
            all_metrics.update(result.metrics)
            if not result.passed:
                return QualityResult(False, f"[{f.name}] {result.reason}", all_metrics)
        return QualityResult(True, "Passed all filters", all_metrics)

    @classmethod
    def default_fundus(cls) -> "CompositeFilter":
        """Default filter set for fundus images."""
        return cls([
            BrightnessFilter(min_mean=20.0, max_mean=220.0),
            ContrastFilter(min_std=25.0),
            BlackRatioFilter(max_black_ratio=0.65),
            ContentRatioFilter(min_content_brightness=30.0, min_content_ratio=0.20),
        ])

    @classmethod
    def default_oct(cls) -> "CompositeFilter":
        """Default filter set for OCT images."""
        return cls([
            BrightnessFilter(min_mean=10.0, max_mean=245.0),
            ContrastFilter(min_std=15.0),
            BlackRatioFilter(max_black_ratio=0.80),
        ])

    @classmethod
    def strict_fundus(cls) -> "CompositeFilter":
        """Strict filter for high-quality fundus only."""
        return cls([
            BrightnessFilter(min_mean=35.0, max_mean=200.0),
            ContrastFilter(min_std=35.0),
            BlackRatioFilter(max_black_ratio=0.55),
            ContentRatioFilter(min_content_brightness=40.0, min_content_ratio=0.30),
            LaplacianBlurFilter(min_laplacian_var=100.0),
        ])


def filter_dataset(
    root: Path,
    quality_filter: CompositeFilter,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif"),
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """
    Filter a dataset directory.

    Returns:
        (passed_images, rejected_images_with_reasons)
    """
    root = Path(root)
    all_images = sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)

    passed = []
    rejected = []

    for img_path in all_images:
        try:
            img = Image.open(img_path).convert("RGB")
            result = quality_filter.check(img)
            if result.passed:
                passed.append(img_path)
            else:
                rejected.append((img_path, result.reason))
        except Exception as e:
            rejected.append((img_path, f"Error loading: {e}"))

    return passed, rejected
