"""
Preprocessing pipelines for CycleGAN Fundus↔OCT.

Strategy pattern: define multiple pipelines, select via config.
Each pipeline is a callable that returns a torchvision.transforms.Compose.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

import torch
from torchvision import transforms as T

_MEAN = [0.5, 0.5, 0.5]
_STD = [0.5, 0.5, 0.5]


class BasePreprocessor(ABC):
    """Abstract base for all preprocessing pipelines."""

    name: str = "base"

    @abstractmethod
    def get_transform(self, image_size: int, train: bool) -> T.Compose:
        """Return a Compose pipeline."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


# =============================================================================
# PRESET 1: Minimal (resize + normalize only)
# =============================================================================
class MinimalPreprocessor(BasePreprocessor):
    """No augmentation, just resize and normalize. Good baseline."""

    name = "minimal"

    def get_transform(self, image_size: int, train: bool) -> T.Compose:
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
        ])


# =============================================================================
# PRESET 2: Standard (current default with flips + color jitter)
# =============================================================================
class StandardPreprocessor(BasePreprocessor):
    """Standard augmentations: flips + light color jitter."""

    name = "standard"

    def get_transform(self, image_size: int, train: bool) -> T.Compose:
        ops = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)]
        if train:
            ops.extend([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
        ops.extend([T.ToTensor(), T.Normalize(_MEAN, _STD)])
        return T.Compose(ops)


# =============================================================================
# PRESET 3: Aggressive (heavy augmentations)
# =============================================================================
class AggressivePreprocessor(BasePreprocessor):
    """Heavy augmentations for regularization."""

    name = "aggressive"

    def get_transform(self, image_size: int, train: bool) -> T.Compose:
        ops = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)]
        if train:
            ops.extend([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            ])
        ops.extend([T.ToTensor(), T.Normalize(_MEAN, _STD)])
        return T.Compose(ops)


# =============================================================================
# PRESET 4: Medical-focused (preserve structure, minimal color changes)
# =============================================================================
class MedicalPreprocessor(BasePreprocessor):
    """Preserve anatomical structures; spatial augmentations only."""

    name = "medical"

    def get_transform(self, image_size: int, train: bool) -> T.Compose:
        ops = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)]
        if train:
            ops.extend([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(10),
                # No color jitter to preserve diagnostic features
            ])
        ops.extend([T.ToTensor(), T.Normalize(_MEAN, _STD)])
        return T.Compose(ops)


# =============================================================================
# PRESET 5: CLAHE-inspired (contrast enhancement via histogram equalization)
# =============================================================================
class ContrastEnhancedPreprocessor(BasePreprocessor):
    """Apply histogram equalization for better contrast (useful for OCT)."""

    name = "contrast"

    def get_transform(self, image_size: int, train: bool) -> T.Compose:
        ops = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)]
        if train:
            ops.extend([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAutocontrast(p=0.5),
                T.RandomEqualize(p=0.3),
            ])
        else:
            # Apply consistent enhancement for val/test
            ops.append(T.RandomAutocontrast(p=1.0))
        ops.extend([T.ToTensor(), T.Normalize(_MEAN, _STD)])
        return T.Compose(ops)


# =============================================================================
# PRESET 6: Domain-specific (different transforms per domain)
# =============================================================================
class DomainSpecificPreprocessor(BasePreprocessor):
    """Different pipelines for Fundus vs OCT. Returns dict of transforms."""

    name = "domain_specific"

    def get_transform(self, image_size: int, train: bool) -> T.Compose:
        # Default fallback; use get_domain_transforms for split
        return self._get_fundus_transform(image_size, train)

    def get_domain_transforms(self, image_size: int, train: bool) -> Dict[str, T.Compose]:
        return {
            "fundus": self._get_fundus_transform(image_size, train),
            "oct": self._get_oct_transform(image_size, train),
        }

    def _get_fundus_transform(self, image_size: int, train: bool) -> T.Compose:
        ops = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)]
        if train:
            ops.extend([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1, hue=0.02),
            ])
        ops.extend([T.ToTensor(), T.Normalize(_MEAN, _STD)])
        return T.Compose(ops)

    def _get_oct_transform(self, image_size: int, train: bool) -> T.Compose:
        ops = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)]
        if train:
            ops.extend([
                T.RandomHorizontalFlip(),
                # OCT: no vertical flip (layer orientation matters)
                T.RandomAutocontrast(p=0.3),
            ])
        ops.extend([T.ToTensor(), T.Normalize(_MEAN, _STD)])
        return T.Compose(ops)


# =============================================================================
# Registry: map names to classes
# =============================================================================
PREPROCESSOR_REGISTRY: Dict[str, Type[BasePreprocessor]] = {
    "minimal": MinimalPreprocessor,
    "standard": StandardPreprocessor,
    "aggressive": AggressivePreprocessor,
    "medical": MedicalPreprocessor,
    "contrast": ContrastEnhancedPreprocessor,
    "domain_specific": DomainSpecificPreprocessor,
}


def get_preprocessor(name: str) -> BasePreprocessor:
    """Factory function to get a preprocessor by name."""
    if name not in PREPROCESSOR_REGISTRY:
        available = ", ".join(PREPROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown preprocessor '{name}'. Available: {available}")
    return PREPROCESSOR_REGISTRY[name]()


def list_preprocessors() -> list[str]:
    """Return list of available preprocessor names."""
    return list(PREPROCESSOR_REGISTRY.keys())
