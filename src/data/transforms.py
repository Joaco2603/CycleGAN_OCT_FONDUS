"""
Transforms module - bridges old API to new preprocessing system.

Use `build_transforms` for backward compatibility, or use
`src.data.preprocessing` directly for the new Strategy-based system.
"""
from torchvision import transforms as T

from .preprocessing import get_preprocessor, list_preprocessors, PREPROCESSOR_REGISTRY

_MEAN = [0.5, 0.5, 0.5]
_STD = [0.5, 0.5, 0.5]


def build_transforms(
    image_size: int,
    train: bool = True,
    augment: bool = True,
    preset: str = "standard",
) -> T.Compose:
    """
    Build transforms using a named preset.

    Args:
        image_size: Target resolution (e.g., 256).
        train: Whether this is for training (enables augmentation).
        augment: Legacy flag; if False, forces 'minimal' preset.
        preset: Name of preprocessing pipeline. Options:
            - 'minimal': resize + normalize only
            - 'standard': flips + light color jitter (default)
            - 'aggressive': heavy augmentations
            - 'medical': spatial only, no color changes
            - 'contrast': histogram equalization
            - 'domain_specific': different per domain

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if not augment:
        preset = "minimal"
    preprocessor = get_preprocessor(preset)
    return preprocessor.get_transform(image_size, train)


# Re-export for convenience
__all__ = ["build_transforms", "get_preprocessor", "list_preprocessors", "PREPROCESSOR_REGISTRY"]
