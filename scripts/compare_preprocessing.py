"""
Preview preprocessing pipelines side-by-side.

Usage:
    python scripts/compare_preprocessing.py --image path/to/image.jpg
    python scripts/compare_preprocessing.py --preset standard aggressive medical
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import list_preprocessors, get_preprocessor


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] tensor to [0, 1] for visualization."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def preview_presets(
    image_path: Path,
    presets: list[str] | None = None,
    image_size: int = 256,
    num_samples: int = 3,
    output_path: Path | None = None,
) -> None:
    """Show same image processed with different presets."""
    if presets is None:
        presets = list_preprocessors()

    img = Image.open(image_path).convert("RGB")
    n_presets = len(presets)

    fig, axes = plt.subplots(n_presets, num_samples + 1, figsize=(4 * (num_samples + 1), 4 * n_presets))
    if n_presets == 1:
        axes = [axes]

    for row, preset_name in enumerate(presets):
        preprocessor = get_preprocessor(preset_name)
        transform = preprocessor.get_transform(image_size, train=True)

        # Original (resized only)
        axes[row][0].imshow(img.resize((image_size, image_size)))
        axes[row][0].set_title(f"{preset_name}\n(original)", fontsize=10)
        axes[row][0].axis("off")

        # Multiple samples to show augmentation variability
        for col in range(num_samples):
            transformed = transform(img)
            img_display = denormalize(transformed).permute(1, 2, 0).numpy()
            axes[row][col + 1].imshow(img_display)
            axes[row][col + 1].set_title(f"sample {col + 1}", fontsize=9)
            axes[row][col + 1].axis("off")

    plt.suptitle(f"Preprocessing comparison: {image_path.name}", fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare preprocessing pipelines")
    parser.add_argument("--image", type=Path, required=True, help="Path to test image")
    parser.add_argument("--presets", nargs="*", default=None, help="Presets to compare (default: all)")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples per preset")
    parser.add_argument("--output", type=Path, default=None, help="Save figure to file")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        return

    available = list_preprocessors()
    print(f"Available presets: {', '.join(available)}")

    presets = args.presets if args.presets else available
    invalid = [p for p in presets if p not in available]
    if invalid:
        print(f"Unknown presets: {invalid}")
        return

    preview_presets(args.image, presets, args.size, args.samples, args.output)


if __name__ == "__main__":
    main()
