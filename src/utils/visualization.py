from __future__ import annotations

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 0.5) + 0.5


def write_image_grid(writer: SummaryWriter, tag: str, images: torch.Tensor, step: int, max_images: int = 4) -> None:
    sample = denormalize(images[:max_images].detach().cpu())
    grid = make_grid(sample, nrow=max(1, min(max_images, sample.size(0))))
    writer.add_image(tag, grid, step)
