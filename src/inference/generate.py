from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

from ..models import CycleGAN
from ..utils.config import TrainingConfig


def generate_samples(
    checkpoint: Path,
    inputs: Iterable[Path],
    config: TrainingConfig,
    direction: str = "fundus_to_oct",
    output_dir: Path | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CycleGAN(blocks=config.model.blocks).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.gen_f2o.load_state_dict(state["model_gen_f2o"])
    model.gen_o2f.load_state_dict(state["model_gen_o2f"])
    model.eval()
    transform = T.Compose([T.Resize((config.data.image_size, config.data.image_size)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    output_dir = output_dir or config.paths.generated
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = model.gen_f2o if direction == "fundus_to_oct" else model.gen_o2f
    with torch.inference_mode():
        for path in inputs:
            tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            result = generator(tensor)
            image = (result.squeeze(0).cpu() * 0.5) + 0.5
            save_image(image, output_dir / f"{path.stem}_{direction}.png")
