from __future__ import annotations

from typing import Dict

import torch

from ..metrics import compute_psnr, compute_ssim
from ..models import CycleGAN
from ..utils.config import TrainingConfig


def evaluate_models(checkpoint: str, dataloader, config: TrainingConfig, direction: str = "fundus_to_oct") -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CycleGAN(blocks=config.model.blocks).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.gen_f2o.load_state_dict(state["model_gen_f2o"])
    model.gen_o2f.load_state_dict(state["model_gen_o2f"])
    model.eval()
    psnr_scores, ssim_scores = [], []
    generator = model.gen_f2o if direction == "fundus_to_oct" else model.gen_o2f
    with torch.inference_mode():
        for batch in dataloader:
            real = batch["fundus"].to(device) if direction == "fundus_to_oct" else batch["oct"].to(device)
            target = batch["oct"].to(device) if direction == "fundus_to_oct" else batch["fundus"].to(device)
            fake = generator(real)
            psnr_scores.append(compute_psnr(fake, target))
            ssim_scores.append(compute_ssim(fake, target))
    return {
        "psnr": torch.stack(psnr_scores).mean().item(),
        "ssim": torch.stack(ssim_scores).mean().item(),
    }
