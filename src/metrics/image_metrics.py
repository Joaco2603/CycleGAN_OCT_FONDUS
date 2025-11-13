import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure


def compute_psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    mse = torch.clamp(mse, min=1e-10)
    return 20 * torch.log10(torch.tensor(data_range, device=prediction.device)) - 10 * torch.log10(mse)


def compute_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return structural_similarity_index_measure(prediction, target, data_range=2.0)
