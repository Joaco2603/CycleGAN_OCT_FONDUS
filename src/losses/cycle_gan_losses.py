import torch
import torch.nn.functional as F

_mse = torch.nn.MSELoss()


def adversarial_loss(prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
    return _mse(prediction, target)


def cycle_consistency_loss(real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(reconstructed, real)


def identity_loss(real: torch.Tensor, same_domain: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(same_domain, real)
