from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_lr_scheduler(optimizer: Optimizer, total_epochs: int, decay_start: int) -> LambdaLR:
    decay_start = max(decay_start, 0)

    def lr_lambda(epoch: int) -> float:
        if epoch < decay_start:
            return 1.0
        progress = (epoch - decay_start) / max(1, total_epochs - decay_start)
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
