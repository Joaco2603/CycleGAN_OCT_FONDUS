from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch.optim import Optimizer


def save_checkpoint(state: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state), path)


def load_checkpoint(
    path: Path,
    modules: Mapping[str, torch.nn.Module],
    optimizers: Optional[Mapping[str, Optimizer]] = None,
    schedulers: Optional[Mapping[str, torch.optim.lr_scheduler._LRScheduler]] = None,
    map_location: str | torch.device | None = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    for name, module in modules.items():
        module.load_state_dict(checkpoint[f"model_{name}"])
    if optimizers:
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint[f"optim_{name}"])
    if schedulers:
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(checkpoint[f"sched_{name}"])
    return checkpoint
