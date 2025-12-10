"""
Metrics extraction module.

Provides a standard format for training metrics that can be
consumed by any tracking backend (MLflow, W&B, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingMetrics:
    """Standard format for training metrics (adapter target)."""
    
    # Run info
    run_name: str
    start_time: datetime
    end_time: datetime
    status: str  # completed, failed, interrupted
    
    # Training progress
    epochs_completed: int
    total_epochs: int
    total_steps: int
    
    # Final losses
    final_g_loss: float
    final_d_loss: float
    final_cycle_loss: float = 0.0
    final_identity_loss: float = 0.0
    
    # Best metrics
    best_g_loss: float = float("inf")
    best_epoch: int = 0
    
    # History (for plotting curves)
    g_loss_history: List[float] = field(default_factory=list)
    d_loss_history: List[float] = field(default_factory=list)
    
    # Artifacts
    checkpoint_path: Optional[str] = None
    sample_paths: List[str] = field(default_factory=list)
    
    # Hardware
    device: str = "cpu"
    gpu_name: Optional[str] = None
    max_gpu_temp: Optional[float] = None
    
    # Hyperparameters (flat dict for easy logging)
    params: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        d = asdict(self)
        d["start_time"] = self.start_time.isoformat()
        d["end_time"] = self.end_time.isoformat()
        d["duration_seconds"] = self.duration_seconds
        return d


def extract_metrics(
    run_name: str,
    start_time: datetime,
    end_time: datetime,
    epochs_completed: int,
    total_epochs: int,
    total_steps: int,
    g_loss_history: List[float],
    d_loss_history: List[float],
    config_dict: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    sample_paths: Optional[List[str]] = None,
    device: str = "cpu",
    gpu_name: Optional[str] = None,
    max_gpu_temp: Optional[float] = None,
    status: str = "completed",
) -> TrainingMetrics:
    """
    Extract and adapt training results to standard TrainingMetrics.
    
    This is the adapter function that converts raw training outputs
    to a format consumable by any tracking backend.
    """
    best_g_loss = min(g_loss_history) if g_loss_history else float("inf")
    best_epoch = g_loss_history.index(best_g_loss) + 1 if g_loss_history else 0
    
    return TrainingMetrics(
        run_name=run_name,
        start_time=start_time,
        end_time=end_time,
        status=status,
        epochs_completed=epochs_completed,
        total_epochs=total_epochs,
        total_steps=total_steps,
        final_g_loss=g_loss_history[-1] if g_loss_history else 0.0,
        final_d_loss=d_loss_history[-1] if d_loss_history else 0.0,
        best_g_loss=best_g_loss,
        best_epoch=best_epoch,
        g_loss_history=g_loss_history,
        d_loss_history=d_loss_history,
        checkpoint_path=checkpoint_path,
        sample_paths=sample_paths or [],
        device=device,
        gpu_name=gpu_name,
        max_gpu_temp=max_gpu_temp,
        params=config_dict,
    )
