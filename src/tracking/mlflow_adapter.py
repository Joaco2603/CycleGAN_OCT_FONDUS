"""
MLflow Adapter.

Adapts TrainingMetrics to MLflow's logging interface.
The training code knows nothing about MLflow.
MLflow knows nothing about training internals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .metrics import TrainingMetrics


class MLflowAdapter:
    """
    Adapter that translates TrainingMetrics → MLflow calls.
    
    Usage:
        metrics = extract_metrics(...)
        adapter = MLflowAdapter(experiment_name="cyclegan")
        adapter.log(metrics)
    """
    
    def __init__(
        self,
        experiment_name: str = "cyclegan_fundus_oct",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "mlruns"
        self._mlflow = None
    
    def _init_mlflow(self):
        """Lazy init - only import when needed."""
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
            except ImportError:
                raise ImportError("MLflow not installed. Run: pip install mlflow")
        return self._mlflow
    
    def log(self, metrics: TrainingMetrics) -> str:
        """
        Log TrainingMetrics to MLflow.
        
        Returns the MLflow run_id.
        """
        mlflow = self._init_mlflow()
        
        with mlflow.start_run(run_name=metrics.run_name) as run:
            # Log hyperparameters
            if metrics.params:
                mlflow.log_params(metrics.params)
            
            # Log final metrics
            mlflow.log_metrics({
                "final_g_loss": metrics.final_g_loss,
                "final_d_loss": metrics.final_d_loss,
                "best_g_loss": metrics.best_g_loss,
                "best_epoch": metrics.best_epoch,
                "epochs_completed": metrics.epochs_completed,
                "total_steps": metrics.total_steps,
                "duration_minutes": metrics.duration_minutes,
            })
            
            if metrics.max_gpu_temp:
                mlflow.log_metric("max_gpu_temp", metrics.max_gpu_temp)
            
            # Log loss curves
            for step, (g, d) in enumerate(zip(
                metrics.g_loss_history, metrics.d_loss_history
            )):
                mlflow.log_metrics({"epoch_g_loss": g, "epoch_d_loss": d}, step=step)
            
            # Log artifacts
            if metrics.checkpoint_path and Path(metrics.checkpoint_path).exists():
                mlflow.log_artifact(metrics.checkpoint_path)
            
            for path in metrics.sample_paths:
                if Path(path).exists():
                    mlflow.log_artifact(path)
            
            # Tags
            mlflow.set_tags({
                "status": metrics.status,
                "device": metrics.device,
                "gpu": metrics.gpu_name or "N/A",
            })
            
            return run.info.run_id


def track_experiment(
    metrics: TrainingMetrics,
    experiment_name: str = "cyclegan_fundus_oct",
    tracking_uri: Optional[str] = None,
) -> str:
    """
    Convenience function to log metrics in one call.
    
    Usage:
        metrics = extract_metrics(...)
        run_id = track_experiment(metrics, experiment_name="my_exp")
    """
    adapter = MLflowAdapter(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )
    return adapter.log(metrics)
