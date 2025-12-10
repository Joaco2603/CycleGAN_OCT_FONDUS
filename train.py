from __future__ import annotations

import argparse

from src.training.train import train
from src.tracking import extract_metrics, track_experiment, track_dataset
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CycleGAN for OCT↔Fundus translation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--track_metrics", action="store_true", help="Log metrics to MLflow")
    parser.add_argument("--track_dataset", action="store_true", help="Track dataset with DVC")
    parser.add_argument("--experiment", default="cyclegan_fundus_oct")
    args = parser.parse_args()
    
    config = load_config(args.config)
    result = train(config)
    
    # Extract metrics (adapter)
    metrics = extract_metrics(
        run_name=result.run_name,
        start_time=result.start_time,
        end_time=result.end_time,
        epochs_completed=result.epochs_completed,
        total_epochs=result.total_epochs,
        total_steps=result.total_steps,
        g_loss_history=result.g_loss_history,
        d_loss_history=result.d_loss_history,
        config_dict=result.config_dict,
        checkpoint_path=result.checkpoint_path,
        device=result.device,
        gpu_name=result.gpu_name,
        max_gpu_temp=result.max_gpu_temp,
    )
    
    print(f"\n✅ Training completed: {metrics.epochs_completed} epochs")
    print(f"   Final G: {metrics.final_g_loss:.4f} | Best: {metrics.best_g_loss:.4f} (ep {metrics.best_epoch})")
    print(f"   Duration: {metrics.duration_minutes:.1f} min")
    
    if args.track_dataset:
        dataset_info = track_dataset(
            run_name=result.run_name,
            dataset_root="dataset",
            preprocessing=config.data.preprocessing,
            image_size=config.data.image_size,
        )
        print(f"\n📁 Dataset tracked: {dataset_info.total_images()} images")
        print(f"   Fundus: {dataset_info.fundus_train_count} train / {dataset_info.fundus_val_count} val")
        print(f"   OCT: {dataset_info.oct_train_count} train / {dataset_info.oct_val_count} val")
    
    if args.track_metrics:
        run_id = track_experiment(metrics, experiment_name=args.experiment)
        print(f"\n📊 Logged to MLflow: {args.experiment} (run: {run_id[:8]})")


if __name__ == "__main__":
    main()
