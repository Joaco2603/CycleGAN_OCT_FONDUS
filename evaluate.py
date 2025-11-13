from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from src.data import CycleGANDataset, build_transforms
from src.inference import evaluate_models
from src.utils.config import TrainingConfig, load_config


def build_eval_loader(config: TrainingConfig, batch_size: int) -> DataLoader:
    transforms = build_transforms(config.data.image_size, train=False, augment=False)
    dataset = CycleGANDataset(config.data.fundus.val, config.data.oct.val, transforms, sample_mode="sequential")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN checkpoints")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--direction", choices=["fundus_to_oct", "oct_to_fundus"], default="fundus_to_oct")
    args = parser.parse_args()
    config = load_config(args.config)
    loader = build_eval_loader(config, args.batch_size)
    metrics = evaluate_models(args.checkpoint, loader, config, direction=args.direction)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
