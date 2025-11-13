from __future__ import annotations

import argparse

from src.training.train import train
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CycleGAN for OCT↔Fundus translation")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
