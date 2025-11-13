from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.generate import generate_samples
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OCT↔Fundus translations")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--direction", choices=["fundus_to_oct", "oct_to_fundus"], default="fundus_to_oct")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    generate_samples(args.checkpoint, args.inputs, config, direction=args.direction, output_dir=args.output)


if __name__ == "__main__":
    main()
