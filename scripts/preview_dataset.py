from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview random Fundus and OCT samples")
    parser.add_argument("fundus", type=Path)
    parser.add_argument("oct", type=Path)
    parser.add_argument("--count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fundus = sorted(p for p in args.fundus.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    oct_paths = sorted(p for p in args.oct.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not fundus or not oct_paths:
        raise FileNotFoundError("Datasets are empty")
    for idx in range(args.count):
        Image.open(random.choice(fundus)).show(title=f"fundus_{idx}")
        Image.open(random.choice(oct_paths)).show(title=f"oct_{idx}")


if __name__ == "__main__":
    main()
