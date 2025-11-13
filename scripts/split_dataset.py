from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from shutil import copy2

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a dataset into train/val/test folders")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating hardlinks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No images found in {args.source}")
    random.Random(args.seed).shuffle(files)
    totals = [int(len(files) * ratio) for ratio in args.ratios]
    totals[0] = max(0, totals[0])
    totals[1] = max(0, totals[1])
    totals[2] = len(files) - totals[0] - totals[1]
    splits = {
        "train": files[: totals[0]],
        "val": files[totals[0] : totals[0] + totals[1]],
        "test": files[totals[0] + totals[1] :],
    }
    for split, split_files in splits.items():
        base_target = args.destination / split
        base_target.mkdir(parents=True, exist_ok=True)
        for path in split_files:
            rel_path = path.relative_to(args.source)
            out = base_target / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            if args.copy:
                copy2(path, out)
                continue
            if out.exists():
                out.unlink()
            try:
                os.link(path, out)
            except OSError:
                copy2(path, out)


if __name__ == "__main__":
    main()
