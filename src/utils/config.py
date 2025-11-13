from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class DomainPaths:
    train: Path
    val: Path
    test: Path


@dataclass
class DataConfig:
    fundus: DomainPaths
    oct: DomainPaths
    image_size: int
    augment: bool
    num_workers: int


@dataclass
class ModelConfig:
    blocks: int


@dataclass
class LossConfig:
    lambda_cycle: float
    lambda_identity: float


@dataclass
class OptimConfig:
    lr: float
    betas: tuple[float, float]
    batch_size: int


@dataclass
class PathsConfig:
    logs: Path
    weights: Path
    generated: Path


@dataclass
class ScheduleConfig:
    epochs: int
    decay_start: int
    amp: bool
    sample_interval: int
    save_interval: int
    max_gpu_temp: float
    temp_check_interval: int


@dataclass
class TrainingConfig:
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    optim: OptimConfig
    paths: PathsConfig
    training: ScheduleConfig


def load_config(path: Path | str) -> TrainingConfig:
    cfg_path = Path(path)
    raw = _read_yaml(cfg_path)
    base = cfg_path.parent
    data = DataConfig(
        fundus=_parse_domain(raw["data"]["fundus"], base),
        oct=_parse_domain(raw["data"]["oct"], base),
        image_size=raw["data"]["image_size"],
        augment=raw["data"].get("augment", True),
        num_workers=raw["data"].get("num_workers", 2),
    )
    optim = OptimConfig(
        lr=raw["optim"]["lr"],
        betas=tuple(raw["optim"].get("betas", [0.5, 0.999])),
        batch_size=raw["optim"].get("batch_size", 1),
    )
    paths = PathsConfig(
        logs=(base / raw["paths"]["logs"]).resolve(),
        weights=(base / raw["paths"]["weights"]).resolve(),
        generated=(base / raw["paths"]["generated"]).resolve(),
    )
    return TrainingConfig(
        data=data,
        model=ModelConfig(blocks=raw["model"].get("blocks", 9)),
        loss=LossConfig(
            lambda_cycle=float(raw["loss"].get("lambda_cycle", 10.0)),
            lambda_identity=float(raw["loss"].get("lambda_identity", 0.5)),
        ),
        optim=optim,
        paths=paths,
        training=ScheduleConfig(
            epochs=raw["training"].get("epochs", 200),
            decay_start=raw["training"].get("decay_start", 100),
            amp=raw["training"].get("amp", True),
            sample_interval=raw["training"].get("sample_interval", 100),
            save_interval=raw["training"].get("save_interval", 10),
            max_gpu_temp=float(raw["training"].get("max_gpu_temp", 85.0)),
            temp_check_interval=raw["training"].get("temp_check_interval", 10),
        ),
    )


def _read_yaml(path: Path | str) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parse_domain(section: Mapping[str, Any], base: Path) -> DomainPaths:
    root = base / section.get("root", "")
    return DomainPaths(
        train=(root / section["train"]).resolve(),
        val=(root / section["val"]).resolve(),
        test=(root / section["test"]).resolve(),
    )
