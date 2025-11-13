from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from ..data import build_dataloader, build_transforms
from ..losses import adversarial_loss, cycle_consistency_loss, identity_loss
from ..models import CycleGAN
from ..utils.config import TrainingConfig
from ..utils.gpu_monitor import GPUTempMonitor, get_gpu_temperature
from ..utils.visualization import write_image_grid
from .checkpoint import save_checkpoint
from .schedulers import build_lr_scheduler


@dataclass
class TrainArtifacts:
    epoch: int
    generator_loss: float
    discriminator_loss: float


def train(config: TrainingConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        temp = get_gpu_temperature()
        if temp is not None:
            print(f"   🌡️  Temperatura inicial: {temp}°C (límite: {config.training.max_gpu_temp}°C)")
    config.paths.logs.mkdir(parents=True, exist_ok=True)
    config.paths.weights.mkdir(parents=True, exist_ok=True)
    config.paths.generated.mkdir(parents=True, exist_ok=True)
    transforms_train = build_transforms(config.data.image_size, train=True, augment=config.data.augment)
    loader = build_dataloader(
        config.data.fundus.train,
        config.data.oct.train,
        transforms_train,
        config.optim.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = CycleGAN(blocks=config.model.blocks).to(device)
    for module in model.generators() + model.discriminators():
        module.apply(_init_weights)
    opt_g = Adam(
        list(model.gen_f2o.parameters()) + list(model.gen_o2f.parameters()),
        lr=config.optim.lr,
        betas=config.optim.betas,
    )
    opt_d = Adam(
        list(model.disc_f.parameters()) + list(model.disc_o.parameters()),
        lr=config.optim.lr,
        betas=config.optim.betas,
    )
    sched_g = build_lr_scheduler(opt_g, config.training.epochs, config.training.decay_start)
    sched_d = build_lr_scheduler(opt_d, config.training.epochs, config.training.decay_start)
    writer = SummaryWriter(log_dir=config.paths.logs)
    scaler = GradScaler(enabled=config.training.amp and device.type == "cuda")
    temp_monitor = GPUTempMonitor(
        max_temp=config.training.max_gpu_temp,
        check_every=config.training.temp_check_interval
    )
    global_step = 0
    for epoch in range(config.training.epochs):
        losses_g: Dict[str, float] = {}
        losses_d: Dict[str, float] = {}
        for batch_idx, batch in enumerate(loader):
            real_f = batch["fundus"].to(device)
            real_o = batch["oct"].to(device)
            opt_g.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            with autocast(scaler.is_enabled()):
                fake_o = model.gen_f2o(real_f)
                fake_f = model.gen_o2f(real_o)
                rec_f = model.gen_o2f(fake_o)
                rec_o = model.gen_f2o(fake_f)
                id_f = model.gen_o2f(real_f)
                id_o = model.gen_f2o(real_o)
                loss_d_f = _disc_loss(model.disc_f, real_f, fake_f.detach())
                loss_d_o = _disc_loss(model.disc_o, real_o, fake_o.detach())
                loss_g_adv = adversarial_loss(model.disc_o(fake_o), True) + adversarial_loss(model.disc_f(fake_f), True)
                loss_cycle = cycle_consistency_loss(real_f, rec_f) + cycle_consistency_loss(real_o, rec_o)
                loss_id = identity_loss(real_f, id_f) + identity_loss(real_o, id_o)
                loss_g = loss_g_adv + config.loss.lambda_cycle * loss_cycle + config.loss.lambda_identity * loss_id
                loss_d = (loss_d_f + loss_d_o) * 0.5
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss_d).backward()
            scaler.step(opt_d)
            scaler.update()
            opt_d.zero_grad(set_to_none=True)
            losses_g["total"] = loss_g.item()
            losses_d["total"] = loss_d.item()
            writer.add_scalar("loss/generator", losses_g["total"], global_step)
            writer.add_scalar("loss/discriminator", losses_d["total"], global_step)
            if global_step % config.training.sample_interval == 0:
                write_image_grid(writer, "fundus", real_f, global_step)
                write_image_grid(writer, "oct_fake", fake_o, global_step)
            global_step += 1
            temp_monitor.check()
        sched_g.step()
        sched_d.step()
        temp_monitor.reset()
        writer.flush()
        temp_info = f" | GPU: {temp_monitor.last_temp:.0f}°C" if temp_monitor.last_temp else ""
        print(f"Epoch {epoch+1}/{config.training.epochs} — G: {losses_g['total']:.4f}, D: {losses_d['total']:.4f}{temp_info}")
        if (epoch + 1) % config.training.save_interval == 0:
            _persist_checkpoint(config, epoch, model, opt_g, opt_d, sched_g, sched_d)
    writer.close()


def _disc_loss(module, real, fake):
    return adversarial_loss(module(real), True) + adversarial_loss(module(fake), False)


def _init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, torch.nn.InstanceNorm2d) and m.weight is not None:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def _persist_checkpoint(config: TrainingConfig, epoch: int, model: CycleGAN, opt_g, opt_d, sched_g, sched_d) -> None:
    state = {
        "epoch": epoch,
        "model_gen_f2o": model.gen_f2o.state_dict(),
        "model_gen_o2f": model.gen_o2f.state_dict(),
        "model_disc_f": model.disc_f.state_dict(),
        "model_disc_o": model.disc_o.state_dict(),
        "optim_gen": opt_g.state_dict(),
        "optim_disc": opt_d.state_dict(),
        "sched_gen": sched_g.state_dict(),
        "sched_disc": sched_d.state_dict(),
    }
    save_checkpoint(state, Path(config.paths.weights) / f"cycle_gan_epoch_{epoch+1:03d}.pth")
