# Architecture Guide - OCT CycleGAN

## System overview

Unpaired CycleGAN translating between retinal Fundus photos and OCT images. Two generators learn F→O and O→F mappings with cycle-consistency; two PatchGAN discriminators judge realism in each domain.

## High-level architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       OCT CycleGAN                          │
├─────────────────────────────────────────────────────────────┤
│  Data (unpaired)  →  Training Core  →  CycleGAN (F↔O)  →  Logging │
│                    (losses, sched, AMP)  (G_F→O,G_O→F; D_F,D_O)   │
└─────────────────────────────────────────────────────────────┘
```

## Core components

### 1) Data layer (unpaired)

Purpose: load Fundus and OCT images independently with consistent normalizations.

- Suggested layout
  - `data/fundus/{train,val}/...` (RGB)
  - `data/oct/{train,val}/...` (grayscale or RGB; convert to 3ch if needed)
- Transforms (both domains)
  - Resize to `img_size` (e.g., 256)
  - ToTensor; Normalize mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] → [-1,1]
- Data flow: files → datasets (unpaired) → balanced batches → device

### 2) CycleGAN model

- Generators: `G_F→O`, `G_O→F`
  - Architecture: ResNet-9 blocks (256×256) or U-Net; in/out: 3×H×W → 3×H×W
  - InstanceNorm, reflection padding to reduce artifacts; last layer: Tanh
- Discriminators: `D_F`, `D_O`
  - 70×70 PatchGAN; conv blocks with LeakyReLU + InstanceNorm (no norm on first)
  - Output: patch real/fake map (not single scalar)

### 3) Losses

- Adversarial (per domain)
  - LSGAN recommended for stability; target labels: real=1, fake=0
  - L_adv = E[(D(real)-1)^2] + E[(D(fake))^2]
- Cycle-consistency
  - L_cyc = ||G_O→F(G_F→O(x_F)) - x_F||_1 + ||G_F→O(G_O→F(x_O)) - x_O||_1
  - Weight: λ_cyc ≈ 10
- Identity (optional)
  - L_id = ||G_F→O(x_O)||_1 + ||G_O→F(x_F)||_1 (when fed same-domain)
  - Weight: λ_id ≈ 0.5 · λ_cyc (helps color/intensity preservation)

### 4) Training system

Per batch (unpaired x_F, x_O):
1. Discriminators
   - Update `D_F` with real x_F and fake F̂ = G_O→F(x_O)
   - Update `D_O` with real x_O and fake Ô = G_F→O(x_F)
2. Generators (joint step)
   - Recompute Ô, F̂ with current G; minimize L_adv(G) + λ_cyc·L_cyc + λ_id·L_id

Notes
- Optimizer: Adam(lr=2e-4, betas=(0.5, 0.999))
- Mixed precision (AMP) on CUDA; gradient accumulation if memory is tight
- Image buffers (replay) optional; keep pipeline simple unless instability appears

### 5) Monitoring & logging

- TensorBoard scalars: adv/cycle/identity losses per domain, total G/D
- Image grids every N steps: x_F, Ô, recon F↔, x_O, F̂, recon O↔
- Directory layout: `logs/` for events; `generated/` for periodic samples

### 6) Checkpoints

Save after each epoch and on interrupt:

```
weights/
  G_FtoO_v{N}.pth  # state_dict, epoch, lr, λ
  G_OtoF_v{N}.pth
  D_F_v{N}.pth
  D_O_v{N}.pth
```

Include: model/optimizer states, epoch, global_step, configs. Use descriptive names; avoid absolute paths.

### 7) Configuration (centralized)

- Paths: `data/fundus`, `data/oct`, `weights/`, `generated/`, `logs/`
- Image size: 256 (start) — keep both domains equal
- Batch size: 1–4 (GPU dependent; RTX 3070: 4 fits with AMP)
- LR: 2e-4 (G and D), betas=(0.5, 0.999)
- Weights: λ_cyc=10, λ_id=5 (or 0 to disable identity)
- Device: CUDA if available; fall back to CPU gracefully (Windows PowerShell env)

### 8) Generation flow

1) Load `G_F→O` (or `G_O→F`) checkpoint; set `eval()`
2) Read and normalize input image(s) to [-1,1]
3) Forward through generator; no gradients
4) Denormalize to [0,1]/[0,255]; save to `generated/`

### 9) Scalability & performance

- RTX 3070 (8GB): 256×256 with batch=4 works with AMP; reduce to 1–2 on CPU
- Use `num_workers=2`, `pin_memory=True`; prefetch and cache augmentations if needed
- Prefer InstanceNorm over BatchNorm for small batches

### 10) Error handling & safeguards

- OOM: lower batch, enable AMP, reduce image size
- NaN/Inf: lower LR, clamp losses, check normalization; disable identity temporarily
- Mode collapse: add image pool, tune λ_id/λ_cyc, verify discriminator capacity
- Emergency save on SIGINT; validate checkpoint load after refactors

---

Related documents
- `copilot-instructions.md` — main development guide
- `copilot-tests.md` — testing procedures
- `copilot-documentation.md` — documentation standards

Last updated: 2025-11-12