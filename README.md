# CycleGAN OCT ↔ Fundus

Modular CycleGAN pipeline for unsupervised translation between retinal fundus photographs and OCT scans.

## Features

- ✅ Unpaired domain translation (Fundus ↔ OCT)
- 🌡️ **GPU temperature control** - Automatic throttling to protect hardware
- 📊 TensorBoard monitoring with sample visualizations
- 💾 Automatic checkpointing and resume capability
- ⚡ Mixed precision training (AMP) support

## Quick start

1. Install dependencies:
   ```powershell
   pwsh scripts/setup_env.ps1
   ```
2. Define paths and hyperparameters in `config.yaml`.
3. Split each domain into train/val/test:
   ```powershell
   python scripts/split_dataset.py "data/EYE FUNDUS" dataset/fundus --ratios 0.7 0.2 0.1 --copy
   python scripts/split_dataset.py data/OCT dataset/oct --ratios 0.7 0.2 0.1 --copy
   ```
4. Train the model:
   ```powershell
   python train.py --config config.yaml
   ```
5. Generate samples:
   ```powershell
   python generate.py weights/cycle_gan_epoch_200.pth samples/fundus/*.png --direction fundus_to_oct
   ```

## Layout

- `data/` — unpaired Fundus/OCT images organised by split.
- `src/` — data pipelines, models, training, and inference utilities.
- `scripts/` — setup, dataset splitting, quick previews.
- `weights/`, `generated/`, `logs/` — checkpoints, outputs, TensorBoard runs.

## Notes

- Configuration, losses, and logging defaults follow the project guardrails.
- Use `TensorBoard --logdir logs` to monitor learning curves and reconstructions.
- **GPU temperature control**: Set `max_gpu_temp` in `config.yaml` to prevent overheating (see `docs/GPU_TEMPERATURE.md`)
- Add smoke tests from `.github/copilot-tests.md` before pushing major changes.
