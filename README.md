# CycleGAN OCT ↔ Fundus

Modular CycleGAN pipeline for unsupervised translation between retinal fundus photographs and OCT scans.

## Features

- ✅ Unpaired domain translation (Fundus ↔ OCT)
- 🌡️ **GPU temperature control** - Automatic throttling to protect hardware
- 📊 TensorBoard monitoring with sample visualizations
- 💾 Automatic checkpointing and resume capability
- ⚡ Mixed precision training (AMP) support
- 📈 **MLflow integration** - Track metrics and experiments
- 🗂️ **DVC integration** - Version datasets

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
   # Basic training
   python train.py --config config.yaml
   
   # With MLflow metrics tracking
   python train.py --config config.yaml --track_metrics
   
   # With dataset versioning (DVC)
   python train.py --config config.yaml --track_dataset
   
   # Full tracking
   python train.py --config config.yaml --track_metrics --track_dataset --experiment my_exp
   ```
5. Generate samples:
   ```powershell
   python generate.py weights/cycle_gan_epoch_200.pth samples/fundus/*.png --direction fundus_to_oct

   # Fundus -> OCT (adjust the path of the image if you want another one)
   python generate.py "weights/cycle_gan_epoch_010.pth" "dataset/fundus/val/EYE FUNDUS1/0002_OD_f_1.jpg" --config config.yaml --direction fundus_to_oct --output "generated/quick_demo"

   # OCT -> Fundus
   python generate.py "weights/cycle_gan_epoch_010.pth" "dataset/oct/val/OCT1/1221_OD_o_2.jpg" --config config.yaml --direction oct_to_fundus --output "generated/quick_demo"
   ```
6. View experiment results:
   ```powershell
   mlflow ui
   # Open http://localhost:5000
   ```

# 1. Create folder structure
New-Item -ItemType Directory -Force -Path "dataset/fundus/train", "dataset/fundus/val", "dataset/fundus/test"
New-Item -ItemType Directory -Force -Path "dataset/oct/train", "dataset/oct/val", "dataset/oct/test"

# 2. Copy original images
Copy-Item -Recurse "data/EYE FUNDUS/*" -Destination "dataset/fundus/train/"
Copy-Item -Recurse "data/OCT/*" -Destination "dataset/oct/train/"

# 3. Verify count
(Get-ChildItem -Recurse "dataset/fundus/train" -Filter "*.jpg").Count
(Get-ChildItem -Recurse "dataset/oct/train" -Filter "*.jpg").Count

# 4. Filter fundus (dry-run first)
python scripts/filter_quality.py --root dataset/fundus/train --domain fundus --dry-run

# 5. Filter OCT (dry-run first)
python scripts/filter_quality.py --root dataset/oct/train --domain oct --dry-run

- Analyze dataset (see statistics before filtering)
python scripts/analyze_oct_dataset.py --root dataset/oct/train

- Preview what would be filtered (dry-run)
python scripts/filter_quality.py --root dataset/oct/train --domain oct --dry-run

- Strict filtering (recommended)
python scripts/filter_quality.py --root dataset/oct/train --domain oct --strict --dry-run

- Move bad images to quarantine
python scripts/filter_quality.py --root dataset/oct/train --domain oct --strict --quarantine

# 6. Apply filter (move to quarantine)
python scripts/filter_quality.py --root dataset/fundus/train --domain fundus --quarantine
python scripts/filter_quality.py --root dataset/oct/train --domain oct --quarantine

# 7. Generate results
python generate.py weights/cycle_gan_epoch_040.pth "data/EYE FUNDUS/EYE FUNDUS1/0001_OD_f_1.jpg" --direction fundus_to_oct --output generated/test_results

python generate.py weights/cycle_gan_epoch_040.pth "data/OCT/OCT1/1221_OD_o_2.jpg" --direction oct_to_fundus --output generated/test_results

python generate.py weights/cycle_gan_epoch_040.pth "data/OCT/OCT1/1221_OD_o_2.jpg" "data/OCT/OCT1/1222_OD_o_2.jpg" "data/OCT/OCT1/1223_OD_o_1.jpg" --direction oct_to_fundus --output generated/test_results_oct2fundus

## Layout

- `data/` — unpaired Fundus/OCT images organised by split.
- `src/` — data pipelines, models, training, and inference utilities.
- `src/tracking/` — experiment tracking (MLflow metrics, DVC datasets).
- `scripts/` — setup, dataset splitting, quick previews.
- `weights/`, `generated/`, `logs/` — checkpoints, outputs, TensorBoard runs.
- `mlruns/` — MLflow experiment logs.
- `logs/dataset_versions/` — dataset version snapshots per run.

## Notes

- Configuration, losses, and logging defaults follow the project guardrails.
- Use `tensorboard --logdir logs` to monitor learning curves and reconstructions.
- **GPU temperature control**: Set `max_gpu_temp` in `config.yaml` to prevent overheating (see `docs/GPU_TEMPERATURE.md`)
- Add smoke tests from `.github/copilot-tests.md` before pushing major changes.
- MLflow logs are stored in `mlruns/` (local by default).
- Dataset versions are tracked in `logs/dataset_versions/` for reproducibility.
