# Tests and validation (short version)

Practical plan for an unpaired CycleGAN Fundus↔OCT. Goal: catch breakages fast and validate outputs.

## Test levels
- Smoke (30–60s): env OK, data readable, loop starts and saves 1 checkpoint.  
- Unit (≈5m): shapes/outputs for key functions (losses, norms, IO).  
- Integration (≤30m): 1–2 epochs on a small subset; export 4–8 samples per domain.

## Minimal smoke criteria
- CUDA available or clean CPU fallback.  
- Both domains detected.  
- Logs in `logs/` and samples in `generated/`.  
- Create and reload a `.pth` in `weights/`.

## CycleGAN checks
- Losses: `adv`, `cycle_F`, `cycle_O`, `identity` (if used) behave across steps.  
- Matched sizes/normalization between domains.  
- Stability: no NaN/Inf; reasonable gradients.  
- Checkpoint load/compatibility after refactors.

## Test skeleton
- Data: loader returns [B,C,H,W] normalized to [-1,1].  
- Model: `G_F→O(x_F)`, `G_O→F(x_O)` → expected sizes.  
- Losses: finite; trend improves over ~10 synthetic steps.  
- Checkpoint: `save→load` preserves state (allclose on weights).

## Execution rules
- Skip GPU tests if no CUDA; free memory after tests.  
- No absolute paths; use tiny fixtures/datasets.  
- Limit runtimes; mark long tests as "slow".

## Manual acceptance
- 4–8 samples/domain: no obvious artifacts, variety, anatomically plausible.  
- Coherent losses; no mode collapse.

—  
Last updated: 2025-11-12

## Test Levels

### 1. Smoke Tests (30 seconds)

**Purpose**: Quick sanity check before committing code

**Run**: Before every commit
```powershell
# From repository root
python ProGAN/config/check_gpu.py
python -c "import torch; print(torch.cuda.is_available())"
```

**Validates**:
- Python environment is functional
- PyTorch with CUDA is installed
- GPU is accessible

### 2. Unit Tests (5 minutes)

**Purpose**: Test individual components in isolation

#### Model Architecture Tests

```python
# test_generator.py
import torch
from ProGAN.model.progan_local import Generator

def test_generator_output_shape():
    """Generator should produce correct output dimensions"""
    gen = Generator(Z_DIM=256, IN_CHANNELS=256, img_channels=3)
    z = torch.randn(4, 256, 1, 1)
    
    # Test each resolution
    for step in range(9):  # 4x4 to 512x512
        out = gen(z, alpha=1.0, steps=step)
        expected_size = 4 * (2 ** step)
        assert out.shape == (4, 3, expected_size, expected_size), \
            f"Step {step}: Expected {expected_size}x{expected_size}, got {out.shape}"

def test_generator_fade_in():
    """Alpha parameter should smoothly blend resolutions"""
    gen = Generator(Z_DIM=256, IN_CHANNELS=256, img_channels=3)
    z = torch.randn(1, 256, 1, 1)
    
    out_alpha_0 = gen(z, alpha=0.0, steps=1)
    out_alpha_1 = gen(z, alpha=1.0, steps=1)
    out_alpha_half = gen(z, alpha=0.5, steps=1)
    
    # Output at alpha=0.5 should be between alpha=0 and alpha=1
    assert not torch.allclose(out_alpha_0, out_alpha_1, atol=1e-3)
    assert torch.all(out_alpha_half != out_alpha_0)
    assert torch.all(out_alpha_half != out_alpha_1)

def test_pixel_norm():
    """PixelNorm should normalize feature vectors"""
    from ProGAN.model.progan_local import PixelNorm
    norm = PixelNorm()
    
    x = torch.randn(16, 256, 8, 8)
    out = norm(x)
    
    # Check per-pixel norm
    per_pixel_norm = (out ** 2).mean(dim=1, keepdim=True).sqrt()
    assert torch.allclose(per_pixel_norm, torch.ones_like(per_pixel_norm), atol=1e-5)
```

#### Data Loading Tests

```python
# test_data_loader.py
from pathlib import Path
from ProGAN.model.progan_local import get_loader

def test_loader_creates_correct_batches():
    """DataLoader should produce batches of correct size"""
    data_dir = Path("ProGAN/data/OCT2017/train")
    
    if not data_dir.exists():
        pytest.skip("Dataset not available")
    
    loader = get_loader(data_dir, image_size=128, batch_size=4)
    batch, _ = next(iter(loader))
    
    assert batch.shape[0] <= 4  # Batch size
    assert batch.shape[1] == 3  # RGB channels
    assert batch.shape[2] == 128  # Height
    assert batch.shape[3] == 128  # Width
    assert batch.min() >= -1.0 and batch.max() <= 1.0  # Normalized

def test_loader_handles_different_resolutions():
    """DataLoader should support all progressive resolutions"""
    data_dir = Path("ProGAN/data/OCT2017/train")
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512]
    
    for res in resolutions:
        loader = get_loader(data_dir, image_size=res, batch_size=2)
        batch, _ = next(iter(loader))
        assert batch.shape[2] == res and batch.shape[3] == res
```

#### Checkpoint Tests

```python
# test_checkpoints.py
import torch
from ProGAN.model.progan_local import save_checkpoint, load_checkpoint, Generator
import tempfile

def test_checkpoint_save_and_load():
    """Checkpoints should preserve model state exactly"""
    gen = Generator(Z_DIM=256, IN_CHANNELS=256, img_channels=3)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)
    
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name
    
    # Save
    save_checkpoint(gen, opt, checkpoint_path)
    
    # Modify model
    gen_new = Generator(Z_DIM=256, IN_CHANNELS=256, img_channels=3)
    opt_new = torch.optim.Adam(gen_new.parameters(), lr=1e-3)
    
    # Load
    load_checkpoint(checkpoint_path, gen_new, opt_new, lr=1e-3)
    
    # Compare state dicts
    for (k1, v1), (k2, v2) in zip(gen.state_dict().items(), gen_new.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2)

def test_emergency_save_creates_files():
    """Emergency save should create timestamped checkpoints"""
    import signal
    from ProGAN.model.progan_local import setup_emergency_save
    
    gen = Generator(Z_DIM=256, IN_CHANNELS=256, img_channels=3)
    critic = Generator(Z_DIM=256, IN_CHANNELS=256, img_channels=3)  # Placeholder
    opt_gen = torch.optim.Adam(gen.parameters())
    opt_critic = torch.optim.Adam(critic.parameters())
    
    setup_emergency_save(gen, critic, opt_gen, opt_critic)
    
    # Simulate Ctrl+C (don't actually send signal in test)
    # Just verify handler is registered
    assert signal.getsignal(signal.SIGINT) != signal.SIG_DFL
```

### 3. Integration Tests (30 minutes)

**Purpose**: Test complete workflows end-to-end

#### Training Pipeline Test

```python
# test_training_integration.py
import torch
from ProGAN.model.progan_local import train_progressive_gan

def test_one_epoch_training():
    """Full training loop should run for 1 epoch without errors"""
    config = {
        'START_TRAIN_AT_IMG_SIZE': 4,
        'PROGRESSIVE_EPOCHS': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'BATCH_SIZES': [16, 16, 12, 8, 6, 4, 3, 2, 1],
        'LEARNING_RATE': 1e-3,
        'Z_DIM': 256,
        'SAVE_MODEL': False,  # Don't clutter weights directory
        'LOAD_MODEL': False
    }
    
    # Run training (should complete without exceptions)
    train_progressive_gan(**config)
    
    # Verify GPU memory was released
    torch.cuda.empty_cache()
    memory_after = torch.cuda.memory_allocated() / 1024**3
    assert memory_after < 1.0, f"Memory leak detected: {memory_after:.2f} GB still allocated"

def test_progressive_resolution_increase():
    """Training should progress through all resolutions"""
    # This test runs a mini training session and checks that
    # the model successfully transitions from 4x4 to 8x8
    pass  # Implementation details
```

#### Generation Pipeline Test

```python
# test_generation_integration.py
def test_generate_images_from_checkpoint():
    """Should generate images from saved checkpoint"""
    from ProGAN.model.progan_local import generate_samples
    
    checkpoint_path = "weights/generator_DRUSEN_v11.pth"
    
    if not Path(checkpoint_path).exists():
        pytest.skip("Pretrained checkpoint not available")
    
    images = generate_samples(
        checkpoint_path=checkpoint_path,
        num_samples=8,
        resolution=512
    )
    
    assert images.shape == (8, 3, 512, 512)
    assert images.min() >= -1.0 and images.max() <= 1.0
```

### 4. Performance Tests (1 hour)

**Purpose**: Validate GPU memory and speed constraints

```python
# test_performance.py
import torch
import time

def test_memory_usage_within_limits():
    """Training should not exceed 8GB VRAM (RTX 3070)"""
    from ProGAN.model.progan_local import Generator, Discriminator
    
    device = torch.device("cuda:0")
    batch_sizes = [16, 16, 12, 8, 6, 4, 3, 2, 1]
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512]
    
    for res, batch_size in zip(resolutions, batch_sizes):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        gen = Generator(Z_DIM=256, IN_CHANNELS=256, img_channels=3).to(device)
        disc = Discriminator(IN_CHANNELS=256, img_channels=3).to(device)
        
        z = torch.randn(batch_size, 256, 1, 1).to(device)
        fake = gen(z, alpha=1.0, steps=resolutions.index(res))
        loss = disc(fake, alpha=1.0, steps=resolutions.index(res)).mean()
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        # RTX 3070 has 8GB, leave 0.5GB buffer
        assert peak_memory < 7.5, \
            f"Resolution {res}x{res} exceeded memory: {peak_memory:.2f} GB"
        
        del gen, disc, z, fake, loss

def test_training_speed_baseline():
    """Measure time per epoch for performance regression"""
    # Run 1 epoch at 128x128 and ensure it completes in < 25 minutes
    pass
```

### 5. Acceptance Tests (Manual)

**Purpose**: Human validation of generated images

#### Quality Checklist

- [ ] Generated images are sharp (not blurry)
- [ ] DRUSEN pathology is visible and realistic
- [ ] No obvious artifacts (checkerboard patterns, color banding)
- [ ] Variety in generated samples (not mode collapse)
- [ ] Resolution is 512×512 as expected
- [ ] Images are anatomically plausible

#### Example Test Protocol

```powershell
# Generate 100 samples
python ProGAN/model/progan_local.py generate --num-samples 100

# Review in image viewer
explorer ProGAN\generated_images\

# Compare with real dataset
explorer ProGAN\data\OCT2017\train\DRUSEN\
```

## Test Infrastructure

### Running Tests

```powershell
# Install test dependencies
pip install pytest pytest-cov pytest-timeout

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ProGAN --cov-report=html

# Run specific test file
pytest tests/test_generator.py -v

# Run tests matching pattern
pytest tests/ -k "test_generator" -v
```

### Continuous Testing (Planned)

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
          pip install pytest numpy pillow
      - name: Run tests
        run: pytest tests/ -v --ignore=tests/test_performance.py
```

## Test Data Management

### Fixtures

```python
# conftest.py
import pytest
import torch

@pytest.fixture
def sample_noise():
    """Fixed noise vector for reproducibility"""
    torch.manual_seed(42)
    return torch.randn(8, 256, 1, 1)

@pytest.fixture
def tiny_generator():
    """Lightweight generator for fast tests"""
    from ProGAN.model.progan_local import Generator
    return Generator(Z_DIM=64, IN_CHANNELS=64, img_channels=3)

@pytest.fixture
def mock_oct_image():
    """Synthetic OCT image for testing"""
    return torch.randn(1, 3, 512, 512)
```

### Test Dataset

Create a minimal test dataset:
```
tests/data/
  └── test_oct/
      └── DRUSEN/
          ├── test_001.jpg
          ├── test_002.jpg
          └── test_003.jpg
```

## Debugging Tests

### Verbose Mode

```powershell
pytest tests/test_generator.py -v -s  # Show print statements
```

### Debugging Individual Tests

```powershell
pytest tests/test_generator.py::test_generator_output_shape --pdb
```

### Visual Debugging

```python
# In test function
import matplotlib.pyplot as plt

def test_generator_visual_debug():
    gen = Generator(...)
    z = torch.randn(1, 256, 1, 1)
    out = gen(z, alpha=1.0, steps=5)
    
    # Save for inspection
    plt.imshow(out[0].permute(1, 2, 0).detach().numpy())
    plt.savefig('tests/debug_output.png')
```

## GPU Testing Best Practices

### 1. Memory Management

```python
@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Automatically clear GPU memory after each test"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 2. Skip Tests Without GPU

```python
@pytest.mark.skipif(not torch.cuda.is_available(), 
                    reason="CUDA not available")
def test_gpu_training():
    # Test that requires GPU
    pass
```

### 3. Temperature Monitoring

```python
def test_training_doesnt_overheat():
    """Ensure GPU stays below 85°C during test"""
    import subprocess
    
    # Run training
    # ...
    
    # Check temperature
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=temperature.gpu', 
         '--format=csv,noheader'],
        capture_output=True
    )
    temp = float(result.stdout.decode().strip())
    
    assert temp < 85, f"GPU overheated: {temp}°C"
```

## Test Coverage Goals

| Component           | Current | Target |
|--------------------|---------|--------|
| Model Architecture | 0%      | 80%    |
| Data Loading       | 0%      | 90%    |
| Training Loop      | 0%      | 60%    |
| Checkpointing      | 0%      | 95%    |
| Utilities          | 0%      | 70%    |
| **Overall**        | **0%**  | **75%**|

## Common Test Failures

### 1. CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Fix**: Reduce batch size in test configuration

### 2. Numerical Instability

**Symptom**: `AssertionError: NaN detected in loss`

**Fix**: Add gradient clipping, reduce learning rate

### 3. Checkpoint Incompatibility

**Symptom**: `KeyError` when loading checkpoint

**Fix**: Regenerate test checkpoints with current model version

## Test Documentation

Each test should include:

```python
def test_example():
    """
    One-line summary of what is being tested.
    
    Given: Initial conditions
    When: Action performed
    Then: Expected outcome
    
    Validates: Specific requirement or bug fix
    """
    # Arrange
    setup_code()
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected_value
```

---

**Related Documents**:
- `copilot-arquitecture.md` - System design
- `copilot-instructions.md` - Development guide
- `copilot-contributing.md` - How to submit tested code

**Last Updated**: November 12, 2025