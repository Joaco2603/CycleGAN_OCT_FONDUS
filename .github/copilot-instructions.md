# GitHub Copilot Instructions - OCT_GANs

## Overview

This directory contains comprehensive instructions for AI coding assistants working on the OCT_GANs repository. These documents ensure consistent, high-quality contributions aligned with project standards.

## 📚 Documentation Structure

### Core Instructions
   - Project overview and key locations
   - Developer workflows and conventions
   - Use a maximum of words for code of two hundred
   - Quick reference for common tasks

### Specialized Guides

2. **[copilot-arquitecture.md](copilot-arquitecture.md)**
   - System architecture and design patterns
   - ProGAN model details (Generator, Discriminator)
   - Training pipeline and data flow
   - Performance optimization strategies

3. **[copilot-tests.md](copilot-tests.md)**
   - Testing philosophy and levels (smoke, unit, integration)
   - Test examples for model, data, checkpoints
   - GPU testing best practices
   - Coverage goals and common failures

4. **[copilot-documentation.md](copilot-documentation.md)**
   - Documentation standards and templates
   - README structure and style guide
   - Inline documentation (docstrings, comments)
   - Bilingual documentation (EN/ES)

5. **[copilot-contributing.md](copilot-contributing.md)**
   - Contribution workflow and branching strategy
   - PR process and review guidelines
   - Coding standards (PEP 8, type hints)
   - Recognition and community guidelines

6. **[copilot-guardrails.md](copilot-guardrails.md)**
   - Security policies and vulnerability reporting
   - Medical ethics and HIPAA/GDPR compliance
   - Intellectual property and licensing
   - AI assistant usage guidelines

## 🧠 Model Routing Intelligence
# .github — Quick summary (CycleGAN Fundus↔OCT)

This directory contains operational documentation for AI assistants and collaborators. Project goal: train an unpaired CycleGAN to translate between retinal fundus photos (Fundus) and OCT images for research.

## What's here
- `copilot-instructions.md` — Short guide for AI assistants (routines, flows, conventions).
- `copilot-guardrails.md` — Security, ethics, and IP rules.
- `copilot-contributing.md` — Contribution workflow and PRs.
- `copilot-documentation.md` — Documentation standards (concise and focused).
- `copilot-tests.md` — Minimal tests and smoke checks.
- `copilot-arquitecture.md` — Architecture details (left unchanged).

## Key points (condensed)
- Data: unpaired Fundus↔OCT training. Keep splits consistent and normalizations aligned between domains.
- Model: CycleGAN with adversarial, cycle-consistency (L1), and optional identity losses. Avoid risky changes without a smoke test.
- Artifacts: weights in `weights/`, samples in `generated/`, logs in `logs/` (use descriptive names and dates).
- System: Windows + PowerShell. Prefer PyTorch with CUDA if a GPU is available; gracefully fall back to CPU.
- AI assistants: follow `copilot-instructions.md` and respect `copilot-guardrails.md`.

## When to update
Update these files when the training/inference flow, data layout, artifact paths, or security policies change.

—
Maintainer: OCT CycleGAN Team · Last updated: 2025-11-12