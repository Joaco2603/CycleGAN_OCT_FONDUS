# Contributing guide (short)

This repo researches an unpaired CycleGAN Fundus↔OCT. Keep changes small, tested, and clearly described.

## Workflow
- Branches: `feat/`, `fix/`, `docs/`, `exp/` (experiments).
- Commits: clear and atomic; reference area (data, model, training).
- PRs: small; include a short description, brief results (logs/images), and known risks.

## PR checklist
- [ ] Local smoke test (small subset) passes.  
- [ ] Checkpoints save and load.  
- [ ] Attach 2–4 samples in `generated/` or link to logs.  
- [ ] Docs updated if flows/data/artifacts change.  
- [ ] Relative paths and coherent names (`{model}_{domain}_v{N}.pth`).

## Style and quality
- Pythonic; optional typing when it clarifies.  
- Avoid heavy dependencies unless justified.  
- Centralized config; no absolute paths.  
- Lint/format if configured; comments only when they add value.

## Sensitive changes (need review)
- Architectures (G/D), losses, normalizations.  
- Training/optimization scheme.  
- Checkpoint IO or data layout.

## How to test
- Minimal subset per domain; 1–2 epochs.  
- Verify losses: adversarial, cycle F↔O, identity (if used).  
- Export at least 4 samples per domain to `generated/`.  
- Save and reload a checkpoint.

## PR resolution
- Prioritize correctness and reproducibility.  
- Share results and known limitations.  
- Use `copilot-guardrails.md` for ethics/security.

—  
Last updated: 2025-11-12

