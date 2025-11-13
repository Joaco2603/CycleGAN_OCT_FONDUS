# Documentation Standards (short version)

Goal: minimal, useful, and up-to-date documentation for an unpaired CycleGAN Fundus↔OCT.

## Principles
- Concise and actionable (short examples, no noise).
- Synchronized with code; update docs in the same PR as the change.
- English primary; keep terms consistent across files.

## Where things belong
- `README.md` (root): project summary and quick start.
- `.github/` (these files): operational guides and policies.
- `docs/` (if present): extended guides (only if necessary).
- Docstrings: API and non-obvious design decisions in code.

## Minimums by doc type
- Module/script README: purpose (1–2 lines), inputs/outputs, how to run (1 block), limitations.
- Guides: H2/H3 sections, short lists, max 1 image/table per topic.
- Examples: only runnable examples; avoid fictitious commands.

## When to update
- Changes in data/layout, losses/architecture, artifact paths, CLI/flags, or system requirements.

## Style
- Clear headings, short sentences, no unnecessary jargon.
- Self-contained code blocks; one command per line.
- Avoid absolute paths and unjustified dependencies.

## Review checklist
- [ ] Commands work when copy-pasted (Windows PowerShell).  
- [ ] Relative, consistent paths.  
- [ ] Examples produce logs and 2–4 samples in `generated/`.  
- [ ] Docstrings for critical public functions.  
- [ ] Last updated date at the end.

## CycleGAN terminology (consistent)
- Generators: `G_F→O`, `G_O→F`; Discriminators: `D_F`, `D_O`.
- Losses: `adv`, `cycle_F`, `cycle_O`, `identity` (optional).
- Artifacts: `weights/`, `generated/`, `logs/`.

—
Last updated: 2025-11-12