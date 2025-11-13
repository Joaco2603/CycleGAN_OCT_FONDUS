# Security & ethics guardrails (short version)

Applies to the entire CycleGAN Fundus↔OCT repository. Prioritize security, privacy, and legal compliance.

## Security & dependencies
- No absolute paths or unvalidated user inputs.
- Store checkpoints only in `weights/`; validate file size and extension.
- No secrets in the repository; use environment variables and keep `.env` ignored.
- Pin versions in `requirements.txt`; perform periodic audits.

## Privacy & medical ethics
- Use de-identified datasets; OCT2017 is acceptable.
- PHI/PII is prohibited; for private datasets: de-identify and control access.
- Permitted use: research and education; not for clinical diagnosis.
- Label synthetic data clearly and disclose limitations on use.

## Intellectual property
- Respect third-party licenses and add attribution when applicable.
- Ensure license compatibility (MIT/BSD/Apache recommended).
- Cite OCT2017 if used (Kermany et al., doi:10.17632/rscbjbr9sj.2).

## Quick review checklist
- [ ] Paths validated (no traversal) and no secrets committed.
- [ ] File sizes and formats controlled.
- [ ] Dependencies pinned and audited.
- [ ] Error messages do not leak sensitive information.
- [ ] No unjustified network/IO calls.

## AI assistants
- Allowed: refactoring, documentation, tests, and utility code.
- Not allowed: bypassing license or guardrails, misleading documentation, or using proprietary data.
- AI-generated code: must be reviewed, tested, and have risks documented.

## Incidents
1) Contain. 2) Assess impact. 3) Notify maintainers. 4) Remediate and document.

—  
Last updated: 2025-11-12
