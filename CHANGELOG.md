# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Project rebuilt as **MedSignal** (drug-safety signals from patient reviews); see `docs/DECISIONS.md` ADR-0001/0002.
- Agent and management docs: `AGENTS.md`, `CLAUDE.md`, `docs/PLAN.md`, `docs/STATUS.md`, `docs/DECISIONS.md`, `docs/DATA.md`, `docs/RESEARCH.md`.
- Tooling: uv project (`pyproject.toml`), ruff, mypy, pytest, pre-commit, GitHub Actions CI, Dependabot, issue/PR templates.
- Package skeleton `src/medsignal` with config loading, seeding and git-SHA helpers.

### Added (Phase 1)
- `medsignal.data.download`: config-driven dataset download with pinned SHA-256 checksums and manifests (DrugRev, Druglib; manual-source checks for UCI Drugs.com and WebMD).
- ADR-0011: merged UCI Drugs.com + DrugRev corpus (UCI #462 withdrawn).

### Removed
- Legacy DistilBERT sentiment code, report and figures (preserved at tag `v0-minor-project`). The legacy confusion matrix was simulated, not measured, and is withdrawn.

## [0.0.1] - 2025-09-02
- Legacy minor project (tag `v0-minor-project`).
