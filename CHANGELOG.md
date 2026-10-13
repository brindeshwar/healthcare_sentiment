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
- `medsignal.data.build`: cleaning, within-source de-duplication, UCI↔DrugRev matching (prefers UCI full text), stable `review_id`s → `data/interim/reviews.parquet` (263,862 reviews) + build report.
- ADR-0012: all DrugRev-only rows flagged `possibly_truncated` (row-level detection measured as unreliable).
- Sentiment baselines (`medsignal.baselines`: classical, encoder, report) + `medsignal.evaluation.metrics`; `gpu` extra (torch 2.14.1+cu130). ADR-0015.
- `medsignal.data.eda`: generated data report (figures + summary.json + README) incl. token lengths with the student tokenizer (99.7% of reviews fit 256 tokens).
- ADR-0014: repair of 67 mid-word-truncated UCI condition names.
- `medsignal.utils.load_env`: loads `.env`, resolves relative cache paths inside the project.
- `medsignal.data.splits`: brand/generic drug groups (union-find), drug-held-out split + stratified train/val/test, leakage checks → `data/processed/splits.parquet` + report (ADR-0013).

### Changed
- All caches and local tools kept inside the project folder (uv `cache-dir`, `HF_HOME`, Ollama paths); owner checklist in `docs/SETUP.md`.

### Removed
- Legacy DistilBERT sentiment code, report and figures (preserved at tag `v0-minor-project`). The legacy confusion matrix was simulated, not measured, and is withdrawn.

## [0.0.1] - 2025-09-02
- Legacy minor project (tag `v0-minor-project`).
