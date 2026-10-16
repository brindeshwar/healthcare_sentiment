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
- Review-level student + full-corpus inference (ADR-0022): `extraction.review_heads` (discontinued / effectiveness / stop reasons, one 17-output classifier), `extraction.infer` (rule-based seed choice, resumable chunks with ID verification) -> 523,161 mentions in 263,862 reviews.
- Phase 3 encoder student (ADR-0021): `extraction.bio` (span/BIO conversion), `extraction.train` (BioClinical ModernBERT token classifier, 3 seeds, CADEC eval), `extraction.gliner_score` + `scripts/gliner_predict.py` (zero-shot GLiNER-BioMed in an isolated venv), `extraction.report` (comparison table). CADEC test relaxed F1: student 0.795 ± 0.003, teacher 0.806, GLiNER 0.766.
- Phase 2 LLM teacher: `labeling.benchmark` (CADEC strict/relaxed span F1, bootstrap CIs, paired comparisons), prompt v2 / guideline + schema v1.1 (ADR-0019), teacher gemma4:e4b; CADEC test relaxed F1 0.806. `labeling.silver`: 10K silver labels with provenance (ADR-0020); robust retries for Ollama server errors.
- External expert gold (ADR-0018): CADEC v2 download (checksum-pinned), loader with discontinuous spans and multi-code MedDRA parsing, fixed dev/test split; `evaluation.spans` strict/relaxed span P/R/F1; PsyTAR as manual source.
- Gold-set tooling: `labeling.sample_gold` (100 dev from val / 200 test from test, committed IDs), `labeling.llm` (Ollama structured output, retries on invalid spans), `labeling.preannotate` (resumable; optional lenient span fallback for pre-annotation), `labeling.labelstudio` (interface generated from schema + SDK-validated, task export with predictions, export import). ADR-0017.
- Annotation guidelines and schema v1.0 (`docs/annotation_guidelines.md`, `medsignal.labeling.schema`; span alignment with word boundaries; guideline examples as tests). ADR-0016.
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
