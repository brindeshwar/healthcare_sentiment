# Status

> **Hand-off file.** Every agent/person reads this first and updates it last (see AGENTS.md §2-3).
> Keep "Now" short and current; append to the Session log, never rewrite history.

**Current phase:** Phase 0 (foundation) → starting Phase 1
**Branch:** `rebuild`
**Last updated:** 2026-10-13

## Now (in progress)

- Phase 0 wrap-up: local checks pass (ruff, mypy, pytest 5/5, pre-commit). Changes staged on `rebuild`, **not committed/pushed** (awaiting owner). CI runs on first push.

## Next (ordered)

1. Owner: install Ollama; create HF + Kaggle accounts; request openFDA API key.
2. Owner approval needed: download UCI Drugs.com zip (~40 MB) into `data/raw/drugscom/`.
3. Phase 1: data download scripts → cleaning → de-duplication → splits → EDA.
4. Phase 1: annotation guidelines + annotation tool + LLM pre-annotation.

## Blockers / waiting on owner

- Ollama not installed (needed from Phase 1 pre-annotation onward).
- WebMD dataset requires Kaggle login (owner downloads).

## Known facts about the data (verified 2026-10-13)

- Local `drugsComTrain_raw.csv`: 161,297 rows (official test file, 53,766 rows, missing).
- 48,879 duplicate (review, rating) rows (~30%); 62% of reviews contain HTML entities (`&#039;`).
- Rating distribution skewed: 10 stars = 32%, 1 star = 13%.
- Legacy data still at `healthcare_sentiment/` (untracked); will be moved to `data/raw/` in Phase 1.

## Session log

### 2026-10-13: Claude Code (Opus 5.5)
- Reviewed legacy project: found simulated confusion matrix, wrong class stats in report, train/test leakage via duplicates, unused LR config, HTML entities not decoded.
- Researched field (2025-26) and chose direction: Option 3 = ADE extraction + discontinuation reasons vs FAERS/labels (see DECISIONS.md, docs/RESEARCH.md).
- Tagged legacy as `v0-minor-project`; created `rebuild` branch; removed legacy code from branch.
- Created agent docs, management docs, repo tooling, CI, GitHub templates, package skeleton.
- Verified locally: `uv sync`, ruff, mypy (strict), pytest, pre-commit all pass; pre-commit hook installed.
- Next: owner setup tasks, then Phase 1.
