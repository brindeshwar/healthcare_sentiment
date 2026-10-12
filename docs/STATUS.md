# Status

> **Hand-off file.** Every agent/person reads this first and updates it last (see AGENTS.md §2-3).
> Keep "Now" short and current; append to the Session log, never rewrite history.

**Current phase:** Phase 1 (data & baseline)
**Branch:** `feat/clean-merge-dedup`
**Last updated:** 2026-10-13

## Now (in progress)

- Issue #4 (cleaning + merge + dedup): done on `feat/clean-merge-dedup`; PR pending. 263,862 unique reviews (uci 18,534 / both 93,222 / drugrev 152,106).
- Dependabot PRs #16-#18: owner approved merging when green; rebases requested (branch protection requires up-to-date).

## Next (ordered)

1. Phase 1: splits (#5) → EDA (#6) → baselines (#7).
2. Phase 1: annotation guidelines (#8) + annotation tool + LLM pre-annotation (#9).
3. Owner (#2): install Ollama; HF + Kaggle accounts; openFDA key; download WebMD CSV to `data/raw/webmd/webmd.csv`.

## Blockers / waiting on owner

- Ollama not installed (needed from Phase 1 pre-annotation onward).
- WebMD dataset requires Kaggle login (owner downloads).

## Known facts about the data (verified 2026-10-13)

- **UCI #462 (Drugs.com) withdrawn** from UCI (403 / API not found). Using local train copy (161,297 rows) + DrugRev (Mendeley, CC BY 4.0, 392,510 rows, 2008-2023). See ADR-0011.
- DrugRev truncates ~28% of overlapping reviews (19% lose >50%); 37% duplicate texts; 25,036 rows with shifted `IntakeTime`; has `UserName` (drop at load).
- Local `drugsComTrain_raw.csv`: 161,297 rows (official test file unobtainable).
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
- Committed + pushed `rebuild` and tag `v0-minor-project`; opened PR #15.
- GitHub: description/topics, squash-only merges, auto-delete branches, labels (task/experiment/bug/phase-N/owner/blocked), 6 milestones, issues #1-#10 (Phase 0-1 tasks) + #11-#14 (Phase 2-5 epics), private vulnerability reporting, `main` protection (PR required, both CI checks required + up to date, linear history, no force-push/deletion, conversations resolved; admins may bypass).
- Next: CI green → owner merges #15; owner setup (#2); then Phase 1.

### 2026-10-13: Claude Code (Opus 5.5), session 2
- Renamed repo to `brindeshwar/medsignal`; updated URLs; CI green; squash-merged PR #15; closed #1.
- Found UCI #462 withdrawn; evaluated alternatives (DrugRev, Kaggle mirror, Druglib). Chose merge of UCI copy + DrugRev (ADR-0011) after measuring DrugRev truncation.
- Added `medsignal.data.download` (config-driven, sha256-pinned, idempotent, manifests) + 5 tests; downloaded DrugRev + Druglib; copied legacy UCI train into `data/raw/drugscom_uci/`.
- Next: #4 cleaning + merge + de-duplication.

### 2026-10-13: Claude Code (Opus 5.5), session 3
- Merged PR #19 (closes #3). Requested Dependabot rebases for #16-#18 (approved to merge when green).
- #4: `medsignal.data.clean` (source loaders → standard schema, drops DrugRev `UserName`), `medsignal.data.dedup` (within-source dedup on text+date with aliases; cross-source match on date+rating+40-char prefix, chosen because it gave 100% rating agreement vs 87% for prefix-only), `medsignal.data.build` (CLI + report). 24 tests.
- Found DrugRev truncates at sentence boundaries → row-level detection unreliable (precision 0.46 / recall 0.18) → ADR-0012.
- Next: #5 splits (stratified, drug-held-out), #6 EDA.
