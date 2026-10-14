# Status

> **Hand-off file.** Every agent/person reads this first and updates it last (see AGENTS.md §2-3).
> Keep "Now" short and current; append to the Session log, never rewrite history.

**Current phase:** Phase 1 (data & baseline)
**Branch:** `feat/teacher-benchmark`
**Last updated:** 2026-10-13

## Now (in progress)

- ADR-0018: owner annotation replaced by expert corpora. CADEC v2 downloaded, loaded, split (dev 313 / test 937 posts). PsyTAR waiting for owner's one-minute browser download (`docs/SETUP.md` §5). PR pending on `feat/external-gold`.

## Next (ordered)

1. Phase 2: teacher benchmark harness on CADEC dev (strict/relaxed span F1), candidate local LLMs.
2. Owner (whenever convenient): download PsyTAR xlsx (`docs/SETUP.md` §5).
3. Optional: encoder with class weighting (neutral F1 is the weak spot); not required for Phase 1.
4. Owner (#2), later: HF token, openFDA key, WebMD CSV (see `docs/SETUP.md`). Ollama done.

## Blockers / waiting on owner

- WebMD dataset requires Kaggle login (owner downloads).

## Known facts about the data (verified 2026-10-13)

- **UCI #462 (Drugs.com) withdrawn** from UCI (403 / API not found). Using local train copy (161,297 rows) + DrugRev (Mendeley, CC BY 4.0, 392,510 rows, 2008-2023). See ADR-0011.
- DrugRev truncates ~28% of overlapping reviews (19% lose >50%); 37% duplicate texts; 25,036 rows with shifted `IntakeTime`; has `UserName` (drop at load).
- Local `drugsComTrain_raw.csv`: 161,297 rows (official test file unobtainable).
- 48,879 duplicate (review, rating) rows (~30%); 62% of reviews contain HTML entities (`&#039;`).
- Rating distribution skewed: 10 stars = 32%, 1 star = 13%.
- Legacy folders `healthcare_sentiment/`, `healthcare_bert_final/` (untracked, ~0.5 GB) no longer used; UCI file copied to `data/raw/drugscom_uci/`. Owner decides on deletion (SETUP.md §5).

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

### 2026-10-13: Claude Code (Opus 5.5), session 4
- Owner asked to keep everything inside the project folder: uv `cache-dir = .cache/uv` (pyproject), `HF_HOME=.cache/huggingface` (.env), Ollama install `/DIR` → `tools/ollama`, `OLLAMA_MODELS` → `models/ollama`. Added `docs/SETUP.md` (owner checklist), created local `.env` from `.env.example`.
- Next: owner works through SETUP.md; agent continues #5 splits after PR #20 merges.
- Owner's manual install failed (0-byte installer download; no residue). Agent downloaded the signed installer, set user `OLLAMA_MODELS=D:\healthcare_sentiment\models\ollama`, installed Ollama 0.40.3 silently into `tools/ollama`. Only tiny unavoidable files on C: (`~/.ollama`, `%LOCALAPPDATA%/Ollama`).

### 2026-10-13: Claude Code (Opus 5.5), session 5
- Merged Dependabot #17, #18 and PR #20 (closes #4); rebased #21 (resolved STATUS.md session-log conflict by keeping both entries).
- #5: `medsignal.data.splits` (union-find drug groups, whole-group drug holdout capped at 1%-share groups, stratified 80/10/10 by sentiment bucket + source, leakage checks that abort the build). ADR-0013. 29 tests.
- Next: #6 EDA report, #7 baselines.

### 2026-10-13: Claude Code (Opus 5.5), session 6
- Merged PR #22 (closes #5).
- #6: `medsignal.data.eda` (5 figures, summary.json, generated README). Tokenizer `thomas-sounack/BioClinical-ModernBERT-base`: median 84 tokens, p99 231, 99.7% within 256 (confirms max_length). Added `utils.load_env` (HF cache resolved into `.cache/huggingface`).
- EDA surfaced 67 mid-word-truncated UCI condition names → repaired (ADR-0014); first attempt over-matched ("Acne" → "Acnex"), fixed by restricting to UCI-only names. DrugRev-only reviews ~2x more one-star than UCI-era.
- Pre-existing 6 GB Hugging Face cache on C: (`~/.cache/huggingface`, from before the rebuild) left for the owner to decide.
- Next: #7 baselines.

### 2026-10-13: Claude Code (Opus 5.5), session 7
- Merged PR #23 (closes #6). #7 baselines on `feat/baselines` (stacked on the pre-squash EDA branch; rebased before PR).
- Installed `gpu` extra (torch 2.14.1+cu130; CUDA OK on RTX 4060 Laptop, bf16). transformers v5: `warmup_ratio` → `warmup_steps` float, `group_by_length` → `train_sampling_strategy`.
- Results (`reports/baselines/`): majority 0.245 macro-F1; TF-IDF+LR 0.634 test / 0.594 drug_holdout; BioClinical ModernBERT 1 epoch, 3 seeds 0.646 ± 0.002 / 0.613 ± 0.002 (accuracy 0.832). Encoder weak on neutral (F1 0.22 vs 0.31) — unweighted loss + noisy 5-6-star labels.
- Leakage demo with control: legacy setup 0.727 macro-F1 (48% of test texts in train) vs de-duplicated 0.631 → +9.6 points from leakage alone.
- Seed 2026 trained ~3x slower than the others (73 vs 26-30 min), likely thermal throttling / GPU contention on the laptop; results unaffected.
- Next: #8 annotation guidelines.

### 2026-10-13: Claude Code (Opus 5.5), session 8
- Merged PR #24 (closes #7); fixed CI mypy `unused-ignore` (GPU-only ignores on CPU CI).
- #8: `docs/annotation_guidelines.md` (fields, include/exclude rules, span rules, severity cues, withdrawal/beneficial flags, discontinuation + 9 stop reasons, effectiveness, 7 worked examples, process: 100 dev / 200 test, self-agreement on 30). Examples are constructed sentences modelled on real reviews (review IDs given) because only 9/20 real excerpts exist verbatim in CC-BY DrugRev.
- `medsignal.labeling.schema` (Pydantic, validators, `align_spans` with case-insensitive, word-boundary, reading-order matching); guideline examples are tests. 55 tests.
- Next: owner review → #9 annotation tool + LLM pre-annotation.
- Owner approved the annotation guidelines (all five judgment calls as drafted); ADR-0016 accepted.

### 2026-10-13: Claude Code (Opus 5.5), session 9
- Merged PR #25 (closes #8). Owner chose: dev pre-filled by LLM, test annotated blind (ADR-0017).
- Gold sample drawn (`data/annotations/gold_sample.csv`, seed 7): dev 100 from `val`, test 200 from `test`, 40/20/40 neg/neu/pos, <=2 per drug group, full-text only; 248 drugs.
- Label Studio 1.23.2 installed in isolated venv `tools/label-studio`; interface `configs/labelstudio.xml` generated from schema and validated with the SDK. Owner steps in `docs/SETUP.md` §5.
- Ollama: tray app's server times out at startup; pipeline runs `ollama serve` itself with `OLLAMA_MODELS=models/ollama` (CUDA 13.4 detected). Tag is `qwen3.5:9b-q4_K_M` (not `9b-q4`).
- Next: pre-annotate 100 dev reviews, export tasks, hand over to owner (#10).
- Pre-annotation (qwen3.5:9b-q4_K_M, temp 0): 100/100 dev reviews; median 2.9 s/review after model load (~52 s first call); 94 first-try valid. Failure mode: the model stitches non-adjacent words into one span ("gained... 53 pounds") and repeats it on retry → added `drop_unaligned_on_failure` (pre-annotation only; 2 spans dropped and recorded). Severity skews to "moderate" (132/238 spans): watch in Phase 2 prompt work.

### 2026-10-15: Claude Code (Opus 5.5), session 10
- Merged PR #26 (closes #9). Owner set up Label Studio but has no time to annotate, so ADR-0018: evaluate on expert corpora. Agent declined to label the gold set itself (circular, misrepresents human gold).
- CADEC: v1 (collection 10948) has annotations but no texts; v2 (collection 17190, `CADEC.v2.zip`) has all 1,250 texts. Offsets verified (9,107/9,111 exact); 6,318 ADRs, 1,066 discontinuous; MedDRA lines include `+`-joined codes, `a/b` alternatives and `CONCEPT_LESS`; 5,947 ADRs with codes (687 unique).
- PsyTAR: AskaPatient is bot-protected (not bypassed), HF copy is a loader only, publisher CDN 404, so owner downloads in a browser. MedHelp DDE: repo has no licence and only scrapers, so discontinuation is not validated against human labels.
- Lesson: generating Python or Markdown through shell heredocs corrupted backslash escapes three times; write patch scripts with the file tool instead.
- Next: Phase 2 teacher benchmark on CADEC dev.

### 2026-10-15: Claude Code (Opus 5.5), session 11 (paused by owner)
- Merged PR #27 (closes #10). Phase 2 started on `feat/teacher-benchmark` (benchmark harness, prompt versions, bootstrap CIs committed).
- CADEC dev, prompt v1 (relaxed F1 [95% CI]): qwen3.5:9b 0.768 [0.735, 0.796], gemma4:e4b 0.761 [0.728, 0.789], qwen3.5:4b 0.755 [0.723, 0.784]; CIs overlap, gemma ~2.8x faster (0.92 s/doc). Error analysis: 410 missed, 423 boundary-only (CADEC keeps intensity/location words), 61 false positives, 3.5% failures from paraphrased spans.
- Prompt v2 (keep reviewer's words, label every mention): gemma4:e4b relaxed F1 0.792, strict 0.540 (report in `reports/teacher/cadec_dev/`). qwen3.5:9b v2 run was stopped part-way; it resumes automatically.
- **Resume:** start Ollama (`docs/SETUP.md` §1), then `uv run python -m medsignal.labeling.benchmark --model qwen3.5:9b-q4_K_M --prompt v2`. Then compare v1/v2 with CIs, and ask the owner whether to adopt v2 span convention (intensity words inside spans) as guideline v1.1 (ADR-0019, not yet written).
