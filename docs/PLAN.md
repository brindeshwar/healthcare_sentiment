# Project Plan: MedSignal

> Living roadmap. Tick a box only when its acceptance criteria are met. Scope changes need an ADR in `DECISIONS.md`.

## Goal

Build an evaluated, reproducible pipeline that extracts **adverse drug events (ADEs)** and **discontinuation reasons** from
patient drug reviews at scale, normalises them to MedDRA Preferred Terms, and compares them with FDA FAERS reports and
drug labels, shipped as an evidence explorer with open models and an honest write-up.

**Headline question:** *What do patients report in reviews that FAERS and drug labels under-represent, automated across
3,000+ drugs, with every pipeline stage measured?*

**Secondary:** *Why do patients stop their medication, and how does that differ between drugs in the same class?*

## Success criteria (project level)

- [ ] Every stage has a reported metric on held-out human-labelled data.
- [ ] Encoder student is within a stated margin of the LLM teacher on the gold set and >= 50x faster.
- [ ] Cross-platform result: trained on Drugs.com, evaluated on WebMD.
- [ ] Signal table for all drugs with >= N reviews (N set in config), with PRR/ROR and confidence intervals.
- [ ] Public demo + models on Hugging Face Hub with model cards.
- [ ] README results reproducible from a clean clone with documented commands.

## Time budget (estimate)

| | Estimate |
|---|---|
| GPU time (mostly unattended) | ~25-35 h |
| Owner hands-on time | ~25-35 h (incl. ~6-10 h gold annotation) |
| Calendar | ~2.5-3 weeks fast / ~4 weeks comfortable |

---

## Phase 0: Project foundation

- [x] New branch `rebuild`; legacy code preserved at tag `v0-minor-project`
- [x] Agent/management docs: AGENTS.md, CLAUDE.md, PLAN, STATUS, DECISIONS, DATA, RESEARCH
- [x] Repo hygiene: pyproject (uv), ruff, mypy, pytest, pre-commit, CI, issue/PR templates, licence, changelog
- [x] `uv sync` works; `uv run pre-commit run --all-files` passes locally
- [ ] CI green on GitHub (after first push)
- [ ] Owner: install Ollama; create Hugging Face + Kaggle accounts; request free openFDA API key

**Acceptance:** clean clone, then `uv sync && uv run pytest` passes locally and in GitHub Actions.

## Phase 1: Data foundation & baseline  (~1.5 weeks; GPU 2-3 h; owner 8-12 h)

- [x] Download script with pinned checksums: DrugRev, Druglib (auto); UCI Drugs.com local copy, WebMD (manual check). UCI #462 withdrawn (ADR-0011)
- [x] Merge UCI + DrugRev (prefer UCI full text; `source`, `possibly_truncated`, duration fields)
- [x] Cleaning: HTML-entity decoding, whitespace, junk `condition` values (`</span>` artefacts), empty reviews
- [x] **De-duplication** (normalised text + date; cross-source date + rating + prefix) before any split; counts in `reports/data/build_report.json`
- [x] Splits: own train/val/test after cross-source de-duplication (official UCI test unavailable); drug-held-out split (ADR-0013)
- [x] EDA report (`reports/eda/`): rating distribution, token lengths, conditions, duration, years, splits
- [x] Sentiment baselines (sanity only): majority, TF-IDF + LR, one encoder; macro-F1, QWK, 3 seeds (ADR-0015; `reports/baselines/`)
- [x] **Annotation guidelines** (`docs/annotation_guidelines.md`): ADE span, severity, discontinued, reason taxonomy (schema in code, ADR-0016; owner approved)
- [ ] Annotation tool set up (Label Studio or Argilla, local); LLM pre-annotation of 300 sampled reviews
- [ ] Owner annotates 300 reviews (gold set); 30 double-annotated after 1 week for self-agreement

**Acceptance:** `python -m medsignal.data.build` reproduces processed data; baseline table in `reports/`; gold set committed.

## Phase 2: LLM teacher  (~1.5 weeks; GPU 10-14 h; owner 2 h)

- [ ] Output JSON schema (Pydantic): ADE mentions (span, text, severity), discontinued (bool), reason category, effectiveness
- [ ] Prompt v1 + few-shot examples drawn from a **dev** part of the gold set only
- [ ] Benchmark 2-3 local LLMs (e.g. Qwen3.5-9B, Gemma 4, verify current options) via Ollama structured outputs
- [ ] Measure teacher vs gold: span F1 (strict + relaxed), discontinuation F1, reason macro-F1, invalid-JSON rate, s/review
- [ ] Optional: self-consistency (k samples, majority vote) if it improves dev F1 enough for the cost
- [ ] Label ~10K stratified reviews (resumable job), store with prompt + model version

**Acceptance:** teacher metrics table on gold test; 10K silver labels with provenance.

## Phase 3: Encoder student & extraction at scale  (~1.5 weeks; GPU 4-6 h; owner 2 h)

- [ ] Convert silver labels to BIO tags; train BioClinical ModernBERT token classifier + discontinuation/reason heads
- [ ] Baselines: GLiNER-BioMed zero-shot; teacher itself
- [ ] Evaluate all on gold test (3 seeds); speed (reviews/s) and cost comparison
- [ ] Cross-platform: evaluate on WebMD sample (owner annotates 100 WebMD reviews, or use rating-derived proxy where valid)
- [ ] Run student over all de-duplicated reviews; store mentions table (parquet)
- [ ] Publish model to HF Hub with model card

**Acceptance:** results table (teacher vs student vs GLiNER) with mean ± std; full-corpus mentions table.

## Phase 4: Normalisation & signal detection  (~1.5 weeks; GPU 4-8 h; owner 2-3 h)

- [ ] Target vocabulary: MedDRA PTs as they appear in FAERS reactions (see ADR-0007)
- [ ] Candidate retrieval (SapBERT embeddings + lexical/synonym match) → cross-encoder rerank → LLM verify for low-confidence
- [ ] Evaluate on 100-200 owner-checked mentions: Acc@1, Acc@5
- [ ] openFDA: per-drug FAERS reaction counts; label `adverse_reactions` text (cached locally)
- [ ] Disproportionality (PRR, ROR with 95% CI) on review data; compare with FAERS; flag "frequent in reviews, absent from label"
- [ ] Discontinuation-reason profiles per drug and per drug class
- [ ] Case-study validation: reproduce known findings (e.g. zolpidem amnesia, Borchert et al. 2019)

**Acceptance:** signal tables + one validated case study; method limitations documented.

## Phase 5: Product & write-up  (~1 week; GPU ~1 h; owner 5-8 h)

- [ ] Evidence explorer (Streamlit/Gradio): drug search, ADE profile vs FAERS/label, stop-reasons, review quotes, disclaimers
- [ ] Deploy on HF Spaces (CPU; precomputed tables + small models)
- [ ] Dockerfile; `docker run` reproduces the app locally
- [ ] README rewritten with final results, figures, demo link
- [ ] Technical report (`reports/medsignal_report.md` → PDF), 4-6 pages, owner's own words
- [ ] Release `v1.0.0`; CHANGELOG updated

**Acceptance:** a stranger can open the demo, understand the finding in 1 minute, and reproduce the numbers.

---

## Out of scope (unless an ADR changes it)

- Free-form RAG chatbot (hallucination risk; see ADR-0008)
- Fine-tuning LLMs > 2B parameters (hardware; ADR-0005)
- Clinical/causal claims; medical advice
- Non-English data (Russian facility reviews parked; possible stretch goal)
