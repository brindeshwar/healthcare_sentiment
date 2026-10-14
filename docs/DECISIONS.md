# Architecture & Design Decisions (ADR log)

> Lightweight ADRs. Append new entries; never delete. To reverse a decision, add a new ADR that **supersedes** the old one.
> Write the *why* in plain language; the owner must be able to defend each one in an interview.

Template:

```
## ADR-XXXX: Title
- Date: YYYY-MM-DD · Status: Proposed | Accepted | Superseded by ADR-YYYY
- Context: what problem / constraint
- Decision: what we chose
- Alternatives: what else was considered and why not
- Consequences: trade-offs we accept
```

---

## ADR-0001: Rebuild instead of patching the legacy project
- Date: 2026-10-13 · Status: Accepted
- Context: Legacy repo reported a simulated confusion matrix, incorrect class statistics, and had train/test leakage (~30% duplicate reviews). The task itself (5-class sentiment on Drugs.com) is a saturated student project.
- Decision: Start fresh on branch `rebuild`; keep legacy at tag `v0-minor-project`; reuse only ideas.
- Alternatives: Fix bugs and re-run (fast, but leaves a commodity project).
- Consequences: More work up front; no legacy numbers are reused.

## ADR-0002: Project direction: ADE extraction + discontinuation reasons vs FAERS/labels
- Date: 2026-10-13 · Status: Accepted
- Context: Owner wants resume value, current tech, real-world value, $0. Prior work compares reviews with FAERS only for single drugs (hypnotics 2019, tramadol 2022, semaglutide 2024), manually. No automated all-drug comparison found.
- Decision: Build a pipeline covering 3,000+ drugs: extract ADEs + discontinuation reasons, normalise to MedDRA PTs, compare with FAERS and labels.
- Alternatives: Pure sentiment benchmark (saturated); RAG chatbot product (low value, hallucination risk); ADE-only (misses an easy, high-interest feature).
- Consequences: Several stages each need evaluation; findings are hypotheses, not causal claims.

## ADR-0003: LLM as teacher, encoder as student (distillation)
- Date: 2026-10-13 · Status: Accepted
- Context: No budget for human annotation at scale; running an LLM on 215K reviews on a laptop GPU would take ~200 h.
- Decision: Local LLM labels ~10K reviews → train an encoder token classifier on those labels → run encoder on everything.
- Alternatives: LLM on everything (too slow); encoder on public ADE corpora only (CADEC/PsyTAR are small and domain-shifted).
- Evidence: Vedula et al. 2025 (arXiv 2501.00031): distilled BioBERT beat its LLM teacher, ~12x faster.
- Consequences: Student inherits teacher errors → teacher must be measured against a human gold set.

## ADR-0004: Human gold set of 300 reviews, annotated by the owner
- Date: 2026-10-13 · Status: Superseded by ADR-0018
- Context: LLM labels on noisy social text are unreliable (5-shot F1 ~49-66 on SMM4H/CADEC in 2025 benchmarks). Credible work validates on human labels.
- Decision: Owner annotates 300 Drugs.com reviews (LLM pre-annotation, human correction), split dev/test. Optional 100 WebMD reviews for cross-platform eval.
- Consequences: ~6-10 h owner time; small test set → report confidence intervals.

## ADR-0005: No LLM fine-tuning above 2B parameters
- Date: 2026-10-13 · Status: Accepted
- Context: Unsloth advises against 4-bit QLoRA on Qwen3.5; bf16 LoRA needs ~10 GB for 4B. GPU has 8 GB.
- Decision: LLMs used for inference only (Ollama, ~9B @ Q4). Optional experiment: LoRA on a <=2B model.
- Consequences: Extraction at scale relies on the encoder student.

## ADR-0006: Encoder choice: BioClinical ModernBERT, GLiNER-BioMed as baseline
- Date: 2026-10-13 · Status: Accepted (revisit if benchmarks disagree)
- Context: Need a fast, accurate span tagger for medical-ish informal text.
- Decision: BioClinical ModernBERT-base (2025, biomedical+clinical pretraining, 8K context) as student; GLiNER-BioMed zero-shot as baseline.
- Alternatives: DeBERTa-v3, PubMedBERT, BioBERT (keep as fallback if ModernBERT is problematic on Windows).

## ADR-0007: Normalise to MedDRA Preferred Terms as used in FAERS
- Date: 2026-10-13 · Status: Accepted
- Context: Comparison with FAERS requires the same vocabulary. Full MedDRA LLT linking tops out at ~40% Acc@1 (ALTA 2025).
- Decision: Target = PT vocabulary observed in FAERS reactions (via openFDA). Pipeline: SapBERT + lexical retrieval → cross-encoder rerank → LLM verify low-confidence.
- Consequences: Report Acc@1/Acc@5; MedDRA licence terms to be checked before redistributing any mapping table.

## ADR-0008: Evidence explorer, not a RAG chatbot
- Date: 2026-10-13 · Status: Accepted
- Context: Hallucination is the most-cited LLM limitation in pharmacovigilance (2025 systematic review).
- Decision: UI shows filters, statistics and verbatim review quotes. Any LLM summary must cite quotes and is optional.

## ADR-0009: Tooling: uv, ruff, mypy, pytest, pre-commit, GitHub Actions, Ollama
- Date: 2026-10-13 · Status: Accepted
- Context: Reproducibility on Windows, multiple coding agents, $0.
- Decision: uv for env/lock; ruff for lint+format; mypy; pytest; pre-commit; GitHub Actions CI; Ollama (Windows-native) for local LLMs; openFDA for FDA data; HF Hub for weights.
- Alternatives: conda/pip (slower, less reproducible); vLLM (no native Windows); MLflow (optional later).

## ADR-0010: Sentiment kept only as a baseline
- Date: 2026-10-13 · Status: Accepted
- Context: Star-rating sentiment on Drugs.com is saturated in the literature.
- Decision: Majority, TF-IDF+LR and one encoder, reported as a sanity baseline and to show the de-duplication effect; not a headline.

## ADR-0011: Corpus = UCI Drugs.com (full text) + DrugRev (newer reviews), merged
- Date: 2026-10-13 · Status: Accepted
- Context: UCI withdrew the Drugs.com dataset (#462); its official test file is unobtainable from the source. DrugRev (Mendeley, CC BY 4.0, 392K rows, 2008-2023, adds duration of use) is available, but truncates ~28% of reviews that also appear in UCI (19% lose over half their text).
- Decision: Merge both. Where a review exists in both, keep the **UCI full text** and DrugRev's extra fields (duration). Add DrugRev-only reviews (mostly 2018-2023) with a `source` column and a `possibly_truncated` heuristic flag. De-duplicate across sources before splitting; make our own train/val/test (+ drug-held-out) splits. Do not use the Hugging Face mirror of the withdrawn data.
- Alternatives: UCI only (no post-2017 drugs, no duration); DrugRev only (truncation hurts ADE recall); HF mirror for the official test file (relies on a mirror of withdrawn data).
- Consequences: No "official" test split, so results are not directly comparable with papers using the UCI test file (state this). Truncation lowers recall for DrugRev-only reviews; report metrics by source and flag in signal analyses.

## ADR-0012: Flag every DrugRev-only review as possibly truncated (no row-level heuristic)
- Date: 2026-10-13 · Status: Accepted
- Context: DrugRev cuts reviews at sentence boundaries (keeps the first N sentences), so truncated text looks complete. On 93,517 matched pairs the best simple heuristic (no terminal punctuation) had precision 0.46 and recall 0.18.
- Decision: `possibly_truncated = True` for all DrugRev-only rows; `False` for `uci`/`both` (UCI full text). Gold set and teacher-evaluation samples use only full-text rows. Report extraction metrics by source.
- Alternatives: punctuation heuristic (misses 82% of truncations, false sense of safety); dropping DrugRev-only rows (loses 2018-2023 data and newer drugs).
- Consequences: ~152K reviews carry the flag; signal analyses must treat DrugRev-only mention rates as lower bounds.

## ADR-0013: Splits: brand/generic drug groups held out whole; rest stratified 80/10/10
- Date: 2026-10-13 · Status: Accepted
- Context: The official UCI test file is unavailable (ADR-0011), so we make our own splits. The same review is often listed under brand and generic names (Zoloft / Sertraline), so a naive "hold out drug X" split leaks X in under another name. We also want to test generalisation to unseen drugs.
- Decision: (1) Build drug groups with union-find over names that co-occur on a review (3,613 names → 1,592 groups). (2) Randomly hold out whole groups until ~10% of reviews (`drug_holdout`), excluding groups larger than 1% of reviews so one contraceptive cannot dominate it. (3) Split the rest 80/10/10 into train/val/test, stratified by (sentiment bucket, source). Seeded; leakage checks run on every build and abort on failure.
- Alternatives: random split only (no unseen-drug test); time-based split (confounded by DrugRev covering only post-2017 additions); holding out by `drug` string (leaks via aliases).
- Consequences: Results on `test` = in-distribution; on `drug_holdout` = unseen drugs. Both reported. Not comparable to papers using the official UCI test split.

## ADR-0014: Repair mid-word-truncated condition names from the UCI scrape
- Date: 2026-10-13 · Status: Accepted
- Context: The EDA showed "Bipolar Disorde" and "Bipolar Disorder" as two top-15 conditions. The UCI scrape cut 67 condition names mid-word ("Cance", "Ulce", "Tum", "Pain/Feve"), splitting one condition into two and distorting per-condition statistics.
- Decision: Map a name to its completed form only if it occurs in UCI but not DrugRev, stops mid-word (next character of a longer name is a letter), has exactly one completed-word candidate, and that completed name exists. All 67 mappings were reviewed by hand; the count and examples are in `build_report.json`.
- Alternatives: no repair (split conditions); generic prefix matching over all names (tried first: wrongly mapped "Acne" → "Acnex" and "Asthma" → "Asthmanefrin", drug names that leak into DrugRev's condition field); a hand-written list (does not generalise to re-scrapes).
- Consequences: Conditions are comparable across sources. DrugRev's condition field still contains some drug names; to be handled when conditions matter (Phase 4).

## ADR-0015: Sentiment baseline protocol
- Date: 2026-10-13 · Status: Accepted
- Context: Phase 1 needs sanity baselines (ADR-0010) that are honest and cheap, and a measured answer to "how much did duplicate leakage inflate the legacy score?".
- Decision: 3-class ordinal task from stars (neg 1-4 / neu 5-6 / pos 7-10). Metrics: macro-F1 (primary), accuracy, quadratic-weighted kappa. Models: majority class; TF-IDF (1-2 grams) + logistic regression with `class_weight=balanced`, C chosen on `val`; BioClinical ModernBERT fine-tuned for 1 epoch, 3 seeds, best checkpoint by `val` macro-F1. `test` and `drug_holdout` are scored once. Leakage demo: legacy setup (raw UCI, random 80/20) vs the same setup after de-duplication.
- Alternatives: 10-class or 5-class ratings (noisier, mostly a measure of rating noise); more encoder epochs (2 epochs ≈ 2x GPU time for a baseline that is not the project's headline).
- Consequences: Numbers are not comparable with papers using the official UCI test split (unavailable, ADR-0011). TF-IDF is deterministic → single run; encoder → mean ± std over seeds.

## ADR-0016: Annotation schema v1.0 (ADEs with attributes, discontinuation, stop reasons, effectiveness)
- Date: 2026-10-13 · Status: Accepted (owner approved the guidelines 2026-10-13)
- Context: The gold set, the LLM teacher and the evaluation must share one label definition, small enough for one person to annotate 300 reviews in 6-10 h, yet rich enough for FAERS comparison and the "why patients stop" analysis.
- Decision: Per review: ADE spans (verbatim text; attributes severity mild/moderate/severe/not_stated, withdrawal, beneficial), discontinued (yes/planned/no/not_stated), multi-label stop reasons (adverse_event, lack_of_efficacy, cost_access, convenience, course_completed, pregnancy_related, clinician_decision, other, not_stated), effectiveness (effective/partial/not_effective/not_stated). Encoded once as Pydantic models (`labeling/schema.py`); spans are given as text and aligned to offsets (case-insensitive, word-boundary, reading order). Guideline examples are unit tests.
- Alternatives: PsyTAR's full scheme (ADR/WD/SSI/DI/EF/INF plus symptom categories; too heavy for one annotator); CADEC's five entity types (Drug/Disease/Symptom/Finding need more time and are not needed for our questions); free-text reasons (not countable).
- Consequences: Withdrawal effects are ADEs with a flag rather than a separate entity; welcome effects (e.g. amenorrhoea on contraceptives) are kept with `beneficial=true` so FAERS-style counts include them. Reason taxonomy is close to the 2026 Estonian EHR study, which supports comparison.

## ADR-0017: Gold-set protocol: sample, tool, and blind test annotation
- Date: 2026-10-13 · Status: Superseded by ADR-0018 (sample, pre-annotations and Label Studio setup kept, parked)
- Context: The gold set evaluates the LLM teacher (Phase 2) and the encoder student (Phase 3). LLM pre-annotation saves time but anchors the annotator toward the LLM's answers, which would inflate the teacher's test score.
- Decision:
  - Sample: 100 **dev** reviews from the `val` split and 200 **test** reviews from the `test` split (never from `train`, which the student trains on); full-text only; sentiment buckets oversampled to 40/20/40 neg/neu/pos (ADEs are rarer in positive reviews); at most 2 per drug group; drawn once with a fixed seed and committed as IDs only (`data/annotations/gold_sample.csv`).
  - Pre-annotation: **dev only**, by a local LLM (`qwen3.5:9b-q4_K_M` via Ollama structured output). **Test is annotated blind.**
  - Tool: Label Studio 1.23 (Apache-2.0), run locally from an isolated venv in `tools/label-studio`, data in `.cache/label-studio`. Interface generated from the schema enums and validated with the Label Studio SDK.
- Alternatives: all 300 pre-filled (~2-3 h faster, biased test); pre-filling test with a non-LLM model such as GLiNER-BioMed (less bias, more setup); Argilla (needs a search backend); a custom Streamlit tool (no native span selection).
- Consequences: Owner time ~8-11 h. Sample metrics reflect the 40/20/40 mix, not the population (stated in reports). Dev may be used freely for prompt design; test is touched once per final model.

## ADR-0018: Evaluate on published expert-annotated corpora instead of an owner-annotated gold set
- Date: 2026-10-15 · Status: Accepted (owner: no time for manual annotation)
- Context: ADR-0004/0017 required ~8-11 h of owner annotation. The owner cannot provide this time. Letting an LLM (or the agent) label the gold set would make the evaluation circular and would misrepresent "human gold" (AGENTS.md §7.5).
- Decision: Use independent, expert-annotated public corpora as gold, each split once into dev (tuning) and test (reported once):
  - **CADEC v2** (CSIRO; 1,250 AskaPatient posts; 6,318 expert ADR spans, 1,066 discontinuous; 5,947 with MedDRA codes): ADE span extraction (strict + relaxed F1) and, in Phase 4, MedDRA normalisation. Split 25/75 dev/test stratified by drug (313/937 posts, 1,484/4,834 ADEs; seed 11, `data/annotations/external/cadec_split.csv`).
  - **PsyTAR** (891 AskaPatient reviews of 4 psychiatric drugs; CC BY 4.0): ADEs, withdrawal symptoms, severity qualifiers, effectiveness/ineffectiveness sentences. Owner downloads the xlsx once in a browser (the source is bot-protected; no scripted download).
- Not covered by any obtainable expert corpus: discontinuation and stop reasons (the MedHelp DDE study releases only scraper scripts, no licence; scraping MedHelp is not acceptable), the `beneficial` flag. These are reported as **not validated against human labels**; only reliability measures (agreement between two different local LLMs) are reported for them.
- Alternatives: owner gold set (no time); smaller owner gold set of 50-60 reviews (still manual); LLM-labelled "gold" (circular, rejected); scraping MedHelp for the DDE labels (terms of use, unlicensed code, rejected).
- Consequences: Evaluation is **out-of-domain** (AskaPatient, not Drugs.com): this is reported as external validation, a gap highlighted by the 2025 systematic review. Corpora use their own annotation conventions (e.g. CADEC span boundaries, "Symptom" vs "ADR"), so scores reflect agreement with those conventions; dev splits are used to adapt prompts, test splits are used once. CADEC is 80% Lipitor: results are also reported per drug. CADEC licence is non-commercial, attribution, no redistribution: raw files stay in `data/raw/` (git-ignored); only IDs/splits are committed. The 300-review Drugs.com sample, LLM pre-annotations and Label Studio setup are kept for an optional later in-domain spot check.
