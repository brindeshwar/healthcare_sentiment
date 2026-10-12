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
- Date: 2026-10-13 · Status: Accepted
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
