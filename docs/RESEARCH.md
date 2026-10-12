# Related Work & Evidence

> Literature that justifies our design. Cite these in the write-up. Add entries as you read; note what we take from each.
> Survey done 2026-10-13. "Verified" = read beyond the abstract; otherwise treat numbers as indicative.

## Field overview
- **LLMs in ADR detection & pharmacovigilance: systematic review** (Diagnostics, 2025/26), [PMC13465311](https://pmc.ncbi.nlm.nih.gov/articles/PMC13465311/). LLMs are best as supervised, task-specific tools; hallucination, prompt sensitivity, **weak external validation** are common gaps. *Takeaway:* our cross-platform + gold-set evaluation targets the gap.
- **Pharmacovigilance from social media** (review, 2025), [PMC12149966](https://pmc.ncbi.nlm.nih.gov/articles/PMC12149966). BERT-style models work but need labelled data.

## ADE extraction (LLM vs encoder)
- **Benchmarking LLMs for ADR extraction** (Results in Engineering, 2025), [link](https://www.sciencedirect.com/science/article/pii/S2590123025034176). Fine-tuned GPT-4o-mini: relaxed F1 79.1 CADEC / 66.8 SMM4H; 5-shot 65.6 / 49.3. *Takeaway:* prompting alone is weak on noisy text → validate teacher; fine-tuning matters.
- **SFT or ICL? LLMs for clinical NER** (arXiv 2510.22285). GPT-4o SFT F1 ≈ 87.1 on CADEC; encoders plateau; simple prompts beat complex ones.
- **SMM4H 2025 overview**, [PDF](https://workshop-proceedings.icwsm.org/pdf/2025_55.pdf). Most teams fine-tuned BERT-family; top system fine-tuned Mistral-Nemo + LLM paraphrase augmentation.
- **SRCB @ SMM4H 2024**, [ACL](https://aclanthology.org/2024.smm4h-1.8/). LLM data augmentation for ADE extraction/normalisation.
- **Scaboro et al., Transformer ADE extraction** (arXiv 2306.05276). 19 encoders on informal text; useful baseline numbers.

## Distillation / synthetic labels
- **Vedula et al., Distilling LLMs for clinical IE** (arXiv 2501.00031). LLM teacher → BioBERT student; student beat teacher (disease F1 0.84 vs 0.82), below human-label model (0.89); ~12x faster. *Basis for ADR-0003.*
- **GLiNER-BioMed** (arXiv 2504.00676; Bioinformatics). Zero-shot biomedical NER, built via LLM distillation; SOTA zero-shot micro-F1 59.8. *Our baseline.*

## Normalisation (mention → MedDRA)
- **ALTA 2025 shared task** (ADE → MedDRA): best Acc@1 ≈ 0.40, Acc@5 ≈ 0.78. Winning pattern: synonym-augmented retrieval → cross-encoder rerank → LLM discriminator. [SCaLER](https://preview.aclanthology.org/setup/2025.alta-main.17), [hybrid system](https://preview.aclanthology.org/setup/2025.alta-main.16). *Basis for ADR-0007.*
- **BERT-based normalisation to MedDRA SOCs** (BDCC 2025), [doi](https://doi.org/10.3390/bdcc10050141). Macro-F1 drops 0.78 → 0.45 going from 6 to 25 classes.

## Reviews vs FAERS (closest prior work; all single-drug, manual)
- **Borchert et al. 2019, hypnotics: Drugs.com vs FAERS** (JMIR), [PMC6874799](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6874799/). Reviews over-report amnesia for zolpidem; "valid supplementary source". *Our validation case study.*
- **Park et al. 2022, tramadol: WebMD vs FAERS** (JMIR PH&S), [PMC8767477](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8767477/). PRR/ROR/IC; 125 signals not on label.
- **Zhang et al. 2024, semaglutide: FAERS vs reviews** (Front. Pharmacol.), [PMC11534668](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11534668/).
- **Social media vs FAERS timing** (Drug Saf., 2016), [PMC5157765](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5157765/). Mixed evidence of earlier detection; social media skews mild.

## Discontinuation
- **Drug discontinuation events in forum posts** (JMIR 2025, e54601): GPT-4/DeBERTa/BART on MedHelp; released DDE dataset.
- **LLMs for discontinuation reasons in Estonian EHR** (JMIR 2026, e86183): 93-98% phrase accuracy; reason weighted-F1 0.81-0.84; adverse reactions top reason.

## Sentiment on drug reviews (saturated; context only)
- Gräßer et al. 2018: aspect-based sentiment, cross-domain/cross-data (dataset origin).
- 2025 transformer/lexicon papers on Drugs.com (JDSIS 2025; IEEE Access 2025 doi 10.1109/ACCESS.2025.3590326).

## Models & tooling (check currency before use)
- **BioClinical ModernBERT** (arXiv 2506.10896): base 150M / large 396M, 8K context.
- Small LLMs for 8 GB (2026 blogs, unverified): Qwen3.5-9B Q4 (~6 GB), Qwen3.5-4B, Gemma 4 E4B. **Benchmark locally.**
- **Unsloth Qwen3.5 guide**: no QLoRA recommended; bf16 LoRA 4B ≈ 10 GB. [docs](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Ollama structured outputs (JSON schema in `format`); vLLM richer but not native on Windows.
- openFDA: `drug/event.json?search=patient.drug.openfda.generic_name:"X"&count=patient.reaction.reactionmeddrapt.exact`; counts = reporting volume, not incidence.
