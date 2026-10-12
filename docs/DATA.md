# Data Card

> What data we use, where it comes from, licence terms, and known issues. Raw data is **never committed**.

| Dataset | Source | Size | Labels | Use | Licence / terms |
|---|---|---|---|---|---|
| Drugs.com reviews | [UCI #462](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com), Gräßer et al. 2018 | 215,063 (161,297 train / 53,766 test) | 1-10 rating, condition, usefulCount, date | main corpus | per UCI page; cite authors; do not redistribute |
| WebMD reviews | [Kaggle](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset) | ~360K | effectiveness, ease of use, satisfaction (1-5), age, sex | cross-platform eval | check Kaggle page |
| PsyTAR | [Zolnoori et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6495095/) | 891 reviews, sentence-level | expert ADR / WD / SSI / effectiveness | external check (optional) | CC BY 4.0 |
| FAERS | [openFDA drug/event](https://open.fda.gov/apis/drug/event/) | API | MedDRA PT reactions | comparison | public domain (US gov) |
| Drug labels | [openFDA drug/label](https://open.fda.gov/apis/drug/label/) | API | `adverse_reactions` text | "on label?" check | public domain |
| Gold set | owner-annotated | 300 (+100 WebMD optional) | ADE spans, severity, discontinued, reason | evaluation | ours; review text excerpts subject to source terms |

## Layout

```
data/raw/<dataset>/        untouched downloads (+ checksum file)
data/interim/              cleaned, de-duplicated parquet
data/processed/            splits, silver labels, mentions, signal tables
data/annotations/          gold set exports (see below)
```

Gold annotations are stored as **review IDs + labels** (not full text) so they can be committed without redistributing source text; text is re-joined from raw data at load time.

## Known issues (Drugs.com, verified 2026-10-13 on train file)

- ~30% duplicate reviews (same text listed under multiple conditions/drug names) → de-duplicate before splitting.
- HTML entities in ~62% of reviews (`&#039;`, `&quot;`) → `html.unescape`.
- ~900 `condition` values are scraping artefacts (`"3</span> users found this comment helpful."`); 899 missing.
- Ratings are bimodal (10 = 32%, 1 = 13%); a star rating measures satisfaction, not text sentiment.
- Birth control ≈ 18% of reviews → report per-condition metrics; consider capping per drug when sampling.
- Median review ≈ 84 words; 99th percentile 154 words, 99.9th 253 words → `max_length=256` tokens covers ~99% (word-based estimate; confirm with tokenizer in Phase 1).

## Sampling for annotation / LLM labelling

Stratified by rating bucket and condition; cap reviews per drug; fixed seed in `configs/`. Exact procedure is recorded in the config used.
