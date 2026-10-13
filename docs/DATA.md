# Data Card

> What data we use, where it comes from, licence terms, and known issues. Raw data is **never committed**.
> Fetch with `uv run python -m medsignal.data.download` (see `configs/data.yaml`); manual sources are checked for presence.

| Dataset | Source | Size | Labels / fields | Use | Licence / terms |
|---|---|---|---|---|---|
| **DrugRev** (Drugs.com, 2008-2023) | [Mendeley Data 10.17632/64cc5w5dxy.1](https://data.mendeley.com/datasets/64cc5w5dxy/1), Uttara Univ., 2023 | 392,510 rows (~248K unique texts) | rating 1-10, condition, **duration of use**, likes, date | main corpus (2018-2023 + duration) | CC BY 4.0 (credit + link; review text originates from Drugs.com) |
| **Drugs.com UCI** (2008-2017) | UCI #462, Gräßer et al. 2018 (**withdrawn from UCI**, see below) | 161,297 train rows (local copy; test file not available) | rating 1-10, condition, usefulCount, date | main corpus (full-length text) | research use, cite authors, do not redistribute |
| **Druglib.com** | [UCI #461](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com) | 4,143 (official train/test) | separate benefits / side-effects / comments text; effectiveness + side-effect ratings | aspect-level check set | research only, no redistribution, cite authors |
| WebMD reviews | [Kaggle](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset) (manual, login) | ~360K | effectiveness, ease of use, satisfaction (1-5), age, sex | cross-platform eval | listed CC0 on OpenML; verify on Kaggle |
| PsyTAR | [Zolnoori et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6495095/) | 891 reviews | expert ADR / WD / SSI / effectiveness | external check (optional) | CC BY 4.0 |
| FAERS | [openFDA drug/event](https://open.fda.gov/apis/drug/event/) | API | MedDRA PT reactions | comparison | public domain (US gov) |
| Drug labels | [openFDA drug/label](https://open.fda.gov/apis/drug/label/) | API | `adverse_reactions` text | "on label?" check | public domain |
| Gold set | owner-annotated | 300 (+100 WebMD optional) | ADE spans, severity, discontinued, reason | evaluation | ours |

Checksums are pinned in `configs/data.yaml`; each `data/raw/<source>/manifest.json` records URL, size, sha256, retrieval time.

## Interim table: `data/interim/reviews.parquet`

Built by `uv run python -m medsignal.data.build` (counts in `reports/data/build_report.json`).

| Column | Meaning |
|---|---|
| `review_id` | stable 16-hex ID = sha1(date + normalised text); used by gold set and all later stages |
| `source` | `uci`, `drugrev`, or `both` (matched: UCI full text kept) |
| `drug`, `drug_aliases` | first alias alphabetically; all drug names the review was listed under |
| `condition`, `conditions` | cleaned condition(s); `None` for missing/artefacts |
| `text` | cleaned review text (HTML decoded, whitespace collapsed, wrapping quotes removed) |
| `rating`, `date`, `helpful_count` | 1-10; review date; max of usefulCount/likes |
| `duration` | DrugRev "Taken for …" bucket, else `None` |
| `n_listings`, `rating_conflict` | raw rows collapsed into this review; whether their ratings disagreed |
| `possibly_truncated` | `True` for all DrugRev-only rows (ADR-0012) |

## Layout

```
data/raw/<source>/         untouched downloads + manifest.json
data/interim/              cleaned, merged, de-duplicated parquet
data/processed/            splits, silver labels, mentions, signal tables
data/annotations/          gold set exports (IDs + labels, no review text)
```

## Provenance notes

- **UCI #462 (Drugs.com) withdrawn.** On 2026-10-13 its page returned HTTP 403 and the UCI API returned "Dataset with ID 462 not found"; #461 (Druglib) was still online. No public reason found. We use a copy of the train file obtained while it was public, for research only, and never redistribute it. A Hugging Face mirror exists (`Zakia/drugscom_reviews`, lists CC BY 4.0); deliberately **not** used.
- **Personal data.** DrugRev has a `UserName` column; it is dropped at load time and never stored in interim/processed data.

## Known issues

### DrugRev (verified 2026-10-13)
- **Truncated reviews.** Of 75,807 reviews also present in the UCI file (prefix-only match), only 27.6% are identical after whitespace normalisation; **27.7% are >10% shorter, 18.8% lost more than half their text**. With the stricter build match (93,517 pairs, see `reports/data/build_report.json`) 34.8% are >10% shorter.
  - Pattern: DrugRev keeps the **first N sentences** (often one) and cuts at a sentence boundary; 62% of truncated texts are an exact prefix of the UCI text. Multi-paragraph reviews are cut far more often (76% vs 26%). DrugRev also strips double quotes inside text.
  - Because cuts land on sentence boundaries, truncation is **not detectable per row** ("no terminal punctuation": precision 0.46, recall 0.18). All DrugRev-only rows are therefore flagged `possibly_truncated` (ADR-0012).
  - Truncation removes content but adds none: extraction precision unaffected, recall lower for DrugRev-only reviews.
- ~37% duplicate texts (144,080 rows).
- 25,036 rows have a date in `IntakeTime` (column shift); treat as "Not Specified".
- `IntakeTime` is specified for ~63% of rows (8 buckets from "<1 month" to "10 years or more").
- No HTML entities. Conditions are prefixed "For …" (e.g. "For Birth Control"); "Not Mentioned" = missing.
- Ratings bimodal (10 = 28%, 1 = 21%).

### Drugs.com UCI (train file)
- ~30% duplicate reviews (same text listed under multiple conditions/drug names) → de-duplicate before splitting.
- HTML entities in ~62% of reviews (`&#039;`, `&quot;`) → `html.unescape`.
- ~900 `condition` values are scraping artefacts (`"3</span> users found this comment helpful."`); 899 missing.
- Ratings bimodal (10 = 32%, 1 = 13%); a star rating measures satisfaction, not text sentiment.
- Median review ≈ 84 words; 99th percentile 154 words → `max_length=256` tokens covers ~99% (word-based estimate; confirm with tokenizer).

### Both
- Birth control is the largest condition (~7-18%) → report per-condition metrics; cap reviews per drug when sampling.

## Sampling for annotation / LLM labelling

Stratified by rating bucket, condition and source; cap reviews per drug; fixed seed in `configs/`. Gold set drawn only from reviews with full text (`possibly_truncated == False`, i.e. source `uci` or `both`).
