# MedSignal

**Drug-safety signals from patient reviews.** A local-LLM-distilled pipeline that extracts adverse drug events and
reasons for stopping medication from ~215K patient reviews, maps them to MedDRA terms, and compares them with FDA
adverse-event reports (FAERS) and drug labels.

[![CI](https://github.com/brindeshwar/medsignal/actions/workflows/ci.yml/badge.svg)](https://github.com/brindeshwar/medsignal/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in%20development-orange)

> **Status:** under active development (Phase 1 of 5). No results are reported yet; every number that appears here
> will be produced by code in this repo. See [`docs/STATUS.md`](docs/STATUS.md) and [`docs/PLAN.md`](docs/PLAN.md).

> **Not medical advice.** Outputs are research hypotheses derived from unverified patient reports.

## Why

Drug labels and FDA reports miss part of the patient experience. Patients describe side effects, and why they quit a
drug, in their own words online. Prior studies compared reviews with FAERS by hand for single drugs. MedSignal
automates this across 3,000+ drugs and **measures every stage** against human-labelled data.

## How it works

```
Reviews (Drugs.com, WebMD)
  → LLM teacher (local, Ollama) labels ~10K reviews        → validated on a 300-review human gold set
  → encoder student (BioClinical ModernBERT) labels all    → compared with teacher and GLiNER-BioMed
  → normalise mentions to MedDRA PTs (retrieve → rerank → verify)
  → compare with FAERS + drug labels (PRR / ROR)
  → evidence explorer: per-drug side-effect profile, stop reasons, real review quotes
```

Design decisions and their reasoning: [`docs/DECISIONS.md`](docs/DECISIONS.md) · Related work: [`docs/RESEARCH.md`](docs/RESEARCH.md) · Data: [`docs/DATA.md`](docs/DATA.md)

## Results

TBD. Filled in as phases complete.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. GPU stages need an NVIDIA GPU (developed on an 8 GB RTX 4060 Laptop) and [Ollama](https://ollama.com). Accounts, keys and project-local install paths: [`docs/SETUP.md`](docs/SETUP.md).

```bash
git clone https://github.com/brindeshwar/medsignal.git
cd medsignal
uv sync
uv run pytest
```

Fetch data (DrugRev + Druglib are downloaded and checksum-verified; see [`docs/DATA.md`](docs/DATA.md) for the manual sources):

```bash
uv run python -m medsignal.data.download
```

Further pipeline commands are added here as phases complete.

## Repository layout

```
src/medsignal/   package: data, labeling, extraction, normalization, signals, evaluation, app
configs/         YAML configs for every stage
docs/            plan, status, decisions, data card, related work
tests/           pytest suite
```

## Contributing / working with AI agents

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Coding agents follow [`AGENTS.md`](AGENTS.md).

## Licence & data

Code: MIT. Datasets are **not** redistributed; download scripts fetch them from their sources under their own terms (see [`docs/DATA.md`](docs/DATA.md)).

## Acknowledgements

Drug review data: Gräßer, Kallumadi, Malberg & Zaunseder (2018), UCI ML Repository. FDA data via [openFDA](https://open.fda.gov).
