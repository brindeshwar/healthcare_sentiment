# AGENTS.md

Instructions for **any** coding agent (Claude Code, Codex, Cursor, Copilot, Gemini CLI, …) and for human contributors.
This file is the single source of truth for *how* to work in this repo. Tool-specific files (e.g. `CLAUDE.md`) only point here.

---

## 1. Project in one paragraph

**MedSignal** turns ~215K patient drug reviews (Drugs.com, WebMD) into structured drug-safety evidence.
A local LLM labels a sample of reviews (side effects, severity, discontinuation + reason); those labels are
validated against a human gold set and distilled into a fast encoder (BioClinical ModernBERT) that processes every review.
Extracted side effects are normalised to MedDRA Preferred Terms (as used by FDA FAERS), then compared against
FAERS reports and drug labels with standard disproportionality statistics (PRR/ROR).
A Streamlit/Gradio evidence explorer shows per-drug findings with the real review quotes behind them.

Constraints: **$0 budget** (no paid APIs), **single RTX 4060 Laptop GPU (8 GB)**, Windows 11.

---

## 2. Before you start any task (mandatory)

1. Read [`docs/STATUS.md`](docs/STATUS.md): current phase, what's in progress, blockers, next steps.
2. Read the relevant phase in [`docs/PLAN.md`](docs/PLAN.md): scope and acceptance criteria.
3. Skim [`docs/DECISIONS.md`](docs/DECISIONS.md) before proposing a change of approach. **Do not silently reverse a recorded decision.** If you think one is wrong, say so and propose a new ADR.

## 3. When you finish a task or end a session (mandatory)

1. Update `docs/STATUS.md`: move items, add a dated entry to the **Session log** (date, agent/person, what changed, what's next).
2. Tick checkboxes in `docs/PLAN.md` that are now complete (only if acceptance criteria are actually met).
3. If you made a non-obvious choice (library, method, threshold, data rule), add an ADR to `docs/DECISIONS.md`.
4. If user-visible behaviour changed, add a line under `## [Unreleased]` in `CHANGELOG.md`.

These four files are how agents hand work to each other. Keeping them accurate matters more than any single feature.

---

## 4. Repository layout

```
src/medsignal/        Python package (all importable code lives here)
  data/               loading, cleaning, de-duplication, splits
  labeling/           LLM teacher: prompts, schemas, Ollama client
  extraction/         encoder student: training + inference (span tagging, classifiers)
  normalization/      mention → MedDRA PT (retrieve → rerank → verify)
  signals/            FAERS / label comparison, PRR/ROR
  evaluation/         metrics, gold-set scoring, reports
  app/                evidence explorer UI
configs/              YAML configs (one per experiment/stage); no hard-coded paths in code
data/                 NOT committed. raw/ interim/ processed/ annotations/ (gold set is the exception, see DATA.md)
models/               NOT committed. Trained weights go to Hugging Face Hub.
reports/              generated metrics/figures (committed only when referenced by README/write-up)
notebooks/            exploration only; never imported by src/
tests/                pytest
docs/                 PLAN, STATUS, DECISIONS, DATA, RESEARCH, SETUP
```

## 5. Commands

Environment is managed with **uv** (Python 3.12). Never `pip install` into the global interpreter.

```bash
uv sync                              # install/update env from uv.lock
uv sync --extra gpu                  # + CUDA torch & training deps (extra added in Phase 1)
uv run pytest                        # tests
uv run ruff check . --fix            # lint
uv run ruff format .                 # format
uv run mypy src                      # type-check
uv run pre-commit run --all-files    # everything CI runs
uv add <pkg>   /  uv add --dev <pkg> # add dependency (updates pyproject + lock)
```

Pipeline entry points are run as modules, e.g. `uv run python -m medsignal.data.build --config configs/data.yaml`.
Each stage reads a config, writes outputs under `data/` or `reports/`, and logs the config + git SHA next to its outputs.

---

## 6. Coding conventions

- Python 3.12, type hints on all public functions, `ruff` (lint + format) is the style authority.
- Small, pure functions; I/O at the edges. No notebooks in the pipeline.
- **No hard-coded paths or magic numbers.** Put them in `configs/*.yaml`.
- **Reproducibility:** every stochastic step takes a seed from config; call `medsignal.utils.seed_everything`.
- Logging via `logging` (not `print`) in library code. No emoji in logs or docs.
- Docstrings: one-line summary; add detail only where the *why* isn't obvious.
- Tests: every data transformation and metric gets a unit test with a tiny fixture under `tests/fixtures/`.

## 7. Scientific-integrity rules (non-negotiable)

This project exists partly because a previous version reported fabricated results. Therefore:

1. **Every number** in README, reports, or the write-up must be produced by code in this repo and traceable to a saved output (`reports/…json`) and config.
2. **Never** simulate, hard-code, or "estimate" a result and present it as measured. Placeholders must be visibly marked `TBD`.
3. Splits are made **after** de-duplication; test data is never used for model selection or prompt tuning.
4. Report macro-F1 (not only accuracy), with mean ± std over seeds where training is involved.
5. Gold-set annotations are made by a human. LLM pre-annotation is allowed but must be reviewed and is recorded as such.
6. State limitations honestly (reviews are biased, signals are hypotheses, not causal findings).

## 8. Data & safety rules

- Raw datasets are **not redistributed** (licence terms). Commit download scripts, not data.
- Never commit secrets. Use `.env` (see `.env.example`); `OPENFDA_API_KEY`, `HF_TOKEN`.
- No paid API calls. LLMs run locally via Ollama.
- Outputs are research artefacts, **not medical advice**; UI and model cards must say so.
- Files > 5 MB are blocked by pre-commit; weights go to the HF Hub.

## 9. Git workflow

- `main` is always releasable. Work on branches: `feat/…`, `fix/…`, `exp/…` (experiments), `docs/…`, `chore/…`.
- Commit messages: [Conventional Commits](https://www.conventionalcommits.org/) (`feat(labeling): add ADE JSON schema`).
- Open a PR using the template; CI (lint, type-check, tests) must pass. Squash-merge.
- Agents: **do not push, merge, tag, or change GitHub settings unless the user explicitly asks.**

## 10. Hardware & local paths

- **Everything stays inside the project folder** (owner preference): uv cache `.cache/uv`, Hugging Face cache `HF_HOME=.cache/huggingface`, Ollama program `tools/ollama`, Ollama weights `models/ollama` (`OLLAMA_MODELS`). Never write caches or downloads elsewhere. See `docs/SETUP.md`.

- RTX 4060 Laptop, 8 GB VRAM. Use bf16/fp16, gradient accumulation, `max_length` 256 (covers ~99% of reviews).
- LLM inference: Ollama, ~9B models at Q4 (~6 GB). LLM fine-tuning is limited to ≤2B models (see ADR-0005).
- Long GPU jobs must be resumable (checkpoint + skip already-processed IDs) because laptops sleep and throttle.
