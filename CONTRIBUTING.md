# Contributing

Thanks for helping. This project is worked on by a human owner and several AI coding agents, so the process below is
deliberately explicit. AI agents should also read [`AGENTS.md`](AGENTS.md).

## Setup

```bash
uv sync                      # create .venv from uv.lock
uv run pre-commit install    # run checks on every commit
cp .env.example .env         # then fill in keys (never commit .env)
```

## Workflow

1. Pick or open an issue (templates: **Task**, **Experiment**, **Bug**). Tasks map to phases in `docs/PLAN.md`.
2. Branch from `main`: `feat/<short-name>`, `fix/…`, `exp/…`, `docs/…`, `chore/…`.
3. Commit with [Conventional Commits](https://www.conventionalcommits.org/): `feat(extraction): add BIO conversion`.
4. Before pushing: `uv run pre-commit run --all-files` and `uv run pytest`.
5. Open a PR with the template; CI must pass; squash-merge.
6. Update `docs/STATUS.md` (and PLAN/DECISIONS/CHANGELOG as applicable) in the same PR.

## Experiments

- One experiment = one issue (hypothesis, metric, decision rule) + one config file in `configs/`.
- Outputs go to `reports/<stage>/<run-name>/` with `metrics.json`, the resolved config and the git SHA.
- Report mean ± std over seeds. Never tune on the test split.
- Record the outcome in the issue and, if it changes the approach, add an ADR to `docs/DECISIONS.md`.

## Code standards

Ruff (lint + format), mypy (strict), pytest. Type hints on public functions; config-driven; no hard-coded paths;
`logging` not `print` in library code. See AGENTS.md §6-8.

## Releases

Semantic versioning. Move `Unreleased` entries in `CHANGELOG.md` under a new version heading, tag `vX.Y.Z` on `main`.
