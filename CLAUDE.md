# CLAUDE.md

@AGENTS.md

## Claude Code specifics

- The shared instructions above (AGENTS.md) are authoritative; this file only adds Claude-specific notes.
- Start each session by reading `docs/STATUS.md` (and the active phase in `docs/PLAN.md`); end it by updating them per AGENTS.md §3.
- Use the Bash tool with `uv run …` for all Python commands; the shell is Git Bash on Windows (use `/d/healthcare_sentiment` paths in Bash).
- Long GPU jobs (LLM labelling, training): run in the background and make them resumable; never block the session waiting on them.
- Prefer asking the user before: downloading datasets, creating accounts/keys, pushing, or any GitHub settings change.
- The user wants to be able to explain every decision in interviews: when you make a choice, record the *why* in `docs/DECISIONS.md`, in plain language.
