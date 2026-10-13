"""Shared helpers: config loading, seeding, run metadata."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# .env variables holding paths; relative values are resolved against the project root so caches
# stay inside the project folder regardless of the working directory (AGENTS.md §10).
_PATH_VARS = ("HF_HOME",)


def load_env() -> None:
    """Load the project's .env (if present) without overriding variables already set."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    for var in _PATH_VARS:
        value = os.environ.get(var)
        if value and not Path(value).is_absolute():
            os.environ[var] = str(PROJECT_ROOT / value)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config; relative paths are resolved against the project root."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {p} must be a mapping, got {type(cfg).__name__}")
    return cfg


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)  # noqa: NPY002 - seeds libraries that still use the global RNG
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_sha() -> str:
    """Current commit SHA (+ "-dirty" if tracked non-report files changed), or "unknown".

    Generated outputs under reports/ are excluded: they do not change the code that made them.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!reports"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
