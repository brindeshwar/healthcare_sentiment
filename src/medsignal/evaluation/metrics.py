"""Classification metrics used across the project (AGENTS.md §7.4: macro-F1 first)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

SENTIMENT_LABELS = ("neg", "neu", "pos")


def sentiment3(rating: int) -> int:
    """Star rating -> ordinal 3-class sentiment: 0 = neg (1-4), 1 = neu (5-6), 2 = pos (7-10)."""
    if rating <= 4:
        return 0
    if rating <= 6:
        return 1
    return 2


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Sequence[str] = SENTIMENT_LABELS,
) -> dict[str, Any]:
    """Macro-F1, accuracy, quadratic-weighted kappa (ordinal), per-class F1, confusion matrix."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    idx = list(range(len(labels)))
    per_class = f1_score(yt, yp, labels=idx, average=None, zero_division=0)
    return {
        "n": len(yt),
        "macro_f1": round(float(f1_score(yt, yp, labels=idx, average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "qwk": round(float(cohen_kappa_score(yt, yp, labels=idx, weights="quadratic")), 4),
        "f1_per_class": {lab: round(float(v), 4) for lab, v in zip(labels, per_class, strict=True)},
        "confusion_matrix": confusion_matrix(yt, yp, labels=idx).tolist(),
    }


def mean_std(runs: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    """Mean and sample std of a scalar metric across seeds."""
    vals = np.array([r[key] for r in runs], dtype=float)
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return {"mean": round(float(vals.mean()), 4), "std": round(std, 4), "n_runs": len(vals)}
