"""Span-level ADE metrics (strict and relaxed), the standard way CADEC results are reported.

A span is a set of character ranges (``parts``) so CADEC's discontinuous gold spans are represented
exactly. Predictions are contiguous (one part).

- strict: predicted character set == gold character set
- relaxed: predicted and gold spans share at least one character

Matching is one-to-one (greedy, in document order) so a single prediction cannot satisfy two gold
spans. Micro-averaged precision/recall/F1 over all documents.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Span:
    parts: tuple[tuple[int, int], ...]

    @classmethod
    def contiguous(cls, start: int, end: int) -> Span:
        return cls(((start, end),))

    def chars(self) -> frozenset[int]:
        return frozenset(i for s, e in self.parts for i in range(s, e))

    @property
    def start(self) -> int:
        return min(s for s, _ in self.parts)


def _match(gold: Sequence[Span], pred: Sequence[Span], relaxed: bool) -> int:
    used: set[int] = set()
    tp = 0
    for g in sorted(gold, key=lambda s: s.start):
        gc = g.chars()
        for j, p in enumerate(pred):
            if j in used:
                continue
            pc = p.chars()
            if (gc & pc) if relaxed else (gc == pc):
                used.add(j)
                tp += 1
                break
    return tp


def span_prf(
    docs: Iterable[tuple[Sequence[Span], Sequence[Span]]], relaxed: bool = False
) -> dict[str, float]:
    """Micro P/R/F1 over (gold_spans, predicted_spans) pairs, one pair per document."""
    tp = n_gold = n_pred = 0
    for gold, pred in docs:
        tp += _match(gold, pred, relaxed)
        n_gold += len(gold)
        n_pred += len(pred)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "tp": tp,
        "n_gold": n_gold,
        "n_pred": n_pred,
    }


def bootstrap_f1_ci(
    docs: Sequence[tuple[Sequence[Span], Sequence[Span]]],
    relaxed: bool = False,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap CI for micro F1, resampling documents with replacement."""
    per_doc = [(_match(g, p, relaxed), len(g), len(p)) for g, p in docs]
    arr = np.array(per_doc, dtype=float).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    f1s = []
    for _ in range(n_boot):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        tp, n_gold, n_pred = s
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gold if n_gold else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    lo, hi = np.percentile(f1s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"f1_lo": round(float(lo), 4), "f1_hi": round(float(hi), 4), "n_boot": n_boot}


def _counts(docs: Sequence[tuple[Sequence[Span], Sequence[Span]]], relaxed: bool) -> np.ndarray:
    return np.array(
        [(_match(g, p, relaxed), len(g), len(p)) for g, p in docs], dtype=float
    ).reshape(-1, 3)


def _f1_from(sums: np.ndarray) -> float:
    tp, n_gold, n_pred = sums
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return float(2 * p * r / (p + r)) if p + r else 0.0


def paired_bootstrap_diff(
    docs_a: Sequence[tuple[Sequence[Span], Sequence[Span]]],
    docs_b: Sequence[tuple[Sequence[Span], Sequence[Span]]],
    relaxed: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """F1(B) - F1(A) on the same documents, with a paired percentile bootstrap CI.

    ``p_b_not_better`` is the bootstrap share of resamples where B does not beat A.
    """
    a, b = _counts(docs_a, relaxed), _counts(docs_b, relaxed)
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same documents in the same order")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        diffs.append(_f1_from(b[idx].sum(axis=0)) - _f1_from(a[idx].sum(axis=0)))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "diff": round(_f1_from(b.sum(axis=0)) - _f1_from(a.sum(axis=0)), 4),
        "diff_lo": round(float(lo), 4),
        "diff_hi": round(float(hi), 4),
        "p_b_not_better": round(float(np.mean(np.array(diffs) <= 0)), 4),
        "n_boot": n_boot,
    }
