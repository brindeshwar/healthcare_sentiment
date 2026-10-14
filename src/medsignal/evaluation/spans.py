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
