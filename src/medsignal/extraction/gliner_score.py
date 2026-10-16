"""Score zero-shot GLiNER-BioMed predictions on CADEC (Phase 3 baseline, ADR-0021).

Usage (after ``scripts/gliner_predict.py`` for dev and test):
    uv run python -m medsignal.extraction.gliner_score --config configs/student.yaml

The label subset and score threshold are chosen on **dev** (relaxed span F1), then applied once to
**test**. Overlapping predictions are merged (keep the highest-scoring span) because GLiNER may
return the same mention under several labels.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medsignal.evaluation.spans import Span, bootstrap_f1_ci, span_prf
from medsignal.extraction.train import cadec_docs
from medsignal.utils import PROJECT_ROOT, git_sha, load_config

log = logging.getLogger(__name__)

LABEL_SETS = [
    ("adverse drug reaction",),
    ("side effect",),
    ("adverse drug reaction", "side effect"),
    ("adverse drug reaction", "side effect", "symptom"),
]
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def select_spans(
    spans: list[dict[str, Any]], labels: tuple[str, ...], threshold: float
) -> list[tuple[int, int]]:
    """Spans with an allowed label above threshold; overlapping spans resolved by highest score."""
    cand = sorted(
        (s for s in spans if s["label"] in labels and s["score"] >= threshold),
        key=lambda s: -s["score"],
    )
    kept: list[tuple[int, int]] = []
    for s in cand:
        if all(s["end"] <= a or s["start"] >= b for a, b in kept):
            kept.append((s["start"], s["end"]))
    return sorted(kept)


def _load(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        r["doc_id"]: r["spans"]
        for r in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    }


def evaluate(
    docs: list[Any], preds: dict[str, list[dict[str, Any]]], labels: tuple[str, ...], thr: float
) -> list[tuple[list[Span], list[Span]]]:
    return [
        (d.ades, [Span.contiguous(a, b) for a, b in select_spans(preds[d.doc_id], labels, thr)])
        for d in docs
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--pred-dir", default="data/processed/gliner")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    pred_dir = PROJECT_ROOT / args.pred_dir

    dev_docs, test_docs = cadec_docs(cfg, "dev"), cadec_docs(cfg, "test")
    dev, test = _load(pred_dir / "cadec_dev.jsonl"), _load(pred_dir / "cadec_test.jsonl")

    grid: list[dict[str, Any]] = []
    for labels, thr in itertools.product(LABEL_SETS, THRESHOLDS):
        m = span_prf(evaluate(dev_docs, dev, labels, thr), relaxed=True)
        grid.append({"labels": list(labels), "threshold": thr, "dev_relaxed_f1": m["f1"]})
    best = max(grid, key=lambda g: g["dev_relaxed_f1"])
    labels, thr = tuple(best["labels"]), float(best["threshold"])
    log.info("chosen on dev: %s @ %.1f (relaxed F1 %.3f)", labels, thr, best["dev_relaxed_f1"])

    result: dict[str, Any] = {
        "model": "Ihor/gliner-biomed-large-v1.0",
        "chosen": best,
        "dev_grid": grid,
    }
    for name, docs, preds in (("cadec_dev", dev_docs, dev), ("cadec_test", test_docs, test)):
        pairs = evaluate(docs, preds, labels, thr)
        result[name] = {
            "docs": len(docs),
            "strict": {**span_prf(pairs), **bootstrap_f1_ci(pairs)},
            "relaxed": {**span_prf(pairs, relaxed=True), **bootstrap_f1_ci(pairs, relaxed=True)},
        }
    result["meta"] = {
        "git_sha": git_sha(),
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = PROJECT_ROOT / cfg["report_dir"] / "gliner_biomed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("test relaxed F1 %.3f -> %s", result["cadec_test"]["relaxed"]["f1"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
