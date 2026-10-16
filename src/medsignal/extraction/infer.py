"""Run the student over the whole corpus (Phase 3, ADR-0022).

Usage:
    uv run --extra gpu python -m medsignal.extraction.infer --config configs/student.yaml [--limit N]

Model choice is by rule, from saved reports: span model = seed with the best CADEC **dev** relaxed
F1; review model = seed with the best held-out silver agreement (discontinued macro-F1); ties go to
the lowest seed. Reviews are processed in chunks written as parquet parts (resumable: existing parts
are skipped), then merged into ``data/processed/student/mentions.parquet`` (one row per ADE mention)
and ``review_labels.parquet`` (one row per review).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from medsignal.extraction.review_heads import decode, predict_probs
from medsignal.extraction.train import predict_spans
from medsignal.utils import PROJECT_ROOT, git_sha, load_config, load_env

log = logging.getLogger(__name__)


def choose_seed(reports: dict[int, dict[str, Any]], key: list[str]) -> int:
    """Seed with the highest value at ``key`` path; ties -> lowest seed."""

    def val(r: dict[str, Any]) -> float:
        for k in key:
            r = r[k]
        return float(r)  # type: ignore[arg-type]

    return sorted(reports, key=lambda s: (-val(reports[s]), s))[0]


def _reports(report_dir: Path, prefix: str) -> dict[int, dict[str, Any]]:
    out = {}
    for p in report_dir.glob(f"{prefix}*.json"):
        if "smoke" in p.name or "summary" in p.name:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        out[int(r["seed"])] = r
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--limit", type=int, help="only the first N reviews (smoke test)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    cfg = load_config(args.config)
    inf = cfg["inference"]
    report_dir = PROJECT_ROOT / cfg["report_dir"]
    span_seed = choose_seed(_reports(report_dir, "seed"), ["cadec_dev", "relaxed", "f1"])
    review_seed = choose_seed(
        _reports(report_dir, "reviews_seed"), ["agreement_with_teacher", "discontinued", "macro_f1"]
    )
    log.info("span model seed %s, review model seed %s", span_seed, review_seed)

    models_dir = PROJECT_ROOT / cfg["output_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    span_path = models_dir / f"seed{span_seed}" / "best"
    review_path = models_dir / f"reviews_seed{review_seed}" / "best"
    tok = AutoTokenizer.from_pretrained(span_path)
    span_model = AutoModelForTokenClassification.from_pretrained(
        span_path, dtype=torch.bfloat16
    ).to(device)
    review_model = AutoModelForSequenceClassification.from_pretrained(
        review_path, dtype=torch.bfloat16
    ).to(device)

    data_cfg = load_config(cfg["data_config"])
    reviews = pd.read_parquet(
        PROJECT_ROOT / data_cfg["paths"]["interim"] / "reviews.parquet",
        columns=["review_id", "text", "drug", "condition", "date", "source", "rating"],
    )
    if args.limit:
        reviews = reviews.head(args.limit)
    out_dir = PROJECT_ROOT / inf["out_dir"]
    # Smoke runs (--limit) write to their own folder so they can never be mistaken for finished
    # chunks of the full run.
    parts = out_dir / ("parts" if not args.limit else f"parts_smoke{args.limit}")
    parts.mkdir(parents=True, exist_ok=True)
    chunk = int(inf["chunk_size"])
    t0 = time.perf_counter()
    for start in range(0, len(reviews), chunk):
        part = parts / f"part-{start // chunk:04d}"
        df = reviews.iloc[start : start + chunk]
        done_path = part.with_suffix(".reviews.parquet")
        if done_path.exists():
            done_ids = pd.read_parquet(done_path, columns=["review_id"])["review_id"]
            if done_ids.tolist() != df["review_id"].tolist():
                raise RuntimeError(f"{done_path} does not match chunk {start}; delete it and rerun")
            continue
        texts = df["text"].tolist()
        spans, _ = predict_spans(
            span_model, tok, texts, int(inf["max_length"]), int(inf["batch_size"])
        )
        probs, _ = predict_probs(
            review_model, tok, texts, int(inf["max_length"]), int(inf["batch_size"])
        )
        mentions = [
            {"review_id": rid, "start": a, "end": b, "text": text[a:b]}
            for rid, text, ss in zip(df["review_id"], texts, spans, strict=True)
            for a, b in ss
        ]
        labels = [
            {"review_id": rid, **decode(p), "n_ades": len(ss)}
            for rid, p, ss in zip(df["review_id"], probs, spans, strict=True)
        ]
        pd.DataFrame(mentions, columns=["review_id", "start", "end", "text"]).to_parquet(
            part.with_suffix(".mentions.parquet"), index=False
        )
        pd.DataFrame(labels).to_parquet(part.with_suffix(".reviews.parquet"), index=False)
        done = min(start + chunk, len(reviews))
        log.info("%s/%s reviews (%.0f/s)", done, len(reviews), done / (time.perf_counter() - t0))

    mentions = pd.concat(
        [pd.read_parquet(p) for p in sorted(parts.glob("*.mentions.parquet"))], ignore_index=True
    )
    labels = pd.concat(
        [pd.read_parquet(p) for p in sorted(parts.glob("*.reviews.parquet"))], ignore_index=True
    )
    meta_cols = reviews.drop(columns=["text"])
    mentions = mentions.merge(meta_cols, on="review_id", how="left")
    labels = labels.merge(meta_cols, on="review_id", how="left")
    suffix = f"_smoke{args.limit}" if args.limit else ""
    mentions.to_parquet(out_dir / f"mentions{suffix}.parquet", index=False)
    labels.to_parquet(out_dir / f"review_labels{suffix}.parquet", index=False)
    summary = {
        "reviews": len(labels),
        "mentions": len(mentions),
        "reviews_with_ade": int((labels["n_ades"] > 0).sum()),
        "span_model_seed": span_seed,
        "review_model_seed": review_seed,
        "discontinued": labels["discontinued"].value_counts().to_dict(),
        "effectiveness": labels["effectiveness"].value_counts().to_dict(),
        "stop_reasons": labels["stop_reasons"].explode().dropna().value_counts().to_dict(),
        "top_mentions": mentions["text"].str.lower().value_counts().head(25).to_dict(),
        "git_sha": git_sha(),
    }
    if not args.limit:
        (report_dir / "corpus_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    log.info("done: %s reviews, %s mentions", summary["reviews"], summary["mentions"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
