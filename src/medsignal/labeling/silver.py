"""Silver-label Drugs.com reviews with the chosen LLM teacher (Phase 2, ADR-0020).

Usage:
    uv run python -m medsignal.labeling.silver sample --config configs/silver.yaml
    uv run python -m medsignal.labeling.silver label  --config configs/silver.yaml [--limit N]

``sample`` draws the review IDs once (committed). ``label`` annotates them with the teacher model and
prompt from ``configs/teacher.yaml`` and appends one JSON line per review (labels with offsets +
provenance, no review text). It is resumable: re-running skips reviews already labelled.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from medsignal.data.splits import sentiment_bucket
from medsignal.labeling.benchmark import prompt_version
from medsignal.labeling.llm import AnnotationFailedError, OllamaAnnotator
from medsignal.labeling.sample_gold import draw
from medsignal.labeling.schema import SCHEMA_VERSION
from medsignal.utils import PROJECT_ROOT, git_sha, load_config, load_env

log = logging.getLogger(__name__)


def sample_silver(reviews: pd.DataFrame, splits: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Stratified sample from the student's training split (any source, truncated rows allowed)."""
    s = cfg["sample"]
    pool = reviews.merge(splits[["review_id", "drug_group", "split"]], on="review_id")
    pool = pool[
        (pool["split"] == s["source_split"]) & (pool["text"].str.len() >= int(s["min_chars"]))
    ]
    pool = pool.assign(bucket=pool["rating"].map(sentiment_bucket))
    rng = np.random.default_rng(int(cfg["seed"]))
    picked = draw(pool, int(s["size"]), s["sentiment_weights"], int(s["max_per_drug_group"]), rng)
    picked = picked.sample(frac=1, random_state=int(cfg["seed"]))
    cols = ["review_id", "drug", "condition", "rating", "bucket", "source", "drug_group"]
    return picked[cols].reset_index(drop=True)


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["review_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def label(cfg: dict[str, Any], limit: int | None) -> int:
    data_cfg = load_config(cfg["data_config"])
    teacher = load_config(cfg["teacher_config"])["teacher"]
    sample = pd.read_csv(PROJECT_ROOT / cfg["sample"]["out"])
    reviews = pd.read_parquet(PROJECT_ROOT / data_cfg["paths"]["interim"] / "reviews.parquet")
    todo = sample.merge(reviews[["review_id", "text"]], on="review_id", how="left")
    if todo["text"].isna().any():
        raise RuntimeError("silver sample contains review IDs missing from reviews.parquet")

    out = PROJECT_ROOT / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out)
    todo = todo[~todo["review_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    log.info(
        "silver: %s to label (%s done) with %s [%s]",
        len(todo),
        len(done),
        teacher["model"],
        teacher["prompt"],
    )

    llm = cfg["llm"]
    annotator = OllamaAnnotator(
        model=teacher["model"],
        temperature=float(llm["temperature"]),
        num_ctx=int(llm["num_ctx"]),
        max_retries=int(llm["max_retries"]),
        seed=int(cfg["seed"]),
        drop_unaligned_on_failure=bool(llm["drop_unaligned_on_failure"]),
        prompt=teacher["prompt"],
    )
    base_prov = {
        "annotator": f"llm:{teacher['model']}",
        "prompt_version": prompt_version(teacher["prompt"]),
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha(),
    }
    failures = 0
    with httpx.Client() as client, out.open("a", encoding="utf-8", newline="\n") as fh:
        for i, row in enumerate(todo.itertuples(index=False), 1):
            t0 = time.perf_counter()
            condition = row.condition if isinstance(row.condition, str) else None
            try:
                ann, meta = annotator.annotate(client, row.review_id, row.drug, condition, row.text)
                rec = {**ann.model_dump(mode="json"), "status": "ok"}
                prov = {
                    "attempts": meta["attempts"],
                    "dropped_spans": meta.get("dropped_spans", []),
                }
            except AnnotationFailedError as exc:
                failures += 1
                rec = {"review_id": row.review_id, "status": "failed", "error": str(exc)[:300]}
                prov = {}
            rec["provenance"] = {
                **base_prov,
                **prov,
                "seconds": round(time.perf_counter() - t0, 2),
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 250 == 0 or i == len(todo):
                log.info("silver: %s/%s (%s failures)", i, len(todo), failures)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "label"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/silver.yaml")
        if name == "label":
            p.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_env()
    cfg = load_config(args.config)

    if args.cmd == "sample":
        out = PROJECT_ROOT / cfg["sample"]["out"]
        if out.exists():
            raise SystemExit(f"{out} exists; the silver sample is fixed once drawn")
        data_cfg = load_config(cfg["data_config"])
        reviews = pd.read_parquet(PROJECT_ROOT / data_cfg["paths"]["interim"] / "reviews.parquet")
        splits = pd.read_parquet(PROJECT_ROOT / data_cfg["paths"]["processed"] / "splits.parquet")
        sample = sample_silver(reviews, splits, cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(out, index=False, lineterminator="\n")
        log.info("silver sample: %s", sample.groupby(["bucket", "source"]).size().to_dict())
        return 0
    label(cfg, args.limit)  # per-review failures are recorded in the output, not fatal
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
