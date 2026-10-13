"""Pre-annotate gold-dev reviews with the local LLM (resumable).

Usage:
    uv run python -m medsignal.labeling.preannotate --config configs/labeling.yaml

Appends one JSON line per review to the output file (labels + provenance, no review text) and skips
reviews already present, so an interrupted run can simply be restarted. Only the splits listed in
``preannotate.splits`` are processed; the gold test set is annotated blind (ADR-0017).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from medsignal.labeling.llm import OllamaAnnotator, provenance
from medsignal.utils import PROJECT_ROOT, load_config, load_env

log = logging.getLogger(__name__)


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["review_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/labeling.yaml")
    parser.add_argument("--limit", type=int, help="process at most N new reviews (smoke test)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_env()

    cfg = load_config(args.config)
    pcfg = cfg["preannotate"]
    data_cfg = load_config(cfg["data_config"])
    gold = pd.read_csv(PROJECT_ROOT / cfg["gold"]["out"])
    gold = gold[gold["gold_split"].isin(pcfg["splits"])]
    reviews = pd.read_parquet(PROJECT_ROOT / data_cfg["paths"]["interim"] / "reviews.parquet")
    todo = gold.merge(reviews[["review_id", "text"]], on="review_id").sort_values("order")

    out = PROJECT_ROOT / pcfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = done_ids(out)
    todo = todo[~todo["review_id"].isin(skip)]
    if args.limit:
        todo = todo.head(args.limit)
    log.info("%s reviews to annotate (%s already done)", len(todo), len(skip))

    annotator = OllamaAnnotator(
        model=pcfg["model"],
        temperature=float(pcfg["temperature"]),
        num_ctx=int(pcfg["num_ctx"]),
        max_retries=int(pcfg["max_retries"]),
        seed=int(cfg["seed"]),
        drop_unaligned_on_failure=bool(pcfg["drop_unaligned_on_failure"]),
    )
    failures = 0
    with httpx.Client() as client, out.open("a", encoding="utf-8", newline="\n") as fh:
        for i, row in enumerate(todo.itertuples(index=False), 1):
            t0 = time.perf_counter()
            try:
                ann, meta = annotator.annotate(
                    client,
                    row.review_id,
                    row.drug,
                    row.condition if isinstance(row.condition, str) else None,
                    row.text,
                )
            except RuntimeError as exc:
                failures += 1
                log.warning("%s", exc)
                continue
            record = {
                **ann.model_dump(mode="json"),
                "provenance": {
                    **provenance(annotator.model),
                    **meta,
                    "seconds": round(time.perf_counter() - t0, 2),
                    "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(todo):
                log.info("%s/%s done", i, len(todo))
    log.info("finished; %s failures", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
