"""Build the cleaned, merged, de-duplicated review table.

Usage:
    uv run python -m medsignal.data.build --config configs/data.yaml

Writes ``data/interim/reviews.parquet`` and ``reports/data/build_report.json`` (counts for every
step, cross-source match quality, truncation-heuristic check, config + git SHA).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from medsignal.data.clean import load_drugrev, load_uci, truncated_name_map
from medsignal.data.dedup import dedup_within, merge_sources
from medsignal.utils import PROJECT_ROOT, git_sha, load_config

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "review_id",
    "source",
    "drug",
    "drug_aliases",
    "condition",
    "conditions",
    "text",
    "rating",
    "date",
    "helpful_count",
    "duration",
    "n_listings",
    "rating_conflict",
    "possibly_truncated",
]


def truncation_heuristic_quality(pairs: pd.DataFrame, threshold: float = 0.9) -> dict[str, float]:
    """How well 'no terminal punctuation' would predict real truncation, on matched pairs.

    Kept as evidence: it is too weak to use (see ADR-0011), so all DrugRev-only rows are flagged.

    Ground truth: the DrugRev text is shorter than ``threshold`` x the UCI text.
    """
    truth = pairs["length_ratio"] < threshold
    pred = pairs["drugrev_no_terminal_punct"]
    tp = int((truth & pred).sum())
    return {
        "truncated_rate": round(float(truth.mean()), 4),
        "heuristic_precision": round(tp / max(int(pred.sum()), 1), 4),
        "heuristic_recall": round(tp / max(int(truth.sum()), 1), 4),
    }


def build(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run load -> clean -> de-dup -> merge -> filter; return table and report."""
    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    min_chars = int(cfg["cleaning"]["min_review_chars"])
    report: dict[str, Any] = {}

    uci_raw = load_uci(raw / "drugscom_uci" / "drugsComTrain_raw.csv")
    drugrev_raw = load_drugrev(raw / "drugrev" / "DrugReviews.csv")
    report["rows_raw"] = {"uci": len(uci_raw), "drugrev": len(drugrev_raw)}

    uci_conds = set(uci_raw["condition"].dropna())
    drugrev_conds = set(drugrev_raw["condition"].dropna())
    cond_map = truncated_name_map(uci_conds - drugrev_conds, uci_conds | drugrev_conds)
    for frame in (uci_raw, drugrev_raw):
        frame["condition"] = frame["condition"].map(lambda c: cond_map.get(c, c) if c else c)
    report["truncated_condition_names_fixed"] = len(cond_map)
    report["truncated_condition_examples"] = dict(sorted(cond_map.items())[:10])

    uci = dedup_within(uci_raw)
    drugrev = dedup_within(drugrev_raw)
    report["rows_after_within_source_dedup"] = {"uci": len(uci), "drugrev": len(drugrev)}
    report["rating_conflicts_within_source"] = {
        "uci": int(uci["rating_conflict"].sum()),
        "drugrev": int(drugrev["rating_conflict"].sum()),
    }

    merged, pairs = merge_sources(uci, drugrev)
    report["cross_source_matches"] = len(pairs)
    report["truncation_on_matched_pairs"] = truncation_heuristic_quality(pairs)

    n_before = len(merged)
    merged = merged.drop_duplicates("review_id")
    report["dropped_review_id_collisions"] = n_before - len(merged)

    short = merged["text"].str.len() < min_chars
    report["dropped_short_reviews"] = int(short.sum())
    merged = merged.loc[~short, OUTPUT_COLUMNS].sort_values(["date", "review_id"])
    merged = merged.reset_index(drop=True)

    report["rows_final"] = len(merged)
    report["rows_final_by_source"] = merged["source"].value_counts().to_dict()
    report["possibly_truncated"] = int(merged["possibly_truncated"].sum())
    report["with_duration"] = int(merged["duration"].notna().sum())
    report["unique_drugs"] = int(merged["drug"].nunique())
    report["date_range"] = [f"{merged['date'].min():%Y-%m-%d}", f"{merged['date'].max():%Y-%m-%d}"]
    report["rating_counts"] = {
        int(k): int(v) for k, v in merged["rating"].value_counts().sort_index().items()
    }
    return merged, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    table, report = build(cfg)

    out = PROJECT_ROOT / cfg["paths"]["interim"] / "reviews.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    report_path = PROJECT_ROOT / "reports" / "data" / "build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["meta"] = {
        "config": args.config,
        "git_sha": git_sha(),
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "output": str(Path(out).relative_to(PROJECT_ROOT).as_posix()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("wrote %s rows to %s", len(table), out)
    log.info("report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
