"""Draw the 300-review gold sample (100 dev from `val`, 200 test from `test`).

Usage:
    uv run python -m medsignal.labeling.sample_gold --config configs/labeling.yaml

Writes review IDs and metadata only (no review text) to ``data/annotations/gold_sample.csv``, which
is committed so the sample is fixed and auditable. Refuses to overwrite an existing sample.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd

from medsignal.data.splits import sentiment_bucket
from medsignal.utils import PROJECT_ROOT, load_config

log = logging.getLogger(__name__)


def eligible_pool(reviews: pd.DataFrame, splits: pd.DataFrame, min_chars: int) -> pd.DataFrame:
    """Full-text reviews with their split, drug group and sentiment bucket."""
    df = reviews.merge(splits[["review_id", "drug_group", "split"]], on="review_id")
    df = df[~df["possibly_truncated"] & (df["text"].str.len() >= min_chars)]
    return df.assign(bucket=df["rating"].map(sentiment_bucket))


def draw(
    pool: pd.DataFrame,
    n: int,
    weights: dict[str, float],
    max_per_group: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample ``n`` reviews with bucket quotas and at most ``max_per_group`` per drug group."""
    quotas = {b: round(n * w) for b, w in weights.items()}
    quotas[max(quotas, key=lambda b: quotas[b])] += n - sum(quotas.values())  # fix rounding
    picked: list[pd.DataFrame] = []
    per_group: dict[str, int] = {}
    for bucket, quota in quotas.items():
        cand = pool[pool["bucket"] == bucket]
        order = rng.permutation(len(cand))
        rows = []
        for i in order:
            row = cand.iloc[i]
            g = row["drug_group"]
            if per_group.get(g, 0) >= max_per_group:
                continue
            rows.append(i)
            per_group[g] = per_group.get(g, 0) + 1
            if len(rows) == quota:
                break
        if len(rows) < quota:
            raise ValueError(
                f"only {len(rows)} eligible reviews for bucket {bucket!r}, need {quota}"
            )
        picked.append(cand.iloc[rows])
    return pd.concat(picked)


def sample_gold(
    reviews: pd.DataFrame, splits: pd.DataFrame, cfg: dict[str, Any], seed: int
) -> pd.DataFrame:
    g = cfg["gold"]
    rng = np.random.default_rng(seed)
    pool = eligible_pool(reviews, splits, int(g["min_chars"]))
    parts = []
    for gold_split, n in g["sizes"].items():
        src = pool[pool["split"] == g["source_split"][gold_split]]
        part = draw(src, int(n), g["sentiment_weights"], int(g["max_per_drug_group"]), rng)
        parts.append(part.assign(gold_split=gold_split))
    out = pd.concat(parts)
    out = out.sample(frac=1, random_state=seed)  # annotation order: shuffled, not grouped by bucket
    out = out.assign(order=range(1, len(out) + 1))
    cols = [
        "order",
        "review_id",
        "gold_split",
        "split",
        "drug",
        "condition",
        "rating",
        "bucket",
        "source",
    ]
    return out[cols].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/labeling.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    data_cfg = load_config(cfg["data_config"])
    out = PROJECT_ROOT / cfg["gold"]["out"]
    if out.exists():
        raise SystemExit(
            f"{out} exists; the gold sample is fixed once drawn (delete it deliberately to redraw)"
        )

    reviews = pd.read_parquet(PROJECT_ROOT / data_cfg["paths"]["interim"] / "reviews.parquet")
    splits = pd.read_parquet(PROJECT_ROOT / data_cfg["paths"]["processed"] / "splits.parquet")
    gold = sample_gold(reviews, splits, cfg, int(cfg["seed"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(out, index=False, lineterminator="\n")
    log.info("gold sample: %s", gold.groupby(["gold_split", "bucket"]).size().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
