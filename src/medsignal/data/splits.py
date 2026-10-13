"""Create train / val / test and drug-held-out splits from the de-duplicated review table.

Usage:
    uv run python -m medsignal.data.splits --config configs/data.yaml

Design (ADR-0013):
- Splits are made after de-duplication (AGENTS.md §7.3).
- Drug names that co-occur on one review (brand <-> generic, e.g. Zoloft / Sertraline) are merged into
  one *drug group* with union-find, so a held-out drug cannot leak in under another name.
- ``drug_holdout``: whole drug groups sampled until ~``drug_holdout_fraction`` of reviews.
- Remaining reviews: stratified by (sentiment bucket, source) into train / val / test.

Writes ``data/processed/splits.parquet`` (review_id, drug_group, split) and ``reports/data/splits_report.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from medsignal.utils import PROJECT_ROOT, git_sha, load_config, seed_everything

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test", "drug_holdout")


def sentiment_bucket(rating: int) -> str:
    """1-4 negative, 5-6 neutral, 7-10 positive (used only for stratification)."""
    if rating <= 4:
        return "neg"
    if rating <= 6:
        return "neu"
    return "pos"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _key(name: str) -> str:
    return " ".join(name.lower().split())


def drug_groups(alias_lists: Iterable[Iterable[str]]) -> list[str]:
    """Assign each review a drug-group id: the smallest normalised alias in its connected component."""
    lists = [[_key(a) for a in aliases] for aliases in alias_lists]
    uf = _UnionFind()
    for aliases in lists:
        for a in aliases:
            uf.find(a)
        for a, b in itertools.pairwise(aliases):
            uf.union(a, b)
    return [uf.find(aliases[0]) if aliases else "<unknown>" for aliases in lists]


def pick_holdout_groups(
    group_sizes: pd.Series, fraction: float, rng: np.random.Generator, max_group_share: float
) -> set[str]:
    """Randomly add whole groups until ``fraction`` of reviews is reached.

    Groups larger than ``max_group_share`` of all reviews are never held out (one giant group
    would make the held-out set a single-drug benchmark).
    """
    total = int(group_sizes.sum())
    target = fraction * total
    eligible = group_sizes[group_sizes <= max_group_share * total]
    order = rng.permutation(eligible.index.to_numpy())
    chosen: set[str] = set()
    count = 0
    for g in order:
        if count >= target:
            break
        size = int(eligible[g])
        if count + size > target * 1.05:
            continue
        chosen.add(str(g))
        count += size
    return chosen


def stratified_assign(
    strata: pd.Series, val_fraction: float, test_fraction: float, rng: np.random.Generator
) -> pd.Series:
    """Assign train/val/test within each stratum by shuffled position."""
    out = pd.Series("train", index=strata.index, dtype=object)
    for _, idx in strata.groupby(strata).groups.items():
        idx_arr = rng.permutation(np.asarray(idx))
        n = len(idx_arr)
        n_test = round(n * test_fraction)
        n_val = round(n * val_fraction)
        out.loc[idx_arr[:n_test]] = "test"
        out.loc[idx_arr[n_test : n_test + n_val]] = "val"
    return out


def make_splits(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return review_id, drug_group, split."""
    s = cfg["splits"]
    rng = np.random.default_rng(int(cfg["seed"]))

    groups = pd.Series(drug_groups(df["drug_aliases"]), index=df.index, name="drug_group")
    holdout = pick_holdout_groups(
        groups.value_counts(),
        float(s["drug_holdout_fraction"]),
        rng,
        float(s["max_holdout_group_share"]),
    )
    split = pd.Series("train", index=df.index, dtype=object)
    in_holdout = groups.isin(holdout)
    split[in_holdout] = "drug_holdout"

    rest = df.loc[~in_holdout]
    strata = rest["rating"].map(sentiment_bucket) + "|" + rest["source"]
    split.loc[rest.index] = stratified_assign(
        strata, float(s["val_fraction"]), float(s["test_fraction"]), rng
    )
    return pd.DataFrame({"review_id": df["review_id"], "drug_group": groups, "split": split})


def split_report(df: pd.DataFrame, splits: pd.DataFrame) -> dict[str, Any]:
    """Counts per split and leakage checks."""
    merged = df[["review_id", "rating", "source"]].merge(splits, on="review_id")
    report: dict[str, Any] = {
        "rows": len(splits),
        "counts": merged["split"].value_counts().reindex(SPLITS, fill_value=0).to_dict(),
        "drug_groups": int(splits["drug_group"].nunique()),
        "drug_groups_held_out": int(
            splits.loc[splits["split"] == "drug_holdout", "drug_group"].nunique()
        ),
        "largest_drug_group_share": round(
            float(splits["drug_group"].value_counts(normalize=True).iloc[0]), 4
        ),
    }
    report["sentiment_share_by_split"] = {
        sp: merged.loc[merged["split"] == sp, "rating"]
        .map(sentiment_bucket)
        .value_counts(normalize=True)
        .round(4)
        .to_dict()
        for sp in SPLITS
    }
    report["source_share_by_split"] = {
        sp: merged.loc[merged["split"] == sp, "source"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict()
        for sp in SPLITS
    }
    held = set(splits.loc[splits["split"] == "drug_holdout", "drug_group"])
    others = set(splits.loc[splits["split"] != "drug_holdout", "drug_group"])
    report["leakage"] = {
        "duplicate_review_ids": int(splits["review_id"].duplicated().sum()),
        "drug_groups_in_holdout_and_other_splits": len(held & others),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    df = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["interim"] / "reviews.parquet")
    splits = make_splits(df, cfg)
    report = split_report(df, splits)
    if any(report["leakage"].values()):
        raise RuntimeError(f"Leakage detected: {report['leakage']}")

    out = PROJECT_ROOT / cfg["paths"]["processed"] / "splits.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(out, index=False)

    report["meta"] = {
        "config": args.config,
        "seed": int(cfg["seed"]),
        "git_sha": git_sha(),
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = PROJECT_ROOT / "reports" / "data" / "splits_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("splits: %s", report["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
