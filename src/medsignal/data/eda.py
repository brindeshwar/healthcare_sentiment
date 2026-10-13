"""Exploratory data report for the merged corpus and its splits.

Usage:
    uv run python -m medsignal.data.eda --config configs/data.yaml

Writes ``reports/eda/summary.json``, figures ``reports/eda/*.png`` and ``reports/eda/README.md``.
Every number in the README is read from ``summary.json`` (AGENTS.md §7).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from medsignal.data.clean import DURATION_BUCKETS
from medsignal.data.splits import SPLITS
from medsignal.utils import PROJECT_ROOT, git_sha, load_config, load_env

log = logging.getLogger(__name__)

# Reference palette, light mode (dataviz skill references/palette.md). Sources use slots 1-3,
# which validate for all pairs; single-series charts use slot 1.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SOURCE_COLORS = {"uci": "#2a78d6", "both": "#eb6834", "drugrev": "#1baf7a"}
SOURCE_LABELS = {"uci": "UCI only", "both": "In both (UCI text)", "drugrev": "DrugRev only"}
SERIES_1 = "#2a78d6"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "font.size": 10,
            "legend.frameon": False,
            "legend.labelcolor": TEXT,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def token_lengths(texts: pd.Series, tokenizer_name: str, batch: int = 4096) -> np.ndarray:
    """Token counts (incl. special tokens) with the student model's tokenizer."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(tokenizer_name)
    tok.no_truncation()
    tok.no_padding()
    values = texts.tolist()
    out: list[int] = []
    for i in range(0, len(values), batch):
        out.extend(len(e.ids) for e in tok.encode_batch(values[i : i + batch]))
    return np.asarray(out)


def summarize(
    df: pd.DataFrame, splits: pd.DataFrame, lengths: np.ndarray, max_len: int
) -> dict[str, Any]:
    """All numbers used by figures and the README."""
    df = df.assign(tokens=lengths, year=df["date"].dt.year)
    merged = df.merge(splits[["review_id", "split"]], on="review_id")
    cond = df["condition"].value_counts()
    return {
        "reviews": len(df),
        "by_source": df["source"].value_counts().to_dict(),
        "unique_drug_names": int(df["drug"].nunique()),
        "drug_groups": int(splits["drug_group"].nunique()),
        "conditions_known": int(df["condition"].notna().sum()),
        "unique_conditions": int(cond.size),
        "top_conditions": {k: int(v) for k, v in cond.head(15).items()},
        "date_range": [f"{df['date'].min():%Y-%m-%d}", f"{df['date'].max():%Y-%m-%d}"],
        "reviews_per_year_by_source": {
            str(y): {s: int(n) for s, n in g["source"].value_counts().items()}
            for y, g in df.groupby("year")
        },
        "rating_share_by_source": {
            s: {
                int(r): round(float(v), 4)
                for r, v in g["rating"].value_counts(normalize=True).sort_index().items()
            }
            for s, g in df.groupby("source")
        },
        "tokens": {
            "tokenizer": None,  # filled by caller
            "median": int(np.median(lengths)),
            "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
            f"share_within_{max_len}": round(float((lengths <= max_len).mean()), 4),
        },
        "duration_counts": {b: int((df["duration"] == b).sum()) for b in DURATION_BUCKETS},
        "duration_known_share": round(float(df["duration"].notna().mean()), 4),
        "possibly_truncated_share": round(float(df["possibly_truncated"].mean()), 4),
        "split_counts": merged["split"].value_counts().reindex(SPLITS, fill_value=0).to_dict(),
        "max_length": max_len,
    }


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


def plot_reviews_per_year(s: dict[str, Any], path: Path) -> None:
    years = sorted(s["reviews_per_year_by_source"], key=int)
    x = np.array([int(y) for y in years])
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(years))
    for src in ("uci", "both", "drugrev"):
        vals = np.array([s["reviews_per_year_by_source"][y].get(src, 0) for y in years])
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=SOURCE_COLORS[src],
            label=SOURCE_LABELS[src],
            edgecolor=SURFACE,
            linewidth=1.5,
            width=0.8,
        )
        bottom += vals
    ax.set_title("Reviews per year by source")
    ax.set_ylabel("Reviews")
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_rating_by_source(s: dict[str, Any], path: Path) -> None:
    ratings = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.27
    for i, src in enumerate(("uci", "both", "drugrev")):
        share = [s["rating_share_by_source"][src].get(int(r), 0) * 100 for r in ratings]
        ax.bar(
            ratings + (i - 1) * width,
            share,
            width=width - 0.03,
            color=SOURCE_COLORS[src],
            label=SOURCE_LABELS[src],
        )
    ax.set_title("Star rating distribution by source (% of each source)")
    ax.set_xticks(ratings)
    ax.set_xlabel("Star rating")
    ax.set_ylabel("% of reviews")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center")
    _save(fig, path)


def plot_token_lengths(lengths: np.ndarray, s: dict[str, Any], path: Path) -> None:
    max_len = s["max_length"]
    share = s["tokens"][f"share_within_{max_len}"] * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    clipped = np.clip(lengths, 0, 512)
    ax.hist(clipped, bins=64, color=SERIES_1, edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(max_len, color=TEXT_2, linewidth=1.5, linestyle="--")
    ax.text(
        max_len + 6,
        ax.get_ylim()[1] * 0.9,
        f"max_length = {max_len}\n{share:.1f}% of reviews fit",
        color=TEXT,
        va="top",
    )
    ax.set_title("Review length in tokens (BioClinical ModernBERT tokenizer; >512 shown at 512)")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Reviews")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def _hbar(labels: list[str], values: list[int], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(labels) + 1))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=SERIES_1, height=0.7)
    ax.set_yticks(y, labels)
    for yi, v in zip(y, values, strict=True):
        ax.text(v, yi, f" {v:,}", va="center", color=TEXT_2, fontsize=9)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.15)
    _save(fig, path)


def write_readme(s: dict[str, Any], path: Path) -> None:
    t = s["tokens"]
    src = s["by_source"]
    sp = s["split_counts"]
    max_len = s["max_length"]
    lines = [
        "# Data overview (generated)",
        "",
        "> Generated by `uv run python -m medsignal.data.eda`. Do not edit by hand; numbers come from `summary.json`.",
        f"> Commit `{s['meta']['git_sha']}`, built {s['meta']['built_utc']}.",
        "",
        "## Corpus",
        "",
        "| | |",
        "|---|---|",
        f"| Reviews (de-duplicated) | {s['reviews']:,} |",
        f"| By source | UCI only {src.get('uci', 0):,} · in both {src.get('both', 0):,} · DrugRev only {src.get('drugrev', 0):,} |",
        f"| Date range | {s['date_range'][0]} to {s['date_range'][1]} |",
        f"| Drug names / drug groups | {s['unique_drug_names']:,} / {s['drug_groups']:,} |",
        f"| Reviews with a condition | {s['conditions_known']:,} ({s['unique_conditions']:,} distinct) |",
        f"| Duration of use known | {s['duration_known_share']:.1%} |",
        f"| Possibly truncated (DrugRev only) | {s['possibly_truncated_share']:.1%} |",
        "",
        "![Reviews per year](reviews_per_year.png)",
        "",
        "DrugRev adds the 2018-2023 reviews; the UCI file ends in 2017.",
        "",
        "## Ratings",
        "",
        "![Rating distribution](rating_by_source.png)",
        "",
        "## Length",
        "",
        f"Tokenizer `{t['tokenizer']}`: median {t['median']} tokens, p95 {t['p95']}, p99 {t['p99']}, max {t['max']}. "
        f"**{t[f'share_within_{max_len}']:.1%} of reviews fit in `max_length={max_len}`.**",
        "",
        "![Token lengths](token_lengths.png)",
        "",
        "## Conditions and duration",
        "",
        "![Top conditions](top_conditions.png)",
        "",
        "![Duration of use](duration.png)",
        "",
        "## Splits (ADR-0013)",
        "",
        "| train | val | test | drug_holdout |",
        "|---|---|---|---|",
        f"| {sp['train']:,} | {sp['val']:,} | {sp['test']:,} | {sp['drug_holdout']:,} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()
    cfg = load_config(args.config)
    eda_cfg = cfg["eda"]

    df = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["interim"] / "reviews.parquet")
    splits = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["processed"] / "splits.parquet")
    log.info("tokenizing %s reviews with %s", len(df), eda_cfg["tokenizer"])
    lengths = token_lengths(df["text"], eda_cfg["tokenizer"])

    s = summarize(df, splits, lengths, int(eda_cfg["max_length"]))
    s["tokens"]["tokenizer"] = eda_cfg["tokenizer"]
    s["meta"] = {
        "config": args.config,
        "git_sha": git_sha(),
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    out = PROJECT_ROOT / "reports" / "eda"
    out.mkdir(parents=True, exist_ok=True)
    _style()
    plot_reviews_per_year(s, out / "reviews_per_year.png")
    plot_rating_by_source(s, out / "rating_by_source.png")
    plot_token_lengths(lengths, s, out / "token_lengths.png")
    _hbar(
        list(s["top_conditions"]),
        list(s["top_conditions"].values()),
        "Top 15 conditions (reviews)",
        out / "top_conditions.png",
    )
    _hbar(
        [b.replace("Taken for ", "") for b in DURATION_BUCKETS],
        list(s["duration_counts"].values()),
        "Duration of use (DrugRev field, where reported)",
        out / "duration.png",
    )

    (out / "summary.json").write_text(
        json.dumps(s, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    write_readme(s, out / "README.md")
    log.info("EDA written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
