"""Majority-class and TF-IDF + logistic-regression sentiment baselines, plus a leakage demo.

Usage:
    uv run python -m medsignal.baselines.classical --config configs/baselines.yaml

Writes ``reports/baselines/classical.json``. Deterministic (no seed-dependent steps except the
leakage-demo split), so a single run is reported.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from medsignal.data.clean import load_uci
from medsignal.evaluation.metrics import classification_metrics, sentiment3
from medsignal.utils import PROJECT_ROOT, git_sha, load_config

log = logging.getLogger(__name__)

EVAL_SPLITS = ("val", "test", "drug_holdout")


def load_labelled(data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Reviews joined with their split and 3-class label."""
    reviews = pd.read_parquet(PROJECT_ROOT / data_cfg["paths"]["interim"] / "reviews.parquet")
    splits = pd.read_parquet(PROJECT_ROOT / data_cfg["paths"]["processed"] / "splits.parquet")
    df = reviews[["review_id", "text", "rating"]].merge(
        splits[["review_id", "split"]], on="review_id"
    )
    return df.assign(label=df["rating"].map(sentiment3))


def make_vectorizer(cfg: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=int(cfg["min_df"]),
        max_features=int(cfg["max_features"]),
        sublinear_tf=bool(cfg["sublinear_tf"]),
        lowercase=True,
    )


def majority_baseline(df: pd.DataFrame) -> dict[str, Any]:
    train = df[df["split"] == "train"]
    majority = int(train["label"].mode().iloc[0])
    return {
        sp: classification_metrics(g["label"], np.full(len(g), majority))
        for sp, g in df[df["split"].isin(EVAL_SPLITS)].groupby("split")
    }


def tfidf_baseline(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit on train, choose C on val macro-F1, report all eval splits with the chosen C."""
    train = df[df["split"] == "train"]
    vec = make_vectorizer(cfg)
    x_train = vec.fit_transform(train["text"])
    xs = {sp: vec.transform(g["text"]) for sp, g in df.groupby("split") if sp in EVAL_SPLITS}
    ys = {sp: g["label"].to_numpy() for sp, g in df.groupby("split") if sp in EVAL_SPLITS}

    search: dict[str, float] = {}
    best: tuple[float, float, LogisticRegression] | None = None
    for c in cfg["C_grid"]:
        clf = LogisticRegression(C=float(c), max_iter=2000, class_weight=cfg["class_weight"])
        clf.fit(x_train, train["label"])
        f1 = classification_metrics(ys["val"], clf.predict(xs["val"]))["macro_f1"]
        search[str(c)] = f1
        log.info("tfidf C=%s val macro-F1=%.4f", c, f1)
        if best is None or f1 > best[1]:
            best = (float(c), f1, clf)
    assert best is not None
    c_best, _, clf = best
    return {
        "chosen_C": c_best,
        "val_search": search,
        "vocab_size": len(vec.vocabulary_),
        **{sp: classification_metrics(ys[sp], clf.predict(xs[sp])) for sp in EVAL_SPLITS},
    }


def leakage_demo(
    cfg: dict[str, Any], tfidf_cfg: dict[str, Any], data_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Legacy-style evaluation: random split of the raw UCI file with duplicates left in."""
    raw = load_uci(
        PROJECT_ROOT / data_cfg["paths"]["raw"] / "drugscom_uci" / "drugsComTrain_raw.csv"
    )
    raw = raw.assign(label=raw["rating"].map(sentiment3))

    def run(frame: pd.DataFrame) -> dict[str, Any]:
        train, test = train_test_split(
            frame,
            test_size=float(cfg["test_fraction"]),
            random_state=int(cfg["seed"]),
            stratify=frame["label"],
        )
        vec = make_vectorizer(tfidf_cfg)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight=tfidf_cfg["class_weight"])
        clf.fit(vec.fit_transform(train["text"]), train["label"])
        return {
            "rows": len(frame),
            "test_reviews_whose_text_is_in_train": round(
                float(test["text"].isin(set(train["text"])).mean()), 4
            ),
            "test": classification_metrics(test["label"], clf.predict(vec.transform(test["text"]))),
        }

    # Control: identical data and procedure, duplicates removed (same text + date = same review),
    # so the score difference isolates the effect of leakage.
    deduped = raw.drop_duplicates(subset=["text", "date"])
    return {
        "description": "UCI train file only, random 80/20 split, TF-IDF+LR (C=1)",
        "legacy_with_duplicates": run(raw),
        "control_deduplicated": run(deduped),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/baselines.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    data_cfg = load_config(cfg["data_config"])
    df = load_labelled(data_cfg)

    t0 = time.perf_counter()
    report: dict[str, Any] = {
        "task": "3-class sentiment from star rating (neg 1-4 / neu 5-6 / pos 7-10)",
        "label_share_train": df[df["split"] == "train"]["label"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "majority": majority_baseline(df),
        "tfidf_logreg": tfidf_baseline(df, cfg["tfidf"]),
        "leakage_demo": leakage_demo(cfg["leakage_demo"], cfg["tfidf"], data_cfg),
    }
    report["meta"] = {
        "config": args.config,
        "git_sha": git_sha(),
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    out = PROJECT_ROOT / "reports" / "baselines" / "classical.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
