"""Within-source de-duplication and cross-source (UCI <-> DrugRev) matching.

Why these keys (measured 2026-10-13, see docs/DATA.md and ADR-0011):
- Drugs.com lists one review under several drug names (brand/generic), so duplicates share text *and* date.
- DrugRev truncates many reviews, so cross-source matching uses a text prefix. Prefix alone gives false
  matches (13% rating disagreement at 40 chars); prefix + date + rating gives 100% rating agreement.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from medsignal.data.clean import normalize_for_matching


def _unique_sorted(values: pd.Series) -> list[str]:
    return sorted({v for v in values if isinstance(v, str) and v})


def review_id(date: pd.Timestamp, norm_text: str) -> str:
    """Stable ID from date + normalised text (survives re-runs and row reordering)."""
    key = f"{date:%Y-%m-%d}|{norm_text}".encode()
    return hashlib.sha1(key, usedforsecurity=False).hexdigest()[:16]


def dedup_within(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows with the same (normalised text, date) into one review.

    Keeps every drug name / condition the review was listed under (``drug_aliases``,
    ``conditions``), the modal rating, the max helpful count, and ``n_listings``.
    """
    work = df.assign(norm=df["text"].map(normalize_for_matching))
    grouped = work.groupby(["norm", "date"], sort=False)
    out = grouped.agg(
        source=("source", "first"),
        text=("text", "first"),
        rating=("rating", lambda s: int(s.mode().iloc[0])),
        rating_conflict=("rating", lambda s: bool((s != s.iloc[0]).any())),
        helpful_count=("helpful_count", "max"),
        duration=("duration", "first"),
        drug_aliases=("drug", _unique_sorted),
        conditions=("condition", _unique_sorted),
        n_listings=("text", "size"),
    ).reset_index()
    out["drug"] = out["drug_aliases"].map(lambda a: a[0] if a else None)
    out["condition"] = out["conditions"].map(lambda c: c[0] if c else None)
    return out


def match_cross_source(
    uci: pd.DataFrame, drugrev: pd.DataFrame, prefix_chars: int = 40, min_short_chars: int = 15
) -> pd.DataFrame:
    """Return pairs (uci_idx, drugrev_idx) of the same review in both de-duplicated sources.

    Pass 1: same date + rating + first ``prefix_chars`` normalised characters.
    Pass 2: DrugRev texts shorter than ``prefix_chars`` (but >= ``min_short_chars``) that are a
    prefix of a UCI text with the same date + rating.
    """
    u = uci[["norm", "date", "rating"]].assign(uci_idx=uci.index)
    d = drugrev[["norm", "date", "rating"]].assign(drugrev_idx=drugrev.index)
    u["p"] = u["norm"].str[:prefix_chars]
    d["p"] = d["norm"].str[:prefix_chars]

    long_d = d[d["norm"].str.len() >= prefix_chars]
    pass1 = long_d.merge(u, on=["p", "date", "rating"])[["uci_idx", "drugrev_idx"]]

    short_d = d[(d["norm"].str.len() < prefix_chars) & (d["norm"].str.len() >= min_short_chars)]
    cand = short_d.merge(u, on=["date", "rating"], suffixes=("_d", "_u"))
    hit = [ut.startswith(dt) for ut, dt in zip(cand["norm_u"], cand["norm_d"], strict=True)]
    pass2 = cand.loc[hit, ["uci_idx", "drugrev_idx"]]

    pairs = pd.concat([pass1, pass2], ignore_index=True)
    # One-to-one: if a DrugRev review matches several UCI reviews (or vice versa), keep the first.
    pairs = pairs.drop_duplicates("drugrev_idx").drop_duplicates("uci_idx")
    return pairs.reset_index(drop=True)


def ends_without_terminal_punctuation(text: str) -> bool:
    """Candidate truncation signal (text does not end in . ! ? ) or a quote).

    Evaluated on matched pairs in build_report.json; too weak to use as a row-level flag.
    """
    return not text.rstrip().endswith((".", "!", "?", ")", '"', "'"))


def merge_sources(uci: pd.DataFrame, drugrev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge de-duplicated UCI and DrugRev reviews.

    Matched reviews keep the UCI text (full length) and gain DrugRev's duration and aliases.
    Returns (merged reviews, matched pairs with length ratio for reporting).
    """
    pairs = match_cross_source(uci, drugrev)
    uci = uci.copy()
    drugrev = drugrev.copy()

    u_rows = uci.loc[pairs["uci_idx"]].reset_index(drop=True)
    d_rows = drugrev.loc[pairs["drugrev_idx"]].reset_index(drop=True)
    pairs = pairs.assign(
        length_ratio=d_rows["text"].str.len().to_numpy() / u_rows["text"].str.len().to_numpy(),
        drugrev_no_terminal_punct=d_rows["text"].map(ends_without_terminal_punctuation).to_numpy(),
    )

    uci["possibly_truncated"] = False
    uci.loc[pairs["uci_idx"], "source"] = "both"
    uci.loc[pairs["uci_idx"], "duration"] = d_rows["duration"].to_numpy()
    uci.loc[pairs["uci_idx"], "helpful_count"] = (
        pd.concat([u_rows["helpful_count"], d_rows["helpful_count"]], axis=1).max(axis=1).to_numpy()
    )
    uci.loc[pairs["uci_idx"], "drug_aliases"] = pd.Series(
        [
            sorted(set(a) | set(b))
            for a, b in zip(u_rows["drug_aliases"], d_rows["drug_aliases"], strict=True)
        ],
        index=pairs["uci_idx"].to_numpy(),
        dtype=object,
    )

    # DrugRev cuts reviews at sentence boundaries, so truncation is not detectable per row
    # (the punctuation heuristic is reported in build_report.json for evidence). Every
    # DrugRev-only review is therefore flagged as possibly truncated.
    rest = drugrev.drop(index=pairs["drugrev_idx"]).assign(possibly_truncated=True)

    merged = pd.concat([uci, rest], ignore_index=True)
    merged["review_id"] = [
        review_id(dt, n) for dt, n in zip(merged["date"], merged["norm"], strict=True)
    ]
    return merged, pairs
