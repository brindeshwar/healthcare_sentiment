import pandas as pd

from medsignal.data.dedup import (
    dedup_within,
    ends_without_terminal_punctuation,
    merge_sources,
    review_id,
)

D1 = pd.Timestamp("2015-03-01")
D2 = pd.Timestamp("2016-07-09")
LONG = "I took this for three weeks and the nausea was unbearable, so I stopped it."


def _rows(source: str, rows: list[tuple[str, str, int, pd.Timestamp]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": source,
                "drug": drug,
                "condition": "Pain",
                "text": text,
                "rating": rating,
                "date": date,
                "helpful_count": 1,
                "duration": "Taken for 1 to 6 months" if source == "drugrev" else None,
            }
            for drug, text, rating, date in rows
        ]
    )


def test_dedup_within_collapses_aliases() -> None:
    df = _rows("uci", [("Brand", LONG, 2, D1), ("generic", LONG, 2, D1), ("Brand", LONG, 2, D2)])
    out = dedup_within(df)
    assert len(out) == 2  # same text on another date is a different review
    first = out[out["date"] == D1].iloc[0]
    assert first["drug_aliases"] == ["Brand", "generic"]
    assert first["n_listings"] == 2


def test_dedup_within_flags_rating_conflict() -> None:
    df = _rows("uci", [("A", LONG, 2, D1), ("B", LONG, 3, D1), ("C", LONG, 3, D1)])
    out = dedup_within(df)
    assert out.iloc[0]["rating"] == 3
    assert bool(out.iloc[0]["rating_conflict"])


def test_merge_prefers_uci_full_text_for_truncated_match() -> None:
    uci = dedup_within(_rows("uci", [("Zoloft", LONG, 2, D1)]))
    truncated = LONG[:50]
    dr = dedup_within(_rows("drugrev", [("Zoloft", truncated, 2, D1)]))
    merged, pairs = merge_sources(uci, dr)
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["source"] == "both"
    assert row["text"] == LONG
    assert row["duration"] == "Taken for 1 to 6 months"
    assert not row["possibly_truncated"]
    assert pairs.iloc[0]["length_ratio"] < 1


def test_merge_requires_same_date_and_rating() -> None:
    uci = dedup_within(_rows("uci", [("Zoloft", LONG, 2, D1)]))
    dr = dedup_within(_rows("drugrev", [("Zoloft", LONG, 2, D2), ("Zoloft", LONG, 9, D1)]))
    merged, pairs = merge_sources(uci, dr)
    assert len(pairs) == 0
    assert len(merged) == 3


def test_short_drugrev_text_matches_as_prefix() -> None:
    uci = dedup_within(
        _rows("uci", [("X", "Works great for my headaches, no side effects.", 10, D1)])
    )
    dr = dedup_within(_rows("drugrev", [("X", "Works great for my head", 10, D1)]))
    _, pairs = merge_sources(uci, dr)
    assert len(pairs) == 1


def test_drugrev_only_rows_get_truncation_flag() -> None:
    uci = dedup_within(_rows("uci", [("X", LONG, 2, D1)]))
    dr = dedup_within(_rows("drugrev", [("Y", "Gave me a rash. I stopped.", 1, D2)]))
    merged, _ = merge_sources(uci, dr)
    assert bool(merged.loc[merged["source"] == "drugrev", "possibly_truncated"].iloc[0])


def test_review_id_is_stable() -> None:
    assert review_id(D1, "abc") == review_id(D1, "abc")
    assert review_id(D1, "abc") != review_id(D2, "abc")


def test_terminal_punctuation_heuristic() -> None:
    assert ends_without_terminal_punctuation("and then I")
    assert not ends_without_terminal_punctuation("It worked.")
