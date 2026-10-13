import numpy as np
import pandas as pd

from medsignal.data.splits import (
    drug_groups,
    make_splits,
    pick_holdout_groups,
    sentiment_bucket,
    split_report,
    stratified_assign,
)


def test_sentiment_bucket() -> None:
    assert [sentiment_bucket(r) for r in (1, 4, 5, 6, 7, 10)] == [
        "neg",
        "neg",
        "neu",
        "neu",
        "pos",
        "pos",
    ]


def test_drug_groups_link_brand_and_generic_transitively() -> None:
    groups = drug_groups(
        [["Zoloft", "Sertraline"], ["sertraline"], ["Lustral", "Sertraline"], ["Ibuprofen"]]
    )
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_pick_holdout_respects_max_group_share() -> None:
    sizes = pd.Series({"giant": 900, **{f"g{i}": 10 for i in range(10)}})
    chosen = pick_holdout_groups(sizes, 0.05, np.random.default_rng(0), max_group_share=0.5)
    assert "giant" not in chosen
    assert sum(sizes[g] for g in chosen) <= 0.05 * 1000 * 1.05


def test_stratified_assign_fractions() -> None:
    strata = pd.Series(["a"] * 100 + ["b"] * 50)
    out = stratified_assign(strata, 0.1, 0.2, np.random.default_rng(0))
    assert (out[:100] == "test").sum() == 20
    assert (out[:100] == "val").sum() == 10
    assert (out[100:] == "test").sum() == 10


def _toy() -> pd.DataFrame:
    rows = []
    for i in range(400):
        drug = f"drug{i % 20}"
        aliases = [drug, f"Brand{i % 20}"] if i % 2 else [drug]
        rows.append(
            {
                "review_id": f"r{i}",
                "drug_aliases": aliases,
                "rating": (i % 10) + 1,
                "source": "uci" if i % 3 else "drugrev",
            }
        )
    return pd.DataFrame(rows)


CFG = {
    "seed": 1,
    "splits": {
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "drug_holdout_fraction": 0.1,
        "max_holdout_group_share": 0.2,
    },
}


def test_make_splits_has_no_leakage_and_is_deterministic() -> None:
    df = _toy()
    a = make_splits(df, CFG)
    b = make_splits(df, CFG)
    pd.testing.assert_frame_equal(a, b)
    report = split_report(df, a)
    assert report["leakage"] == {
        "duplicate_review_ids": 0,
        "drug_groups_in_holdout_and_other_splits": 0,
    }
    assert set(a["split"]) == {"train", "val", "test", "drug_holdout"}
    assert a["drug_group"].nunique() == 20  # drugN and BrandN merged
