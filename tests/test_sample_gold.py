import numpy as np
import pandas as pd
import pytest

from medsignal.labeling.sample_gold import draw, sample_gold


def _data(n: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(n)],
            "text": ["x" * 80] * n,
            "rating": rng.integers(1, 11, n),
            "possibly_truncated": [i % 5 == 0 for i in range(n)],
            "drug": [f"d{i % 50}" for i in range(n)],
            "condition": ["c"] * n,
            "source": ["both"] * n,
        }
    )
    splits = pd.DataFrame(
        {
            "review_id": reviews["review_id"],
            "drug_group": [f"g{i % 50}" for i in range(n)],
            "split": ["val" if i % 2 else "test" for i in range(n)],
        }
    )
    return reviews, splits


CFG = {
    "gold": {
        "sizes": {"dev": 20, "test": 30},
        "source_split": {"dev": "val", "test": "test"},
        "sentiment_weights": {"neg": 0.4, "neu": 0.2, "pos": 0.4},
        "max_per_drug_group": 3,
        "min_chars": 60,
    }
}


def test_sample_gold_respects_sizes_splits_and_truncation() -> None:
    reviews, splits = _data()
    gold = sample_gold(reviews, splits, CFG, seed=1)
    assert gold["gold_split"].value_counts().to_dict() == {"test": 30, "dev": 20}
    assert set(gold.loc[gold["gold_split"] == "dev", "split"]) == {"val"}
    assert set(gold.loc[gold["gold_split"] == "test", "split"]) == {"test"}
    truncated = set(reviews.loc[reviews["possibly_truncated"], "review_id"])
    assert not set(gold["review_id"]) & truncated
    assert gold["review_id"].is_unique
    assert list(gold["order"]) == list(range(1, 51))


def test_sample_gold_bucket_quotas_and_determinism() -> None:
    reviews, splits = _data()
    a = sample_gold(reviews, splits, CFG, seed=1)
    b = sample_gold(reviews, splits, CFG, seed=1)
    pd.testing.assert_frame_equal(a, b)
    dev = a[a["gold_split"] == "dev"]["bucket"].value_counts().to_dict()
    assert dev == {"neg": 8, "neu": 4, "pos": 8}


def test_draw_raises_when_pool_too_small() -> None:
    pool = pd.DataFrame({"bucket": ["neg"] * 3, "drug_group": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        draw(pool, 10, {"neg": 1.0}, 5, np.random.default_rng(0))
