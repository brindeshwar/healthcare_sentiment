import numpy as np
import pandas as pd

from medsignal.labeling.silver import sample_silver


def _data(n: int = 800) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(n)],
            "text": ["x" * 80] * n,
            "rating": rng.integers(1, 11, n),
            "drug": [f"d{i % 40}" for i in range(n)],
            "condition": ["c"] * n,
            "source": ["drugrev" if i % 3 else "both" for i in range(n)],
        }
    )
    splits = pd.DataFrame(
        {
            "review_id": reviews["review_id"],
            "drug_group": [f"g{i % 40}" for i in range(n)],
            "split": ["train" if i % 4 else "test" for i in range(n)],
        }
    )
    return reviews, splits


CFG = {
    "seed": 5,
    "sample": {
        "size": 100,
        "source_split": "train",
        "sentiment_weights": {"neg": 0.4, "neu": 0.2, "pos": 0.4},
        "max_per_drug_group": 5,
        "min_chars": 40,
    },
}


def test_sample_silver_train_only_quotas_and_cap() -> None:
    reviews, splits = _data()
    s = sample_silver(reviews, splits, CFG)
    train_ids = set(splits.loc[splits["split"] == "train", "review_id"])
    assert len(s) == 100
    assert set(s["review_id"]) <= train_ids
    assert s["bucket"].value_counts().to_dict() == {"neg": 40, "pos": 40, "neu": 20}
    assert s["drug_group"].value_counts().max() <= 5
    pd.testing.assert_frame_equal(s, sample_silver(reviews, splits, CFG))


def test_summarize_counts() -> None:
    from medsignal.labeling.silver import summarize

    prov = {"attempts": 1, "seconds": 1.0, "prompt_version": "v2-x", "annotator": "llm:m"}
    records = [
        {
            "review_id": "a",
            "status": "ok",
            "ades": [{"severity": "severe", "withdrawal": False, "beneficial": False}],
            "discontinued": "yes",
            "stop_reasons": ["adverse_event"],
            "effectiveness": "not_effective",
            "provenance": prov,
        },
        {
            "review_id": "b",
            "status": "ok",
            "ades": [],
            "discontinued": "not_stated",
            "stop_reasons": [],
            "effectiveness": "effective",
            "provenance": prov,
        },
        {"review_id": "c", "status": "failed", "provenance": {**prov, "attempts": 3}},
    ]
    sample = pd.DataFrame({"review_id": ["a", "b", "c"], "bucket": ["neg", "pos", "neu"]})
    s = summarize(records, sample)
    assert s["labelled_ok"] == 2 and s["failed"] == 1
    assert s["stop_reasons"] == {"adverse_event": 1}
    assert s["by_sentiment_bucket"]["neg"]["share_with_ade"] == 1.0
    assert s["by_sentiment_bucket"]["pos"]["share_stopped_or_planned"] == 0.0
    assert s["severity"] == {"severe": 1}
