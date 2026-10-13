import pandas as pd

from medsignal.baselines.classical import majority_baseline, tfidf_baseline

POS = ["great drug, worked well", "excellent results, love it", "really helped me, great"]
NEG = ["terrible side effects, awful", "made me sick, horrible", "awful nausea, stopped it"]
NEU = ["it was okay, nothing special", "some help, some side effects, okay", "okay I guess"]


def _toy() -> pd.DataFrame:
    rows = []
    for split, reps in (("train", 6), ("val", 2), ("test", 2), ("drug_holdout", 2)):
        for _ in range(reps):
            rows += [{"text": t, "label": 2, "split": split} for t in POS]
            rows += [{"text": t, "label": 0, "split": split} for t in NEG]
            rows += [{"text": t, "label": 1, "split": split} for t in NEU]
    return pd.DataFrame(rows)


def test_majority_baseline_predicts_one_class() -> None:
    df = _toy()
    df = pd.concat([df, df[(df["split"] == "train") & (df["label"] == 2)]])
    out = majority_baseline(df)
    assert set(out) == {"val", "test", "drug_holdout"}
    assert out["test"]["f1_per_class"]["neg"] == 0.0


def test_tfidf_baseline_learns_toy_task() -> None:
    cfg = {
        "ngram_range": [1, 2],
        "min_df": 1,
        "max_features": 1000,
        "sublinear_tf": True,
        "C_grid": [1.0, 10.0],
        "class_weight": "balanced",
    }
    out = tfidf_baseline(_toy(), cfg)
    assert out["chosen_C"] in (1.0, 10.0)
    assert out["test"]["macro_f1"] == 1.0


def test_aggregate_mean_std_over_seeds() -> None:
    from medsignal.baselines.report import aggregate

    def m(f1: float) -> dict[str, float]:
        return {"macro_f1": f1, "accuracy": 0.8, "qwk": 0.7, "reviews_per_second": 100.0}

    classical = {
        k: {sp: m(0.5) for sp in ("val", "test", "drug_holdout")}
        for k in ("majority", "tfidf_logreg")
    }
    runs = [
        {"seed": s, **{sp: m(f) for sp in ("val", "test", "drug_holdout")}}
        for s, f in ((1, 0.6), (2, 0.7))
    ]
    agg = aggregate(classical, runs)
    assert agg["encoder"]["test"]["macro_f1"]["mean"] == 0.65
    assert agg["encoder"]["seeds"] == [1, 2]
    assert agg["tfidf_logreg"]["test"]["macro_f1"]["n_runs"] == 1
