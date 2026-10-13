import pytest

from medsignal.evaluation.metrics import classification_metrics, mean_std, sentiment3


def test_sentiment3_boundaries() -> None:
    assert [sentiment3(r) for r in range(1, 11)] == [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]


def test_perfect_predictions() -> None:
    m = classification_metrics([0, 1, 2, 2], [0, 1, 2, 2])
    assert m["macro_f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["qwk"] == 1.0
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]


def test_majority_class_has_low_macro_f1() -> None:
    y = [0] * 30 + [1] * 10 + [2] * 60
    m = classification_metrics(y, [2] * 100)
    assert m["accuracy"] == 0.6
    assert m["macro_f1"] == pytest.approx(0.25, abs=1e-4)  # F1 0.75 for pos, 0 for others
    assert m["f1_per_class"]["neg"] == 0.0


def test_qwk_penalises_distant_errors_more() -> None:
    y = [0, 0, 2, 2]
    near = classification_metrics(y, [1, 0, 2, 2])["qwk"]
    far = classification_metrics(y, [2, 0, 2, 2])["qwk"]
    assert near > far


def test_mean_std() -> None:
    out = mean_std([{"f": 0.5}, {"f": 0.7}], "f")
    assert out == {"mean": 0.6, "std": pytest.approx(0.1414, abs=1e-4), "n_runs": 2}
