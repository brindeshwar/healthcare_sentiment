from medsignal.evaluation.external import GoldDoc
from medsignal.evaluation.spans import Span
from medsignal.labeling.benchmark import prompt_version, safe_name, score


def test_prompt_version_is_stable_hash() -> None:
    assert prompt_version() == prompt_version()
    assert len(prompt_version()) == 10


def test_safe_name() -> None:
    assert safe_name("qwen3.5:9b-q4_K_M") == "qwen3.5_9b-q4_K_M"


def test_score_counts_failures_as_empty_predictions() -> None:
    docs = [
        GoldDoc("LIPITOR.1", "LIPITOR", "muscle pain and cramps", [Span.contiguous(0, 11)]),
        GoldDoc("ARTHROTEC.1", "ARTHROTEC", "bad rash", [Span.contiguous(4, 8)]),
    ]
    preds = {
        "LIPITOR.1": {
            "status": "ok",
            "attempts": 1,
            "seconds": 2.0,
            "ades": [{"start": 0, "end": 11}],
        },
        "ARTHROTEC.1": {"status": "failed", "seconds": 9.0, "ades": []},
    }
    m = score(docs, preds)
    assert m["docs"] == 2
    assert m["strict"]["tp"] == 1 and m["strict"]["n_gold"] == 2
    assert m["failure_rate"] == 0.5
    assert m["first_try_rate"] == 0.5
    assert m["relaxed_by_drug"]["LIPITOR"]["f1"] == 1.0
    assert m["relaxed_by_drug"]["diclofenac family"]["recall"] == 0.0
    assert m["seconds_median"] == 5.5
