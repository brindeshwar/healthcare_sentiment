import numpy as np
import pandas as pd

from medsignal.extraction.review_heads import OUTPUTS, agreement, decode, encode_targets


def _probs(disc: str, eff: str, reasons: dict[str, float]) -> np.ndarray:
    p = np.full(len(OUTPUTS), 0.05)
    p[OUTPUTS.index(f"disc:{disc}")] = 0.9
    p[OUTPUTS.index(f"eff:{eff}")] = 0.8
    for r, v in reasons.items():
        p[OUTPUTS.index(f"reason:{r}")] = v
    return p


def test_encode_targets_one_hot_groups() -> None:
    y = encode_targets("yes", "partial", ["adverse_event", "cost_access"])
    assert len(y) == len(OUTPUTS) == 17
    assert sum(y) == 4
    assert y[OUTPUTS.index("disc:yes")] == 1.0
    assert y[OUTPUTS.index("reason:cost_access")] == 1.0


def test_decode_reasons_only_when_stopping() -> None:
    out = decode(_probs("no", "effective", {"adverse_event": 0.9}))
    assert out == {"discontinued": "no", "effectiveness": "effective", "stop_reasons": []}


def test_decode_forces_one_reason_and_drops_not_stated_combo() -> None:
    weak = decode(_probs("yes", "not_stated", {"lack_of_efficacy": 0.3}))
    assert weak["stop_reasons"] == ["lack_of_efficacy"]
    combo = decode(_probs("planned", "partial", {"adverse_event": 0.8, "not_stated": 0.7}))
    assert combo["stop_reasons"] == ["adverse_event"]


def test_agreement_perfect() -> None:
    gold = pd.DataFrame(
        {
            "discontinued": ["yes", "no"],
            "effectiveness": ["effective", "partial"],
            "stop_reasons": [["adverse_event"], []],
        }
    )
    pred = [
        {"discontinued": "yes", "effectiveness": "effective", "stop_reasons": ["adverse_event"]},
        {"discontinued": "no", "effectiveness": "partial", "stop_reasons": []},
    ]
    a = agreement(gold, pred)
    assert a["discontinued"]["accuracy"] == 1.0
    assert a["stop_reasons_micro_f1"] == 1.0
    assert a["stopped_or_planned_f1"] == 1.0


def test_choose_seed_best_then_lowest() -> None:
    from medsignal.extraction.infer import choose_seed

    reports = {42: {"m": {"f1": 0.8}}, 13: {"m": {"f1": 0.8}}, 7: {"m": {"f1": 0.7}}}
    assert choose_seed(reports, ["m", "f1"]) == 13
