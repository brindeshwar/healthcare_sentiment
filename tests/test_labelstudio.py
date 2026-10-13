import xml.etree.ElementTree as ET

from medsignal.labeling.labelstudio import from_result, labeling_config, to_prediction
from medsignal.labeling.schema import ReviewAnnotation, StopReason, align_spans

TEXT = "Bad headache and a mild rash. I stopped it because of the cost too."


def _ann() -> ReviewAnnotation:
    ann = ReviewAnnotation(
        review_id="r1",
        ades=[
            {"text": "headache", "severity": "moderate"},
            {"text": "rash", "severity": "mild", "withdrawal": True, "beneficial": False},
        ],
        discontinued="yes",
        stop_reasons=["adverse_event", "cost_access"],
        effectiveness="not_stated",
    )
    return align_spans(ann, TEXT)


def test_labeling_config_is_valid_xml_with_all_enums() -> None:
    xml = labeling_config()
    root = ET.fromstring(xml)
    names = {c.get("name") for c in root.iter("Choices")}
    assert names == {"severity", "flags", "discontinued", "stop_reasons", "effectiveness"}
    reasons = {c.get("value") for c in root.iter("Choice")}
    assert {r.value for r in StopReason} <= reasons


def test_prediction_round_trip() -> None:
    ann = _ann()
    pred = to_prediction(ann, "llm:test")
    back = from_result("r1", pred["result"])
    assert back.model_dump() == ann.model_dump()


def test_from_result_defaults_and_notes() -> None:
    result = [
        {
            "from_name": "discontinued",
            "to_name": "review",
            "type": "choices",
            "value": {"choices": ["no"]},
        },
        {
            "from_name": "notes",
            "to_name": "review",
            "type": "textarea",
            "value": {"text": ["unclear drug"]},
        },
    ]
    ann = from_result("r2", result)
    assert ann.ades == []
    assert ann.effectiveness.value == "not_stated"
    assert ann.notes == "unclear drug"
