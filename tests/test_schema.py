import pytest
from pydantic import ValidationError

from medsignal.labeling.schema import (
    ADEMention,
    AlignmentError,
    Discontinued,
    ReviewAnnotation,
    Severity,
    StopReason,
    align_spans,
)

TEXT = "Gave me Nausea and a rash. The nausea faded but I stopped after a month."


def _ann(**kw: object) -> ReviewAnnotation:
    return ReviewAnnotation(review_id="r1", **kw)  # type: ignore[arg-type]


def test_defaults_are_not_stated() -> None:
    a = _ann()
    assert a.ades == []
    assert a.discontinued == Discontinued.NOT_STATED


def test_stopping_requires_a_reason() -> None:
    with pytest.raises(ValidationError):
        _ann(discontinued="yes")
    assert _ann(discontinued="yes", stop_reasons=["adverse_event"]).stop_reasons == [
        StopReason.ADVERSE_EVENT
    ]


def test_reasons_forbidden_when_not_stopping() -> None:
    with pytest.raises(ValidationError):
        _ann(discontinued="no", stop_reasons=["cost_access"])


def test_not_stated_reason_is_exclusive() -> None:
    with pytest.raises(ValidationError):
        _ann(discontinued="planned", stop_reasons=["not_stated", "adverse_event"])


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        ADEMention(text="rash", colour="red")  # type: ignore[call-arg]


def test_align_spans_case_insensitive_and_repeated() -> None:
    a = _ann(
        ades=[
            {"text": "nausea", "severity": "mild"},
            {"text": "nausea"},
            {"text": "a rash"},
        ]
    )
    out = align_spans(a, TEXT)
    starts = [m.start for m in out.ades]
    assert starts[0] == TEXT.index("Nausea")
    assert starts[1] == TEXT.index("nausea")
    assert out.ades[0].text == "Nausea"  # copied from the review, original casing
    assert out.ades[0].severity == Severity.MILD
    assert all(TEXT[m.start : m.end] == m.text for m in out.ades)  # type: ignore[misc]


def test_align_spans_respects_word_boundaries() -> None:
    text = "After the car crash I started it, and got a rash."
    out = align_spans(_ann(ades=[{"text": "rash"}]), text)
    assert out.ades[0].start == text.index(" rash") + 1


def test_align_spans_missing_span_raises() -> None:
    with pytest.raises(AlignmentError):
        align_spans(_ann(ades=[{"text": "headache"}]), TEXT)


# Worked examples from docs/annotation_guidelines.md §4: must validate and align (keeps doc and code in sync).
GUIDELINE_EXAMPLES = [
    (
        "Within two weeks on Paxil I had constant migraines, extreme fatigue and constipation. "
        "My doctor moved me to Zoloft and I feel much better.",
        {
            "ades": [
                {"text": "migraines", "severity": "moderate"},
                {"text": "fatigue", "severity": "severe"},
                {"text": "constipation", "severity": "moderate"},
            ],
            "discontinued": "yes",
            "stop_reasons": ["adverse_event"],
        },
    ),
    (
        "Worked great for four years with no side effects. Had to stop because it got too expensive.",
        {"discontinued": "yes", "stop_reasons": ["cost_access"], "effectiveness": "effective"},
    ),
    (
        "Ended up in hospital with a severely long QT interval. Get an ECG before starting. "
        "The withdrawal was horrific too.",
        {
            "ades": [
                {"text": "long QT interval", "severity": "severe"},
                {"text": "withdrawal", "severity": "severe", "withdrawal": True},
            ],
            "discontinued": "yes",
            "stop_reasons": ["adverse_event"],
        },
    ),
    (
        "No period at all after month three, which is great because my cramps were awful. But I've "
        "become very depressed and gained a lot of weight. After this pack I'm going off it.",
        {
            "ades": [
                {"text": "No period", "beneficial": True},
                {"text": "depressed", "severity": "moderate"},
                {"text": "gained a lot of weight", "severity": "moderate"},
            ],
            "discontinued": "planned",
            "stop_reasons": ["adverse_event"],
            "effectiveness": "effective",
        },
    ),
    (
        "It seemed to make my shoulder spasm worse, and on the second try I also got a rash. Not for me.",
        {
            "ades": [{"text": "rash", "severity": "moderate"}],
            "discontinued": "yes",
            "stop_reasons": ["lack_of_efficacy", "adverse_event"],
            "effectiveness": "not_effective",
        },
    ),
    (
        "Insertion felt like a bad cramp. A few hours later, minor cramping and very little spotting.",
        {
            "ades": [
                {"text": "cramp", "severity": "moderate"},
                {"text": "cramping", "severity": "mild"},
                {"text": "spotting", "severity": "mild"},
            ]
        },
    ),
]


@pytest.mark.parametrize(("text", "labels"), GUIDELINE_EXAMPLES)
def test_guideline_examples_validate_and_align(text: str, labels: dict[str, object]) -> None:
    ann = ReviewAnnotation(review_id="example", **labels)  # type: ignore[arg-type]
    aligned = align_spans(ann, text)
    starts = [m.start for m in aligned.ades]
    assert starts == sorted(starts)  # spans listed in reading order
