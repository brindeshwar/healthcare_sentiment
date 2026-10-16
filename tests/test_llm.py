import json
from collections.abc import Iterator

import httpx
import pytest

from medsignal.labeling.llm import OllamaAnnotator, build_messages
from medsignal.labeling.schema import Discontinued

TEXT = "Gave me a rash so I stopped after a week."


def _client(replies: Iterator[dict[str, object]], seen: list[dict[str, object]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(next(replies))}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_build_messages_contains_examples_and_review() -> None:
    msgs = build_messages("Zoloft", "Depression", TEXT)
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["content"].endswith(TEXT)
    assert "Reviewed drug: Zoloft" in msgs[-1]["content"]
    assert sum(m["role"] == "assistant" for m in msgs) >= 6  # guideline demonstrations


def test_annotate_valid_first_try_sends_schema() -> None:
    good = {"ades": [{"text": "rash"}], "discontinued": "yes", "stop_reasons": ["adverse_event"]}
    seen: list[dict[str, object]] = []
    ann, meta = OllamaAnnotator(model="m").annotate(
        _client(iter([good]), seen), "r1", "X", None, TEXT
    )
    assert meta["attempts"] == 1
    assert ann.discontinued == Discontinued.YES
    assert ann.ades[0].start == TEXT.index("rash")
    assert "properties" in seen[0]["format"]  # type: ignore[operator]
    assert seen[0]["options"]["temperature"] == 0.0  # type: ignore[index]


def test_annotate_retries_on_hallucinated_span() -> None:
    bad = {"ades": [{"text": "hives"}]}
    good = {"ades": [{"text": "rash"}], "discontinued": "yes", "stop_reasons": ["adverse_event"]}
    seen: list[dict[str, object]] = []
    _ann, meta = OllamaAnnotator(model="m").annotate(
        _client(iter([bad, good]), seen), "r1", "X", None, TEXT
    )
    assert meta["attempts"] == 2
    assert "hives" in meta["errors"][0]
    assert "invalid" in seen[1]["messages"][-1]["content"]  # type: ignore[index]


def test_annotate_gives_up_after_retries() -> None:
    bad = {"discontinued": "yes"}  # missing stop reason
    with pytest.raises(RuntimeError):
        OllamaAnnotator(model="m", max_retries=1).annotate(
            _client(iter([bad, bad]), []), "r1", "X", None, TEXT
        )


def test_drop_unaligned_on_failure_keeps_other_labels() -> None:
    stitched = {
        "ades": [{"text": "rash"}, {"text": "gave... a week"}],
        "discontinued": "yes",
        "stop_reasons": ["adverse_event"],
    }
    ann, meta = OllamaAnnotator(model="m", max_retries=1, drop_unaligned_on_failure=True).annotate(
        _client(iter([stitched, stitched]), []), "r1", "X", None, TEXT
    )
    assert [a.text for a in ann.ades] == ["rash"]
    assert meta["dropped_spans"] == ["gave... a week"]
    assert ann.discontinued == Discontinued.YES


def _flaky_client(statuses: list[int], body: dict[str, object]) -> httpx.Client:
    calls = iter(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(calls)
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        return httpx.Response(200, json={"message": {"content": json.dumps(body)}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_server_errors_are_retried() -> None:
    good = {"ades": [{"text": "rash"}], "discontinued": "yes", "stop_reasons": ["adverse_event"]}
    annotator = OllamaAnnotator(model="m", server_backoff_s=(0.0, 0.0))
    ann, meta = annotator.annotate(_flaky_client([500, 500, 200], good), "r1", "X", None, TEXT)
    assert meta["attempts"] == 1
    assert ann.ades[0].text == "rash"


def test_persistent_server_error_raises_unavailable() -> None:
    from medsignal.labeling.llm import AnnotationFailedError, OllamaUnavailableError

    annotator = OllamaAnnotator(model="m", server_backoff_s=(0.0,))
    with pytest.raises(OllamaUnavailableError) as info:
        annotator.annotate(_flaky_client([500, 500], {}), "r1", "X", None, TEXT)
    assert not isinstance(info.value, AnnotationFailedError)


def test_client_errors_are_not_retried() -> None:
    from medsignal.labeling.llm import OllamaUnavailableError

    annotator = OllamaAnnotator(model="m", server_backoff_s=(0.0, 0.0))
    with pytest.raises(OllamaUnavailableError):
        annotator.annotate(_flaky_client([404], {}), "r1", "X", None, TEXT)
