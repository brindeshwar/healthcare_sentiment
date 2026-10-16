import re

import pytest

from medsignal.extraction.bio import IGNORE, B, I, O, bio_to_spans, spans_to_bio

TEXT = "Gave me extreme fatigue and a rash, no nausea."


def _offsets(text: str, leading_space: bool) -> list[tuple[int, int]]:
    """Fake word tokenizer: one token per word/punct; BPE-style tokens include the leading space."""
    offs = [(0, 0)]  # [CLS]
    for m in re.finditer(r"\w+|[^\w\s]", text):
        s = m.start()
        if leading_space and s > 0 and text[s - 1] == " ":
            s -= 1
        offs.append((s, m.end()))
    offs.append((0, 0))  # [SEP]
    return offs


@pytest.mark.parametrize("leading_space", [False, True])
def test_round_trip(leading_space: bool) -> None:
    spans = [(TEXT.index("extreme"), TEXT.index(" and")), (TEXT.index("rash"), TEXT.index(","))]
    offs = _offsets(TEXT, leading_space)
    tags = spans_to_bio(offs, spans, TEXT)
    assert tags[0] == IGNORE and tags[-1] == IGNORE
    assert tags.count(B) == 2 and tags.count(I) == 1
    assert bio_to_spans(offs, tags, TEXT) == spans


def test_adjacent_spans_stay_separate() -> None:
    text = "headache nausea"
    spans = [(0, 8), (9, 15)]
    offs = _offsets(text, leading_space=True)
    tags = spans_to_bio(offs, spans, text)
    assert tags[1:3] == [B, B]
    assert bio_to_spans(offs, tags, text) == spans


def test_stray_inside_tag_starts_new_span() -> None:
    offs = _offsets(TEXT, leading_space=False)
    tags = [IGNORE] + [O] * (len(offs) - 2) + [IGNORE]
    tags[TEXT.split().index("rash,") + 1] = I  # word index -> token index (+1 for CLS)
    spans = bio_to_spans(offs, tags, TEXT)
    assert [TEXT[a:b] for a, b in spans] == ["rash"]


def test_no_spans_all_outside() -> None:
    offs = _offsets(TEXT, leading_space=True)
    tags = spans_to_bio(offs, [], TEXT)
    assert set(tags) == {O, IGNORE}
    assert bio_to_spans(offs, tags, TEXT) == []


def test_split_by_group_holds_out_whole_groups() -> None:
    import pandas as pd

    from medsignal.extraction.train import split_by_group

    df = pd.DataFrame({"drug_group": [f"g{i % 10}" for i in range(100)], "x": range(100)})
    tr, va = split_by_group(df, 0.2, seed=1)
    assert not set(tr["drug_group"]) & set(va["drug_group"])
    assert len(tr) + len(va) == 100
    assert 20 <= len(va) <= 30


def test_token_f1_ignores_padding() -> None:
    from types import SimpleNamespace

    import numpy as np

    from medsignal.extraction.train import token_f1

    labels = np.array([[IGNORE, B, I, O, IGNORE]])
    logits = np.zeros((1, 5, 3))
    for j, t in enumerate([O, B, O, O, B]):  # last position is padding: must not count
        logits[0, j, t] = 1.0
    m = token_f1(SimpleNamespace(label_ids=labels, predictions=logits))
    assert m["token_p"] == 1.0
    assert m["token_r"] == 0.5


def test_gliner_select_spans_threshold_labels_and_overlap() -> None:
    from medsignal.extraction.gliner_score import select_spans

    spans = [
        {"start": 0, "end": 8, "label": "side effect", "score": 0.9},
        {"start": 0, "end": 8, "label": "adverse drug reaction", "score": 0.7},  # same mention
        {"start": 4, "end": 12, "label": "side effect", "score": 0.6},  # overlaps the best one
        {"start": 20, "end": 25, "label": "symptom", "score": 0.95},  # label not allowed
        {"start": 30, "end": 34, "label": "side effect", "score": 0.2},  # below threshold
    ]
    assert select_spans(spans, ("side effect", "adverse drug reaction"), 0.5) == [(0, 8)]
    assert select_spans(spans, ("symptom",), 0.5) == [(20, 25)]
