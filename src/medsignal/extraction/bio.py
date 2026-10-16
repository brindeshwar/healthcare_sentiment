"""Character spans <-> token BIO tags, via the tokenizer's offset mapping.

Labels: 0 = O, 1 = B-ADE, 2 = I-ADE; special/padding tokens get -100 (ignored by the loss).
A token belongs to a span if their character ranges overlap; the first such token is B, the rest I.
Decoding merges B/I runs back into character spans, trims whitespace at the edges (BPE tokenizers
include the leading space in a token's offsets) and treats a stray I as the start of a new span.
"""

from __future__ import annotations

from collections.abc import Sequence

O, B, I = 0, 1, 2  # noqa: E741 - conventional BIO names
IGNORE = -100
LABELS = ("O", "B-ADE", "I-ADE")


def spans_to_bio(
    offsets: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]], text: str
) -> list[int]:
    """Tag tokens (given their ``(start, end)`` char offsets) for the ``spans`` (char ranges)."""
    tags: list[int] = []
    prev_span: int | None = None
    for start, end in offsets:
        if start == end:  # special token or padding
            tags.append(IGNORE)
            prev_span = None
            continue
        # ignore leading whitespace inside the token when testing overlap
        s = start
        while s < end and text[s].isspace():
            s += 1
        hit = next((k for k, (a, b) in enumerate(spans) if s < b and a < end), None)
        if hit is None:
            tags.append(O)
            prev_span = None
        else:
            tags.append(I if hit == prev_span else B)
            prev_span = hit
    return tags


def bio_to_spans(
    offsets: Sequence[tuple[int, int]], tags: Sequence[int], text: str
) -> list[tuple[int, int]]:
    """Merge B/I token runs into character spans, trimming surrounding whitespace."""
    spans: list[tuple[int, int]] = []
    cur: list[int] | None = None
    for (start, end), tag in zip(offsets, tags, strict=True):
        if start == end or tag in (O, IGNORE):
            if cur:
                spans.append((cur[0], cur[1]))
            cur = None
            continue
        if tag == B or cur is None:
            if cur:
                spans.append((cur[0], cur[1]))
            cur = [start, end]
        else:
            cur[1] = end
    if cur:
        spans.append((cur[0], cur[1]))
    trimmed = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            trimmed.append((a, b))
    return trimmed
