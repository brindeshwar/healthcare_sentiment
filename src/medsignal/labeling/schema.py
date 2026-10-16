"""Annotation schema shared by the gold set, the LLM teacher and evaluation (ADR-0016).

The human guidelines are in ``docs/annotation_guidelines.md``; this module is their machine form.
Annotators and the LLM give each ADE as a verbatim ``text`` span; ``align_spans`` adds character
offsets so span-level metrics can be computed.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.1"  # 1.1: span convention (ADR-0019); fields unchanged


class Severity(StrEnum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    NOT_STATED = "not_stated"


class Discontinued(StrEnum):
    YES = "yes"  # stopped or switched away from the reviewed drug
    PLANNED = "planned"  # says they will stop / have it removed
    NO = "no"  # still taking it (or says they will continue)
    NOT_STATED = "not_stated"


class StopReason(StrEnum):
    ADVERSE_EVENT = "adverse_event"
    LACK_OF_EFFICACY = "lack_of_efficacy"  # never worked, stopped working, condition worsened
    COST_ACCESS = "cost_access"  # price, insurance, availability
    CONVENIENCE = "convenience"  # dosing schedule, administration, hassle
    COURSE_COMPLETED = "course_completed"  # finished the prescription / no longer needed
    PREGNANCY_RELATED = "pregnancy_related"  # planning pregnancy, pregnant, breastfeeding
    CLINICIAN_DECISION = "clinician_decision"  # doctor stopped/switched it, no other reason given
    OTHER = "other"
    NOT_STATED = "not_stated"


class Effectiveness(StrEnum):
    EFFECTIVE = "effective"
    PARTIAL = "partial"  # some benefit, mixed, or benefit that wore off
    NOT_EFFECTIVE = "not_effective"  # no benefit or the treated condition got worse
    NOT_STATED = "not_stated"


class ADEMention(BaseModel):
    """One adverse effect the reviewer attributes to the reviewed drug."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, description="Verbatim span copied from the review")
    severity: Severity = Severity.NOT_STATED
    withdrawal: bool = Field(False, description="Occurred when stopping/reducing/missing doses")
    beneficial: bool = Field(False, description="Reviewer presents the effect as welcome")
    start: int | None = Field(None, description="Character offset; filled by align_spans")
    end: int | None = None


def _check_reasons(discontinued: Discontinued, reasons: list[StopReason]) -> None:
    stopping = discontinued in (Discontinued.YES, Discontinued.PLANNED)
    if stopping and not reasons:
        raise ValueError(
            "discontinued yes/planned needs at least one stop reason (or 'not_stated')"
        )
    if not stopping and reasons:
        raise ValueError("stop_reasons only allowed when discontinued is yes/planned")
    if StopReason.NOT_STATED in reasons and len(reasons) > 1:
        raise ValueError("'not_stated' cannot be combined with other stop reasons")


class ADEOut(BaseModel):
    """ADE as produced by the LLM: no offsets (added later by ``align_spans``)."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    severity: Severity = Severity.NOT_STATED
    withdrawal: bool = False
    beneficial: bool = False


class LLMAnnotation(BaseModel):
    """What the LLM teacher returns for one review; its JSON schema constrains decoding."""

    model_config = ConfigDict(extra="forbid")

    ades: list[ADEOut] = Field(default_factory=list)
    discontinued: Discontinued = Discontinued.NOT_STATED
    stop_reasons: list[StopReason] = Field(default_factory=list)
    effectiveness: Effectiveness = Effectiveness.NOT_STATED

    @model_validator(mode="after")
    def _reasons_match_discontinuation(self) -> LLMAnnotation:
        _check_reasons(self.discontinued, self.stop_reasons)
        return self

    def to_review(self, review_id: str) -> ReviewAnnotation:
        return ReviewAnnotation(
            review_id=review_id,
            ades=[ADEMention(**a.model_dump()) for a in self.ades],
            discontinued=self.discontinued,
            stop_reasons=self.stop_reasons,
            effectiveness=self.effectiveness,
        )


class ReviewAnnotation(BaseModel):
    """All labels for one review."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    ades: list[ADEMention] = Field(default_factory=list)
    discontinued: Discontinued = Discontinued.NOT_STATED
    stop_reasons: list[StopReason] = Field(default_factory=list)
    effectiveness: Effectiveness = Effectiveness.NOT_STATED
    notes: str | None = None

    @model_validator(mode="after")
    def _reasons_match_discontinuation(self) -> ReviewAnnotation:
        _check_reasons(self.discontinued, self.stop_reasons)
        return self


class AlignmentError(ValueError):
    """An ADE span does not occur in the review text."""


def align_spans(annotation: ReviewAnnotation, text: str) -> ReviewAnnotation:
    """Return a copy with character offsets for every ADE span.

    Matching is case-insensitive and takes the earliest unused occurrence, so spans keep reading
    order and "nausea ... nausea" labelled twice gets two different offsets. The stored text is
    re-copied from the review, so its casing always matches the source.
    """
    lowered = text.lower()
    used: list[tuple[int, int]] = []
    aligned: list[ADEMention] = []
    for ade in annotation.ades:
        span = ade.text.strip()
        start = _find_unused(lowered, span.lower(), used)
        if start < 0:
            raise AlignmentError(f"span {span!r} not found in review {annotation.review_id}")
        end = start + len(span)
        used.append((start, end))
        aligned.append(ade.model_copy(update={"text": text[start:end], "start": start, "end": end}))
    return annotation.model_copy(update={"ades": aligned})


def align_spans_lenient(
    annotation: ReviewAnnotation, text: str
) -> tuple[ReviewAnnotation, list[str]]:
    """Like ``align_spans`` but drops spans that are not in the text; returns the dropped spans."""
    kept: list[ADEMention] = []
    dropped: list[str] = []
    for ade in annotation.ades:
        trial = annotation.model_copy(update={"ades": [*kept, ade]})
        try:
            kept = align_spans(trial, text).ades
        except AlignmentError:
            dropped.append(ade.text)
    return annotation.model_copy(update={"ades": kept}), dropped


def _find_unused(haystack: str, needle: str, used: list[tuple[int, int]]) -> int:
    """Earliest occurrence that does not overlap a used span and sits on word boundaries."""
    pos = haystack.find(needle)
    while pos >= 0:
        end = pos + len(needle)
        free = all(pos >= e or end <= s for s, e in used)
        if free and _boundary(haystack, pos, end, needle):
            return pos
        pos = haystack.find(needle, pos + 1)
    return -1


def _boundary(haystack: str, start: int, end: int, needle: str) -> bool:
    """Span edges that are word characters must not continue a word ('rash' not in 'crash')."""
    left_ok = not needle[0].isalnum() or start == 0 or not haystack[start - 1].isalnum()
    right_ok = not needle[-1].isalnum() or end == len(haystack) or not haystack[end].isalnum()
    return left_ok and right_ok
