"""LLM annotator: prompt built from the approved guidelines + Ollama structured output.

The JSON schema of ``LLMAnnotation`` is passed to Ollama's ``format`` so decoding is constrained;
responses are still validated with Pydantic and spans are aligned to the review text. On a
validation or alignment error the model is asked again with the error message (``max_retries``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import ValidationError

from medsignal.labeling.examples import GUIDELINE_EXAMPLES, GUIDELINE_EXAMPLES_V2
from medsignal.labeling.schema import (
    SCHEMA_VERSION,
    AlignmentError,
    LLMAnnotation,
    ReviewAnnotation,
    align_spans,
    align_spans_lenient,
)

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You annotate patient drug reviews for a drug-safety research project.
Return JSON matching the schema. Label only what the review says.

ADE mentions (ades): unwanted or neutral effects on body, mind, behaviour or lab values that the
reviewer EXPERIENCED and attributes to THE REVIEWED DRUG.
Do NOT label: negated effects ("no side effects"), worries or hypotheticals, symptoms of the
condition being treated or their relief, effects of other drugs, the treated condition getting worse
(that is effectiveness), generic phrases like "bad side effects".
Spans: copy the shortest exact text from the review that names the effect, including body location
("pain in my legs", "couldn't sleep", "gained 40 lbs"). Put separable intensity words in severity,
not in the span ("extreme fatigue" -> text "fatigue", severity "severe"). One item per effect, in
reading order; label repeated mentions separately.
severity from the reviewer's words: severe = hospital/ER, life-threatening, unable to function,
"unbearable", "horrific", "extreme"; moderate = clearly bothersome or a reason for stopping;
mild = "slight", "minor", "went away quickly"; otherwise not_stated.
withdrawal = true if it happened when stopping, reducing or missing doses.
beneficial = true if the reviewer presents a non-therapeutic effect as welcome (e.g. no periods).

discontinued: yes (stopped, switched away, removed, finished the course), planned (says they will
stop), no (explicitly continuing), not_stated. A dose change alone is not discontinuation.
stop_reasons (only when yes/planned; at least one): adverse_event, lack_of_efficacy, cost_access,
convenience, course_completed, pregnancy_related, clinician_decision (doctor changed it with no other
reason), other, not_stated (alone). If an effect is described and they stop without another reason,
use adverse_event. If a doctor switched them because of side effects, use adverse_event.

effectiveness for the treated condition, judged on the final state: effective, partial (some
benefit, mixed, or wore off), not_effective (no benefit or condition worse), not_stated."""


# v2 (dev-tuned on CADEC, ADR-0019): keep the reviewer's own words in spans (intensity + location),
# never paraphrase, and label every mention (repeats, list items, short generic words). Recall was
# the weak point of v1 (relaxed recall ~0.65 at precision ~0.93).
_SPANS_V1 = """Spans: copy the shortest exact text from the review that names the effect, including body location
("pain in my legs", "couldn't sleep", "gained 40 lbs"). Put separable intensity words in severity,
not in the span ("extreme fatigue" -> text "fatigue", severity "severe"). One item per effect, in
reading order; label repeated mentions separately."""

_SPANS_V2 = """Spans: copy the words of the review that describe the effect EXACTLY as written, including
intensity words and body location ("extreme fatigue", "pain in my legs", "couldn't sleep",
"gained 40 lbs"). Never rephrase, reorder or shorten inside a span: if the review says "pain in my
shoulder", the span is "pain in my shoulder", not "shoulder pain". Still set severity from those
words. Label EVERY mention, in reading order: each item of a list separately, a repeated effect
again each time it is mentioned, and short words such as "pain", "pains", "aching", "weak" when
they describe an effect of the drug. When it is unclear whether a described problem came from the
drug, include it if the reviewer links it to taking the drug."""

assert _SPANS_V1 in SYSTEM_PROMPT
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT.replace(_SPANS_V1, _SPANS_V2)

PROMPTS: dict[str, tuple[str, list[tuple[str, dict[str, Any]]]]] = {
    "v1": (SYSTEM_PROMPT, GUIDELINE_EXAMPLES),
    "v2": (SYSTEM_PROMPT_V2, GUIDELINE_EXAMPLES_V2),
}


def _user(drug: str, condition: str | None, text: str) -> str:
    return f"Reviewed drug: {drug}\nCondition: {condition or 'not given'}\nReview:\n{text}"


def build_messages(
    drug: str, condition: str | None, text: str, prompt: str = "v1"
) -> list[dict[str, str]]:
    """System prompt, guideline examples as demonstrations, then the review."""
    system, examples = PROMPTS[prompt]
    messages = [{"role": "system", "content": system}]
    for example_text, labels in examples:
        demo = LLMAnnotation(**labels)
        messages.append({"role": "user", "content": _user("(see review)", None, example_text)})
        messages.append({"role": "assistant", "content": demo.model_dump_json()})
    messages.append({"role": "user", "content": _user(drug, condition, text)})
    return messages


@dataclass
class OllamaAnnotator:
    model: str
    temperature: float = 0.0
    num_ctx: int = 4096
    max_retries: int = 2
    seed: int = 7
    host: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    timeout_s: float = 300.0
    # After the last retry, keep a schema-valid answer and drop only spans that do not occur in the
    # review (recorded in provenance as dropped_spans). Default off: the whole review then fails.
    drop_unaligned_on_failure: bool = False
    prompt: str = "v1"  # key of PROMPTS

    def _chat(self, client: httpx.Client, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "format": LLMAnnotation.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "seed": self.seed,
            },
        }
        resp = client.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return str(resp.json()["message"]["content"])

    def annotate(
        self, client: httpx.Client, review_id: str, drug: str, condition: str | None, text: str
    ) -> tuple[ReviewAnnotation, dict[str, Any]]:
        """Return the aligned annotation and provenance (attempts, errors)."""
        messages = build_messages(drug, condition, text, self.prompt)
        errors: list[str] = []
        last_valid: ReviewAnnotation | None = None
        for attempt in range(1, self.max_retries + 2):
            raw = self._chat(client, messages)
            try:
                ann = LLMAnnotation.model_validate_json(raw).to_review(review_id)
                last_valid = ann
                aligned = align_spans(ann, text)
                meta = {"attempts": attempt, "errors": errors}
                return aligned, meta
            except (ValidationError, AlignmentError, json.JSONDecodeError) as exc:
                msg = str(exc).splitlines()[0][:300]
                errors.append(msg)
                log.debug("review %s attempt %s failed: %s", review_id, attempt, msg)
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"That output was invalid: {msg}. Every ADE text must be copied "
                        "exactly from the review. Return corrected JSON.",
                    },
                ]
        if self.drop_unaligned_on_failure and last_valid is not None:
            kept, dropped = align_spans_lenient(last_valid, text)
            return kept, {
                "attempts": self.max_retries + 1,
                "errors": errors,
                "dropped_spans": dropped,
            }
        raise RuntimeError(f"review {review_id}: no valid annotation after retries: {errors}")


def provenance(model: str) -> dict[str, str]:
    return {"annotator": f"llm:{model}", "schema_version": SCHEMA_VERSION}
