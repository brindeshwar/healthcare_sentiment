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

from medsignal.labeling.examples import GUIDELINE_EXAMPLES
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


def _user(drug: str, condition: str | None, text: str) -> str:
    return f"Reviewed drug: {drug}\nCondition: {condition or 'not given'}\nReview:\n{text}"


def build_messages(drug: str, condition: str | None, text: str) -> list[dict[str, str]]:
    """System prompt, guideline examples as demonstrations, then the review."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_text, labels in GUIDELINE_EXAMPLES:
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
    # review (recorded in provenance). Off for teacher evaluation, where such outputs count as failures.
    drop_unaligned_on_failure: bool = False

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
        messages = build_messages(drug, condition, text)
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
