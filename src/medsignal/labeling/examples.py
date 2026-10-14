"""Worked examples from docs/annotation_guidelines.md §4.

Single source for (a) tests that keep the guidelines and schema in sync and (b) the LLM prompt's
demonstrations. Constructed sentences modelled on real reviews; no review text is redistributed.
"""

from __future__ import annotations

from typing import Any

GUIDELINE_EXAMPLES: list[tuple[str, dict[str, Any]]] = [
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

# Prompt v2 demonstrations (ADR-0019): same sentences, spans keep the reviewer's intensity words and
# body location; one extra example shows list items, a short generic word and a repeated mention.
GUIDELINE_EXAMPLES_V2: list[tuple[str, dict[str, Any]]] = [
    (
        GUIDELINE_EXAMPLES[0][0],
        {
            "ades": [
                {"text": "constant migraines", "severity": "moderate"},
                {"text": "extreme fatigue", "severity": "severe"},
                {"text": "constipation", "severity": "moderate"},
            ],
            "discontinued": "yes",
            "stop_reasons": ["adverse_event"],
        },
    ),
    GUIDELINE_EXAMPLES[1],
    (
        GUIDELINE_EXAMPLES[2][0],
        {
            "ades": [
                {"text": "severely long QT interval", "severity": "severe"},
                {"text": "withdrawal", "severity": "severe", "withdrawal": True},
            ],
            "discontinued": "yes",
            "stop_reasons": ["adverse_event"],
        },
    ),
    (
        GUIDELINE_EXAMPLES[3][0],
        {
            "ades": [
                {"text": "No period", "beneficial": True},
                {"text": "very depressed", "severity": "moderate"},
                {"text": "gained a lot of weight", "severity": "moderate"},
            ],
            "discontinued": "planned",
            "stop_reasons": ["adverse_event"],
            "effectiveness": "effective",
        },
    ),
    GUIDELINE_EXAMPLES[4],
    (
        GUIDELINE_EXAMPLES[5][0],
        {
            "ades": [
                {"text": "bad cramp", "severity": "moderate"},
                {"text": "minor cramping", "severity": "mild"},
                {"text": "very little spotting", "severity": "mild"},
            ]
        },
    ),
    (
        "Two days in I had pain in my knees, aching all over and some nausea. By week two the pain was "
        "so bad I could barely walk, so I quit.",
        {
            "ades": [
                {"text": "pain in my knees", "severity": "moderate"},
                {"text": "aching all over", "severity": "not_stated"},
                {"text": "nausea", "severity": "mild"},
                {"text": "pain", "severity": "severe"},
            ],
            "discontinued": "yes",
            "stop_reasons": ["adverse_event"],
        },
    ),
]
