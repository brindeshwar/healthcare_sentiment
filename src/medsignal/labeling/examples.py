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
