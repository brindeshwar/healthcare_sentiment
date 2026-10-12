"""Source-specific loaders that map raw files onto one standard review schema.

Standard columns (one row per raw listing, before de-duplication):
    source, drug, condition, text, rating, date, helpful_count, duration
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

STANDARD_COLUMNS = [
    "source",
    "drug",
    "condition",
    "text",
    "rating",
    "date",
    "helpful_count",
    "duration",
]

# DrugRev "IntakeTime" buckets, in order; anything else (e.g. a shifted date) is treated as missing.
DURATION_BUCKETS = [
    "Taken for less than 1 month",
    "Taken for 1 to 6 months",
    "Taken for 6 months to 1 year",
    "Taken for 1 to 2 years",
    "Taken for 2 to 5 years",
    "Taken for 5 to 10 years",
    "Taken for 10 years or more",
]

_WS = re.compile(r"\s+")


def clean_text(raw: object) -> str:
    """Decode HTML entities, collapse whitespace, strip wrapping quotes."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    text = _WS.sub(" ", html.unescape(str(raw))).strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return text


def normalize_for_matching(text: str) -> str:
    """Lower-cased, whitespace-collapsed form used only for duplicate detection."""
    return _WS.sub(" ", text).strip().lower()


def clean_condition(raw: object) -> str | None:
    """Return a usable condition name, or None for missing/scraping artefacts."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    cond = clean_text(raw)
    if not cond or "</span>" in cond or cond.lower() in {"not mentioned", "not listed / othe"}:
        return None
    if cond.startswith("For "):
        cond = cond[4:]
    return cond


def clean_duration(raw: object) -> str | None:
    """Map DrugRev IntakeTime to a known bucket, else None."""
    value = clean_text(raw)
    return value if value in DURATION_BUCKETS else None


def load_uci(path: Path) -> pd.DataFrame:
    """Load the UCI Drugs.com train file (CSV) into the standard schema."""
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "source": "uci",
            "drug": df["drugName"].map(clean_text),
            "condition": df["condition"].map(clean_condition),
            "text": df["review"].map(clean_text),
            "rating": df["rating"].astype("int64"),
            "date": pd.to_datetime(df["date"], format="%B %d, %Y"),
            "helpful_count": df["usefulCount"].astype("int64"),
            "duration": None,
        }
    )
    return out[STANDARD_COLUMNS]


def load_drugrev(path: Path) -> pd.DataFrame:
    """Load DrugRev (Mendeley) into the standard schema. ``UserName`` is dropped here."""
    df = pd.read_csv(path).drop(columns=["UserName"], errors="ignore")
    out = pd.DataFrame(
        {
            "source": "drugrev",
            "drug": df["MedicineName"].map(clean_text),
            "condition": df["MedicineFor"].map(clean_condition),
            "text": df["Reviews"].map(clean_text),
            "rating": df["Rating"].astype("int64"),
            "date": pd.to_datetime(df["ReviewDate"], format="%d-%b-%y"),
            "helpful_count": df["NumberOfLikes"].astype("int64"),
            "duration": df["IntakeTime"].map(clean_duration),
        }
    )
    return out[STANDARD_COLUMNS]
