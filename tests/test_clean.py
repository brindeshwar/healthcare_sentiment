from pathlib import Path

import pandas as pd

from medsignal.data.clean import (
    clean_condition,
    clean_duration,
    clean_text,
    load_drugrev,
    load_uci,
    truncated_name_map,
)


def test_clean_text_unescapes_and_strips_quotes() -> None:
    assert clean_text('"It&#039;s   great\r\n stuff"') == "It's great stuff"


def test_clean_text_handles_missing() -> None:
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_clean_condition() -> None:
    assert clean_condition("For Birth Control") == "Birth Control"
    assert clean_condition("3</span> users found this comment helpful.") is None
    assert clean_condition("Not Mentioned") is None
    assert clean_condition(float("nan")) is None
    assert clean_condition("Depression") == "Depression"


def test_clean_duration_rejects_shifted_dates() -> None:
    assert clean_duration("Taken for 1 to 6 months") == "Taken for 1 to 6 months"
    assert clean_duration("4-Jan-12") is None
    assert clean_duration("Not Specified") is None


def test_load_uci(tmp_path: Path) -> None:
    f = tmp_path / "uci.csv"
    pd.DataFrame(
        {
            "uniqueID": [1],
            "drugName": ["Zoloft"],
            "condition": ["Depression"],
            "review": ['"Helped a lot &amp; no nausea."'],
            "rating": [9],
            "date": ["May 20, 2012"],
            "usefulCount": [3],
        }
    ).to_csv(f, index=False)
    row = load_uci(f).iloc[0]
    assert row["text"] == "Helped a lot & no nausea."
    assert row["date"] == pd.Timestamp("2012-05-20")
    assert row["source"] == "uci"


def test_load_drugrev_drops_username(tmp_path: Path) -> None:
    f = tmp_path / "dr.csv"
    pd.DataFrame(
        {
            "MedicineName": ["Abilify "],
            "MedicineFor": ["For Bipolar Disorder"],
            "ReviewDate": ["4-Jan-12"],
            "UserName": ["someone"],
            "IntakeTime": ["4-Jan-12"],
            "Reviews": ["Felt sick."],
            "ReviewLength": [10],
            "Rating": [1],
            "NumberOfLikes": [13],
        }
    ).to_csv(f, index=False)
    df = load_drugrev(f)
    assert "UserName" not in df.columns
    row = df.iloc[0]
    assert row["drug"] == "Abilify"
    assert row["condition"] == "Bipolar Disorder"
    assert row["duration"] is None
    assert row["date"] == pd.Timestamp("2012-01-04")


def test_truncated_name_map_completes_only_mid_word_truncations() -> None:
    names = {
        "Bipolar Disorde",
        "Bipolar Disorder",
        "Breast Cance",
        "Breast Cancer",
        "Breast Cancer, Male",
        "Hypercalcemia",
        "Hypercalcemia of Malignancy",
        "Anxiety",
        "Anxiety and Stress",
        "Ulce",
        "Ulcer",
        "Ulceration",
    }
    m = truncated_name_map(names, names | {"Acnex"})
    assert m["Bipolar Disorde"] == "Bipolar Disorder"
    assert m["Breast Cance"] == "Breast Cancer"
    assert "Hypercalcemia" not in m
    assert "Anxiety" not in m
    assert "Ulce" not in m  # ambiguous: Ulcer vs Ulceration
    assert "Acne" not in m  # 'Acnex' is a drug name; Acne ends a word


def test_truncated_name_map_only_touches_candidates() -> None:
    names = {"Acne", "Acnex", "Panic Disorde", "Panic Disorder"}
    m = truncated_name_map({"Panic Disorde"}, names)
    assert m == {"Panic Disorde": "Panic Disorder"}
