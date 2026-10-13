from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medsignal.data.eda import summarize, write_readme
from medsignal.utils import load_env


def _toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "review_id": ["a", "b", "c", "d"],
            "source": ["uci", "both", "drugrev", "drugrev"],
            "drug": ["X", "X", "Y", "Z"],
            "condition": ["Pain", None, "Pain", "Acne"],
            "rating": [1, 10, 10, 5],
            "date": pd.to_datetime(["2012-01-01", "2016-05-05", "2020-02-02", "2021-03-03"]),
            "duration": [None, "Taken for 1 to 6 months", None, "Taken for less than 1 month"],
            "possibly_truncated": [False, False, True, True],
        }
    )
    splits = pd.DataFrame(
        {
            "review_id": ["a", "b", "c", "d"],
            "drug_group": ["x", "x", "y", "z"],
            "split": ["train", "val", "test", "drug_holdout"],
        }
    )
    return df, splits


def test_summarize_and_readme(tmp_path: Path) -> None:
    df, splits = _toy()
    s = summarize(df, splits, np.array([10, 300, 50, 20]), max_len=256)
    assert s["reviews"] == 4
    assert s["tokens"]["share_within_256"] == 0.75
    assert s["top_conditions"] == {"Pain": 2, "Acne": 1}
    assert s["duration_known_share"] == 0.5
    assert s["split_counts"] == {"train": 1, "val": 1, "test": 1, "drug_holdout": 1}
    assert s["reviews_per_year_by_source"]["2020"] == {"drugrev": 1}
    s["tokens"]["tokenizer"] = "tok"
    s["meta"] = {"git_sha": "abc", "built_utc": "now"}
    write_readme(s, tmp_path / "README.md")
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "75.0% of reviews fit" in text
    assert "| 1 | 1 | 1 | 1 |" in text


def test_load_env_resolves_relative_hf_home(monkeypatch: pytest.MonkeyPatch) -> None:
    from medsignal import utils

    monkeypatch.setenv("HF_HOME", ".cache/huggingface")
    load_env()
    assert Path(utils.os.environ["HF_HOME"]).is_absolute()
    assert Path(utils.os.environ["HF_HOME"]).parts[-2:] == (".cache", "huggingface")
