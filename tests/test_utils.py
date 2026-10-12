import random
from pathlib import Path

import pytest

import medsignal
from medsignal.utils import git_sha, load_config, seed_everything


def test_version() -> None:
    assert medsignal.__version__


def test_load_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("seed: 7\npaths:\n  raw: data/raw\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg["seed"] == 7
    assert cfg["paths"]["raw"] == "data/raw"


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(cfg_file)


def test_seed_everything_is_deterministic() -> None:
    seed_everything(123)
    a = random.random()
    seed_everything(123)
    assert random.random() == a


def test_git_sha_returns_string() -> None:
    assert isinstance(git_sha(), str)
