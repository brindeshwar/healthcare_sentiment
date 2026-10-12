import json
import zipfile
from pathlib import Path

import pytest

from medsignal.data.download import check_manual, download_source, sha256_file


def _file_url(p: Path) -> str:
    return p.resolve().as_uri()


def test_download_writes_file_and_manifest(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    raw = tmp_path / "raw"
    spec = {"url": _file_url(src), "filename": "x.csv", "sha256": None}

    m = download_source("demo", spec, raw)

    assert (raw / "demo" / "x.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert m["sha256"] == sha256_file(src)
    assert json.loads((raw / "demo" / "manifest.json").read_text())["file"] == "x.csv"


def test_checksum_mismatch_raises(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    src.write_text("data", encoding="utf-8")
    spec = {"url": _file_url(src), "filename": "x.csv", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_source("demo", spec, tmp_path / "raw")


def test_existing_file_with_matching_checksum_is_not_refetched(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    src.write_text("data", encoding="utf-8")
    digest = sha256_file(src)
    raw = tmp_path / "raw"
    spec = {"url": _file_url(src), "filename": "x.csv", "sha256": digest}
    download_source("demo", spec, raw)

    src.unlink()  # a re-fetch would now fail
    assert download_source("demo", spec, raw)["sha256"] == digest


def test_zip_is_extracted(tmp_path: Path) -> None:
    z = tmp_path / "d.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("train.tsv", "id\tx\n")
    spec = {"url": _file_url(z), "filename": "d.zip", "sha256": None, "extract": True}
    download_source("demo", spec, tmp_path / "raw")
    assert (tmp_path / "raw" / "demo" / "train.tsv").exists()


def test_check_manual(tmp_path: Path) -> None:
    spec = {"files": ["a.csv"], "note": "get it"}
    assert not check_manual("m", spec, tmp_path)
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.csv").write_text("", encoding="utf-8")
    assert check_manual("m", spec, tmp_path)
