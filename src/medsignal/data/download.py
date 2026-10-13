"""Fetch raw datasets into data/raw/<source>/ with checksum verification.

Usage:
    uv run python -m medsignal.data.download --config configs/data.yaml [--only drugrev druglib]

Idempotent: a file that already exists with the expected checksum is not downloaded again.
Each source directory gets a ``manifest.json`` recording URL, size, sha256 and download time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medsignal.utils import PROJECT_ROOT, load_config

log = logging.getLogger(__name__)

CHUNK = 1 << 20
USER_AGENT = "medsignal-data-download/0.1 (+https://github.com/brindeshwar/medsignal)"


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` atomically (via a .part file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, CHUNK)
    tmp.replace(dest)


def download_source(name: str, spec: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    """Download (or verify) one URL source; return its manifest entry."""
    target_dir = raw_dir / name
    dest = target_dir / spec["filename"]
    expected = spec.get("sha256")

    if dest.exists() and expected and sha256_file(dest) == expected:
        log.info("%s: already present, checksum OK", name)
    else:
        log.info("%s: downloading %s", name, spec["url"])
        fetch(spec["url"], dest)

    digest = sha256_file(dest)
    if expected and digest != expected:
        raise ValueError(f"{name}: checksum mismatch (expected {expected}, got {digest})")
    if not expected:
        log.warning("%s: no pinned sha256; got %s. Pin it in the config.", name, digest)

    if spec.get("extract") and zipfile.is_zipfile(dest):
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(target_dir)
        log.info("%s: extracted %s", name, dest.name)

    manifest = {
        "source": name,
        "url": spec["url"],
        "file": dest.name,
        "bytes": dest.stat().st_size,
        "sha256": digest,
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def check_manual(name: str, spec: dict[str, Any], raw_dir: Path) -> bool:
    """Return True if all files of a manual source are present; log instructions otherwise."""
    missing = [f for f in spec.get("files", []) if not (raw_dir / name / f).exists()]
    if missing:
        log.warning("%s: missing %s in %s. %s", name, missing, raw_dir / name, spec.get("note", ""))
        return False
    log.info("%s: manual files present", name)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--only", nargs="*", help="subset of source names")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw"]
    ok = True
    for name, spec in cfg["sources"].items():
        if args.only and name not in args.only:
            continue
        if spec["kind"] == "url":
            download_source(name, spec, raw_dir)
        elif spec["kind"] == "manual":
            ok &= check_manual(name, spec, raw_dir)
        else:
            raise ValueError(f"{name}: unknown kind {spec['kind']!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
