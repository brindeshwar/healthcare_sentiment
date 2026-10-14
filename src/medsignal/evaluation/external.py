"""External expert gold sets (ADR-0018): loaders and fixed dev/test splits.

CADEC (CSIRO, 1,250 AskaPatient posts): ``ADR`` annotations are our ADEs. ``Drug``, ``Disease``,
``Symptom`` and ``Finding`` are not ADEs (symptoms/findings of the condition, or other entities).
Discontinuous gold spans are kept as multi-part ``Span``s.

Usage:
    uv run python -m medsignal.evaluation.external split --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medsignal.evaluation.spans import Span
from medsignal.utils import PROJECT_ROOT, load_config

log = logging.getLogger(__name__)


@dataclass
class GoldDoc:
    doc_id: str
    drug: str
    text: str
    ades: list[Span] = field(default_factory=list)
    meddra: list[list[str]] = field(
        default_factory=list
    )  # MedDRA codes per ADE ([] = none/CONCEPT_LESS)


def _parse_offsets(raw: str) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(map(int, p.split())) for p in raw.split(";"))  # type: ignore[misc]


def load_cadec(root: Path) -> list[GoldDoc]:
    """Load CADEC v2 (``text/``, ``original/``, ``meddra/``)."""
    docs: list[GoldDoc] = []
    for ann_path in sorted((root / "original").glob("*.ann")):
        doc_id = ann_path.stem
        text = (root / "text" / f"{doc_id}.txt").read_text(encoding="utf-8")
        meddra_by_span = _load_meddra(root / "meddra" / f"{doc_id}.ann")
        doc = GoldDoc(doc_id=doc_id, drug=doc_id.split(".")[0], text=text)
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("T"):
                continue
            _, middle, _span_text = line.split("\t", 2)
            etype, offsets = middle.split(" ", 1)
            if etype != "ADR":
                continue
            parts = _parse_offsets(offsets)
            doc.ades.append(Span(parts))
            doc.meddra.append(meddra_by_span.get(parts, []))
        docs.append(doc)
    return docs


_OFFSETS_AT_END = re.compile(r"(\d+ \d+(?:;\d+ \d+)*)$")
_MEDDRA_CODE = re.compile(r"\b1\d{7}\b")


def _load_meddra(path: Path) -> dict[tuple[tuple[int, int], ...], list[str]]:
    """MedDRA codes keyed by span offsets.

    Real lines include ``10028294 165 172``, ``10025483 + 10016379 285 313`` (several codes),
    ``10013781/10003077 ...`` (alternatives) and ``CONCEPT_LESS 12 20`` (no code): offsets are read
    from the end, every 8-digit MedDRA code from the rest.
    """
    out: dict[tuple[tuple[int, int], ...], list[str]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        m = _OFFSETS_AT_END.search(cols[1].strip())
        if not m:
            continue
        codes = _MEDDRA_CODE.findall(cols[1][: m.start()])
        out[_parse_offsets(m.group(1))] = codes
    return out


def split_docs(docs: list[GoldDoc], dev_fraction: float, seed: int) -> pd.DataFrame:
    """Fixed dev/test split stratified by drug (dev is the only part used for tuning)."""
    rng = np.random.default_rng(seed)
    rows = []
    by_drug: dict[str, list[str]] = {}
    for d in docs:
        by_drug.setdefault(d.drug, []).append(d.doc_id)
    for drug, ids in sorted(by_drug.items()):
        ids = list(rng.permutation(sorted(ids)))
        n_dev = round(len(ids) * dev_fraction)
        rows += [
            {"doc_id": i, "drug": drug, "split": "dev" if k < n_dev else "test"}
            for k, i in enumerate(ids)
        ]
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--config", default="configs/external.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg: dict[str, Any] = load_config(args.config)

    c = cfg["cadec"]
    out = PROJECT_ROOT / c["split_out"]
    if out.exists():
        raise SystemExit(f"{out} exists; external splits are fixed once drawn")
    docs = load_cadec(PROJECT_ROOT / c["root"])
    split = split_docs(docs, float(c["dev_fraction"]), int(cfg["seed"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(out, index=False, lineterminator="\n")
    ades = {d.doc_id: len(d.ades) for d in docs}
    log.info(
        "cadec: %s",
        split.assign(ades=split["doc_id"].map(ades))
        .groupby("split")["ades"]
        .agg(["count", "sum"])
        .to_dict(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
