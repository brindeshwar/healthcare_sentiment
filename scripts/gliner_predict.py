"""Zero-shot GLiNER-BioMed ADE predictions on CADEC (Phase 3 baseline, ADR-0021).

Runs in its own environment because gliner pins transformers<5.17 (the project uses 5.19):
    tools\\gliner\\Scripts\\python.exe -I scripts\\gliner_predict.py --split dev --out data\\processed\\gliner\\cadec_dev.jsonl

Standard library + gliner only (no project imports). Writes one JSON line per post with every
predicted span above ``--min-score`` (label + score) so the label set and threshold can be chosen
on dev afterwards, without re-running the model. Long posts are processed in word windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

LABELS = ["adverse drug reaction", "side effect", "symptom"]


def windows(text: str, max_words: int, stride: int) -> list[tuple[int, str]]:
    """(char offset, chunk) word windows covering the text."""
    words = list(re.finditer(r"\S+", text))
    if len(words) <= max_words:
        return [(0, text)]
    out = []
    for i in range(0, len(words), stride):
        chunk = words[i : i + max_words]
        start, end = chunk[0].start(), chunk[-1].end()
        out.append((start, text[start:end]))
        if i + max_words >= len(words):
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cadec-root", default="data/raw/cadec/cadec")
    ap.add_argument("--split-file", default="data/annotations/external/cadec_split.csv")
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="Ihor/gliner-biomed-large-v1.0")
    ap.add_argument("--min-score", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from gliner import GLiNER

    with Path(args.split_file).open(encoding="utf-8") as f:
        ids = sorted(r["doc_id"] for r in csv.DictReader(f) if r["split"] == args.split)
    model = GLiNER.from_pretrained(args.model)
    if torch.cuda.is_available():
        model = model.to("cuda")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for n, doc_id in enumerate(ids, 1):
            text = (Path(args.cadec_root) / "text" / f"{doc_id}.txt").read_text(encoding="utf-8")
            spans = []
            seen = set()
            for offset, chunk in windows(text, max_words=250, stride=200):
                for e in model.predict_entities(chunk, LABELS, threshold=args.min_score):
                    key = (offset + e["start"], offset + e["end"], e["label"])
                    if key not in seen:
                        seen.add(key)
                        spans.append(
                            {
                                "start": key[0],
                                "end": key[1],
                                "label": e["label"],
                                "score": round(float(e["score"]), 4),
                            }
                        )
            fh.write(json.dumps({"doc_id": doc_id, "model": args.model, "spans": spans}) + "\n")
            if n % 100 == 0:
                print(f"{n}/{len(ids)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
