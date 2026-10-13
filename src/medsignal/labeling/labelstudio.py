"""Bridge between our schema and Label Studio (local annotation tool, ADR-0017).

Usage:
    uv run python -m medsignal.labeling.labelstudio export --config configs/labeling.yaml
    uv run python -m medsignal.labeling.labelstudio import --export <label-studio-export.json> [--split dev]

``export`` writes the labelling interface (``configs/labelstudio.xml``, generated from the schema enums)
and one task file per gold split under ``data/interim/labelstudio/`` (git-ignored: contains review
text). Dev tasks carry the LLM pre-annotations as Label Studio *predictions*; test tasks carry none.

``import`` converts a Label Studio JSON export into validated ``ReviewAnnotation`` lines
(``data/annotations/gold_<split>.jsonl``: labels + provenance, no review text).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd

from medsignal.labeling.schema import (
    SCHEMA_VERSION,
    ADEMention,
    Discontinued,
    Effectiveness,
    ReviewAnnotation,
    Severity,
    StopReason,
)
from medsignal.utils import PROJECT_ROOT, load_config

log = logging.getLogger(__name__)

FLAG_CHOICES = ("withdrawal", "beneficial")


def _choices(name: str, values: list[str], **attrs: str) -> str:
    extra = " ".join(f'{k}="{escape(v)}"' for k, v in attrs.items())
    opts = "".join(f'<Choice value="{escape(v)}"/>' for v in values)
    return f'  <Choices name="{name}" toName="review" {extra}>{opts}</Choices>\n'


def labeling_config() -> str:
    """Label Studio XML generated from the schema enums (stays in sync with schema.py)."""
    stopping = f"{Discontinued.YES.value},{Discontinued.PLANNED.value}"
    return (
        "<View>\n"
        '  <Header value="$drug | $condition | $rating stars | #$order"/>\n'
        '  <Labels name="ade" toName="review"><Label value="ADE" background="#e34948"/></Labels>\n'
        '  <Text name="review" value="$text" granularity="word"/>\n'
        '  <Header value="Selected ADE: severity and flags" size="6"/>\n'
        + _choices("severity", [s.value for s in Severity], perRegion="true", choice="single-radio")
        + _choices("flags", list(FLAG_CHOICES), perRegion="true", choice="multiple")
        + '  <Header value="Discontinued?" size="6"/>\n'
        + _choices(
            "discontinued", [d.value for d in Discontinued], choice="single-radio", required="true"
        )
        + _choices(
            "stop_reasons",
            [r.value for r in StopReason],
            choice="multiple",
            visibleWhen="choice-selected",
            whenTagName="discontinued",
            whenChoiceValue=stopping,
        )
        + '  <Header value="Effectiveness" size="6"/>\n'
        + _choices(
            "effectiveness",
            [e.value for e in Effectiveness],
            choice="single-radio",
            required="true",
        )
        + '  <TextArea name="notes" toName="review" placeholder="notes (optional)" maxSubmissions="1"/>\n'
        "</View>\n"
    )


def to_prediction(ann: ReviewAnnotation, model: str) -> dict[str, Any]:
    """ReviewAnnotation (with offsets) -> Label Studio prediction."""
    result: list[dict[str, Any]] = []
    for i, ade in enumerate(ann.ades):
        rid = f"ade{i}"
        span = {"start": ade.start, "end": ade.end, "text": ade.text}
        result.append(
            {
                "id": rid,
                "from_name": "ade",
                "to_name": "review",
                "type": "labels",
                "value": {**span, "labels": ["ADE"]},
            }
        )
        result.append(
            {
                "id": rid,
                "from_name": "severity",
                "to_name": "review",
                "type": "choices",
                "value": {**span, "choices": [ade.severity.value]},
            }
        )
        flags = [f for f in FLAG_CHOICES if getattr(ade, f)]
        if flags:
            result.append(
                {
                    "id": rid,
                    "from_name": "flags",
                    "to_name": "review",
                    "type": "choices",
                    "value": {**span, "choices": flags},
                }
            )
    for name, values in (
        ("discontinued", [ann.discontinued.value]),
        ("stop_reasons", [r.value for r in ann.stop_reasons]),
        ("effectiveness", [ann.effectiveness.value]),
    ):
        if values:
            result.append(
                {
                    "from_name": name,
                    "to_name": "review",
                    "type": "choices",
                    "value": {"choices": values},
                }
            )
    return {"model_version": model, "result": result}


def from_result(review_id: str, result: list[dict[str, Any]]) -> ReviewAnnotation:
    """Label Studio annotation result -> validated ReviewAnnotation."""
    spans: dict[str, dict[str, Any]] = {}
    doc: dict[str, list[str]] = {}
    notes: str | None = None
    for r in result:
        value = r.get("value", {})
        if r["from_name"] == "notes":
            notes = " ".join(value.get("text", [])) or None
        elif "start" in value and r.get("id"):
            span = spans.setdefault(
                r["id"], {"start": value["start"], "end": value["end"], "text": value["text"]}
            )
            if r["from_name"] == "severity":
                span["severity"] = value["choices"][0]
            elif r["from_name"] == "flags":
                for f in value["choices"]:
                    span[f] = True
        else:
            doc[r["from_name"]] = value.get("choices", [])
    ades = [ADEMention(**s) for s in sorted(spans.values(), key=lambda s: s["start"])]
    return ReviewAnnotation(
        review_id=review_id,
        ades=ades,
        discontinued=Discontinued((doc.get("discontinued") or [Discontinued.NOT_STATED.value])[0]),
        stop_reasons=[StopReason(r) for r in doc.get("stop_reasons", [])],
        effectiveness=Effectiveness(
            (doc.get("effectiveness") or [Effectiveness.NOT_STATED.value])[0]
        ),
        notes=notes,
    )


def export_tasks(cfg: dict[str, Any]) -> dict[str, Path]:
    data_cfg = load_config(cfg["data_config"])
    gold = pd.read_csv(PROJECT_ROOT / cfg["gold"]["out"])
    reviews = pd.read_parquet(PROJECT_ROOT / data_cfg["paths"]["interim"] / "reviews.parquet")
    gold = gold.merge(reviews[["review_id", "text"]], on="review_id").sort_values("order")

    pre_path = PROJECT_ROOT / cfg["preannotate"]["out"]
    pre: dict[str, ReviewAnnotation] = {}
    if pre_path.exists():
        for line in pre_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            rec.pop("provenance", None)
            pre[rec["review_id"]] = ReviewAnnotation(**rec)
    model = cfg["preannotate"]["model"]

    out_dir = PROJECT_ROOT / cfg["labelstudio"]["tasks_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, g in gold.groupby("gold_split"):
        tasks = []
        for row in g.itertuples(index=False):
            task: dict[str, Any] = {
                "data": {
                    "review_id": row.review_id,
                    "order": int(row.order),
                    "drug": row.drug,
                    "condition": row.condition if isinstance(row.condition, str) else "not given",
                    "rating": int(row.rating),
                    "text": row.text,
                }
            }
            if split in cfg["preannotate"]["splits"] and row.review_id in pre:
                task["predictions"] = [to_prediction(pre[row.review_id], f"llm:{model}")]
            tasks.append(task)
        path = out_dir / f"tasks_{split}.json"
        path.write_text(json.dumps(tasks, indent=1), encoding="utf-8", newline="\n")
        paths[str(split)] = path
        with_pred = sum("predictions" in t for t in tasks)
        log.info("%s: %s tasks (%s with pre-annotations) -> %s", split, len(tasks), with_pred, path)

    xml = PROJECT_ROOT / cfg["labelstudio"]["config_out"]
    xml.write_text(labeling_config(), encoding="utf-8", newline="\n")
    log.info("labelling interface -> %s", xml)
    return paths


def import_export(export_path: Path, split: str, cfg: dict[str, Any]) -> Path:
    """Convert a Label Studio JSON export to gold_<split>.jsonl (latest annotation per task)."""
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    out = PROJECT_ROOT / "data" / "annotations" / f"gold_{split}.jsonl"
    lines = []
    for task in tasks:
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        latest = max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        ann = from_result(task["data"]["review_id"], latest["result"])
        record = {
            **ann.model_dump(mode="json"),
            "provenance": {
                "annotator": "human:owner",
                "schema_version": SCHEMA_VERSION,
                "pre_annotated_by": (task.get("predictions") or [{}])[0].get("model_version"),
                "lead_time_s": latest.get("lead_time"),
                "updated_at": latest.get("updated_at"),
            },
        }
        lines.append(json.dumps(record))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    log.info("%s annotated reviews -> %s", len(lines), out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--config", default="configs/labeling.yaml")
    i = sub.add_parser("import")
    i.add_argument("--config", default="configs/labeling.yaml")
    i.add_argument("--export", required=True, type=Path)
    i.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.cmd == "export":
        export_tasks(cfg)
    else:
        import_export(args.export, args.split, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
