"""Benchmark local LLM teachers against an expert gold corpus (Phase 2, ADR-0018/0019).

Usage:
    uv run python -m medsignal.labeling.benchmark --config configs/teacher.yaml \
        [--model M] [--prompt v1|v2] [--limit N]

For each candidate model and prompt version: annotate every document of the chosen corpus split
(resumable JSONL under ``predictions_dir``), then score ADE spans against the gold with strict and
relaxed span F1 plus a document-bootstrap 95% CI. A document whose output never validates counts as
"no ADEs predicted" (failures cost recall) and is reported in ``failure_rate``. With
``llm.drop_unaligned_on_failure`` only non-verbatim spans are dropped (counted in ``dropped_spans``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from medsignal.evaluation.external import GoldDoc, load_cadec
from medsignal.evaluation.spans import Span, bootstrap_f1_ci, span_prf
from medsignal.labeling.llm import PROMPTS, OllamaAnnotator
from medsignal.utils import PROJECT_ROOT, git_sha, load_config, load_env

log = logging.getLogger(__name__)


def prompt_version(prompt: str = "v1") -> str:
    """``<name>-<hash>``: the hash changes whenever the prompt text or demonstrations change."""
    system, examples = PROMPTS[prompt]
    blob = system + json.dumps(examples, sort_keys=True)
    return f"{prompt}-{hashlib.sha1(blob.encode(), usedforsecurity=False).hexdigest()[:8]}"


def safe_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def load_split(cfg: dict[str, Any]) -> list[GoldDoc]:
    ext = load_config(cfg["external_config"])
    if cfg["corpus"] != "cadec":
        raise ValueError(f"unsupported corpus {cfg['corpus']!r}")
    docs = {d.doc_id: d for d in load_cadec(PROJECT_ROOT / ext["cadec"]["root"])}
    split = pd.read_csv(PROJECT_ROOT / ext["cadec"]["split_out"])
    ids = split.loc[split["split"] == cfg["split"], "doc_id"]
    return [docs[i] for i in sorted(ids)]


def _done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["doc_id"]: r for r in recs}


def predict(
    model: str,
    prompt: str,
    docs: list[GoldDoc],
    cfg: dict[str, Any],
    out: Path,
    limit: int | None,
) -> dict[str, dict[str, Any]]:
    """Annotate docs with one model + prompt; resumable."""
    done = _done(out)
    todo = [d for d in docs if d.doc_id not in done]
    if limit:
        todo = todo[:limit]
    log.info("%s [%s]: %s to annotate (%s done)", model, prompt, len(todo), len(done))
    llm = cfg["llm"]
    annotator = OllamaAnnotator(
        model=model,
        temperature=float(llm["temperature"]),
        num_ctx=int(llm["num_ctx"]),
        max_retries=int(llm["max_retries"]),
        seed=int(cfg["seed"]),
        drop_unaligned_on_failure=bool(llm.get("drop_unaligned_on_failure", False)),
        prompt=prompt,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pv = prompt_version(prompt)
    with httpx.Client() as client, out.open("a", encoding="utf-8", newline="\n") as fh:
        for i, doc in enumerate(todo, 1):
            t0 = time.perf_counter()
            rec: dict[str, Any] = {"doc_id": doc.doc_id, "model": model, "prompt_version": pv}
            try:
                ann, meta = annotator.annotate(client, doc.doc_id, doc.drug.title(), None, doc.text)
                rec.update(
                    status="ok",
                    attempts=meta["attempts"],
                    dropped_spans=meta.get("dropped_spans", []),
                    ades=[a.model_dump(mode="json") for a in ann.ades],
                    discontinued=ann.discontinued.value,
                    effectiveness=ann.effectiveness.value,
                )
            except RuntimeError as exc:
                rec.update(status="failed", error=str(exc)[:300], ades=[])
            rec["seconds"] = round(time.perf_counter() - t0, 2)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[doc.doc_id] = rec
            if i % 25 == 0 or i == len(todo):
                log.info("%s [%s]: %s/%s", model, prompt, i, len(todo))
    return done


def score(docs: list[GoldDoc], preds: dict[str, dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Strict/relaxed span P/R/F1 (+ bootstrap CI) overall and per drug, failures, timing."""
    scored = [d for d in docs if d.doc_id in preds]

    def pairs(subset: list[GoldDoc]) -> list[tuple[list[Span], list[Span]]]:
        return [
            (d.ades, [Span.contiguous(a["start"], a["end"]) for a in preds[d.doc_id]["ades"]])
            for d in subset
        ]

    secs = [preds[d.doc_id]["seconds"] for d in scored]
    by_drug: dict[str, list[GoldDoc]] = {}
    for d in scored:
        by_drug.setdefault("LIPITOR" if d.drug == "LIPITOR" else "diclofenac family", []).append(d)
    n = max(len(scored), 1)
    all_pairs = pairs(scored)
    return {
        "docs": len(scored),
        "strict": {**span_prf(all_pairs), **bootstrap_f1_ci(all_pairs, seed=seed)},
        "relaxed": {
            **span_prf(all_pairs, relaxed=True),
            **bootstrap_f1_ci(all_pairs, relaxed=True, seed=seed),
        },
        "relaxed_by_drug": {
            k: span_prf(pairs(v), relaxed=True) for k, v in sorted(by_drug.items())
        },
        "failure_rate": round(sum(preds[d.doc_id]["status"] != "ok" for d in scored) / n, 4),
        "first_try_rate": round(sum(preds[d.doc_id].get("attempts") == 1 for d in scored) / n, 4),
        "dropped_spans": sum(len(preds[d.doc_id].get("dropped_spans", [])) for d in scored),
        "seconds_median": round(statistics.median(secs), 2) if secs else None,
        "seconds_total": round(sum(secs), 1),
    }


def _f1(m: dict[str, Any]) -> str:
    ci = f" [{m['f1_lo']:.3f}, {m['f1_hi']:.3f}]" if "f1_lo" in m else ""
    return f"{m['precision']:.3f} / {m['recall']:.3f} / **{m['f1']:.3f}**{ci}"


def write_readme(results: list[dict[str, Any]], path: Path, corpus: str, split: str) -> None:
    lines = [
        f"# LLM teacher benchmark: {corpus} {split} (generated)",
        "",
        "> Generated by `uv run python -m medsignal.labeling.benchmark`. ADE span metrics vs expert gold",
        "> (ADR-0018). Strict = exact character match; relaxed = any overlap; one-to-one matching.",
        "> F1 95% CI: percentile bootstrap over documents (1,000 resamples).",
        "",
        "| Model | Prompt | Docs | Strict P / R / F1 [CI] | Relaxed P / R / F1 [CI] | Failures | s/doc |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: -r["metrics"]["relaxed"]["f1"]):
        m = r["metrics"]
        lines.append(
            f"| `{r['model']}` | `{r['prompt_version']}` | {m['docs']} | {_f1(m['strict'])} | "
            f"{_f1(m['relaxed'])} | {m['failure_rate']:.1%} | {m['seconds_median']} |"
        )
    shas = sorted({r["git_sha"] for r in results})
    lines += ["", f"Commits: {', '.join(f'`{s}`' for s in shas)}.", ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/teacher.yaml")
    parser.add_argument("--model", help="run only this candidate")
    parser.add_argument("--prompt", help="prompt version (default: llm.prompt in config)")
    parser.add_argument("--limit", type=int, help="annotate at most N new docs per model")
    parser.add_argument(
        "--split", help="corpus split (default: split in config); test = report once"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_env()

    cfg = load_config(args.config)
    if args.split:
        cfg["split"] = args.split
    prompt = args.prompt or cfg["llm"]["prompt"]
    docs = load_split(cfg)
    models = [args.model] if args.model else list(cfg["candidates"])
    report_dir = PROJECT_ROOT / cfg["report_dir"] / f"{cfg['corpus']}_{cfg['split']}"
    report_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        stem = f"{cfg['corpus']}_{cfg['split']}_{prompt}"
        out = PROJECT_ROOT / cfg["predictions_dir"] / safe_name(model) / f"{stem}.jsonl"
        preds = predict(model, prompt, docs, cfg, out, args.limit)
        result = {
            "model": model,
            "corpus": cfg["corpus"],
            "split": cfg["split"],
            "prompt_version": prompt_version(prompt),
            "drop_unaligned_on_failure": bool(cfg["llm"].get("drop_unaligned_on_failure", False)),
            "git_sha": git_sha(),
            "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "metrics": score(docs, preds, seed=int(cfg["seed"])),
        }
        (report_dir / f"{safe_name(model)}__{prompt}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        log.info(
            "%s [%s]: relaxed F1 %.3f strict F1 %.3f",
            model,
            prompt,
            result["metrics"]["relaxed"]["f1"],
            result["metrics"]["strict"]["f1"],
        )

    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(report_dir.glob("*.json"))]
    write_readme(results, report_dir / "README.md", cfg["corpus"], cfg["split"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
