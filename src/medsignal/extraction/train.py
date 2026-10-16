"""Train the encoder student on silver ADE spans and evaluate it on CADEC (Phase 3, ADR-0021).

Usage:
    uv run --extra gpu python -m medsignal.extraction.train --config configs/student.yaml [--seed 13]
    [--max-train N]   # smoke run

Per seed: token classification (O / B-ADE / I-ADE) fine-tuning of BioClinical ModernBERT on the
teacher's silver labels (90%; 10% held out by drug group selects the checkpoint by token F1), then
span prediction on CADEC dev/test scored with the same strict/relaxed span F1 as the teacher.
Finished seeds are skipped; an interrupted seed resumes from its last checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medsignal.evaluation.external import GoldDoc, load_cadec
from medsignal.evaluation.spans import Span, bootstrap_f1_ci, span_prf
from medsignal.extraction.bio import IGNORE, LABELS, bio_to_spans, spans_to_bio
from medsignal.utils import PROJECT_ROOT, git_sha, load_config, load_env, seed_everything

log = logging.getLogger(__name__)


def load_silver(cfg: dict[str, Any]) -> pd.DataFrame:
    """review_id, text, spans [(start, end)], drug_group for successfully labelled reviews."""
    silver_cfg = load_config(cfg["silver_config"])
    data_cfg = load_config(cfg["data_config"])
    recs = [
        json.loads(line)
        for line in (PROJECT_ROOT / silver_cfg["out"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = [
        {
            "review_id": r["review_id"],
            "spans": [(a["start"], a["end"]) for a in r["ades"]],
            "discontinued": r["discontinued"],
            "effectiveness": r["effectiveness"],
            "stop_reasons": r["stop_reasons"],
        }
        for r in recs
        if r["status"] == "ok"
    ]
    df = pd.DataFrame(rows)
    reviews = pd.read_parquet(
        PROJECT_ROOT / data_cfg["paths"]["interim"] / "reviews.parquet",
        columns=["review_id", "text"],
    )
    sample = pd.read_csv(
        PROJECT_ROOT / silver_cfg["sample"]["out"], usecols=["review_id", "drug_group"]
    )
    return df.merge(reviews, on="review_id").merge(sample, on="review_id")


def split_by_group(
    df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole drug groups so validation measures generalisation to unseen drugs."""
    groups = np.array(sorted(df["drug_group"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    target = val_fraction * len(df)
    sizes = df["drug_group"].value_counts()
    val_groups: set[str] = set()
    n = 0
    for g in groups:
        if n >= target:
            break
        val_groups.add(str(g))
        n += int(sizes[g])
    is_val = df["drug_group"].isin(val_groups)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


class _TokenDataset:
    def __init__(self, items: list[dict[str, list[int]]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.items[i]


def encode(tok: Any, df: pd.DataFrame, max_length: int) -> _TokenDataset:
    items = []
    for text, spans in zip(df["text"], df["spans"], strict=True):
        enc = tok(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        offs = [tuple(o) for o in enc["offset_mapping"]]
        items.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": spans_to_bio(offs, spans, text),
            }
        )
    return _TokenDataset(items)


def token_f1(pred: Any) -> dict[str, float]:
    """Micro F1 over B/I tokens (ignoring O and padding): cheap checkpoint-selection signal."""
    labels = np.asarray(pred.label_ids)
    preds = np.argmax(np.asarray(pred.predictions), axis=-1)
    mask = labels != IGNORE
    tp = int(((preds == labels) & (labels > 0) & mask).sum())
    n_pred = int(((preds > 0) & mask).sum())
    n_gold = int(((labels > 0) & mask).sum())
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"token_f1": 2 * p * r / (p + r) if p + r else 0.0, "token_p": p, "token_r": r}


def predict_spans(
    model: Any, tok: Any, texts: list[str], max_length: int, batch_size: int
) -> tuple[list[list[tuple[int, int]]], float]:
    """Predict ADE character spans; returns spans per text and texts/second."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    out: list[list[tuple[int, int]]] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping").tolist()
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            tags = logits.argmax(-1).cpu().tolist()
            mask = enc["attention_mask"].tolist()
            for text, offs, tg, m in zip(batch, offsets, tags, mask, strict=True):
                keep = [(tuple(o), t) for o, t, mm in zip(offs, tg, m, strict=True) if mm]
                out.append(bio_to_spans([k[0] for k in keep], [k[1] for k in keep], text))
    return out, len(texts) / (time.perf_counter() - t0)


def evaluate_cadec(
    model: Any, tok: Any, docs: list[GoldDoc], cfg: dict[str, Any], seed: int
) -> dict[str, Any]:
    e = cfg["eval"]
    spans, speed = predict_spans(
        model, tok, [d.text for d in docs], int(e["max_length"]), int(e["batch_size"])
    )
    pairs = [
        (d.ades, [Span.contiguous(a, b) for a, b in s]) for d, s in zip(docs, spans, strict=True)
    ]
    by_drug: dict[str, list[tuple[list[Span], list[Span]]]] = {}
    for d, p in zip(docs, pairs, strict=True):
        by_drug.setdefault("LIPITOR" if d.drug == "LIPITOR" else "diclofenac family", []).append(p)
    return {
        "docs": len(docs),
        "strict": {**span_prf(pairs), **bootstrap_f1_ci(pairs, seed=seed)},
        "relaxed": {
            **span_prf(pairs, relaxed=True),
            **bootstrap_f1_ci(pairs, relaxed=True, seed=seed),
        },
        "relaxed_by_drug": {k: span_prf(v, relaxed=True) for k, v in sorted(by_drug.items())},
        "docs_per_second": round(speed, 1),
    }


def cadec_docs(cfg: dict[str, Any], split: str) -> list[GoldDoc]:
    teacher_cfg = load_config(cfg["teacher_config"])
    ext = load_config(teacher_cfg["external_config"])
    docs = {d.doc_id: d for d in load_cadec(PROJECT_ROOT / ext["cadec"]["root"])}
    ids = pd.read_csv(PROJECT_ROOT / ext["cadec"]["split_out"])
    return [docs[i] for i in sorted(ids.loc[ids["split"] == split, "doc_id"])]


def run_seed(cfg: dict[str, Any], seed: int, max_train: int | None) -> dict[str, Any]:
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    silver = load_silver(cfg)
    train_df, val_df = split_by_group(silver, float(cfg["silver_val_fraction"]), seed)
    if max_train:
        train_df = train_df.sample(n=min(max_train, len(train_df)), random_state=seed)
        val_df = val_df.head(max(max_train // 5, 50))
    t = cfg["train"]
    train_ds = encode(tok, train_df, int(t["max_length"]))
    val_ds = encode(tok, val_df, int(t["max_length"]))

    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model"],
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={lab: i for i, lab in enumerate(LABELS)},
    )
    tag = f"seed{seed}" + (f"_smoke{max_train}" if max_train else "")
    out_dir = PROJECT_ROOT / cfg["output_dir"] / tag
    args = TrainingArguments(
        output_dir=str(out_dir),
        seed=seed,
        num_train_epochs=float(t["epochs"]),
        learning_rate=float(t["lr"]),
        per_device_train_batch_size=int(t["batch_size"]),
        per_device_eval_batch_size=int(t["batch_size"]) * 2,
        gradient_accumulation_steps=int(t["grad_accum"]),
        warmup_steps=float(t["warmup_ratio"]),  # transformers v5: float in [0, 1) = ratio
        weight_decay=float(t["weight_decay"]),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="token_f1",
        greater_is_better=True,
        logging_steps=50,
        train_sampling_strategy="group_by_length",
        dataloader_num_workers=0,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,  # type: ignore[arg-type,unused-ignore]
        eval_dataset=val_ds,  # type: ignore[arg-type,unused-ignore]
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=token_f1,
    )
    last = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None  # type: ignore[no-untyped-call,unused-ignore]
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=last)
    result: dict[str, Any] = {
        "seed": seed,
        "train_reviews": len(train_df),
        "silver_val_reviews": len(val_df),
        "train_seconds": round(time.perf_counter() - t0, 1),
        "silver_val": {
            k: round(float(v), 4) for k, v in trainer.evaluate().items() if "token" in k
        },
        "best_checkpoint": Path(trainer.state.best_model_checkpoint or "").name,
    }
    trainer.save_model(str(out_dir / "best"))
    tok.save_pretrained(str(out_dir / "best"))
    for split in cfg["eval"]["splits"]:
        result[f"cadec_{split}"] = evaluate_cadec(
            trainer.model, tok, cadec_docs(cfg, split), cfg, seed
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()
    cfg = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else [int(s) for s in cfg["seeds"]]
    report_dir = PROJECT_ROOT / cfg["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        name = f"seed{seed}" + (f"_smoke{args.max_train}" if args.max_train else "")
        path = report_dir / f"{name}.json"
        if path.exists():
            log.info("%s done, skipping", name)
            continue
        result = run_seed(cfg, seed, args.max_train)
        result["meta"] = {
            "model": cfg["model"],
            "git_sha": git_sha(),
            "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        log.info(
            "%s: CADEC dev relaxed F1 %.3f, %.0f docs/s",
            name,
            result["cadec_dev"]["relaxed"]["f1"],
            result["cadec_dev"]["docs_per_second"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
