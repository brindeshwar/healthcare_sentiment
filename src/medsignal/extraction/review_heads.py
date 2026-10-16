"""Review-level student: discontinued, effectiveness and stop reasons in one classifier (ADR-0022).

Usage:
    uv run --extra gpu python -m medsignal.extraction.review_heads --config configs/student.yaml [--max-train N]

One BioClinical ModernBERT sequence classifier with 17 sigmoid outputs (multi-label BCE):
4 discontinued + 4 effectiveness + 9 stop reasons. Decoding takes the argmax inside each
single-choice group and stop reasons above 0.5 only when the review is predicted as stopped/planned
(at least one reason, the most likely, if none passes 0.5). No expert gold exists for these fields
(ADR-0018), so the reported metric is **agreement with the teacher** on silver reviews of held-out
drug groups.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from medsignal.evaluation.metrics import mean_std
from medsignal.extraction.train import load_silver, split_by_group
from medsignal.labeling.schema import Discontinued, Effectiveness, StopReason
from medsignal.utils import PROJECT_ROOT, git_sha, load_config, load_env, seed_everything

log = logging.getLogger(__name__)

DISC = [d.value for d in Discontinued]
EFF = [e.value for e in Effectiveness]
REASONS = [r.value for r in StopReason]
OUTPUTS = [f"disc:{v}" for v in DISC] + [f"eff:{v}" for v in EFF] + [f"reason:{v}" for v in REASONS]
_D = slice(0, len(DISC))
_E = slice(len(DISC), len(DISC) + len(EFF))
_R = slice(len(DISC) + len(EFF), len(OUTPUTS))
STOPPING = {Discontinued.YES.value, Discontinued.PLANNED.value}


def encode_targets(discontinued: str, effectiveness: str, reasons: list[str]) -> list[float]:
    y = [0.0] * len(OUTPUTS)
    y[DISC.index(discontinued)] = 1.0
    y[len(DISC) + EFF.index(effectiveness)] = 1.0
    for r in reasons:
        y[len(DISC) + len(EFF) + REASONS.index(r)] = 1.0
    return y


def decode(probs: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Probabilities (17,) -> consistent review-level labels."""
    disc = DISC[int(np.argmax(probs[_D]))]
    eff = EFF[int(np.argmax(probs[_E]))]
    reasons: list[str] = []
    if disc in STOPPING:
        rp = probs[_R]
        reasons = [REASONS[i] for i in np.flatnonzero(rp >= threshold)]
        if not reasons:
            reasons = [REASONS[int(np.argmax(rp))]]
        if StopReason.NOT_STATED.value in reasons and len(reasons) > 1:
            reasons.remove(StopReason.NOT_STATED.value)
    return {"discontinued": disc, "effectiveness": eff, "stop_reasons": reasons}


def agreement(gold: pd.DataFrame, pred: list[dict[str, Any]]) -> dict[str, Any]:
    """Agreement with teacher labels: accuracy + macro-F1 per field, micro-F1 for reasons."""
    from sklearn.metrics import accuracy_score, f1_score

    out: dict[str, Any] = {"reviews": len(gold)}
    for field, labels in (("discontinued", DISC), ("effectiveness", EFF)):
        y, p = gold[field].tolist(), [x[field] for x in pred]
        out[field] = {
            "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(
                float(f1_score(y, p, labels=labels, average="macro", zero_division=0)), 4
            ),
        }
    stop_true = [d in STOPPING for d in gold["discontinued"]]
    stop_pred = [x["discontinued"] in STOPPING for x in pred]
    out["stopped_or_planned_f1"] = round(float(f1_score(stop_true, stop_pred, zero_division=0)), 4)
    yt = np.array([[r in rs for r in REASONS] for rs in gold["stop_reasons"]], dtype=int)
    yp = np.array([[r in x["stop_reasons"] for r in REASONS] for x in pred], dtype=int)
    out["stop_reasons_micro_f1"] = round(
        float(f1_score(yt, yp, average="micro", zero_division=0)), 4
    )
    return out


class _Dataset:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.items[i]


def predict_probs(
    model: Any, tok: Any, texts: list[str], max_length: int, batch_size: int
) -> tuple[np.ndarray, float]:
    import torch

    model.eval()
    device = next(model.parameters()).device
    chunks = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(
                texts[i : i + batch_size],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(chunks), len(texts) / (time.perf_counter() - t0)


def run_seed(cfg: dict[str, Any], seed: int, max_train: int | None) -> dict[str, Any]:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    silver = load_silver(cfg)
    train_df, val_df = split_by_group(silver, float(cfg["silver_val_fraction"]), seed)
    if max_train:
        train_df = train_df.sample(n=min(max_train, len(train_df)), random_state=seed)
    t = cfg["train"]

    def ds(df: pd.DataFrame) -> _Dataset:
        enc = tok(df["text"].tolist(), truncation=True, max_length=int(t["max_length"]))
        ys = [
            encode_targets(d, e, r)
            for d, e, r in zip(
                df["discontinued"], df["effectiveness"], df["stop_reasons"], strict=True
            )
        ]
        return _Dataset(
            [
                {"input_ids": a, "attention_mask": m, "labels": y}
                for a, m, y in zip(enc["input_ids"], enc["attention_mask"], ys, strict=True)
            ]
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"],
        num_labels=len(OUTPUTS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(OUTPUTS)),
        label2id={o: i for i, o in enumerate(OUTPUTS)},
    )
    tag = f"reviews_seed{seed}" + (f"_smoke{max_train}" if max_train else "")
    out_dir = PROJECT_ROOT / cfg["output_dir"] / tag
    args = TrainingArguments(
        output_dir=str(out_dir),
        seed=seed,
        num_train_epochs=float(t["epochs"]),
        learning_rate=float(t["lr"]),
        per_device_train_batch_size=int(t["batch_size"]),
        warmup_steps=float(t["warmup_ratio"]),
        weight_decay=float(t["weight_decay"]),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        save_strategy="no",
        logging_steps=50,
        train_sampling_strategy="group_by_length",
        dataloader_num_workers=0,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds(train_df),  # type: ignore[arg-type,unused-ignore]
        data_collator=DataCollatorWithPadding(tok),
    )
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    trainer.save_model(str(out_dir / "best"))
    tok.save_pretrained(str(out_dir / "best"))

    probs, speed = predict_probs(
        trainer.model, tok, val_df["text"].tolist(), int(t["max_length"]), 64
    )
    preds = [decode(p) for p in probs]
    teacher_majority = {
        "discontinued": train_df["discontinued"].mode().iloc[0],
        "effectiveness": train_df["effectiveness"].mode().iloc[0],
    }
    baseline = [{**teacher_majority, "stop_reasons": []} for _ in range(len(val_df))]
    return {
        "seed": seed,
        "train_reviews": len(train_df),
        "train_seconds": round(train_s, 1),
        "agreement_with_teacher": agreement(val_df, preds),
        "majority_baseline": agreement(val_df, baseline),
        "reviews_per_second": round(speed, 1),
    }


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
    results = []
    for seed in seeds:
        name = f"reviews_seed{seed}" + (f"_smoke{args.max_train}" if args.max_train else "")
        path = report_dir / f"{name}.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        r = run_seed(cfg, seed, args.max_train)
        r["meta"] = {
            "model": cfg["model"],
            "git_sha": git_sha(),
            "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8", newline="\n")
        results.append(r)
        a = r["agreement_with_teacher"]
        log.info(
            "%s: discontinued macro-F1 %.3f, effectiveness macro-F1 %.3f, reasons micro-F1 %.3f",
            name,
            a["discontinued"]["macro_f1"],
            a["effectiveness"]["macro_f1"],
            a["stop_reasons_micro_f1"],
        )
    if not args.max_train and len(results) > 1:
        summary = {
            f: mean_std([{"v": r["agreement_with_teacher"][f]["macro_f1"]} for r in results], "v")
            for f in ("discontinued", "effectiveness")
        }
        summary["stop_reasons_micro_f1"] = mean_std(
            [{"v": r["agreement_with_teacher"]["stop_reasons_micro_f1"]} for r in results], "v"
        )
        summary["stopped_or_planned_f1"] = mean_std(
            [{"v": r["agreement_with_teacher"]["stopped_or_planned_f1"]} for r in results], "v"
        )
        (report_dir / "reviews_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
