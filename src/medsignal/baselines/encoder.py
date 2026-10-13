"""Fine-tune the student encoder (BioClinical ModernBERT) on 3-class sentiment, one run per seed.

Usage:
    uv run --extra gpu python -m medsignal.baselines.encoder --config configs/baselines.yaml [--seed 42]
    [--max-train N]   # quick smoke run on N training reviews

Resumable: re-running continues from the latest checkpoint of an unfinished seed and skips
finished seeds (their ``reports/baselines/encoder_seed{seed}.json`` exists).
Model selection uses ``val`` only; ``test`` and ``drug_holdout`` are scored once at the end.
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

from medsignal.baselines.classical import EVAL_SPLITS, load_labelled
from medsignal.evaluation.metrics import SENTIMENT_LABELS, classification_metrics
from medsignal.utils import PROJECT_ROOT, git_sha, load_config, load_env, seed_everything

log = logging.getLogger(__name__)


class _Encoded:
    """Minimal torch-style dataset over pre-tokenized inputs."""

    def __init__(self, enc: dict[str, list[list[int]]], labels: np.ndarray) -> None:
        self.enc = enc
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item: dict[str, Any] = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def run_seed(
    df: pd.DataFrame, cfg: dict[str, Any], seed: int, max_train: int | None
) -> dict[str, Any]:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    frames = {sp: g.reset_index(drop=True) for sp, g in df.groupby("split")}
    if max_train:
        frames["train"] = (
            frames["train"].sample(n=max_train, random_state=seed).reset_index(drop=True)
        )
        for sp in EVAL_SPLITS:
            frames[sp] = frames[sp].sample(
                n=min(len(frames[sp]), max_train // 4), random_state=seed
            )
            frames[sp] = frames[sp].reset_index(drop=True)

    def encode(frame: pd.DataFrame) -> _Encoded:
        enc = tok(frame["text"].tolist(), truncation=True, max_length=int(cfg["max_length"]))
        return _Encoded(
            {k: enc[k] for k in ("input_ids", "attention_mask")}, frame["label"].to_numpy()
        )

    data = {sp: encode(frames[sp]) for sp in ("train", *EVAL_SPLITS)}
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"],
        num_labels=len(SENTIMENT_LABELS),
        id2label=dict(enumerate(SENTIMENT_LABELS)),
        label2id={lab: i for i, lab in enumerate(SENTIMENT_LABELS)},
    )

    def compute_metrics(pred: Any) -> dict[str, float]:
        m = classification_metrics(pred.label_ids, np.argmax(pred.predictions, axis=-1))
        return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "qwk": m["qwk"]}

    tag = f"seed{seed}" + (f"_smoke{max_train}" if max_train else "")
    out_dir = PROJECT_ROOT / cfg["output_dir"] / tag
    args = TrainingArguments(
        output_dir=str(out_dir),
        seed=seed,
        num_train_epochs=float(cfg["epochs"]),
        learning_rate=float(cfg["lr"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["batch_size"]) * 2,
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        warmup_steps=float(cfg["warmup_ratio"]),  # v5: float in [0, 1) = ratio of total steps
        weight_decay=float(cfg["weight_decay"]),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        eval_strategy="steps",
        eval_steps=int(cfg["eval_every_steps"]),
        save_strategy="steps",
        save_steps=int(cfg["eval_every_steps"]),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        train_sampling_strategy="group_by_length",
        dataloader_num_workers=0,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    last = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None  # type: ignore[no-untyped-call]
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=last)
    train_s = time.perf_counter() - t0

    result: dict[str, Any] = {
        "seed": seed,
        "train_reviews": len(data["train"]),
        "train_seconds": round(train_s, 1),
    }
    for sp in EVAL_SPLITS:
        t1 = time.perf_counter()
        pred = trainer.predict(data[sp])  # type: ignore[arg-type]  # duck-typed dataset
        result[sp] = classification_metrics(
            np.asarray(pred.label_ids), np.argmax(np.asarray(pred.predictions), axis=-1)
        )
        result[sp]["reviews_per_second"] = round(len(data[sp]) / (time.perf_counter() - t1), 1)
    result["best_checkpoint"] = Path(trainer.state.best_model_checkpoint or "").name
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--seed", type=int, help="run only this seed")
    parser.add_argument("--max-train", type=int, help="smoke run on N training reviews")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()

    cfg = load_config(args.config)
    enc_cfg = cfg["encoder"]
    df = load_labelled(load_config(cfg["data_config"]))
    seeds = [args.seed] if args.seed is not None else [int(s) for s in enc_cfg["seeds"]]

    out_dir = PROJECT_ROOT / "reports" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        suffix = f"_smoke{args.max_train}" if args.max_train else ""
        path = out_dir / f"encoder_seed{seed}{suffix}.json"
        if path.exists():
            log.info("seed %s already done (%s), skipping", seed, path.name)
            continue
        result = run_seed(df, enc_cfg, seed, args.max_train)
        result["meta"] = {
            "config": args.config,
            "model": enc_cfg["model"],
            "git_sha": git_sha(),
            "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        log.info("seed %s: test macro-F1 %.4f -> %s", seed, result["test"]["macro_f1"], path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
