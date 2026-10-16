from pathlib import Path

from medsignal.evaluation.external import load_cadec, split_docs
from medsignal.evaluation.spans import Span, span_prf

TEXT = "Muscle pain, leg cramps and pain in my legs. Took Lipitor for cholesterol."


def _cadec(tmp_path: Path) -> Path:
    root = tmp_path / "cadec"
    for sub in ("text", "original", "meddra"):
        (root / sub).mkdir(parents=True)
    (root / "text" / "LIPITOR.1.txt").write_text(TEXT, encoding="utf-8")
    (root / "original" / "LIPITOR.1.ann").write_text(
        "T1\tADR 0 11\tMuscle pain\n"
        "T2\tADR 13 23\tleg cramps\n"
        "T3\tADR 28 32;36 43\tpain legs\n"
        "#1\tAnnotatorNotes T1\tnote\n"
        "T4\tDrug 50 57\tLipitor\n"
        "T5\tDisease 62 73\tcholesterol\n",
        encoding="utf-8",
    )
    (root / "meddra" / "LIPITOR.1.ann").write_text(
        "TT1\t10028411 0 11\tMuscle pain\n"
        "TT2\t10013781 + 10003077 13 23\tleg cramps\n"
        "TT3\tCONCEPT_LESS 28 32;36 43\tpain legs\n",
        encoding="utf-8",
    )
    return root


def test_load_cadec_keeps_only_adr_and_discontinuous_parts(tmp_path: Path) -> None:
    [doc] = load_cadec(_cadec(tmp_path))
    assert doc.drug == "LIPITOR"
    assert [a.parts for a in doc.ades] == [((0, 11),), ((13, 23),), ((28, 32), (36, 43))]
    assert doc.meddra == [["10028411"], ["10013781", "10003077"], []]


def test_split_docs_stratified_and_deterministic(tmp_path: Path) -> None:
    from medsignal.evaluation.external import GoldDoc

    docs = [GoldDoc(f"A.{i}", "A", "") for i in range(8)] + [
        GoldDoc(f"B.{i}", "B", "") for i in range(4)
    ]
    a = split_docs(docs, 0.25, seed=3)
    b = split_docs(docs, 0.25, seed=3)
    assert a.equals(b)
    assert a.groupby("drug")["split"].apply(lambda s: (s == "dev").sum()).to_dict() == {
        "A": 2,
        "B": 1,
    }


def test_span_prf_strict_vs_relaxed() -> None:
    gold = [Span.contiguous(0, 11), Span(((28, 32), (36, 43)))]
    pred = [Span.contiguous(0, 11), Span.contiguous(28, 43), Span.contiguous(50, 57)]
    strict = span_prf([(gold, pred)])
    relaxed = span_prf([(gold, pred)], relaxed=True)
    assert strict["tp"] == 1 and strict["n_pred"] == 3 and strict["n_gold"] == 2
    assert relaxed["tp"] == 2
    assert relaxed["recall"] == 1.0


def test_span_prf_one_to_one() -> None:
    gold = [Span.contiguous(0, 4), Span.contiguous(5, 9)]
    pred = [Span.contiguous(0, 9)]  # overlaps both gold spans but may match only one
    assert span_prf([(gold, pred)], relaxed=True)["tp"] == 1


def test_paired_bootstrap_diff_detects_better_system() -> None:
    from medsignal.evaluation.spans import paired_bootstrap_diff

    gold = [Span.contiguous(0, 4), Span.contiguous(5, 9)]
    weak = [(gold, [Span.contiguous(0, 4)]) for _ in range(30)]
    strong = [(gold, [Span.contiguous(0, 4), Span.contiguous(5, 9)]) for _ in range(30)]
    out = paired_bootstrap_diff(weak, strong, relaxed=True)
    assert out["diff"] > 0
    assert out["diff_lo"] > 0
    assert out["p_b_not_better"] == 0.0
    same = paired_bootstrap_diff(weak, weak)
    assert same["diff"] == 0.0


def test_span_prf_empty() -> None:
    assert span_prf([([], [])])["f1"] == 0.0
