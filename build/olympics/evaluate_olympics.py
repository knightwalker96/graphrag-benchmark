"""Olympics answer evaluation — a thin wrapper over the shared evaluator.

One evaluator, two datasets. `build/eval/evaluate.py` already took `--ref-file` and
`--corpus`; this round additionally parameterized its judge prompt (`--domain`) and added
retrieval precision/recall@k. The SEC rendering of that prompt is asserted byte-identical to
the literal that produced every the SEC corpus number, so no SEC score can have moved.

Run:
    JUDGE_MODEL=gpt-4o-mini python "build/olympics/evaluate_olympics.py" \
        results/olympics/bench_olympics_dev_<tag>_run1.json \
        --out results/olympics/eval_<tag>_run1.json

WHAT IS HELD IDENTICAL TO THE SEC EVALUATION (the comparability contract)
    judge model gpt-4o-mini, blind to pipeline, different family from the Gemini generator ·
    PASS/FAIL + FULLY/MOSTLY/PARTIAL/INCORRECT with the same 1.0/0.66/0.33/0.0 mapping ·
    the citation judge and citation precision, unchanged · `answer_only()` citation stripping ·
    BERTScore pinned to bert-score==0.3.13, roberta-large, lang=en, rescale_with_baseline=True,
    idf=False · the token-accounting fields, untouched.

WHAT DIFFERS, AND WHY — three deviations, all declared
 1. THE DOMAIN SENTENCE. "SEC filings" -> "Olympic events". Anticipated by the contract.
 2. AN ANTI-PARAMETRIC CLAUSE, which is MORE than a domain sentence. The judge is told to grade
    against the supplied reference only. The SEC corpus was synthetic and future-dated so the
    judge had no opinion of its own; this corpus is real Wikipedia and gpt-4o-mini knows the
    Olympics. The dataset README settles the authority question — "If Wikipedia today disagrees
    with corpus.jsonl, the corpus wins" — and without the clause the judge's own recall leaks
    into the score. It is applied identically to all three pipelines and the judge stays blind,
    so it cannot bias P1 vs P2 vs P3 *within* this dataset; it does make the absolute strict-pass
    number less comparable ACROSS datasets, in a predictable direction (it should lower P1), and
    the results document must say so.
 3. SET-F1 IS NOT REPORTED. The scaffold expected it to carry over. It does not: no Olympics
    answer is set-valued. Aggregation answers are a count ("5"), superlative answers are one
    event title. `gold_members` is empty on every row, `set_f1()` correctly returns None, and
    `set_f1_mean` is null. Recorded as a decision, not an omission.

WHAT IS NEW — retrieval precision/recall@k
    `gold_doc_ids` gives per-question retrieval ground truth and 547/547 public references
    resolve in the corpus. The SEC round had no equivalent, so retrieval quality was never
    separable from answer quality there. Scored at DOCUMENT level after de-duplicating chunks
    of the same document. Read RECALL as the cross-pipeline number: k differs by pipeline by
    design (P2 adaptive-k, P3 branches return what the graph returns), so precision@k is not
    comparable across pipelines and is reported with the mean k beside it.
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = "datasets/olympics/corpus.jsonl"


def _evaluator():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "build" / "eval"))
    spec = importlib.util.spec_from_file_location(
        "evaluate", ROOT / "build" / "eval" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="benchmark results JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-cite-judge", action="store_true")
    ap.add_argument("--ref-file", default=None,
                    help="datasets/olympics/{dev,test}_set.jsonl — joins the gold answer, "
                         "route_gold (qtype) and gold_doc_ids onto each row")
    ap.add_argument("--corpus", default=CORPUS)
    a = ap.parse_args()

    ev = _evaluator()
    # Guard the one deviation that would silently invalidate the cross-dataset comparison:
    # evaluating Olympics answers with the SEC judge prompt.
    print(f"[eval-olympics] domain=olympics  judge={ev.JUDGE_MODEL}  corpus={a.corpus}")
    ev.evaluate(a.results, a.out, a.skip_cite_judge, a.ref_file, a.corpus, domain="olympics")


if __name__ == "__main__":
    main()
