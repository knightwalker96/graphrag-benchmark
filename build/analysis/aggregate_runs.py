"""Evaluate the 3 held-out runs and report mean ± std per metric (PS reproducibility).

Evaluates each run's benchmark JSON (strict/graded/BERTScore/citation), then aggregates
across runs into mean and standard deviation for every pipeline. Identifies which metrics
are deterministic (BERTScore) vs nondeterministic (judge-based). Retains all per-run raw
outputs. Writes results/sec/final_summary.json.

Usage:
  python "build/analysis/aggregate_runs.py" --tag final --runs 3
"""
from __future__ import annotations
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

PIPES = ["llm_only", "basic_rag", "graphrag"]
METRICS = ["strict_pass_rate", "graded_quality_mean", "bertscore_f1_mean",
           "citation_quality_mean", "avg_total_inference_tokens", "set_f1_mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--skip-eval", action="store_true", help="reuse existing eval_*.json")
    ap.add_argument("--split", default="test", help="bench filename split segment (test|extra100|combined150)")
    ap.add_argument("--ref-file", default="datasets/sec/eval_set.jsonl",
                    help="question-set JSONL holding the official reference answers")
    ap.add_argument("--cite-judge", action="store_true",
                    help="also run the citation judge (v7final's citation column was produced WITH it)")
    a = ap.parse_args()

    # 1. evaluate each run (adds official refs, judges, BERTScore)
    eval_paths = []
    eset = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(a.ref_file)}
    for i in range(1, a.runs + 1):
        bench = f"results/sec/bench_{a.split}_{a.tag}_run{i}.json"
        ev = f"results/sec/eval_{a.tag}_run{i}.json"
        eval_paths.append(ev)
        if a.skip_eval and Path(ev).exists():
            continue
        # patch official refs into the bench file for all pipelines
        d = json.load(open(bench))
        for pipe, rows in d["pipelines"].items():
            for r in rows:
                if r.get("id") in eset:
                    r["correct_answer"] = eset[r["id"]]
        json.dump(d, open(bench, "w"), indent=2)
        print(f"[agg] evaluating run {i}…")
        cmd = [sys.executable, "build/eval/evaluate.py", bench,
               "--out", ev, "--ref-file", a.ref_file]
        if not a.cite_judge:
            cmd.append("--skip-cite-judge")
        subprocess.run(cmd, check=True)

    # 2. collect per-run aggregates
    per_pipe = {p: {m: [] for m in METRICS} for p in PIPES}
    for ev in eval_paths:
        d = json.load(open(ev))
        for p in PIPES:
            a_ = d["pipelines"][p]["aggregate"]
            for m in METRICS:
                if a_.get(m) is not None:
                    per_pipe[p][m].append(a_[m])

    # 3. mean ± std
    summary = {"tag": a.tag, "runs": a.runs, "pipelines": {}}
    print(f"\n{'metric':26} {'LLM-only':>18} {'Basic RAG':>18} {'GraphRAG':>18}")
    rows = {}
    for m in METRICS:
        cells = []
        for p in PIPES:
            vals = per_pipe[p][m]
            if not vals:
                cells.append("—"); continue
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            summary["pipelines"].setdefault(p, {})[m] = {"mean": round(mean, 4), "std": round(std, 4),
                                                          "runs": vals}
            cells.append(f"{mean:.3f}±{std:.3f}")
        print(f"{m:26} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}")

    # token reduction (mean)
    r2 = statistics.mean(per_pipe["basic_rag"]["avg_total_inference_tokens"])
    r3 = statistics.mean(per_pipe["graphrag"]["avg_total_inference_tokens"])
    summary["token_reduction_vs_rag_pct"] = round((1 - r3 / r2) * 100, 1)
    print(f"\nToken reduction (GraphRAG vs RAG): {summary['token_reduction_vs_rag_pct']}%")
    print("Note: BERTScore is fully deterministic (same model+text); strict/graded/citation "
          "vary run-to-run via the LLM judge + generator — hence mean±std.")

    json.dump(summary, open("results/sec/final_summary.json", "w"), indent=2)
    print("\n[agg] wrote results/sec/final_summary.json")


if __name__ == "__main__":
    main()
