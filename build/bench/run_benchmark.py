"""Three-pipeline benchmark harness.

Runs P1 (LLM-only), P2 (Basic RAG), P3 (TigerGraph GraphRAG) over a question set,
records EVERY PS-required per-question field, and supports repeated runs (reproducibility).

Records per (pipeline, question): answer, retrieved chunk IDs, citations, the full token
breakdown (system/question/context/optimizer/output/total-inference, actual vs estimated),
latency, route, and P3 graph provenance (tg_live, graph_doc_ids).

Usage:
  DATASET=sec LLM_PROVIDER=gemini GEMINI_MODEL=gemini-2.5-flash EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    python "build/bench/run_benchmark.py" --split dev --runs 1 --pipelines 1,2,3
  # held-out final: --split test --runs 3
  # new sets:      --split extra100 | combined150  (see SPLITS below)
Ablations (P3 only): pass --p3-env "P3_GRAPH_OFF=1" etc. to tag a variant run.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")


# Split registry. "dev"/"test" keep their original files byte-for-byte so every earlier
# result reproduces unchanged; new splits are added here rather than by overwriting a file.
SPLITS = {
    "dev":          "datasets/sec/dev_set.jsonl",            # 24 Q, tuning
    "test":         "datasets/sec/eval_set.jsonl",           # official held-out 50 (EQ01-EQ50)
    "extra100":     "datasets/sec/extra100_set.jsonl",       # 100 new SEC Q (SQ001-SQ100)
    "combined150":  "datasets/sec/combined150_set.jsonl",    # SQ100 + EQ50, one batch
    # --- the optimization phasing splits (ADDITIVE ONLY: no existing entry altered, so every earlier
    # reproduction command behaves identically). Built by make_tune_holdout_split.py, seed
    # 20261005, stratified on `shape` (extra100) and the organiser `tier` (held-out 50).
    "tune120":      "datasets/sec/split_tune120.jsonl",      # the optimization tuning split
    "holdout30":    "datasets/sec/split_holdout30.jsonl",    # the optimization Phase-1 holdout
    "holdout23":    "datasets/sec/split_holdout23_clean.jsonl",  # holdout30 minus the 7
                                                                #  questions inspected during
                                                                #  the optimization development

    # --- the Olympics corpus, Olympics corpus. Separate names so no SEC artifact is overwritten
    # and `bench_<split>_<tag>_run<i>.json` filenames stay distinct.
    "olympics_dev":  "datasets/olympics/dev_set.jsonl",         # 30 Q, tuning ONLY
    "olympics_test": "datasets/olympics/test_set.jsonl",        # held-out 70, once per config
}

# P3 module path per dataset. Parameterized because the Olympics corpus adds a second P3 implementation;
# the DEFAULT stays the SEC path so every existing the SEC corpus reproduction command is unaffected.
P3_MODULE_DEFAULT = "pipeline3_sec/pipeline.py"


def load_questions(split: str) -> list[dict]:
    return [json.loads(l) for l in open(SPLITS[split])]


def run_pipeline(which: str, questions: list[dict], vectorstore,
                 p3_module: str = P3_MODULE_DEFAULT):
    import config
    rows = []
    if which == "1":
        import pipeline1_llm_only as p1
        for qd in questions:
            m = p1.run_single(qd["question"])
            rows.append(_row(qd, m))
    elif which == "2":
        import pipeline2_basic_rag as p2
        for qd in questions:
            m = p2.run_single(qd["question"], vectorstore)
            rows.append(_row(qd, m))
    elif which == "3":
        import importlib.util
        spec = importlib.util.spec_from_file_location("p3mod", p3_module)
        p3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p3)
        p3._load(); p3._connect()
        for qd in questions:
            m = p3.run_single(qd["question"])
            rows.append(_row(qd, m))
    return rows


def _row(qd: dict, m) -> dict:
    d = m.to_dict()
    d["id"] = qd.get("id", "")
    d["tier"] = qd.get("tier", "")
    d["hop"] = qd.get("hop", "")
    d["correct_answer"] = qd.get("answer", "")
    d["source_docs"] = qd.get("source_docs", "")
    # Carried through for the extra100 splits: a GOLD route label (so router accuracy can be
    # scored against d["route"], the route P3 actually chose), the finer `shape`, set-valued
    # gold for set-F1, and `answer_from` — which separates the 70 graph-edge-derived questions
    # from the 30 grounded only in 10-K text. Empty for the official held-out 50.
    d["route_gold"] = qd.get("route_gold", "")
    d["shape"] = qd.get("shape", "")
    d["gold_members"] = qd.get("gold_members", [])
    d["answer_from"] = qd.get("answer_from", "")
    d["set_name"] = qd.get("set_name", "")
    # Olympics (the Olympics corpus): per-question RETRIEVAL ground truth. 547/547 gold_doc_ids resolve
    # against the corpus, so retrieval precision/recall@k is measurable per pipeline — the SEC
    # round had no equivalent and could never separate "found the right document and answered
    # wrong" from "never found it". Empty list for every SEC split, so nothing changes there.
    d["gold_doc_ids"] = qd.get("gold_doc_ids", [])
    d["qtype"] = qd.get("qtype", "")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SPLITS), default="dev")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--pipelines", default="1,2,3")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results/sec")
    ap.add_argument("--p3-module", default=P3_MODULE_DEFAULT,
                    help="path to the P3 pipeline module (default: the SEC one, so the SEC corpus "
                         "reproductions are unaffected; the Olympics corpus passes pipeline3_olympics/pipeline.py)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N questions — for cheap preflight dry runs")
    args = ap.parse_args()

    import config
    from metrics import aggregate_metrics, CallMetrics
    questions = load_questions(args.split)
    if args.limit:
        questions = questions[: args.limit]
    pipes = args.pipelines.split(",")
    print(f"[bench] split={args.split} runs={args.runs} pipelines={pipes} n_questions={len(questions)}")

    vectorstore = None
    if "2" in pipes:
        from chroma_store import load_vectorstore
        vectorstore = load_vectorstore()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    name_map = {"1": "llm_only", "2": "basic_rag", "3": "graphrag"}

    for run_i in range(1, args.runs + 1):
        result = {"meta": {"split": args.split, "run": run_i, "runs": args.runs, "tag": args.tag,
                           "generator": f"{config.LLM_PROVIDER}:{config.GEMINI_MODEL if config.LLM_PROVIDER=='gemini' else config.OPENAI_MODEL}",
                           "dataset": config.DATASET, "p3_module": args.p3_module,
                           "max_tokens": config.MAX_TOKENS,
                           "embedder": config.EMBEDDING_MODEL, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  "pipelines": {}, "aggregates": {}}
        for which in pipes:
            print(f"\n[bench] run {run_i}/{args.runs} — pipeline {which} ({name_map[which]})")
            rows = run_pipeline(which, questions, vectorstore, p3_module=args.p3_module)
            result["pipelines"][name_map[which]] = rows
            # rebuild CallMetrics-lite objects for aggregation
            metrics = [CallMetrics(**{k: v for k, v in r.items() if k in CallMetrics.__annotations__}) for r in rows]
            result["aggregates"][name_map[which]] = aggregate_metrics(metrics)

        suffix = f"_{args.tag}" if args.tag else ""
        out = Path(args.out_dir) / f"bench_{args.split}{suffix}_run{run_i}.json"
        json.dump(result, open(out, "w"), indent=2)
        print(f"[bench] wrote {out}")
        # quick token comparison
        aggs = result["aggregates"]
        if "basic_rag" in aggs and "graphrag" in aggs:
            r2 = aggs["basic_rag"]["avg_total_inference_tokens"]
            r3 = aggs["graphrag"]["avg_total_inference_tokens"]
            print(f"[bench] avg inference tokens — RAG={r2} GraphRAG={r3} "
                  f"reduction={(1-r3/r2)*100:.1f}%" if r2 else "")


if __name__ == "__main__":
    main()
