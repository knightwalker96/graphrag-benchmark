"""Pipeline-2 (Basic RAG) optimization sweep on the Olympics DEV 30 — the honest-baseline step.

WHY THIS EXISTS
    A GraphRAG result is only worth reporting against a baseline that was genuinely tried. The
    user's instruction for this step was explicit: "Do an unbiased optimisation of the second
    pipeline to bring the best numbers irrespective of the metrics of the third pipeline." So
    every arm here is selected on P2's own strict-pass, and arms that INCREASE P2's token spend
    are allowed to win. Nothing in the arm list or the selection rule refers to P3.

    Three SEC-era decisions inside pipeline2_basic_rag.py were fixed before this ran, because
    each would have held the Olympics baseline down (see that file for the comments):
      * adaptive-k borrowed the SEC router, which labels 24/30 Olympics dev questions wrong;
      * adaptive-k was min-capped at the global k, so it could only ever retrieve FEWER docs,
        while Olympics aggregation/superlative questions have ~11-12 gold documents each;
      * the SEC ticker metadata filter was live (measured inert on all 150 Olympics questions,
        but now explicitly dataset-gated).

PROTOCOL
    Stage 1 screens arms single-run. Stage 2 re-runs the shortlist 3x. This matters more here
    than it did on SEC: dev is 30 questions, so ONE question is 0.033 strict and the judge is
    nondeterministic. Single-run gaps below ~0.10 on this set are not evidence of anything —
    stage 1 only nominates, stage 2 decides.

    DEV ONLY. The held-out 70 is not touched by this file.

Run:
    python "build/olympics/p2_sweep_olympics.py" --stage 1
    python "build/olympics/p2_sweep_olympics.py" --stage 2 --arms base,topk_20,...
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT = ROOT / "results/olympics/p2_sweep"
BENCH = ROOT / "build/olympics/run_olympics_benchmark.py"
EVAL = ROOT / "build/olympics/evaluate_olympics.py"

# The dev3 P2 configuration, i.e. the baseline these arms are measured against.
BASE_ENV = {
    "P2_SEARCH": "similarity", "CONTEXT_NUM_CHUNKS": "10", "CONTEXT_CHUNK_CHARS": "2000",
    "P2_ADAPTIVE_K": "0", "P2_COMPRESS": "0", "P2_MMR_LAMBDA": "0.5", "P2_RERANK_POOL": "40",
}

# Per-route k profiles. The Olympics router is exact on dev (30/30), so a profile is a real
# per-question budget, not a guess. "wide" follows the gold-document counts: lookup/multi_hop/
# temporal need 1-2 documents, aggregation and superlative need 11-12.
PROF_WIDE = "lookup=4,multi_hop=6,temporal=6,aggregation=30,superlative=30"
PROF_WIDER = "lookup=6,multi_hop=8,temporal=8,aggregation=45,superlative=45"
PROF_LEAN = "lookup=3,multi_hop=4,temporal=4,aggregation=20,superlative=20"

ARMS: dict[str, dict] = {
    "base":            {},
    # --- retrieval mode -------------------------------------------------------------------
    "mmr":             {"P2_SEARCH": "mmr"},
    "mmr_rerank":      {"P2_SEARCH": "mmr_rerank"},
    # --- global top-k ---------------------------------------------------------------------
    "topk_5":          {"CONTEXT_NUM_CHUNKS": "5"},
    "topk_15":         {"CONTEXT_NUM_CHUNKS": "15"},
    "topk_20":         {"CONTEXT_NUM_CHUNKS": "20"},
    "topk_30":         {"CONTEXT_NUM_CHUNKS": "30"},
    # --- chunk size -----------------------------------------------------------------------
    "chunk_1000":      {"CONTEXT_CHUNK_CHARS": "1000"},
    "chunk_3000":      {"CONTEXT_CHUNK_CHARS": "3000"},
    # --- adaptive-k (zero LLM tokens; Olympics router) --------------------------------------
    "adapt_wide":      {"P2_ADAPTIVE_K": "1", "P2_ROUTE_K": PROF_WIDE},
    "adapt_wider":     {"P2_ADAPTIVE_K": "1", "P2_ROUTE_K": PROF_WIDER},
    "adapt_lean":      {"P2_ADAPTIVE_K": "1", "P2_ROUTE_K": PROF_LEAN},
    # --- extractive compression (zero LLM tokens; same local embedder) ----------------------
    "compress_6":      {"P2_COMPRESS": "1", "P2_COMPRESS_SENTS": "6"},
    "compress_12":     {"P2_COMPRESS": "1", "P2_COMPRESS_SENTS": "12"},
    "compress_20":     {"P2_COMPRESS": "1", "P2_COMPRESS_SENTS": "20"},
    "compress_30":     {"P2_COMPRESS": "1", "P2_COMPRESS_SENTS": "30"},
    "compress_40":     {"P2_COMPRESS": "1", "P2_COMPRESS_SENTS": "40"},
    # --- combinations nominated by stage 1 ---------------------------------------------------
    # Stage 1 found two independent wins: k=20 (+0.200 strict over k=10) and chunk_1000 (same
    # strict as 2000 at 41% fewer tokens — the infobox that carries the answer sits at the head
    # of chunk 0, so truncating the tail costs nothing). This combines them.
    "topk_20_chunk1000": {"CONTEXT_NUM_CHUNKS": "20", "CONTEXT_CHUNK_CHARS": "1000"},
    "topk_30_chunk1000": {"CONTEXT_NUM_CHUNKS": "30", "CONTEXT_CHUNK_CHARS": "1000"},
    # --- combinations (filled in after stage 1 nominates; kept here so they are reproducible)
    "adapt_wide_c30":  {"P2_ADAPTIVE_K": "1", "P2_ROUTE_K": PROF_WIDE,
                        "P2_COMPRESS": "1", "P2_COMPRESS_SENTS": "30"},
    "topk_20_c30":     {"CONTEXT_NUM_CHUNKS": "20",
                        "P2_COMPRESS": "1", "P2_COMPRESS_SENTS": "30"},
    "adapt_wide_3000": {"P2_ADAPTIVE_K": "1", "P2_ROUTE_K": PROF_WIDE,
                        "CONTEXT_CHUNK_CHARS": "3000"},
}


def run_arm(name: str, run: int) -> dict:
    ov = ARMS[name]
    env = {**os.environ, **BASE_ENV, **ov}
    tag = f"p2s_{name}_r{run}"
    bench = OUT / f"bench_olympics_dev_{tag}_run1.json"
    ev = OUT / f"eval_{tag}.json"
    if not ev.exists():
        t0 = time.perf_counter()
        subprocess.run([PY, str(BENCH), "--split", "dev", "--runs", "1", "--pipelines", "2",
                        "--tag", tag, "--out-dir", str(OUT)], env=env, check=True, cwd=str(ROOT))
        subprocess.run([PY, str(EVAL), str(bench), "--out", str(ev), "--skip-cite-judge"],
                       env=env, check=True, cwd=str(ROOT))
        print(f"  [{name} r{run}] {time.perf_counter()-t0:.0f}s", flush=True)
    a = json.load(open(ev))["pipelines"]["basic_rag"]["aggregate"]
    b = json.load(open(bench))["aggregates"]["basic_rag"]
    return {"arm": name, "run": run, "strict": a["strict_pass_rate"],
            "graded": a["graded_quality_mean"], "bert": a["bertscore_f1_mean"],
            "tokens": a["avg_total_inference_tokens"], "recall": a["retrieval_recall_mean"],
            "prec": a["retrieval_precision_mean"], "kdocs": a["retrieval_k_docs_mean"],
            "latency": b.get("avg_latency_seconds"), "cost": b.get("avg_cost_usd")}


def table(rows: list[dict], title: str) -> None:
    print("\n" + "=" * 104)
    print(title)
    print(f"{'arm':20} {'strict':>7} {'graded':>7} {'bert':>7} {'tokens':>8} "
          f"{'recall':>7} {'prec':>6} {'kdocs':>6} {'lat(s)':>7} {'cost$':>9}")
    print("-" * 104)
    for r in rows:
        print(f"{r['arm']:20} {r['strict']:>7.3f} {r['graded']:>7.3f} {r['bert']:>7.3f} "
              f"{r['tokens']:>8.0f} {r['recall']:>7.3f} {r['prec']:>6.3f} {r['kdocs']:>6.1f} "
              f"{str(r['latency']):>7} {r['cost']:>9.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--arms", default="")
    ap.add_argument("--runs", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = [x for x in a.arms.split(",") if x] or list(ARMS)
    for n in names:
        if n not in ARMS:
            raise SystemExit(f"unknown arm {n}; known: {', '.join(ARMS)}")
    runs = a.runs or (1 if a.stage == 1 else 3)

    rows = []
    for n in names:
        for r in range(1, runs + 1):
            try:
                rows.append(run_arm(n, r))
            except Exception as e:
                print(f"  !! arm {n} r{r} failed: {str(e)[:200]}", flush=True)

    json.dump(rows, open(OUT / f"stage{a.stage}_rows.json", "w"), indent=2)
    if runs == 1:
        table(sorted(rows, key=lambda r: (-r["strict"], -r["graded"])),
              "STAGE 1 (single run each — gaps under ~0.10 strict are noise on n=30)")
    else:
        agg = []
        for n in names:
            rs = [r for r in rows if r["arm"] == n]
            if not rs:
                continue
            st = [r["strict"] for r in rs]
            agg.append({"arm": n, "strict": statistics.mean(st),
                        "sd": statistics.pstdev(st) if len(st) > 1 else 0.0,
                        "runs": len(st), "spread": max(st) - min(st),
                        **{k: statistics.mean([r[k] for r in rs])
                           for k in ("graded", "bert", "tokens", "recall", "prec", "kdocs",
                                     "latency", "cost")}})
        agg.sort(key=lambda r: (-r["strict"], -r["graded"]))
        table(agg, f"STAGE {a.stage} ({runs} runs each, mean)")
        print(f"\n{'arm':20} {'strict mean':>12} {'sd':>7} {'spread':>8} {'runs':>5}")
        for r in agg:
            print(f"{r['arm']:20} {r['strict']:>12.3f} {r['sd']:>7.3f} "
                  f"{r['spread']:>8.3f} {r['runs']:>5}")
        json.dump(agg, open(OUT / f"stage{a.stage}_agg.json", "w"), indent=2)
    print(f"\n[p2-sweep] wrote {OUT}")


if __name__ == "__main__":
    main()
