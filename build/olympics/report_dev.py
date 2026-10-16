"""Summarise an Olympics eval artifact: headline metrics, per-qtype split, retrieval P/R@k,
and the two integrity audits the brief asks for (output truncation, byte-identical tokens).

    python "build/olympics/report_dev.py" results/olympics/eval_devbase_run1.json \
        [--bench results/olympics/bench_olympics_dev_devbase_run1.json]
"""
from __future__ import annotations
import argparse, json, statistics
from collections import defaultdict

ORDER = ["llm_only", "basic_rag", "graphrag"]
LABEL = {"llm_only": "P1 LLM-only", "basic_rag": "P2 Basic RAG", "graphrag": "P3 GraphRAG"}
QTYPES = ["lookup", "multi_hop", "temporal", "aggregation", "superlative"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval"); ap.add_argument("--bench", default=None)
    a = ap.parse_args()
    rep = json.load(open(a.eval))
    pipes = [p for p in ORDER if p in rep["pipelines"]]
    items = {p: rep["pipelines"][p]["per_item"] for p in pipes}
    aggs = {p: rep["pipelines"][p]["aggregate"] for p in pipes}
    n = len(items[pipes[0]])

    print(f"judge={rep['judge_model']}  domain={rep.get('domain')}  "
          f"prompt_sha={str(rep.get('strict_prompt_sha256'))[:12]}  n={n}\n")
    rows = [("strict pass", "strict_pass_rate"), ("graded quality", "graded_quality_mean"),
            ("BERTScore F1", "bertscore_f1_mean"), ("citation quality", "citation_quality_mean"),
            ("citation precision", "citation_precision_mean"),
            ("avg inference tokens", "avg_total_inference_tokens"),
            ("retrieval recall@k", "retrieval_recall_mean"),
            ("retrieval precision@k", "retrieval_precision_mean"),
            ("mean docs retrieved", "retrieval_k_docs_mean")]
    w = max(len(LABEL[p]) for p in pipes) + 2
    print(f"{'metric':24s}" + "".join(f"{LABEL[p]:>{w}}" for p in pipes))
    for label, key in rows:
        vals = [aggs[p].get(key) for p in pipes]
        if all(v is None for v in vals):
            continue
        print(f"{label:24s}" + "".join(
            f"{('—' if v is None else (f'{v:,.1f}' if key.startswith('avg') else f'{v:.3f}')):>{w}}"
            for v in vals))
    tok = {p: aggs[p]["avg_total_inference_tokens"] for p in pipes}
    if "basic_rag" in tok and "graphrag" in tok:
        print(f"\ntoken reduction P3 vs P2: {(1 - tok['graphrag']/tok['basic_rag'])*100:+.1f}%")
    if "basic_rag" in aggs and "llm_only" in aggs:
        print(f"P2 − P1 strict lead:      {aggs['basic_rag']['strict_pass_rate'] - aggs['llm_only']['strict_pass_rate']:+.3f}")

    print(f"\n--- strict pass by qtype ---")
    print(f"{'qtype':14s}{'n':>4s}" + "".join(f"{LABEL[p]:>{w}}" for p in pipes))
    for qt in QTYPES:
        idx = [i for i, it in enumerate(items[pipes[0]]) if it.get("qtype") == qt]
        if not idx:
            continue
        print(f"{qt:14s}{len(idx):>4d}" + "".join(
            f"{sum(bool(items[p][i]['strict_pass']) for i in idx)/len(idx):>{w}.2f}" for p in pipes))
    print(f"\n--- avg inference tokens by qtype ---")
    print(f"{'qtype':14s}{'n':>4s}" + "".join(f"{LABEL[p]:>{w}}" for p in pipes))
    for qt in QTYPES:
        idx = [i for i, it in enumerate(items[pipes[0]]) if it.get("qtype") == qt]
        if not idx:
            continue
        print(f"{qt:14s}{len(idx):>4d}" + "".join(
            f"{statistics.mean(items[p][i]['total_inference_tokens'] for i in idx):>{w},.0f}" for p in pipes))
    print(f"\n--- retrieval recall@k by qtype ---")
    print(f"{'qtype':14s}{'n':>4s}{'gold docs':>11s}" + "".join(f"{LABEL[p]:>{w}}" for p in pipes))
    for qt in QTYPES:
        idx = [i for i, it in enumerate(items[pipes[0]]) if it.get("qtype") == qt]
        if not idx or items[pipes[0]][idx[0]].get("retrieval_recall") is None:
            continue
        g = statistics.mean(items[pipes[0]][i]["retrieval_gold"] for i in idx)
        print(f"{qt:14s}{len(idx):>4d}{g:>11.1f}" + "".join(
            f"{statistics.mean(items[p][i]['retrieval_recall'] for i in idx):>{w}.2f}" for p in pipes))

    # ---- P1 as the empirical guess_baseline test -----------------------------------------
    if "llm_only" in items:
        hits = [it for it in items["llm_only"] if it["strict_pass"]]
        print(f"\n--- P1 (no retrieval) vs the dataset's `guess_baseline: 0.0` claim ---")
        print(f"P1 strict pass = {aggs['llm_only']['strict_pass_rate']:.3f} "
              f"({len(hits)}/{n}).  Every question is labelled guess_baseline 0.0.")
        by = defaultdict(int)
        for it in hits:
            by[it.get("qtype", "?")] += 1
        if hits:
            print(f"  P1 passes by qtype: {dict(by)}")
            for it in hits[:8]:
                print(f"    [{it['id']}] {it.get('qtype','')}: answered {it['answer'][:48]!r} "
                      f"ref {it['reference'][:40]!r}")

    # ---- integrity audits ------------------------------------------------------------------
    print("\n--- integrity audits ---")
    mx = {p: max(it.get("output_tokens", 0) for it in items[p]) for p in pipes}
    print(f"max output_tokens: {mx}  (MAX_TOKENS=256 -> truncation risk "
          f"{'NONE' if max(mx.values()) < 200 else 'CHECK'})")
    if len(pipes) > 1:
        ts = {p: aggs[p]["avg_total_inference_tokens"] for p in pipes}
        dup = len(set(ts.values())) != len(ts)
        print(f"byte-identical avg tokens across pipelines: {dup} "
              f"({'HARNESS BUG SUSPECTED' if dup else 'ok — arms differ'})")
    if a.bench:
        b = json.load(open(a.bench))
        print(f"bench meta: {json.dumps(b['meta'])}")


if __name__ == "__main__":
    main()
