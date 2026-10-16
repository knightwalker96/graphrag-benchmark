#!/usr/bin/env python3
"""Offline context-size preview for Pipeline 3. No LLM call, no API spend.

WHAT IT DOES. Monkeypatches the generator to a stub and runs run_single over a split, recording
the route taken and the tiktoken length of the context that WOULD have been sent. That makes the
context-token half of the measurement available for free, which is how a change can be sized
before paying for a real batch.

WHAT IT CANNOT TELL YOU. Without a live TigerGraph the `aggregation` and `bridge` routes do not
execute their real paths (no agg query, no vector seeding), so their numbers here are NOT
representative and are reported separately and labelled. `lookup`, `compare` and `temporal`
resolve from local artifacts and ARE representative. Output tokens are stubbed, so totals here
are context + scaffolding only, not a substitute for the benchmarked total_inference_tokens.

Usage:
  python "build/bench/dryrun_context.py" --split combined150
  python "build/bench/dryrun_context.py" --split tune120 --baseline results/sec/dryrun_v7.json
"""
from __future__ import annotations
import argparse, importlib.util, json, os, statistics as st, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())
os.environ.setdefault("DATASET", "sec")
os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("P3_REQUIRE_TG", "0")      # offline by design; see docstring

SPLITS = {
    "combined150": "datasets/sec/combined150_set.jsonl",
    "tune120": "datasets/sec/split_tune120.jsonl",
    "holdout30": "datasets/sec/split_holdout30.jsonl",
    "extra100": "datasets/sec/extra100_set.jsonl",
    "test": "datasets/sec/eval_set.jsonl",
}
LOCAL_ROUTES = ("lookup", "compare", "temporal")   # representative offline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="combined150", choices=sorted(SPLITS))
    ap.add_argument("--pipeline", default="pipeline3_sec/pipeline.py")
    ap.add_argument("--out", default="")
    ap.add_argument("--baseline", default="", help="a previous dryrun json, to diff against")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("p3dry", a.pipeline)
    p3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p3)

    from metrics import count_tokens_tiktoken
    captured: dict = {}

    def stub(prompt, *args, **kwargs):
        captured["prompt"] = prompt
        return ("STUB", {"usage": {"prompt_tokens": count_tokens_tiktoken(prompt),
                                   "completion_tokens": 4,
                                   "total_tokens": count_tokens_tiktoken(prompt) + 4}})

    p3.get_llm_response = stub
    p3._load()

    rows = [json.loads(l) for l in open(SPLITS[a.split])]
    out = []
    for i, qd in enumerate(rows, 1):
        try:
            m = p3.run_single(qd["question"])
            d = {"id": qd.get("id", ""), "route": m.route,
                 "context_tokens": m.context_tokens, "prompt_tokens": m.prompt_tokens,
                 "retrieved_chunks": m.retrieved_chunks, "tg_queries": m.tg_queries,
                 "optimizer_tokens": m.optimizer_tokens,
                 "shape": qd.get("shape", ""), "tier": qd.get("tier", ""),
                 "set_name": qd.get("set_name", ""), "error": ""}
        except Exception as e:
            d = {"id": qd.get("id", ""), "route": "ERROR", "context_tokens": 0,
                 "prompt_tokens": 0, "retrieved_chunks": 0, "tg_queries": 0,
                 "optimizer_tokens": 0, "shape": qd.get("shape", ""),
                 "tier": qd.get("tier", ""), "set_name": qd.get("set_name", ""),
                 "error": f"{type(e).__name__}: {str(e)[:160]}"}
        out.append(d)
        if i % 25 == 0:
            print(f"  ...{i}/{len(rows)}")

    errs = [d for d in out if d["error"]]
    print(f"\nerrors: {len(errs)}")
    for d in errs[:8]:
        print(f"   {d['id']}: {d['error']}")

    bad_opt = [d["id"] for d in out if d["optimizer_tokens"] != 0]
    print(f"optimizer_tokens == 0 on every row: {not bad_opt}"
          + (f"  VIOLATED by {bad_opt[:5]}" if bad_opt else ""))

    base = {}
    if a.baseline and Path(a.baseline).exists():
        base = {d["id"]: d for d in json.load(open(a.baseline))["rows"]}

    def show(title: str, key) -> None:
        g = defaultdict(list)
        for d in out:
            if d["error"]:
                continue
            g[key(d)].append(d)
        print(f"\n=== {title} ===")
        hdr = f"{'group':26}{'n':>4}{'ctx tok':>9}{'chunks':>8}{'tgq':>5}"
        if base:
            hdr += f"{'the initial implementation ctx':>9}{'delta':>9}"
        print(hdr); print("-" * len(hdr))
        for k in sorted(g, key=lambda k: -st.mean([d["context_tokens"] for d in g[k]])):
            v = g[k]
            ctx = st.mean([d["context_tokens"] for d in v])
            ch = st.mean([d["retrieved_chunks"] for d in v])
            tq = st.mean([d["tg_queries"] for d in v])
            line = f"{str(k):26}{len(v):4d}{ctx:9.0f}{ch:8.1f}{tq:5.1f}"
            if base:
                b = [base[d["id"]]["context_tokens"] for d in v if d["id"] in base]
                if b:
                    bm = st.mean(b)
                    line += f"{bm:9.0f}{ctx - bm:+9.0f}"
            print(line)

    show("by route (aggregation/bridge NOT representative offline)", lambda d: d["route"])
    show("by shape / tier", lambda d: d["shape"] or ("tier" + d["tier"]))

    rep = [d for d in out if not d["error"] and d["route"] in LOCAL_ROUTES]
    if rep:
        print(f"\nREPRESENTATIVE routes {LOCAL_ROUTES}: n={len(rep)} "
              f"mean ctx tok {st.mean([d['context_tokens'] for d in rep]):.0f}")
        if base:
            b = [base[d["id"]]["context_tokens"] for d in rep if d["id"] in base]
            if b:
                print(f"  the initial implementation baseline on the same rows: {st.mean(b):.0f}  "
                      f"=> {st.mean(b) - st.mean([d['context_tokens'] for d in rep]):+.0f} tokens")

    op = a.out or f"results/sec/dryrun_{a.split}.json"
    Path(op).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"split": a.split, "pipeline": a.pipeline, "offline": True, "rows": out},
              open(op, "w"), indent=1)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
