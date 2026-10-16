"""Measure ONE P3 branch on the dev split, without spending generator tokens where possible.

Step 8 of the plan says each route is measured on dev before the next is written. This probe
does that cheaply: it runs the branch's GRAPH work only (no LLM call) and reports, per question,
whether the branch resolved, how many graph queries it issued, which ones, how many evidence
rows it produced, and whether the gold answer string is present in the evidence it would hand
the model. That last column is the branch's CEILING — if the answer is not in the context, no
generator can recover it.

    python "build/olympics/probe_branch.py" --route lookup
    python "build/olympics/probe_branch.py" --route all --split dev
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_pipeline():
    spec = importlib.util.spec_from_file_location("p3oly", ROOT / "pipeline3_olympics/pipeline.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m._load()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="all")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    p3 = load_pipeline()
    conn = p3._connect()
    rows = [json.loads(l) for l in open(ROOT / f"datasets/olympics/{a.split}_set.jsonl")]
    if a.route != "all":
        rows = [r for r in rows if r["qtype"] == a.route]

    ok = res = 0
    per = {}
    for r in rows:
        route = p3.route(r["question"])
        branch = p3.BRANCHES.get(route)
        p3._tg_query_count = 0
        p3._hop_log = []
        out = branch(conn, r["question"]) if branch else None
        nq, hops = p3._tg_query_count, list(p3._hop_log)
        if out:
            facts, prov, doc_ids = out
            res += 1
            ctx = facts + " " + " ".join(e["evidence"] for e in prov)
            hit = r["answer"].strip().lower() in ctx.lower()
            ok += hit
            gold = set(r["gold_doc_ids"])
            recall = len(gold & set(doc_ids)) / len(gold)
        else:
            facts, prov, doc_ids, hit, recall, ctx = "", [], [], False, 0.0, ""
        per.setdefault(route, []).append((hit, nq, recall, len(prov)))
        flag = "ok " if hit else ("CTX" if out else "N/A")
        print(f"  {flag} [{r['id']}] {route:11s} q={nq} hops={hops} rows={len(prov):3d} "
              f"R@gold={recall:.2f}  gold={r['answer'][:34]!r}")
        if not hit:
            print(f"        ctx[:160]: {ctx[:160]!r}")

    print(f"\n  branch resolved {res}/{len(rows)} · gold string present in context "
          f"{ok}/{len(rows)} (this is the branch CEILING, before the generator)")
    print(f"  {'route':12s}{'n':>4s}{'ceiling':>9s}{'avg queries':>13s}{'gold recall':>13s}{'avg rows':>10s}")
    for rt, v in sorted(per.items()):
        print(f"  {rt:12s}{len(v):>4d}{sum(x[0] for x in v)/len(v):>9.2f}"
              f"{sum(x[1] for x in v)/len(v):>13.2f}{sum(x[2] for x in v)/len(v):>13.2f}"
              f"{sum(x[3] for x in v)/len(v):>10.1f}")


if __name__ == "__main__":
    main()
