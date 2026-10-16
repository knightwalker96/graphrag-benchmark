#!/usr/bin/env python3
"""Aggregate the optimization batch into a machine-readable summary + the tables for RESULTS_SEC.md.

Every figure printed here is computed from results/sec/{bench,eval}_*_run{1,2,3}.json. Nothing
is typed by hand. Writes results/sec/summary.json.

Usage:
  python "build/analysis/summarize_results.py" --tag v8final --baseline-tag baseline
"""
from __future__ import annotations
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path

PIPES = {"llm_only": "P1", "basic_rag": "P2", "graphrag": "P3"}


def load(tag: str, runs=(1, 2, 3), split="combined150"):
    B, E = {}, {}
    for i in runs:
        bp = Path(f"results/sec/bench_{split}_{tag}_run{i}.json")
        ep = Path(f"results/sec/eval_{tag}_run{i}.json")
        if not bp.exists():
            continue
        b = json.load(open(bp))
        B[i] = {PIPES[k]: {r["id"]: r for r in v} for k, v in b["pipelines"].items()}
        if ep.exists():
            e = json.load(open(ep))
            E[i] = {PIPES[k]: {r["id"]: r for r in v["per_item"]} for k, v in e["pipelines"].items()}
    return B, E


def agg(B, E, sel, pipe, field, src="e"):
    """(mean, pstdev) across runs of the per-run mean of `field` over `sel`."""
    per = []
    for i in sorted(B):
        tbl = (E.get(i) or {}).get(pipe) if src == "e" else B[i].get(pipe)
        if not tbl:
            continue
        vals = [tbl[q][field] for q in sel if q in tbl and tbl[q].get(field) is not None]
        if vals:
            per.append(sum(vals) / len(vals))
    if not per:
        return None, None
    return st.mean(per), (st.pstdev(per) if len(per) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v8final")
    ap.add_argument("--baseline-tag", default="baseline")
    ap.add_argument("--out", default="results/sec/summary.json")
    a = ap.parse_args()

    B, E = load(a.tag)
    B7, E7 = load(a.baseline_tag)
    assert B, f"no bench files for tag {a.tag}"
    ids = sorted(B[min(B)]["P3"])
    meta = {q: B[min(B)]["P3"][q] for q in ids}

    hold = {json.loads(l)["id"] for l in open("datasets/sec/split_holdout30.jsonl")}
    clean = {json.loads(l)["id"] for l in open("datasets/sec/split_holdout23_clean.jsonl")}
    tune = {json.loads(l)["id"] for l in open("datasets/sec/split_tune120.jsonl")}

    SETS = {
        "extra100": [q for q in ids if meta[q].get("set_name") == "extra100"],
        "heldout50": [q for q in ids if meta[q].get("set_name") != "extra100"],
        "combined150": ids,
        "tune120": [q for q in ids if q in tune],
        "holdout30": [q for q in ids if q in hold],
        "holdout23_clean": [q for q in ids if q in clean],
    }

    out = {"tag": a.tag, "baseline_tag": a.baseline_tag, "runs": sorted(B),
           "n_questions": len(ids), "sets": {}}

    print(f"=== the optimization summary: tag={a.tag}, runs={sorted(B)}, n={len(ids)} ===\n")
    for sname, sel in SETS.items():
        row = {"n": len(sel)}
        for p in ("P1", "P2", "P3"):
            d = {}
            for f in ("strict_pass", "graded_score", "bertscore_f1", "citation_quality"):
                m, s = agg(B, E, sel, p, f)
                if m is not None:
                    d[f] = round(m, 4); d[f + "_std"] = round(s, 4)
            for f in ("total_inference_tokens", "context_tokens", "retrieved_chunks"):
                m, s = agg(B, E, sel, p, f, src="b")
                if m is not None:
                    d[f] = round(m, 1); d[f + "_std"] = round(s, 1)
            row[p] = d
        # the initial implementation baseline on the same ids
        if B7:
            b7 = {}
            for f, src in (("strict_pass", "e"), ("total_inference_tokens", "b")):
                m, s = agg(B7, E7, sel, "P3", f, src=src)
                if m is not None:
                    b7[f] = round(m, 4 if src == "e" else 1)
            row["P3_initial"] = b7
        t2, t3 = row["P2"].get("total_inference_tokens"), row["P3"].get("total_inference_tokens")
        s2, s3 = row["P2"].get("strict_pass"), row["P3"].get("strict_pass")
        if t2 and t3:
            row["token_reduction_pct"] = round((t2 - t3) / t2 * 100, 1)
        if None not in (s2, s3):
            row["strict_lead"] = round(s3 - s2, 4)
            row["quality_per_1k_P2"] = round(s2 / t2 * 1000, 4)
            row["quality_per_1k_P3"] = round(s3 / t3 * 1000, 4)
            row["efficiency_ratio"] = round((s3 / t3) / (s2 / t2), 2)
        out["sets"][sname] = row
        init_s = row.get("P3_initial", {})
        print(f"{sname:18} n={len(sel):3d}  P1 {row['P1'].get('strict_pass'):.3f}  "
              f"P2 {s2:.3f}@{t2:6.0f}  P3 {s3:.3f}±{row['P3'].get('strict_pass_std',0):.3f}@{t3:6.0f}  "
              f"| the initial implementation P3 {init_s.get('strict_pass','?')}@{init_s.get('total_inference_tokens','?')}  "
              f"| red {row['token_reduction_pct']:+5.1f}%  eff {row['efficiency_ratio']}x")

    # ---- per gold route (extra100) ----
    print("\n=== extra100 by GOLD route (3-run mean) ===")
    g = defaultdict(list)
    for q in SETS["extra100"]:
        g[meta[q].get("route_gold", "")].append(q)
    out["by_gold_route"] = {}
    print(f"{'route':14}{'n':>3}{'P2':>7}{'P3init':>7}{'P3fin':>7}{'P2tok':>8}{'inittok':>8}{'fintok':>8}{'red':>8}")
    for k, sel in sorted(g.items()):
        s2 = agg(B, E, sel, "P2", "strict_pass")[0]; s3 = agg(B, E, sel, "P3", "strict_pass")[0]
        s7 = agg(B7, E7, sel, "P3", "strict_pass")[0] if B7 else None
        t2 = agg(B, E, sel, "P2", "total_inference_tokens", "b")[0]
        t3 = agg(B, E, sel, "P3", "total_inference_tokens", "b")[0]
        t7 = agg(B7, E7, sel, "P3", "total_inference_tokens", "b")[0] if B7 else None
        out["by_gold_route"][k] = {"n": len(sel), "P2_strict": round(s2, 4),
                                   "P3_initial_strict": round(s7, 4) if s7 is not None else None,
                                   "P3_final_strict": round(s3, 4), "P2_tok": round(t2, 1),
                                   "P3_initial_tok": round(t7, 1) if t7 else None,
                                   "P3_final_tok": round(t3, 1),
                                   "reduction_pct": round((t2 - t3) / t2 * 100, 1)}
        print(f"{k:14}{len(sel):3d}{s2:7.2f}{(s7 if s7 is not None else float('nan')):7.2f}{s3:7.2f}"
              f"{t2:8.0f}{(t7 or 0):8.0f}{t3:8.0f}{(t2-t3)/t2*100:+7.1f}%")

    # ---- per tier (held-out 50) ----
    print("\n=== held-out 50 by TIER (3-run mean) ===")
    g = defaultdict(list)
    for q in SETS["heldout50"]:
        g[meta[q].get("tier", "")].append(q)
    out["by_tier"] = {}
    print(f"{'tier':14}{'n':>3}{'P2':>7}{'P3init':>7}{'P3fin':>7}{'P2tok':>8}{'inittok':>8}{'fintok':>8}{'red':>8}")
    for k, sel in sorted(g.items()):
        s2 = agg(B, E, sel, "P2", "strict_pass")[0]; s3 = agg(B, E, sel, "P3", "strict_pass")[0]
        s7 = agg(B7, E7, sel, "P3", "strict_pass")[0] if B7 else None
        t2 = agg(B, E, sel, "P2", "total_inference_tokens", "b")[0]
        t3 = agg(B, E, sel, "P3", "total_inference_tokens", "b")[0]
        t7 = agg(B7, E7, sel, "P3", "total_inference_tokens", "b")[0] if B7 else None
        out["by_tier"][k] = {"n": len(sel), "P2_strict": round(s2, 4),
                             "P3_initial_strict": round(s7, 4) if s7 is not None else None,
                             "P3_final_strict": round(s3, 4), "P2_tok": round(t2, 1),
                             "P3_final_tok": round(t3, 1)}
        print(f"{k:14}{len(sel):3d}{s2:7.2f}{(s7 if s7 is not None else float('nan')):7.2f}{s3:7.2f}"
              f"{t2:8.0f}{(t7 or 0):8.0f}{t3:8.0f}{(t2-t3)/t2*100:+7.1f}%")

    # ---- per shape: before/after context + total tokens ----
    print("\n=== per-shape token table (3-run mean) ===")
    g = defaultdict(list)
    for q in ids:
        g[meta[q].get("shape") or ("tier" + meta[q].get("tier", ""))].append(q)
    out["by_shape"] = {}
    print(f"{'shape':24}{'n':>3}{'P2tok':>8}{'inittok':>8}{'fintok':>8}{'red':>8}{'v7ctx':>8}{'v8ctx':>8}")
    for k, sel in sorted(g.items(), key=lambda kv: -agg(B, E, kv[1], "P3", "total_inference_tokens", "b")[0]):
        t2 = agg(B, E, sel, "P2", "total_inference_tokens", "b")[0]
        t3 = agg(B, E, sel, "P3", "total_inference_tokens", "b")[0]
        t7 = agg(B7, E7, sel, "P3", "total_inference_tokens", "b")[0] if B7 else None
        c3 = agg(B, E, sel, "P3", "context_tokens", "b")[0]
        c7 = agg(B7, E7, sel, "P3", "context_tokens", "b")[0] if B7 else None
        out["by_shape"][k] = {"n": len(sel), "P2_tok": round(t2, 1),
                              "P3_initial_tok": round(t7, 1) if t7 else None, "P3_final_tok": round(t3, 1),
                              "reduction_pct": round((t2 - t3) / t2 * 100, 1),
                              "P3_initial_ctx": round(c7, 1) if c7 else None, "P3_final_ctx": round(c3, 1)}
        print(f"{k:24}{len(sel):3d}{t2:8.0f}{(t7 or 0):8.0f}{t3:8.0f}{(t2-t3)/t2*100:+7.1f}%{(c7 or 0):8.0f}{c3:8.0f}")

    # ---- invariants ----
    inv = {}
    inv["optimizer_tokens_all_zero"] = all(
        B[i]["P3"][q]["optimizer_tokens"] == 0 for i in B for q in ids)
    tg = {}
    for sname in ("extra100", "heldout50", "combined150"):
        sel = SETS[sname]
        rates = [sum(1 for q in sel if B[i]["P3"][q]["tg_queries"] > 0) / len(sel) for i in B]
        q_init = [sum(1 for q in sel if B7[i]["P3"][q]["tg_queries"] > 0) / len(sel) for i in B7] if B7 else []
        tg[sname] = {"final": round(st.mean(rates), 3), "initial": round(st.mean(q_init), 3) if q_init else None,
                     "total_queries": sum(B[i]["P3"][q]["tg_queries"] for i in B for q in sel)}
    inv["tg_live"] = tg
    inv["max_output_tokens"] = max(B[i]["P3"][q]["output_tokens"] for i in B for q in ids)
    inv["rows_at_output_cap"] = sum(1 for i in B for q in ids if B[i]["P3"][q]["output_tokens"] >= 500)
    out["invariants"] = inv
    print("\n=== invariants ===")
    print(f"  optimizer_tokens == 0 on every P3 row: {inv['optimizer_tokens_all_zero']}")
    for k, v in tg.items():
        print(f"  tg_live {k:12} the initial implementation={v['initial']}  the optimization={v['final']}  (the optimization live GSQL queries={v['total_queries']})")
    print(f"  max P3 output_tokens={inv['max_output_tokens']} (cap 512); rows at cap={inv['rows_at_output_cap']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
