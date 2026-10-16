"""Slice the combined150 evaluation runs into the three reporting views.

The Task-1 batch is ONE coherent batch of 150 questions (SQ001-SQ100 from eval_extra100 plus
EQ01-EQ50, the official held-out set). Reporting it as a single pooled number would hide two
things, so this script emits three views and never collapses them:

  extra100    the 100 new SEC questions alone
  heldout50   the official 50 alone — a reproducibility check against RESULTS_SEC.md §3
  combined150 all 150

Grouping rules (chosen so nothing is invented):
  * tier tables      -> heldout50 ONLY. The extra100 rows have no tier field.
  * gold-route and
    shape tables     -> extra100 ONLY. The heldout50 rows have no gold route label.
  * predicted-route  -> any view. P3's chosen route exists for every question, so it is the only
                        legitimate common axis across the two sets.
  * answer_from      -> extra100 ONLY, and applied to EVERY extra100 table. 70 of those questions
                        have gold answers that coincide exactly with this project's own KG edges
                        (verified member-for-member on audited_by); the 30 marked
                        "10-K text, LLM-verified" are grounded only in filing text. The 30 are
                        the honest headline, the 70 a graph-recall diagnostic.

Across-run aggregation is mean ± population std, matching aggregate_runs.py.

Usage:
  python "build/analysis/slice_runs.py" --tag baseline --runs 3
Writes: results/sec/initial_summary.json
"""
from __future__ import annotations
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PIPES = ["llm_only", "basic_rag", "graphrag"]
LABEL = {"llm_only": "P1 LLM-only", "basic_rag": "P2 Basic RAG", "graphrag": "P3 GraphRAG"}


def metrics(items: list[dict]) -> dict:
    """Aggregate one group of per-item rows. None where a metric does not apply."""
    if not items:
        return {"n": 0}
    scored = [i for i in items if i.get("strict_pass") is not None]
    cites = [i["citation_quality"] for i in items if i.get("citation_quality") is not None]
    # Set-F1 is split by gold route on purpose. The 20 `compare` questions name BOTH candidate
    # companies in the question text ("Qualcomm or Salesforce?"), so a set-F1 over a singleton gold
    # measures a 50/50 guess, not retrieval — P1 LLM-only scores ~0.5 there with no corpus at all.
    # Pooling those with the genuinely set-valued routes inflates every pipeline equally and hides
    # the signal, so the headline figure covers aggregation + bridge only and compare is reported
    # beside it. (evaluate.py's own console line pools all 60; this is the reporting path.)
    sets = [i["set_f1"] for i in items
            if i.get("set_f1") is not None and i.get("route_gold") in ("aggregation", "bridge")]
    sets_cmp = [i["set_f1"] for i in items
                if i.get("set_f1") is not None and i.get("route_gold") == "compare"]
    toks = [i.get("total_inference_tokens", 0) for i in items]
    n_pass = sum(1 for i in scored if i["strict_pass"])
    avg_tok = sum(toks) / len(toks)
    strict = (n_pass / len(scored)) if scored else None
    return {
        "n": len(items),
        "n_pass": n_pass,
        "strict_pass_rate": round(strict, 4) if strict is not None else None,
        "graded_quality_mean": round(statistics.mean(i["graded_score"] for i in items), 4),
        "bertscore_f1_mean": round(statistics.mean(i["bertscore_f1"] for i in items), 4),
        "citation_quality_mean": round(statistics.mean(cites), 4) if cites else None,
        "set_f1_mean": round(statistics.mean(sets), 4) if sets else None,
        "num_set_scored": len(sets),
        "set_f1_mean_compare": round(statistics.mean(sets_cmp), 4) if sets_cmp else None,
        "num_set_scored_compare": len(sets_cmp),
        "avg_total_inference_tokens": round(avg_tok, 1),
        "strict_per_1k_tokens": round(strict / (avg_tok / 1000), 4) if strict and avg_tok else None,
    }


def mean_std(values: list[float | None]) -> dict | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"mean": round(statistics.mean(vals), 4),
            "std": round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4),
            "runs": vals}


AGG_KEYS = ["strict_pass_rate", "graded_quality_mean", "bertscore_f1_mean",
            "citation_quality_mean", "set_f1_mean", "set_f1_mean_compare",
            "avg_total_inference_tokens", "strict_per_1k_tokens"]


def across_runs(per_run: list[dict]) -> dict:
    """Collapse the same group measured in N runs into mean ± std, plus pooled pass counts."""
    out = {"n": per_run[0]["n"], "runs": len(per_run)}
    for k in AGG_KEYS:
        ms = mean_std([r.get(k) for r in per_run])
        if ms:
            out[k] = ms
    out["pooled_pass"] = sum(r["n_pass"] for r in per_run)
    out["pooled_n"] = sum(r["n"] for r in per_run)
    out["pooled_strict"] = round(out["pooled_pass"] / out["pooled_n"], 4) if out["pooled_n"] else None
    return out


def group_by(items: list[dict], field: str) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for i in items:
        key = i.get(field) or ""
        if key:
            g[key].append(i)
    return dict(g)


def build(tag: str, runs: int) -> dict:
    evals = []
    for i in range(1, runs + 1):
        p = Path(f"results/sec/eval_{tag}_run{i}.json")
        if not p.exists():
            raise SystemExit(f"missing {p} — run evaluate.py for every run first")
        evals.append(json.load(open(p)))

    views = {"extra100": lambda i: i.get("set_name") == "extra100",
             "heldout50": lambda i: i.get("set_name") == "heldout50",
             "combined150": lambda i: True}

    out = {
        "tag": tag, "runs": runs,
        "judge_model": evals[0].get("judge_model"),
        "methodology": {
            "batch": "one batch of 150 (SQ001-SQ100 + EQ01-EQ50), sliced; no cross-batch splicing",
            "config": "frozen the initial implementation — nothing retuned for this measurement",
            "tier_tables": "heldout50 only (extra100 has no tier field)",
            "gold_route_tables": "extra100 only (heldout50 has no gold route label)",
            "common_axis": "P3 predicted route — defined for every question in both sets",
            "answer_from_caveat": (
                "70/100 extra100 questions have gold answers identical to this project's own KG "
                "edges (verified on audited_by: EY 32/32, PwC 31/31, Deloitte 24/24, KPMG 12/12). "
                "Every extra100 table is therefore also split by answer_from; the 30 "
                "'10-K text, LLM-verified' rows are the headline, the 70 'graph edges' rows a "
                "graph-recall diagnostic."),
        },
        "views": {}, "p3_by_predicted_route": {}, "extra100_by_gold_route": {},
        "extra100_by_shape": {}, "extra100_by_answer_from": {}, "heldout50_by_tier": {},
    }

    # ---- the three views, per pipeline ----
    for view, keep in views.items():
        out["views"][view] = {}
        for pipe in PIPES:
            per_run = [metrics([i for i in ev["pipelines"][pipe]["per_item"] if keep(i)]) for ev in evals]
            out["views"][view][pipe] = across_runs(per_run)
        v = out["views"][view]
        t2 = v["basic_rag"].get("avg_total_inference_tokens", {}).get("mean")
        t3 = v["graphrag"].get("avg_total_inference_tokens", {}).get("mean")
        if t2 and t3:
            v["token_reduction_p3_vs_p2_pct"] = round((1 - t3 / t2) * 100, 1)
        s2 = v["basic_rag"].get("strict_pass_rate", {}).get("mean")
        s3 = v["graphrag"].get("strict_pass_rate", {}).get("mean")
        if s2 is not None and s3 is not None:
            v["strict_lead_p3_over_p2"] = round(s3 - s2, 4)
        q2 = v["basic_rag"].get("strict_per_1k_tokens", {}).get("mean")
        q3 = v["graphrag"].get("strict_per_1k_tokens", {}).get("mean")
        if q2 and q3:
            v["quality_per_1k_ratio_p3_over_p2"] = round(q3 / q2, 2)

    # ---- grouped tables ----
    def grouped(dest: str, pipes: list[str], field: str, keep):
        for pipe in pipes:
            buckets: dict[str, list[dict]] = defaultdict(list)
            for ev in evals:
                for k, v in group_by([i for i in ev["pipelines"][pipe]["per_item"] if keep(i)], field).items():
                    buckets[k].append(metrics(v))
            out[dest].setdefault(pipe, {})
            for k in sorted(buckets):
                out[dest][pipe][k] = across_runs(buckets[k])

    grouped("p3_by_predicted_route", ["graphrag"], "route", lambda i: True)
    grouped("extra100_by_gold_route", PIPES, "route_gold", lambda i: i.get("set_name") == "extra100")
    grouped("extra100_by_shape", PIPES, "shape", lambda i: i.get("set_name") == "extra100")
    grouped("extra100_by_answer_from", PIPES, "answer_from", lambda i: i.get("set_name") == "extra100")
    grouped("heldout50_by_tier", PIPES, "tier", lambda i: i.get("set_name") == "heldout50")

    # ---- truncation audit (MAX_TOKENS=512) ----
    trunc = []
    for ri, ev in enumerate(evals, 1):
        for pipe in PIPES:
            for i in ev["pipelines"][pipe]["per_item"]:
                if i.get("output_tokens", 0) >= 500:
                    trunc.append({"run": ri, "pipeline": pipe, "id": i["id"],
                                  "output_tokens": i["output_tokens"]})
    out["truncation_audit"] = {
        "threshold": 500, "max_tokens_config": 512, "n_at_or_above": len(trunc), "rows": trunc[:40],
        "max_output_tokens_seen": max(
            (i.get("output_tokens", 0) for ev in evals for p in PIPES
             for i in ev["pipelines"][p]["per_item"]), default=0),
    }

    # ---- graph provenance on this batch ----
    live, q, n = 0, 0, 0
    for ev in evals:
        for i in ev["pipelines"]["graphrag"]["per_item"]:
            n += 1
            live += int(bool(i.get("tg_live")))
            q += i.get("tg_queries", 0)
    out["p3_graph_provenance"] = {"questions": n, "tg_live": live,
                                 "tg_live_rate": round(live / n, 4) if n else None,
                                 "gsql_queries": q}
    return out


def show(out: dict) -> None:
    for view in ("extra100", "heldout50", "combined150"):
        v = out["views"][view]
        print("=" * 92)
        print(f"VIEW: {view}   (n={v['llm_only']['n']} per run × {out['runs']} runs)")
        print(f"{'metric':<28}{'P1 LLM-only':>18}{'P2 Basic RAG':>18}{'P3 GraphRAG':>18}")
        for k in AGG_KEYS:
            cells = []
            for pipe in PIPES:
                d = v[pipe].get(k)
                cells.append(f"{d['mean']:.3f}±{d['std']:.3f}" if d else "—")
            print(f"{k:<28}{cells[0]:>18}{cells[1]:>18}{cells[2]:>18}")
        extra = [f"{kk}={v[kk]}" for kk in
                 ("token_reduction_p3_vs_p2_pct", "strict_lead_p3_over_p2",
                  "quality_per_1k_ratio_p3_over_p2") if kk in v]
        if extra:
            print("  " + "  ·  ".join(extra))

    for dest, title in (("extra100_by_answer_from", "extra100 by answer_from (THE caveat table)"),
                        ("extra100_by_gold_route", "extra100 by GOLD route"),
                        ("heldout50_by_tier", "heldout50 by tier"),
                        ("p3_by_predicted_route", "P3 by PREDICTED route (all 150)")):
        print("=" * 92)
        print(title)
        keys = sorted({k for p in out[dest] for k in out[dest][p]})
        print(f"{'group':<30}{'n':>5}" + "".join(f"{LABEL[p]:>22}" for p in out[dest]))
        for k in keys:
            cells = []
            n = 0
            for p in out[dest]:
                d = out[dest][p].get(k)
                if not d:
                    cells.append("—"); continue
                n = d["n"]
                s = d.get("strict_pass_rate", {}).get("mean")
                t = d.get("avg_total_inference_tokens", {}).get("mean")
                cells.append(f"strict {s:.2f} / {t:,.0f} tok" if s is not None else "—")
            print(f"{k:<30}{n:>5}" + "".join(f"{c:>22}" for c in cells))

    t = out["truncation_audit"]
    print("=" * 92)
    print(f"TRUNCATION AUDIT: {t['n_at_or_above']} rows at or above {t['threshold']} output tokens "
          f"(MAX_TOKENS={t['max_tokens_config']}); max seen = {t['max_output_tokens_seen']}")
    g = out["p3_graph_provenance"]
    print(f"P3 GRAPH PROVENANCE: tg_live {g['tg_live']}/{g['questions']} = {g['tg_live_rate']} "
          f"· {g['gsql_queries']} live GSQL queries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="results/sec/initial_summary.json")
    a = ap.parse_args()
    out = build(a.tag, a.runs)
    show(out)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n[slice] wrote {a.out}")


if __name__ == "__main__":
    main()
