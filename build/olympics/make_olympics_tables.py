"""Emit every Task-2 table as markdown, straight from results/olympics/olympics_summary.json.

Same contract as summarize_results.py: the narrative in TASK2_OLYMPICS_RESULTS.md is written by
hand, every FIGURE inside a table comes from here. Reads only the summary JSON (which in turn
reads the SEC summary read-only for the cross-dataset block).

Usage:  python "build/olympics/make_olympics_tables.py" [--out FILE]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPES = ["llm_only", "basic_rag", "graphrag"]
HEAD = {"llm_only": "P1 LLM-only", "basic_rag": "P2 Basic RAG", "graphrag": "P3 GraphRAG"}
ROWS = [("strict_pass_rate", "strict pass", 3), ("graded_quality_mean", "graded quality", 3),
        ("bertscore_f1_mean", "BERTScore F1", 3), ("citation_quality_mean", "citation quality", 3),
        ("citation_precision_mean", "citation precision", 3),
        ("avg_total_inference_tokens", "avg inference tokens", 0),
        ("retrieval_recall_mean", "retrieval recall@k", 3),
        ("retrieval_precision_mean", "retrieval precision@k", 3),
        ("retrieval_k_docs_mean", "mean docs retrieved", 2),
        ("avg_latency_seconds", "latency (s)", 2),
        ("thinking_tokens_mean", "thinking tokens (billed, excluded above)", 0)]


def cell(d, prec):
    if d is None:
        return "—"
    if prec == 0:
        return f"{d['mean']:,.0f} ± {d['std']:,.0f}"
    return f"{d['mean']:.{prec}f} ± {d['std']:.{prec}f}"


def t_main(s):
    v = s["heldout70"]
    o = [f"**Held-out {v['llm_only']['n']} · {s['runs']} runs · mean ± std**\n",
         "| metric | " + " | ".join(HEAD[p] for p in PIPES) + " |", "|---|:-:|:-:|:-:|"]
    for k, lab, pr in ROWS:
        o.append(f"| {lab} | " + " | ".join(cell(v[p].get(k), pr) for p in PIPES) + " |")
    o += ["", f"P3 token reduction vs P2: **{v['token_reduction_p3_vs_p2_pct']}%** · "
              f"strict lead: **{v['strict_lead_p3_over_p2']:+.4f}** "
              f"(**{v['strict_relative_gain_p3_over_p2_pct']:+.1f}%** relative)"]
    return "\n".join(o)


def t_qtype(s):
    q = s["heldout70_by_qtype"]
    o = ["| qtype | n | P1 | P2 | P3 | P3 − P2 |", "|---|:-:|:-:|:-:|:-:|:-:|"]
    for k, v in sorted(q.items(), key=lambda x: -x[1]["lead_p3_over_p2"]):
        o.append(f"| {k} | {v['n']} | {v['llm_only']:.2f} | {v['basic_rag']:.2f} | "
                 f"{v['graphrag']:.2f} | **{v['lead_p3_over_p2']:+.2f}** |")
    return "\n".join(o)


def t_abl(s):
    a = s["ablations_heldout70"]
    order = ["frozen_router", "route_force_random", "route_force_lookup", "router_off"]
    lab = {"frozen_router": "frozen router (3-run mean)",
           "route_force_random": "`P3_ROUTE_FORCE=random` — un-confounded",
           "route_force_lookup": "`P3_ROUTE_FORCE=lookup` — un-confounded",
           "router_off": "`P3_ROUTER_OFF=1` — **confounded**"}
    o = ["| arm | strict | tokens | branch resolved | route acc | tg_live | Δ strict | Δ tokens |",
         "|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for k in order:
        if k not in a:
            continue
        v = a[k]
        ds = "—" if k == "frozen_router" else f"**{v['delta_strict_vs_frozen']:+.3f}**"
        dt = "—" if k == "frozen_router" else f"**{v['delta_tokens_vs_frozen']:+,.0f}**"
        o.append(f"| {lab[k]} | {v['strict_pass_rate']:.3f} | "
                 f"{v['avg_total_inference_tokens']:,.0f} | {v['branch_resolved']}/{v['n']} | "
                 f"{v['route_accuracy']:.3f} | {v['tg_live_rate']:.2f} | {ds} | {dt} |")
    return "\n".join(o)


def t_router(s):
    r = s["router"]
    o = ["| split | n | router accuracy vs gold qtype |", "|---|:-:|:-:|"]
    for k in ("dev", "test", "hidden"):
        if k in r:
            o.append(f"| {k} | {r[k]['n']} | **{r[k]['accuracy']:.3f}** |")
    if "sec_lexicon_on_olympics_dev" in r:
        o.append(f"| *SEC lexicon on Olympics dev (quarantine check)* | 30 | "
                 f"*{r['sec_lexicon_on_olympics_dev']['accuracy']:.3f}* |")
    return "\n".join(o)


def t_sweep(s):
    if "p2_sweep_dev30" not in s:
        return "_(sweep rows absent)_"
    o = ["| arm | strict | graded | bert | tokens | recall@k |", "|---|:-:|:-:|:-:|:-:|:-:|"]
    for r in s["p2_sweep_dev30"]:
        star = " ← **FROZEN**" if r["arm"] == "topk_20_c30" else ""
        o.append(f"| `{r['arm']}`{star} | {r['strict']:.3f} | {r['graded']:.3f} | "
                 f"{r['bert']:.3f} | {r['tokens']:,.0f} | {r['recall']:.3f} |")
    return "\n".join(o)


def t_cross(s):
    c = s.get("cross_dataset")
    if not c:
        return "_(cross-dataset block absent)_"
    sec, oly = c["sec"], c["olympics"]
    o = [f"| | SEC filings — `{sec['tag']}`, n={sec['n']} | Olympics — held-out, n={oly['n']} |",
         "|---|:-:|:-:|",
         f"| P1 LLM-only strict | {sec['P1_strict']:.4f} | {oly['P1_strict']:.4f} |",
         f"| P2 Basic RAG strict | {sec['P2_strict']:.4f} | {oly['P2_strict']:.4f} |",
         f"| P3 GraphRAG strict | {sec['P3_strict']:.4f} | {oly['P3_strict']:.4f} |",
         f"| **P3 − P2 strict lead** | **{sec['strict_lead_p3_over_p2']:+.4f}** | "
         f"**{oly['strict_lead_p3_over_p2']:+.4f}** |",
         f"| P2 tokens | {sec['P2_tokens']:,.0f} | {oly['P2_tokens']:,.0f} |",
         f"| P3 tokens | {sec['P3_tokens']:,.0f} | {oly['P3_tokens']:,.0f} |",
         f"| **token reduction** | **{sec['token_reduction_pct']}%** | "
         f"**{oly['token_reduction_pct']}%** |"]
    tl = sec.get("tg_live", {})
    if isinstance(tl, dict) and "combined150" in tl:
        o.append(f"| tg_live (P3 actually reached the graph) | "
                 f"{tl['combined150'].get('final')} | {oly['tg_live']} |")
    return "\n".join(o)


def t_integrity(s):
    i = s["integrity_heldout70"]
    o = [f"- `tg_live`: **{i['tg_live']}/{i['n']}** · live GSQL queries: **{i['tg_queries_total']}** · "
         f"rows with zero queries: **{i['rows_with_zero_tg_queries']}**",
         f"- `branch_resolved`: **{i['branch_resolved']}/{i['n']}** · rows with no graph docs: "
         f"**{i['rows_with_no_graph_docs']}**",
         f"- `optimizer_tokens` all zero: **{i['optimizer_tokens_all_zero']}**", "",
         "| route : hops issued | questions |", "|---|:-:|"]
    for k, v in i["hop_logs"].items():
        o.append(f"| `{k}` | {v} |")
    o += ["", "| pipeline | rows at MAX_TOKENS | of total | finish reasons |", "|---|:-:|:-:|---|"]
    for p in PIPES:
        t = i["truncation_audit"][p]
        o.append(f"| {HEAD[p]} | {t['max_tokens_rows']} | {t['rows_total']} | "
                 f"{', '.join('`'+x+'`' for x in t['finish_reasons'])} |")
    return "\n".join(o)


def t_sig(s):
    a = s["significance_heldout70"]
    o = ["| set | P3-only correct | P2-only correct | discordant | exact McNemar p |",
         "|---|:-:|:-:|:-:|:-:|"]
    if "significance_dev30" in s:
        d = s["significance_dev30"]
        o.append(f"| dev 30 (tuning) | {d['p3_only_correct']} | {d['p2_only_correct']} | "
                 f"{d['discordant']} | {d['p_value']:.3f} — *not significant* |")
    o.append(f"| **held-out 70** | **{a['p3_only_correct']}** | **{a['p2_only_correct']}** | "
             f"{a['discordant']} | **{a['p_value']:.2e}** |")
    return "\n".join(o)


SECTIONS = [("MAIN", t_main), ("BY_QTYPE", t_qtype), ("SIGNIFICANCE", t_sig),
            ("ABLATIONS", t_abl), ("ROUTER", t_router), ("P2_SWEEP", t_sweep),
            ("INTEGRITY", t_integrity), ("CROSS_DATASET", t_cross)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/olympics/olympics_summary.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = json.load(open(ROOT / a.summary))
    blocks = [f"<!-- TABLE:{name} -->\n{fn(s)}\n" for name, fn in SECTIONS]
    text = "\n".join(blocks)
    if a.out:
        (ROOT / a.out).write_text(text)
        print(f"[tables] wrote {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
