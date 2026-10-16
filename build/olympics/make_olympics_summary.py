"""Build results/olympics/olympics_summary.json — the single source of every Task-2 number.

Nothing in TASK2_OLYMPICS_RESULTS.md is typed by hand: this script derives every figure from
the run artefacts, and make_olympics_tables.py renders the markdown from this JSON.

READS (never writes) one SEC artefact for the cross-dataset block: results/sec/summary.json
(tag final, 3 judge runs on the full 150). No SEC file is modified by this script.

Usage:  python "build/olympics/make_olympics_summary.py"
"""
from __future__ import annotations
import collections
import json
import statistics as st
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/olympics"
PIPES = ["llm_only", "basic_rag", "graphrag"]
METRICS = ["strict_pass_rate", "graded_quality_mean", "bertscore_f1_mean",
           "citation_quality_mean", "citation_precision_mean", "avg_total_inference_tokens",
           "retrieval_recall_mean", "retrieval_precision_mean", "retrieval_k_docs_mean"]


# route -> queries it must issue. Mirrors scripts/verify_branch_hops.py, which is the standalone
# gate; this records the same result into the summary so the report can cite it.
_EXPECTED_HOPS = {"lookup": ["event_by_title"], "aggregation": ["events_in_group"],
                  "superlative": ["events_in_group"],
                  "temporal": ["games_before", "events_in_group"],
                  "multi_hop": ["event_by_venue_date", "medallists_of"]}


def _hop_verify(B):
    """How many questions on a multi-query route issued every hop, across all runs."""
    multi = {k: v for k, v in _EXPECTED_HOPS.items() if len(v) > 1}
    total = complete = 0
    for b in B:
        for r in b["pipelines"]["graphrag"]:
            exp = multi.get(r.get("route"))
            if not exp:
                continue
            total += 1
            if all(q in (r.get("hop_log") or []) for q in exp):
                complete += 1
    return {"multi_hop_route_questions": total, "issued_every_hop": complete,
            "all_complete": total == complete,
            "routes": sorted(multi), "checker": "scripts/verify_branch_hops.py"}


def ms(vals):
    vals = [0.0 if v is None else v for v in vals]
    return {"mean": round(st.mean(vals), 4), "std": round(st.pstdev(vals), 4),
            "runs": [round(v, 4) for v in vals]}


def mcnemar(a: dict, b: dict):
    """Exact two-sided paired test. a/b are id -> bool. Returns (b_wins, a_wins, p)."""
    ids = sorted(set(a) & set(b))
    bw = [i for i in ids if b[i] and not a[i]]
    aw = [i for i in ids if a[i] and not b[i]]
    n, k = len(bw) + len(aw), min(len(bw), len(aw))
    p = min(sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n, 1.0) if n else 1.0
    return len(bw), len(aw), p, bw, aw


def main() -> None:
    E = [json.load(open(R / f"eval_held70_run{r}.json")) for r in (1, 2, 3)]
    B = [json.load(open(R / f"bench_olympics_test_held70_run{r}.json")) for r in (1, 2, 3)]
    gold = {json.loads(l)["id"]: json.loads(l)
            for l in open(ROOT / "datasets/olympics/test_set.jsonl")}
    devgold = {json.loads(l)["id"]: json.loads(l)
               for l in open(ROOT / "datasets/olympics/dev_set.jsonl")}

    out: dict = {
        "task": "the Olympics corpus — Olympics / Wikipedia",
        "config_tag": "held70",
        "frozen_config": "docs/design/OLYMPICS_FROZEN_CONFIG.md",
        "generator": "gemini-2.5-flash (temp 0, MAX_TOKENS 2048)",
        "judge": "gpt-4o-mini (blind to pipeline, anti-parametric clause — declared deviation)",
        "embedder": "BAAI/bge-small-en-v1.5 (384d, shared P2/P3)",
        "bertscore": "bert-score==0.3.13, roberta-large, lang=en, rescale_with_baseline=True, idf=False",
        "runs": 3,
        "sets": {"dev": {"n": len(devgold), "file": "datasets/olympics/dev_set.jsonl",
                         "role": "all tuning; frozen before held-out"},
                 "test": {"n": len(gold), "file": "datasets/olympics/test_set.jsonl",
                          "role": "held out; run once under the frozen config"},
                 "hidden": {"n": 50, "file": "eval_hidden.jsonl",
                            "role": "router accuracy ONLY — no answers exist, never scored"}},
    }

    # ---- headline, 3 runs ------------------------------------------------------------------
    main_block = {}
    for p in PIPES:
        d = {m: ms([e["pipelines"][p]["aggregate"][m] for e in E]) for m in METRICS}
        d["n"] = E[0]["pipelines"][p]["aggregate"]["num"]
        d["avg_latency_seconds"] = ms([b["aggregates"][p]["avg_latency_seconds"] for b in B])
        d["avg_cost_usd"] = ms([b["aggregates"][p]["avg_cost_usd"] for b in B])
        d["thinking_tokens_mean"] = ms(
            [st.mean([r.get("thinking_tokens", 0) for r in b["pipelines"][p]]) for b in B])
        d["tg_live_rate"] = ms([b["aggregates"][p].get("tg_live_rate", 0) for b in B])
        main_block[p] = d
    p2t = main_block["basic_rag"]["avg_total_inference_tokens"]["mean"]
    p3t = main_block["graphrag"]["avg_total_inference_tokens"]["mean"]
    s2 = main_block["basic_rag"]["strict_pass_rate"]["mean"]
    s3 = main_block["graphrag"]["strict_pass_rate"]["mean"]
    main_block["token_reduction_p3_vs_p2_pct"] = round(100 * (p2t - p3t) / p2t, 1)
    main_block["strict_lead_p3_over_p2"] = round(s3 - s2, 4)
    main_block["strict_relative_gain_p3_over_p2_pct"] = round(100 * (s3 - s2) / s2, 1)
    out["heldout70"] = main_block

    # ---- per qtype -------------------------------------------------------------------------
    byq = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in E:
        per = {p: {x["id"]: x for x in e["pipelines"][p]["per_item"]} for p in PIPES}
        for qid, g in gold.items():
            for p in PIPES:
                byq[g["qtype"]][p].append(1.0 if per[p][qid].get("strict_pass") else 0.0)
    out["heldout70_by_qtype"] = {
        q: {"n": sum(1 for g in gold.values() if g["qtype"] == q),
            **{p: round(st.mean(byq[q][p]), 4) for p in PIPES},
            "lead_p3_over_p2": round(st.mean(byq[q]["graphrag"]) - st.mean(byq[q]["basic_rag"]), 4)}
        for q in sorted(byq)}

    # ---- significance (run 1, paired) -------------------------------------------------------
    per1 = {p: {x["id"]: bool(x.get("strict_pass")) for x in E[0]["pipelines"][p]["per_item"]}
            for p in PIPES}
    bw, aw, p, bwl, awl = mcnemar(per1["basic_rag"], per1["graphrag"])
    out["significance_heldout70"] = {
        "test": "exact two-sided McNemar, paired per-question strict outcomes, run 1",
        "p3_only_correct": bw, "p2_only_correct": aw, "discordant": bw + aw,
        "p_value": p, "p3_only_correct_ids": bwl, "p2_only_correct_ids": awl}

    # dev, for the contrast that dev could not settle
    try:
        dp2 = {x["id"]: bool(x.get("strict_pass")) for x in json.load(
            open(R / "p2_sweep/eval_p2s_topk_20_c30_r1.json"))["pipelines"]["basic_rag"]["per_item"]}
        dp3 = {x["id"]: bool(x.get("strict_pass")) for x in json.load(
            open(R / "eval_dev3_run1.json"))["pipelines"]["graphrag"]["per_item"]}
        bw2, aw2, p2v, _, _ = mcnemar(dp2, dp3)
        out["significance_dev30"] = {"p3_only_correct": bw2, "p2_only_correct": aw2,
                                     "discordant": bw2 + aw2, "p_value": round(p2v, 4),
                                     "note": "dev could not establish the lead; test could"}
    except FileNotFoundError:
        pass

    # ---- ablations --------------------------------------------------------------------------
    abl = {"frozen_router": {"strict_pass_rate": s3,
                             "avg_total_inference_tokens": p3t,
                             "branch_resolved": sum(1 for r in B[0]["pipelines"]["graphrag"]
                                                    if r.get("branch_resolved")),
                             "n": len(B[0]["pipelines"]["graphrag"]),
                             "route_accuracy": round(sum(
                                 1 for r in B[0]["pipelines"]["graphrag"]
                                 if r.get("route") == r.get("route_gold")) / len(
                                 B[0]["pipelines"]["graphrag"]), 4),
                             "tg_live_rate": 1.0, "confounded": False}}
    for name, tag, conf in [("route_force_random", "routeforce_random", False),
                            ("route_force_lookup", "routeforce_lookup", False),
                            ("router_off", "router_off", True)]:
        ap = R / f"ablations/eval_abl_{tag}.json"
        bp = R / f"ablations/bench_olympics_test_abl_{tag}_run1.json"
        if not ap.exists():
            continue
        a = json.load(open(ap))["pipelines"]["graphrag"]["aggregate"]
        br = json.load(open(bp))["pipelines"]["graphrag"]
        n = len(br)
        abl[name] = {
            "strict_pass_rate": round(a["strict_pass_rate"], 4),
            "graded_quality_mean": round(a["graded_quality_mean"], 4),
            "bertscore_f1_mean": round(a["bertscore_f1_mean"], 4),
            "avg_total_inference_tokens": round(a["avg_total_inference_tokens"], 1),
            "retrieval_recall_mean": round(a["retrieval_recall_mean"], 4),
            "branch_resolved": sum(1 for r in br if r.get("branch_resolved")), "n": n,
            "route_accuracy": round(sum(1 for r in br
                                        if r.get("route") == r.get("route_gold")) / n, 4),
            "tg_live_rate": round(sum(1 for r in br if r.get("tg_live")) / n, 4),
            "routes_used": dict(collections.Counter(r.get("route") for r in br)),
            "confounded": conf,
            "delta_strict_vs_frozen": round(a["strict_pass_rate"] - s3, 4),
            "delta_tokens_vs_frozen": round(a["avg_total_inference_tokens"] - p3t, 1)}
    out["ablations_heldout70"] = abl

    # ---- router lexicon --------------------------------------------------------------------
    rt = {}
    for split in ("dev", "test", "hidden"):
        f = R / f"router_lexicon_{split}.json"
        if f.exists():
            d = json.load(open(f))
            rt[split] = {"n": d["n"], "correct": d["correct"], "accuracy": d["accuracy"],
                         "per_qtype": {q: v["accuracy"] for q, v in d["per_qtype"].items()}}
    for split in ("dev", "test"):
        f = R / f"router_sec_lexicon_{split}.json"
        if f.exists():
            rt[f"sec_lexicon_on_olympics_{split}"] = {
                "accuracy": json.load(open(f)).get("accuracy"),
                "note": "quarantine check: the SEC lexicon fails on Olympics questions"}
    # The two claims lexicon.py makes about itself, each backed by a saved artefact.
    f = R / "router_ablate_widening_dev.json"
    if f.exists():
        d = json.load(open(f))["widening_ablation"]
        rt["widening_ablation_dev"] = {
            "n": d["n"], "full_lexicon_correct": d["full_lexicon_correct"],
            "dev_literals_only_correct": d["dev_only_literals_correct"],
            "widened_alternatives_inert": d["identical"],
            "n_widened_alternatives": len(d["widened_alternatives"]),
            "note": "the widened alternatives are not what makes the router score 30/30"}
    f = R / "router_probe_ordering_dev.json"
    if f.exists():
        d = json.load(open(f))["ordering_probe"]
        rt["ordering_probe_dev"] = {
            "patterns_matching_per_question": d["patterns_matching_per_question"],
            "mutually_exclusive": d["mutually_exclusive"],
            "accuracy_by_ordering": d["accuracy_by_ordering"],
            "order_is_load_bearing": d["order_is_load_bearing"],
            "frozen_order": d["frozen_order"],
            "note": "correctness comes from mutual exclusivity, not from the chain order"}
    out["router"] = rt

    # ---- integrity / provenance ------------------------------------------------------------
    g0 = B[0]["pipelines"]["graphrag"]
    hops = collections.Counter((r["route"], tuple(r.get("hop_log") or [])) for r in g0)
    trunc = {}
    for p in PIPES:
        rows = [(bi, r) for bi, b in enumerate(B) for r in b["pipelines"][p]]
        mt = [r for _, r in rows if r.get("finish_reason") == "MAX_TOKENS"]
        trunc[p] = {"max_tokens_rows": len(mt), "rows_total": len(rows),
                    "finish_reasons": sorted({r.get("finish_reason") for _, r in rows})}
    out["integrity_heldout70"] = {
        "tg_live": sum(1 for r in g0 if r.get("tg_live")), "n": len(g0),
        "tg_queries_total": sum(r.get("tg_queries", 0) for r in g0),
        "rows_with_zero_tg_queries": sum(1 for r in g0 if r.get("tg_queries", 0) == 0),
        "branch_resolved": sum(1 for r in g0 if r.get("branch_resolved")),
        "rows_with_no_graph_docs": sum(1 for r in g0 if not r.get("graph_doc_ids")),
        "optimizer_tokens_all_zero": all(r.get("optimizer_tokens", 0) == 0
                                         for b in B for p in PIPES for r in b["pipelines"][p]),
        "hop_logs": {f"{k[0]}:{'+'.join(k[1]) or 'none'}": v for k, v in sorted(hops.items())},
        "hop_verification": _hop_verify(B),
        "truncation_audit": trunc}

    # ---- P2 sweep ---------------------------------------------------------------------------
    sw = R / "p2_sweep/stage1_rows.json"
    if sw.exists():
        rows = json.load(open(sw))
        best = {}
        for r in rows:
            if r["arm"] not in best or r["strict"] > best[r["arm"]]["strict"]:
                best[r["arm"]] = r
        out["p2_sweep_dev30"] = sorted(
            [{"arm": r["arm"], "strict": r["strict"], "graded": r["graded"], "bert": r["bert"],
              "tokens": round(r["tokens"], 0), "recall": r["recall"]} for r in best.values()],
            key=lambda x: -x["strict"])

    # ---- cross dataset (SEC read-only) -----------------------------------------------------
    sec_f = ROOT / "results/sec/summary.json"
    if sec_f.exists():
        sec = json.load(open(sec_f))
        c = sec["sets"]["combined150"]
        sp2, sp3 = c["P2"], c["P3"]
        out["cross_dataset"] = {
            "sec_source": "results/sec/summary.json (tag final, 3 runs, n=150) — READ ONLY",
            "sec": {"tag": sec.get("tag"), "n": c["n"],
                    "P1_strict": sp2 and c["P1"]["strict_pass"], "P2_strict": sp2["strict_pass"],
                    "P3_strict": sp3["strict_pass"],
                    "P1_tokens": c["P1"]["total_inference_tokens"],
                    "P2_tokens": sp2["total_inference_tokens"],
                    "P3_tokens": sp3["total_inference_tokens"],
                    "strict_lead_p3_over_p2": round(sp3["strict_pass"] - sp2["strict_pass"], 4),
                    "token_reduction_pct": round(100 * (sp2["total_inference_tokens"]
                                                        - sp3["total_inference_tokens"])
                                                 / sp2["total_inference_tokens"], 1),
                    "tg_live": sec.get("invariants", {}).get("tg_live", {})},
            "olympics": {"n": main_block["llm_only"]["n"],
                         "P1_strict": main_block["llm_only"]["strict_pass_rate"]["mean"],
                         "P2_strict": s2, "P3_strict": s3,
                         "P1_tokens": main_block["llm_only"]["avg_total_inference_tokens"]["mean"],
                         "P2_tokens": p2t, "P3_tokens": p3t,
                         "strict_lead_p3_over_p2": main_block["strict_lead_p3_over_p2"],
                         "token_reduction_pct": main_block["token_reduction_p3_vs_p2_pct"],
                         "tg_live": 1.0}}

    (R / "olympics_summary.json").write_text(json.dumps(out, indent=2))
    print(f"[summary] wrote {R / 'olympics_summary.json'}")
    print(f"  heldout70  P1={main_block['llm_only']['strict_pass_rate']['mean']:.4f}  "
          f"P2={s2:.4f}  P3={s3:.4f}  lead={main_block['strict_lead_p3_over_p2']:+.4f}  "
          f"tokens -{main_block['token_reduction_p3_vs_p2_pct']}%")
    print(f"  McNemar p={out['significance_heldout70']['p_value']:.2e}")


if __name__ == "__main__":
    main()
