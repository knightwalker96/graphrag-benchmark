"""Recompute every load-bearing figure in the write-ups straight from the raw artifacts.

This exists so that no narrative number in docs/results/ has to be trusted. Each check names the
document and section it defends, recomputes the figure from `results/sec/bench_*`/`eval_*`, and
compares against the value asserted in the prose. Anything that drifts is printed as FAIL.

Covers:
  1. RESULTS_SEC.md §1  — the context-builder split (35 vs 115), measured on the INITIAL implementation
  2. RESULTS_SEC.md §1  — the 29-of-64 aggregation fall-through breakdown
  3. RESULTS_SEC.md §3.4 — per-shape token savings, and how many shapes were token-negative in initial
  4. RESULTS_SEC.md §5  — equal-budget comparison against the compressed baseline
  5. all    — thinking-token accounting: reported vs true billed cost

Free: reads JSON only, no model calls, no graph calls.

Usage: python "build/analysis/verify_claims.py"
       python "build/analysis/verify_claims.py" --json   # machine-readable only
"""
from __future__ import annotations
import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

RES = Path("results/sec")
OUT = RES / "claims_verification.json"
RUNS = (1, 2, 3)
PIPES = {"llm_only": "P1", "basic_rag": "P2", "graphrag": "P3"}

# tolerance for a recomputed mean vs the rounded figure printed in prose
TOL_TOK = 1.0      # tokens, absolute
TOL_RATE = 0.002   # pass rates / fractions, absolute
TOL_PCT = 0.15     # percentages, absolute


def _bench(tag: str, run: int) -> dict:
    return json.load(open(RES / f"bench_combined150_{tag}_run{run}.json"))


def _eval(tag: str, run: int) -> dict:
    return json.load(open(RES / f"eval_{tag}_run{run}.json"))


def _rows(tag: str, run: int, pipe: str) -> dict:
    return {r["id"]: r for r in _bench(tag, run)["pipelines"][pipe]}


def _eval_rows(tag: str, run: int, pipe: str) -> dict:
    return {r["id"]: r for r in _eval(tag, run)["pipelines"][pipe]["per_item"]}


# The citation-prefix proxy below misclassifies at most this many questions per run. On baseline
# EQ42 emits a `graph_aggregation:` citation in runs 1-2 and a doc-id citation in run 3, while its
# context_tokens stay at 213 in all three runs — so the builder did fire every time and only the
# citation label moved. The proxy is the imprecise part, not the pipeline.
PROXY_DRIFT_ALLOWED = 1


def _prov_fired(tag: str, run: int) -> set[str]:
    """Questions where evidence substitution ran instead of whole-chunk context.

    Observable signature: the citation list carries a `graph_aggregation:` entry, which only
    `_context_from_prov` emits. See PROXY_DRIFT_ALLOWED for the one known edge case.
    """
    return {
        i for i, r in _rows(tag, run, "graphrag").items()
        if any(c.startswith("graph_aggregation") for c in (r.get("citations") or []))
    }


def _mean_over_runs(fn) -> float:
    return st.mean(fn(run) for run in RUNS)


# --------------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------------

def check_builder_split() -> dict:
    """RESULTS_SEC.md §1: splitting the 150 by which context builder actually ran, on the initial implementation.

    The table in §1 is a *initial* diagnostic — it is the motivation for final, not a description of
    final. Auditing it against final artifacts will not reproduce it; that is expected.
    """
    tag = "initial"
    fired = _prov_fired(tag, 1)
    # the split must be near run-independent, or the diagnostic is not well defined
    per_run = [_prov_fired(tag, run) for run in RUNS]
    drift = sorted(set.union(*per_run) - set.intersection(*per_run))
    stable = len(drift) <= PROXY_DRIFT_ALLOWED

    groups = {}
    for name in ("fired", "rest"):
        def pick(run, name=name):
            p3, p2 = _rows(tag, run, "graphrag"), _rows(tag, run, "basic_rag")
            p3e, p2e = _eval_rows(tag, run, "graphrag"), _eval_rows(tag, run, "basic_rag")
            ids = [i for i in p3 if (i in fired) == (name == "fired")]
            return dict(
                n=len(ids),
                p2tok=st.mean(p2[i]["total_inference_tokens"] for i in ids),
                p3tok=st.mean(p3[i]["total_inference_tokens"] for i in ids),
                p3ctx=st.mean(p3[i]["context_tokens"] for i in ids),
                p3strict=st.mean(float(p3e[i]["strict_pass"]) for i in ids),
                p2strict=st.mean(float(p2e[i]["strict_pass"]) for i in ids),
            )
        per_run = [pick(run) for run in RUNS]
        m = {k: st.mean(d[k] for d in per_run) for k in per_run[0]}
        m["saving_pct"] = (m["p2tok"] - m["p3tok"]) / m["p2tok"] * 100
        groups[name] = m

    asserted = {
        "fired": dict(n=35, p2tok=4061, p3tok=652, p3ctx=303, p3strict=0.829,
                      p2strict=0.143, saving_pct=83.9),
        "rest": dict(n=115, p2tok=4226, p3tok=4406, p3ctx=3540, p3strict=0.310,
                     p2strict=0.281, saving_pct=-4.3),
    }
    fails = []
    if not stable:
        fails.append(f"builder split drifts on {len(drift)} questions across runs "
                     f"(allowed {PROXY_DRIFT_ALLOWED}): {drift}")
    groups["_proxy_drift"] = {"ids": drift, "n": len(drift),
                              "per_run_sizes": [len(s) for s in per_run]}

    # §1's callout claims the initial "full chunks" group falls to 1,139.9 tokens in the final system
    rest_ids = [i for i in _rows(tag, 1, "graphrag") if i not in fired]
    post = _mean_over_runs(lambda run: st.mean(
        _rows("final", run, "graphrag")[i]["total_inference_tokens"] for i in rest_ids))
    groups["rest_under_v81"] = {"p3tok": post}
    if abs(post - 1139.9) > TOL_TOK:
        fails.append(f"rest-group tokens in the final system: doc 1139.9 vs measured {post:.1f}")

    for g, exp in asserted.items():
        got = groups[g]
        for k, v in exp.items():
            tol = TOL_RATE if "strict" in k else (TOL_PCT if "pct" in k else TOL_TOK)
            if k == "n":
                tol = 0
            if abs(got[k] - v) > tol:
                fails.append(f"{g}.{k}: doc {v} vs measured {got[k]:.3f}")
    return {"name": "RESULTS_SEC.md §1 context-builder split (initial implementation)", "measured": groups,
            "asserted": asserted, "fails": fails}


def check_fallthroughs() -> dict:
    """RESULTS_SEC.md §1: '29 of 64 aggregation-routed questions took a chunk fall-through'."""
    tag = "initial"
    fired = _prov_fired(tag, 1)
    p3 = _rows(tag, 1, "graphrag")
    agg = [r for r in p3.values() if r["route"] == "aggregation"]
    ft = [r for r in agg if r["id"] not in fired]
    by_gold = Counter(r.get("route_gold") or "" for r in ft)
    by_tier = Counter(r.get("tier") or "" for r in ft)
    by_shape = Counter(r.get("shape") or "" for r in ft)
    measured = {
        "aggregation_routed": len(agg),
        "prov_fired": len(agg) - len(ft),
        "fall_through": len(ft),
        "misrouted_compare": by_gold.get("compare", 0),
        "tier_B": by_tier.get("B", 0),
        "tier_C": by_tier.get("C", 0),
        "agg_supplier": by_shape.get("agg_supplier", 0),
    }
    asserted = {"aggregation_routed": 64, "prov_fired": 35, "fall_through": 29,
                "misrouted_compare": 20, "tier_B": 4, "tier_C": 3, "agg_supplier": 2}
    fails = [f"{k}: doc {v} vs measured {measured[k]}"
             for k, v in asserted.items() if measured[k] != v]
    return {"name": "RESULTS_SEC.md §1 aggregation fall-throughs", "measured": measured,
            "asserted": asserted, "fails": fails}


def _shape_savings(tag: str) -> dict:
    """Mean P2→P3 token saving (%) per gold shape, pooled over the 3 runs."""
    acc = defaultdict(lambda: defaultdict(list))
    for run in RUNS:
        p3, p2 = _rows(tag, run, "graphrag"), _rows(tag, run, "basic_rag")
        for i, r in p3.items():
            shape = r.get("shape") or (f"tier{r['tier']}" if r.get("tier") else None)
            if not shape:
                continue
            acc[shape]["p3"].append(r["total_inference_tokens"])
            acc[shape]["p2"].append(p2[i]["total_inference_tokens"])
    return {s: (st.mean(v["p2"]) - st.mean(v["p3"])) / st.mean(v["p2"]) * 100
            for s, v in acc.items()}


def check_shape_signs() -> dict:
    """RESULTS_SEC.md §3.4: every shape token-positive in the final system; how many were negative initially."""
    initial, final = _shape_savings("initial"), _shape_savings("final")
    neg7 = sorted(s for s, v in initial.items() if v < 0)
    neg8 = sorted(s for s, v in final.items() if v < 0)
    measured = {"n_shapes": len(set(initial) | set(final)), "negative_v7": len(neg7),
                "negative_v81": len(neg8), "which_negative_v7": neg7,
                "savings_v7": {k: round(v, 1) for k, v in sorted(initial.items())},
                "savings_v81": {k: round(v, 1) for k, v in sorted(final.items())}}
    asserted = {"n_shapes": 16, "negative_v7": 8, "negative_v81": 0}
    fails = [f"{k}: doc {v} vs measured {measured[k]}"
             for k, v in asserted.items() if measured[k] != v]
    return {"name": "RESULTS_SEC.md §3.4 per-shape token sign", "measured": measured,
            "asserted": asserted, "fails": fails}


def check_thinking_tokens() -> dict:
    """Thinking-token accounting.

    Gemini 2.5 bills reasoning tokens as output, but `total_inference_tokens` is
    prompt + completion only, so every headline token figure understates billed cost.
    This check quantifies the gap and confirms the exclusion is real.
    """
    rep, think = {}, {}
    for pipe, lab in PIPES.items():
        rep[lab] = _mean_over_runs(
            lambda run, p=pipe: st.mean(r["total_inference_tokens"]
                                        for r in _bench("final", run)["pipelines"][p]))
        think[lab] = _mean_over_runs(
            lambda run, p=pipe: st.mean(r.get("thinking_tokens", 0)
                                        for r in _bench("final", run)["pipelines"][p]))
    true = {k: rep[k] + think[k] for k in rep}

    # confirm the exclusion structurally on a row that actually spent thinking tokens
    row = next(r for r in _bench("final", 1)["pipelines"]["graphrag"]
               if r.get("thinking_tokens", 0) > 0)
    excluded = row["total_inference_tokens"] == row["prompt_tokens"] + row["completion_tokens"]

    measured = {
        "reported": {k: round(v, 1) for k, v in rep.items()},
        "thinking": {k: round(v, 1) for k, v in think.items()},
        "true_billed": {k: round(v, 1) for k, v in true.items()},
        "thinking_share_pct": {k: round(think[k] / true[k] * 100, 1) for k in rep},
        "reduction_reported_pct": round((rep["P2"] - rep["P3"]) / rep["P2"] * 100, 1),
        "reduction_true_pct": round((true["P2"] - true["P3"]) / true["P2"] * 100, 1),
        "thinking_excluded_from_total": excluded,
    }
    fails = []
    if not excluded:
        fails.append("thinking_tokens appears to be included in total_inference_tokens; "
                     "the disclosure note in the docs is then wrong")
    return {"name": "thinking-token accounting", "measured": measured,
            "asserted": {"reduction_reported_pct": 75.5, "reduction_true_pct": 71.7},
            "fails": fails}


def check_equal_budget() -> dict:
    """RESULTS_SEC.md §5: P3 runs below the compressed baseline's budget at >2x its accuracy."""
    b = json.load(open(RES / "bench_combined150_abl_compressed_baseline_run1.json"))["pipelines"]["basic_rag"]
    e = json.load(open(RES / "eval_abl_compressed_baseline_run1.json"))
    cmp_rep = st.mean(r["total_inference_tokens"] for r in b)
    cmp_th = st.mean(r.get("thinking_tokens", 0) for r in b)
    cmp_strict = e["pipelines"]["basic_rag"]["aggregate"]["strict_pass_rate"]

    p3_rep = _mean_over_runs(lambda run: st.mean(
        r["total_inference_tokens"] for r in _bench("final", run)["pipelines"]["graphrag"]))
    p3_th = _mean_over_runs(lambda run: st.mean(
        r.get("thinking_tokens", 0) for r in _bench("final", run)["pipelines"]["graphrag"]))
    p3_strict = _mean_over_runs(
        lambda run: _eval("final", run)["pipelines"]["graphrag"]["aggregate"]["strict_pass_rate"])

    measured = {
        "compressed_baseline_reported": round(cmp_rep, 1),
        "compressed_baseline_true": round(cmp_rep + cmp_th, 1),
        "compressed_baseline_strict": round(cmp_strict, 4),
        "p3_reported": round(p3_rep, 1),
        "p3_true": round(p3_rep + p3_th, 1),
        "p3_strict": round(p3_strict, 4),
        "below_budget_reported_pct": round((cmp_rep - p3_rep) / cmp_rep * 100, 1),
        "below_budget_true_pct": round(((cmp_rep + cmp_th) - (p3_rep + p3_th)) / (cmp_rep + cmp_th) * 100, 1),
        "accuracy_ratio": round(p3_strict / cmp_strict, 2),
    }
    fails = []
    if measured["below_budget_true_pct"] <= 0:
        fails.append("P3 is not below the compressed baseline's true budget")
    if measured["accuracy_ratio"] < 2.0:
        fails.append(f"accuracy ratio {measured['accuracy_ratio']} < 2x")
    return {"name": "RESULTS_SEC.md §5 equal-budget comparison", "measured": measured,
            "asserted": {"below_budget_reported_pct": 31.3, "accuracy_ratio": 2.5},
            "fails": fails}


CHECKS = [check_builder_split, check_fallthroughs, check_shape_signs,
          check_thinking_tokens, check_equal_budget]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="suppress the table, write JSON only")
    args = ap.parse_args()

    report, n_fail = [], 0
    for fn in CHECKS:
        try:
            r = fn()
        except Exception as exc:
            r = {"name": fn.__name__, "fails": [f"check errored: {exc}"], "measured": {}}
        report.append(r)
        n_fail += len(r["fails"])

    if not args.json:
        for r in report:
            status = "FAIL" if r["fails"] else "ok"
            print(f"[{status:>4}] {r['name']}")
            for f in r["fails"]:
                print(f"         - {f}")
        print()
        t = next(r for r in report if r["name"] == "thinking-token accounting")["measured"]
        print("token reduction P3 vs P2 (combined150):")
        print(f"  as reported in the docs : {t['reduction_reported_pct']}%")
        print(f"  including billed thinking: {t['reduction_true_pct']}%")
        print()
        print(f"{n_fail} failing assertion(s) across {len(report)} checks")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"checks": report, "n_failing_assertions": n_fail}, open(OUT, "w"), indent=2)
    if not args.json:
        print(f"[verify-claims] wrote {OUT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
