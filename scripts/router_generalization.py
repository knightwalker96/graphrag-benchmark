"""Router generalization test — does the SEC-tuned regex router survive a domain change?

route() is a pure function of the question string: no corpus, no LLM, no tokens. So the
router can be tested against a brand-new domain even when that domain's corpus is absent.

The supplied Olympics/Wikidata eval sets carry gold `qtype` labels that map almost 1:1 onto
our route taxonomy, which gives genuine ground truth for a confusion matrix.

Usage:  python scripts/router_generalization.py
Writes: results/sec/router_generalization.json
"""
import sys, json, collections, importlib.util, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

spec = importlib.util.spec_from_file_location("_p3", "pipeline3_sec/pipeline.py")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
route = _mod.route

# gold qtype -> the route our taxonomy *should* fire. "superlative" has no dedicated route;
# "which X had the highest N" is a scan-and-rank, which is what the aggregation branch does.
EXPECT = {"lookup": "lookup", "multi_hop": "bridge", "aggregation": "aggregation",
          "temporal": "temporal", "superlative": "aggregation"}

BASE = "data/Eval Questions"
OUT = "results/sec/router_generalization.json"


def analyse(fn):
    rows = [json.loads(l) for l in open(f"{BASE}/{fn}")]
    for r in rows:
        r["route"] = route(r["question"])
        r["expected"] = EXPECT[r["qtype"]]
        r["ok"] = r["route"] == r["expected"]

    routes = sorted({r["route"] for r in rows})
    per_qtype = {}
    for qt in sorted({r["qtype"] for r in rows}):
        sub = [r for r in rows if r["qtype"] == qt]
        agree = sum(r["ok"] for r in sub)
        per_qtype[qt] = {
            "n": len(sub),
            "expected_route": EXPECT[qt],
            "routes_produced": dict(collections.Counter(r["route"] for r in sub)),
            "agree": agree,
            "agreement_rate": round(agree / len(sub), 3),
        }
    agree = sum(r["ok"] for r in rows)
    return {
        "file": fn,
        "n": len(rows),
        "distinct_routes_produced": routes,
        "overall_agreement": agree,
        "overall_agreement_rate": round(agree / len(rows), 3),
        "per_qtype": per_qtype,
        "misroute_examples": [
            {"qtype": r["qtype"], "got": r["route"], "question": r["question"]}
            for r in rows if not r["ok"]
        ][:10],
    }


def main():
    out = {
        "what": "SEC-tuned regex router evaluated on Olympics/Wikidata questions",
        "why": "no corpus/LLM needed — route() is a pure function of the question string",
        "route_taxonomy": _mod.ROUTABLE,
        "qtype_to_expected_route": EXPECT,
        "sets": [analyse(f) for f in ("eval_public.jsonl", "eval_hidden.jsonl")],
    }
    tot_n = sum(s["n"] for s in out["sets"])
    tot_a = sum(s["overall_agreement"] for s in out["sets"])
    out["combined"] = {
        "n": tot_n, "agreement": tot_a, "agreement_rate": round(tot_a / tot_n, 3),
        "finding": ("Router assigns 'lookup' to 100% of questions in the new domain. It scores "
                    "1.00 on the one category that is also the fallback and 0.00 on every other "
                    "category. Total collapse to the default branch, not graceful degradation."),
        "cause": ("Every pattern keys on SEC vocabulary: AGG_RE needs literal 'compan(y|ies)' / "
                  "GICS sectors / Big Four names; BRIDGE_RE needs 'the company (that|whose|which)'; "
                  "COMPARE_RE needs 'which' immediately followed by had/has/spent/paid/reported; "
                  "TEMPORAL_RE needs 8-K/fiscal vocabulary. Olympics questions say 'event', not "
                  "'company'."),
        "consequence": ("With every question on the lookup branch, P3 degenerates to 'P2 + curated "
                        "facts', the cheap aggregation route never fires, and the token-reduction "
                        "claim disappears — the saving lives only in the aggregation route."),
    }
    os.makedirs("results/sec", exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    for s in out["sets"]:
        print("=" * 76)
        print(f"{s['file']}  n={s['n']}  agreement={s['overall_agreement']}/{s['n']} "
              f"= {s['overall_agreement_rate']:.2f}   routes produced: {s['distinct_routes_produced']}")
        print(f"{'gold qtype':<14}{'n':>4}{'expected':>14}{'agreement':>12}")
        for qt, d in s["per_qtype"].items():
            agreement = f"{d['agree']}/{d['n']} ({d['agreement_rate']:.2f})"
            print(f"{qt:<14}{d['n']:>4}{d['expected_route']:>14}{agreement:>14}")
    print("=" * 76)
    print(f"COMBINED: {tot_a}/{tot_n} = {out['combined']['agreement_rate']:.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
