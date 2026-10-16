"""In-domain router accuracy on eval_extra100 — the control for the Olympics collapse.

`scripts/router_generalization.py` measured the SEC-tuned router against a NEW domain and found
total collapse (26/150 = 0.17, every question labelled `lookup`). That number is only
interpretable next to an in-domain control, and `eval_extra100.jsonl` supplies one: 100 SEC
questions carrying GOLD `route` labels, 20 per route.

Like the generalization test this costs nothing — `route()` is a pure function of the question
string: no corpus, no graph, no LLM, no tokens. It is run and committed BEFORE the paid batch so
it cannot be retro-fitted to the outcome.

Usage:  python scripts/router_indomain.py
Writes: results/sec/router_indomain_extra100.json
"""
import sys, json, os, collections, importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

spec = importlib.util.spec_from_file_location("_p3", "pipeline3_sec/pipeline.py")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
route = _mod.route

QSET = "datasets/sec/extra100_set.jsonl"
OLYMPICS = "results/sec/router_generalization.json"
OUT = "results/sec/router_indomain_extra100.json"


def main():
    rows = [json.loads(l) for l in open(QSET)]
    for r in rows:
        r["route_pred"] = route(r["question"])
        r["ok"] = r["route_pred"] == r["route_gold"]

    per_route, per_shape = {}, {}
    for key, field, dest in (("route_gold", "route_gold", per_route), ("shape", "shape", per_shape)):
        for v in sorted({r[field] for r in rows}):
            sub = [r for r in rows if r[field] == v]
            agree = sum(r["ok"] for r in sub)
            dest[v] = {"n": len(sub), "correct": agree, "accuracy": round(agree / len(sub), 3),
                       "routes_produced": dict(collections.Counter(r["route_pred"] for r in sub))}

    # confusion matrix gold -> predicted
    labels = sorted({r["route_gold"] for r in rows} | {r["route_pred"] for r in rows})
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for r in rows:
        confusion[r["route_gold"]][r["route_pred"]] += 1

    correct = sum(r["ok"] for r in rows)
    out = {
        "what": "SEC-tuned regex router evaluated IN DOMAIN against gold route labels",
        "why": "control for results/sec/router_generalization.json (out-of-domain, 0.17)",
        "question_set": QSET,
        "n": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 3),
        "route_taxonomy": _mod.ROUTABLE,
        "per_gold_route": per_route,
        "per_shape": per_shape,
        "confusion_gold_to_predicted": confusion,
        "misroutes": [{"id": r["id"], "gold": r["route_gold"], "pred": r["route_pred"],
                       "shape": r["shape"], "question": r["question"]}
                      for r in rows if not r["ok"]],
        "finding": (
            "80/100 = 0.800 in domain. Four of five routes are perfect (20/20 each). ALL 20 "
            "failures are the single `compare` route, and every one lands on `aggregation`."),
        "cause": (
            "All 20 compare questions are phrased 'Which company reported higher <metric>, X or "
            "Y?'. Two independent faults: (1) AGG_RE's first alternative is "
            "r'\\b(which|list|name|how many|all)\\b.{0,40}\\bcompan(y|ies)\\b', which 'Which "
            "company' matches, and AGG_RE is tested FIRST (the an earlier implementation priority fix that won Tier D "
            "+0.17); (2) COMPARE_RE requires `which` IMMEDIATELY followed by "
            "had|has|spent|paid|reported, but here `which` is followed by `company`, so compare "
            "would have missed even without the priority order. The official held-out 50 phrase "
            "these as 'Which HAD the higher ...', which is why the defect never surfaced there."),
        "status": (
            "Found via a held-out set, so per the dev/test discipline it is REPORTED, not fixed "
            "into the canonical config. The frozen-config number stands as canonical; any fix is "
            "reported as a separately labelled post-hoc arm."),
    }

    if os.path.exists(OLYMPICS):
        oly = json.load(open(OLYMPICS))
        out["contrast_out_of_domain"] = {
            "source": OLYMPICS,
            "n": oly["combined"]["n"],
            "accuracy": oly["combined"]["agreement_rate"],
            "note": ("In domain 0.800 with one phrasing-shaped blind spot; out of domain "
                     f"{oly['combined']['agreement_rate']} with total collapse to the default "
                     "branch. The mechanism is portable; the lexicon is not."),
        }

    os.makedirs("results/sec", exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"{'gold route':<14}{'n':>4}{'correct':>9}{'acc':>7}   routes produced")
    for rt, d in out["per_gold_route"].items():
        print(f"{rt:<14}{d['n']:>4}{d['correct']:>9}{d['accuracy']:>7.2f}   {d['routes_produced']}")
    print(f"{'OVERALL':<14}{out['n']:>4}{out['correct']:>9}{out['accuracy']:>7.3f}")
    if "contrast_out_of_domain" in out:
        c = out["contrast_out_of_domain"]
        print(f"\ncontrast: in-domain {out['accuracy']}  vs  out-of-domain {c['accuracy']} (n={c['n']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
