"""Router-lexicon accuracy. Zero cost — `route()` is a pure function of the question string.

Closes the 2x2 that the two existing siblings started:

    scripts/router_generalization.py   SEC lexicon      on Olympics questions   26/150 = 0.173
    scripts/router_indomain.py         SEC lexicon      on SEC questions        80/100 = 0.800
    this script                        Olympics lexicon on Olympics questions   <- the new cell
    this script --lexicon sec          reproduces the 0.173 cell on the same rows, so both
                                       lexicons are scored by ONE scorer on ONE set of rows

That last point matters: the 0.173 and the Olympics number must come from the same scoring code
over the same questions, or the contrast is between two scripts rather than two lexicons.

Usage
    # step 6 — during development, dev only
    python scripts/router_lexicon_eval.py --split dev
    python scripts/router_lexicon_eval.py --split dev --ablate-widening

    # step 10 — ONCE, after the lexicon is frozen
    python scripts/router_lexicon_eval.py --split test --out results/olympics/router_lexicon_eval.json
    python scripts/router_lexicon_eval.py --split hidden     # router accuracy ONLY

THE HIDDEN 50
    `eval_hidden.jsonl` carries qid/question/qtype and NO answers, so nothing about answer
    quality can ever be scored on it. Gold `qtype` IS present, so router accuracy is the one
    honest number obtainable from that file, and this script is the only thing that should
    ever touch it.

WHAT A HIGH NUMBER HERE DOES AND DOES NOT MEAN
    All 150 Olympics questions come from five rigid templates. A near-perfect score is therefore
    a much weaker result than the SEC router's 0.800 on genuinely varied phrasing, and it must be
    reported as the in-domain control for the 0.173 collapse — not as evidence that regex routing
    generalizes. See pipeline3_olympics/lexicon.py.
"""
from __future__ import annotations
import argparse
import collections
import importlib.util
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pipeline3_olympics"))

SPLITS = {
    "dev":    "datasets/olympics/dev_set.jsonl",
    "test":   "datasets/olympics/test_set.jsonl",
    "hidden": "data/new_dataset/questions/eval_hidden.jsonl",
    "public": "data/new_dataset/questions/eval_public.jsonl",
}

# The SEC taxonomy has no `superlative` or `multi_hop` route, so scoring the SEC lexicon against
# Olympics gold needs a mapping. This is the SAME mapping scripts/router_generalization.py used
# to produce the published 0.173, reproduced here so the two cells are comparable.
SEC_EXPECT = {"lookup": "lookup", "multi_hop": "bridge", "aggregation": "aggregation",
              "temporal": "temporal", "superlative": "aggregation"}


def olympics_router():
    import lexicon
    return lexicon.route_of, {qt: qt for qt in
                              ("lookup", "multi_hop", "aggregation", "temporal", "superlative")}


def sec_router():
    spec = importlib.util.spec_from_file_location("_p3sec",
                                                  os.path.join(ROOT, "pipeline3_sec/pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.route, SEC_EXPECT


def load(split: str) -> list[dict]:
    rows = [json.loads(l) for l in open(os.path.join(ROOT, SPLITS[split])) if l.strip()]
    out = []
    for r in rows:
        out.append({"id": r.get("id") or r.get("qid"),
                    "question": r["question"],
                    "qtype": r.get("qtype") or r.get("route_gold")})
    return out


def score(rows, route_fn, expect_map) -> dict:
    for r in rows:
        r["route"] = route_fn(r["question"])
        r["expected"] = expect_map.get(r["qtype"], r["qtype"])
        r["ok"] = r["route"] == r["expected"]
    per = {}
    for qt in sorted({r["qtype"] for r in rows}):
        sub = [r for r in rows if r["qtype"] == qt]
        per[qt] = {"n": len(sub), "correct": sum(r["ok"] for r in sub),
                   "accuracy": round(sum(r["ok"] for r in sub) / len(sub), 4),
                   "expected": sub[0]["expected"],
                   "produced": dict(collections.Counter(r["route"] for r in sub))}
    n_ok = sum(r["ok"] for r in rows)
    return {"n": len(rows), "correct": n_ok, "accuracy": round(n_ok / len(rows), 4),
            "per_qtype": per,
            "confusion": {f"{r['qtype']}->{r['route']}": c for r, c in
                          collections.Counter((x["qtype"], x["route"]) for x in rows).items()
                          for r in [{"qtype": r[0], "route": r[1]}]},
            "routes_produced": dict(collections.Counter(r["route"] for r in rows)),
            "misses": [{"id": r["id"], "qtype": r["qtype"], "route": r["route"],
                        "question": r["question"]} for r in rows if not r["ok"]]}


def ablate_widening(rows) -> dict:
    """Strip every alternative that the dev set does not evidence, and re-score.

    If accuracy is unchanged, the widened alternatives are inert on dev — they are defensive
    generalization rather than patterns fitted to something. If accuracy FALLS, the widening is
    doing real work on dev and the claim in lexicon.py is false.
    """
    import lexicon
    narrow = [
        (re.compile(r"\bimmediately\s+before\b", re.I), "temporal"),
        (re.compile(r"\bevent\s+held\s+at\b.*\bon\b", re.I | re.S), "multi_hop"),
        (re.compile(r"\bwhich\b.*\bhad\s+the\s+highest\s+number\s+of\b", re.I | re.S), "superlative"),
        (re.compile(r"\bhow\s+many\b.*\bevents\b.*\bmore\s+than\b.*\b\d+\b.*\bcompetitors\b", re.I | re.S), "aggregation"),
        (re.compile(r"\bhow\s+many\s+nations\s+competed\s+in\b", re.I), "lookup"),
    ]

    def narrow_route(q):
        for pat, name in narrow:
            if pat.search(q or ""):
                return name
        return lexicon.FALLBACK

    full = sum(lexicon.route_of(r["question"]) == r["qtype"] for r in rows)
    nar = sum(narrow_route(r["question"]) == r["qtype"] for r in rows)
    return {"n": len(rows), "full_lexicon_correct": full, "dev_only_literals_correct": nar,
            "identical": full == nar,
            "widened_alternatives": lexicon.WIDENED_ALTERNATIVES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=sorted(SPLITS))
    ap.add_argument("--lexicon", default="olympics", choices=["olympics", "sec"])
    ap.add_argument("--ablate-widening", action="store_true")
    ap.add_argument("--probe-ordering", action="store_true",
                    help="is the priority order load-bearing? exhaustively re-scores all 120 "
                         "chain orderings and counts how many patterns match each question")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = load(a.split)
    route_fn, expect = (olympics_router() if a.lexicon == "olympics" else sec_router())
    res = score(rows, route_fn, expect)
    res.update(split=a.split, lexicon=a.lexicon, source=SPLITS[a.split])

    print(f"\n=== {a.lexicon.upper()} lexicon on OLYMPICS `{a.split}` ({res['n']} questions) ===")
    print(f"{'gold qtype':14s}{'n':>4s}{'correct':>9s}{'acc':>7s}   routes produced")
    for qt, d in res["per_qtype"].items():
        print(f"{qt:14s}{d['n']:>4d}{d['correct']:>9d}{d['accuracy']:>7.2f}   {d['produced']}")
    print(f"{'OVERALL':14s}{res['n']:>4d}{res['correct']:>9d}{res['accuracy']:>7.3f}")
    for m in res["misses"][:10]:
        print(f"   MISS [{m['id']}] gold={m['qtype']} -> {m['route']}: {m['question'][:78]}")

    if a.ablate_widening:
        abl = ablate_widening(rows)
        res["widening_ablation"] = abl
        print(f"\n--- widening ablation on `{a.split}` ---")
        print(f"full lexicon        : {abl['full_lexicon_correct']}/{abl['n']}")
        print(f"dev-literals only   : {abl['dev_only_literals_correct']}/{abl['n']}")
        print(f"widened alternatives are inert on this split: {abl['identical']}")
        for w in abl["widened_alternatives"]:
            print(f"    + {w}")

    if a.probe_ordering:
        import itertools
        import lexicon as L

        def sc(chain):
            def rr(q):
                for pat, nm in chain:
                    if pat.search(q or ""):
                        return nm
                return L.FALLBACK
            return sum(rr(r["question"]) == r["qtype"] for r in rows)

        overlap = collections.Counter(
            sum(bool(pat.search(r["question"])) for pat, _ in L.ROUTE_CHAIN) for r in rows)
        perms = collections.Counter(sc(list(p)) for p in itertools.permutations(L.ROUTE_CHAIN))
        res["ordering_probe"] = {
            "patterns_matching_per_question": dict(sorted(overlap.items())),
            "mutually_exclusive": set(overlap) == {1},
            "accuracy_by_ordering": dict(sorted(perms.items(), reverse=True)),
            "order_is_load_bearing": len(perms) > 1,
            "frozen_order": [nm for _, nm in L.ROUTE_CHAIN]}
        o = res["ordering_probe"]
        print(f"\n--- priority-order probe on `{a.split}` ---")
        print(f"patterns matching each question : {o['patterns_matching_per_question']}"
              f"   mutually exclusive: {o['mutually_exclusive']}")
        print(f"all 120 chain orderings         : {o['accuracy_by_ordering']}")
        print(f"order is load-bearing           : {o['order_is_load_bearing']}"
              f"  (False = correctness comes from mutual exclusivity, not from the order)")

    if a.out:
        os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
        json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=2, ensure_ascii=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
