#!/usr/bin/env python3
"""Router diagnostics: routing accuracy, paraphrase stability, and stratified k-fold CV.

Costs nothing and touches no API: `route()` is a pure function of the question string, so this
runs in milliseconds and can be re-run after every pattern edit.

THREE MEASUREMENTS
  1. routing accuracy  — route(q) vs the pre-existing `route_gold` label. Only extra100
     (SQ001-SQ100) carries that label; the official held-out 50 does not, and we do NOT invent
     one for it (a gold label we authored after seeing our own router's output would be
     circular). The 50 are reported as a route DISTRIBUTION only.
  2. paraphrase stability — route(paraphrase) == route(original), over
     datasets/sec/paraphrase_probe.jsonl. Needs no gold label at all, which is exactly why
     it is the strongest generalization signal available here.
  3. stratified 5-fold CV — for each fold, disable the router patterns whose justification
     comes ONLY from that fold's questions, then score the fold. Pattern provenance is declared
     in PATTERN_PROVENANCE below; a pattern justified by pre-existing (dev24 / held-out) work
     is never disabled, because no fold of the 150 is what motivated it.

Usage:
  python "build/analysis/router_probe.py" --tag v7frozen
  P3_COMPARE_FIX=1 python "build/analysis/router_probe.py" --tag v7frozen_comparefix
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())
P3_PATH = "pipeline3_sec/pipeline.py"
DS = Path("datasets/sec")


def load_p3():
    spec = importlib.util.spec_from_file_location("p3probe", P3_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--out-dir", default="results/sec")
    a = ap.parse_args()

    p3 = load_p3()
    rows = jl(DS / "combined150_set.jsonl")
    para = jl(DS / "paraphrase_probe.jsonl")
    tune = {r["id"] for r in jl(DS / "split_tune120.jsonl")} if (DS / "split_tune120.jsonl").exists() else set()
    hold = {r["id"] for r in jl(DS / "split_holdout30.jsonl")} if (DS / "split_holdout30.jsonl").exists() else set()

    route = {r["id"]: p3.route(r["question"]) for r in rows}
    by_id = {r["id"]: r for r in rows}

    out: dict = {"tag": a.tag,
                 "compare_fix": os.getenv("P3_COMPARE_FIX", "0") != "0",
                 "route_force": os.getenv("P3_ROUTE_FORCE", "")}

    # ---------- 1. routing accuracy (extra100 only — the only pre-existing gold) ----------
    lab = [r for r in rows if r.get("route_gold")]
    per_route: dict = defaultdict(lambda: {"n": 0, "correct": 0, "produced": Counter()})
    for r in lab:
        g, got = r["route_gold"], route[r["id"]]
        d = per_route[g]
        d["n"] += 1; d["correct"] += (got == g); d["produced"][got] += 1
    acc = sum(d["correct"] for d in per_route.values()) / max(1, len(lab))
    out["routing_accuracy"] = {
        "n_labelled": len(lab), "overall": round(acc, 4),
        "by_gold_route": {k: {"n": v["n"], "correct": v["correct"],
                              "acc": round(v["correct"] / v["n"], 4),
                              "produced": dict(v["produced"])}
                          for k, v in sorted(per_route.items())}}
    print(f"=== 1. routing accuracy (n={len(lab)} labelled) : {acc:.3f} ===")
    print(f"{'gold route':14}{'n':>4}{'correct':>9}{'acc':>7}   produced")
    for k, v in sorted(per_route.items()):
        print(f"{k:14}{v['n']:4d}{v['correct']:9d}{v['correct']/v['n']:7.3f}   "
              + ", ".join(f"{a}x{b}" for a, b in v["produced"].most_common()))
    # unlabelled 50: distribution only, never scored against an invented gold
    dist = Counter(route[r["id"]] for r in rows if not r.get("route_gold"))
    out["heldout50_route_distribution"] = dict(dist)
    print(f"\n  held-out 50 route distribution (no gold label exists; NOT scored): {dict(dist)}")

    # ---------- 2. paraphrase stability ----------
    stable = 0
    fails: list[dict] = []
    by_rule: dict = defaultdict(lambda: [0, 0])
    by_shape: dict = defaultdict(lambda: [0, 0])
    for p in para:
        o = p["orig_id"]
        if o not in route:
            continue
        got, want = p3.route(p["question"]), route[o]
        ok = got == want
        stable += ok
        key = p.get("shape") or ("tier" + (p.get("tier") or ""))
        by_rule[p["rule"]][1] += 1; by_rule[p["rule"]][0] += ok
        by_shape[key][1] += 1; by_shape[key][0] += ok
        if not ok:
            fails.append({"id": p["id"], "orig_id": o, "rule": p["rule"],
                          "orig_route": want, "para_route": got,
                          "orig_question": p["orig_question"], "para_question": p["question"]})
    n = len(para)
    out["paraphrase_stability"] = {
        "n_paraphrases": n, "n_originals": len({p["orig_id"] for p in para}),
        "stable": stable, "rate": round(stable / n, 4),
        "by_rule": {k: {"stable": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)}
                    for k, v in sorted(by_rule.items())},
        "by_shape": {k: {"stable": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)}
                     for k, v in sorted(by_shape.items())},
        "failures": fails}
    print(f"\n=== 2. paraphrase stability : {stable}/{n} = {stable/n:.3f} ===")
    print(f"{'shape / tier':26}{'stable':>8}{'n':>5}{'rate':>8}")
    for k, v in sorted(by_shape.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"{k:26}{v[0]:8d}{v[1]:5d}{v[0]/v[1]:8.3f}")
    print(f"\n  rules with any failure:")
    for k, v in sorted(by_rule.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        if v[0] < v[1]:
            print(f"    {k:34}{v[0]:4d}/{v[1]:<4d} {v[0]/v[1]:.3f}")

    # stability split by tune/holdout, so Phase 1 can report the holdout number separately
    for name, idset in (("tune120", tune), ("holdout30", hold)):
        if not idset:
            continue
        sub = [p for p in para if p["orig_id"] in idset]
        s = sum(1 for p in sub if p3.route(p["question"]) == route[p["orig_id"]])
        out["paraphrase_stability"][f"rate_{name}"] = round(s / len(sub), 4) if sub else None
        print(f"  stability on {name:10} {s}/{len(sub)} = {s/len(sub):.3f}")

    # ---------- 3. stratified 5-fold CV + leave-one-route-out ----------
    out["cv"] = cross_validate(p3, rows, folds=5)
    global _PARA
    _PARA = para
    out["cv_leave_one_route_out"] = leave_one_route_out(p3, rows)

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    op = Path(a.out_dir) / f"router_probe_{a.tag}.json"
    json.dump(out, open(op, "w"), indent=1)
    print(f"\nwrote {op}")


# ---------------------------------------------------------------------------------------
# Pattern provenance for the k-fold CV. Maps a router pattern symbol -> what justified it.
# "pre150" means the pattern predates this exercise (dev-24 / official-50 work in earlier iterations) and
# is therefore never ablated by a fold of the 150. Patterns added during the optimization are tagged with
# the question ids that motivated them, and a fold disables exactly those whose supporting
# ids all fall inside it.
# ---------------------------------------------------------------------------------------
_COMPARE_IDS = [f"SQ{n:03d}" for n in range(41, 61)]      # the 20 compare questions
_BRIDGE_IDS = [f"SQ{n:03d}" for n in range(21, 41)]       # the 20 bridge questions
_TEMPORAL_IDS = [f"SQ{n:03d}" for n in range(81, 101)]    # the 20 temporal questions

PATTERN_PROVENANCE: dict[str, list[str]] = {
    # pre-existing (earlier iterations, justified by dev-24 and the official held-out 50): no fold of the
    # 150 is what motivated these, so a fold can never ablate them. Stated as a limit of this
    # CV rather than smoothed over.
    "AGG_RE": ["pre150"],
    "BRIDGE_RE": ["pre150"],
    "COMPARE_RE": ["pre150"],
    # the optimization additions, each tagged with the questions that justify it.
    "BINARY_COMPARE_RE": _COMPARE_IDS,
    "COMPARE_METRIC_RE": _COMPARE_IDS,
    # The shared-attribute bridge guard buys paraphrase STABILITY, not routing accuracy on the
    # originals: BRIDGE_RE (pre150) already routes all 20 bridge originals correctly, so LORO
    # will show bridge unaffected when these are ablated. That is the honest reading.
    "BRIDGE_SHARED_RE": _BRIDGE_IDS,
    "BRIDGE_REL_RE": _BRIDGE_IDS,
    # TEMPORAL_RE is the initial implementation core and predates the 150; only the optimization extension is attributable
    # to the temporal questions. Splitting them (see the pipeline) is what lets this CV report
    # what the optimization contributed rather than crediting it with pre-existing work.
    "TEMPORAL_RE": ["pre150"],
    "TEMPORAL_EXT_RE": _TEMPORAL_IDS,
}


def cross_validate(p3, rows: list[dict], folds: int = 5) -> dict:
    """Stratified k-fold over the labelled questions. For each fold, any pattern whose entire
    justification lies inside that fold is neutralized (made to never match), the router is
    re-scored on the fold, and the fold accuracies are averaged. This is the honest
    generalization estimate for a deterministic pattern set."""
    import random, re as _re
    lab = [r for r in rows if r.get("route_gold")]
    if not lab:
        return {"note": "no route_gold labels available", "folds": folds}
    strata: dict = defaultdict(list)
    for r in lab:
        strata[r.get("shape") or r["route_gold"]].append(r)
    rng = random.Random(20261005)
    assign: dict[str, int] = {}
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda r: r["id"])
        rng.shuffle(pool)
        for i, r in enumerate(pool):
            assign[r["id"]] = i % folds

    NEVER = _re.compile(r"(?!x)x")        # matches nothing
    results = []
    for f in range(folds):
        test = [r for r in lab if assign[r["id"]] == f]
        disabled = []
        saved = {}
        for sym, prov in PATTERN_PROVENANCE.items():
            if "pre150" in prov:
                continue
            support_outside = [q for q in prov if assign.get(q, -1) != f]
            if not support_outside:          # justified only by this fold -> ablate
                saved[sym] = getattr(p3, sym)
                setattr(p3, sym, NEVER)
                disabled.append(sym)
        correct = sum(1 for r in test if p3.route(r["question"]) == r["route_gold"])
        for sym, pat in saved.items():
            setattr(p3, sym, pat)
        results.append({"fold": f, "n": len(test), "correct": correct,
                        "acc": round(correct / len(test), 4), "patterns_disabled": disabled})
    mean = sum(r["acc"] for r in results) / len(results)
    print(f"\n=== 3. stratified {folds}-fold router CV : mean acc {mean:.3f} ===")
    if not any(r["patterns_disabled"] for r in results):
        print("  NOTE: no pattern was ablated in any fold — every the optimization pattern's support spans")
        print("  multiple folds, so this number is in-sample by construction. Read 3b instead.")
    for r in results:
        print(f"  fold {r['fold']}  n={r['n']:3d}  acc={r['acc']:.3f}"
              + (f"  (disabled: {', '.join(r['patterns_disabled'])})" if r["patterns_disabled"] else ""))
    return {"folds": folds, "mean_acc": round(mean, 4), "per_fold": results,
            "pattern_provenance": PATTERN_PROVENANCE}


_PARA: list[dict] = []


def leave_one_route_out(p3, rows: list[dict]) -> dict:
    """The informative generalization test, and the harsher one.

    Stratified k-fold is near-vacuous for a pattern supported by many questions: if those
    questions land in every fold, the pattern is justified by the training folds of EVERY fold
    and is never ablated, so the CV just reproduces in-sample accuracy. Leave-one-route-out
    asks the question that actually worries a reviewer: *if we had never seen a question of
    this route, would the router still handle it?* A whole route is held out, every pattern
    whose justification lies entirely within that route is neutralized, and the route is scored.
    """
    import re as _re
    lab = [r for r in rows if r.get("route_gold")]
    if not lab:
        return {"note": "no route_gold labels available"}
    NEVER = _re.compile(r"(?!x)x")
    routes = sorted({r["route_gold"] for r in lab})
    gold_route_of = {r["id"]: r["route_gold"] for r in lab}
    results = []
    for held in routes:
        test = [r for r in lab if r["route_gold"] == held]
        saved, disabled = {}, []
        for sym, prov in PATTERN_PROVENANCE.items():
            if "pre150" in prov:
                continue
            # support that lies OUTSIDE the held-out route
            if not [q for q in prov if gold_route_of.get(q, "?") != held]:
                saved[sym] = getattr(p3, sym)
                setattr(p3, sym, NEVER)
                disabled.append(sym)
        correct = sum(1 for r in test if p3.route(r["question"]) == held)
        # Stability under the same ablation. This is where the optimization broadenings actually earn
        # their keep: they were added for robustness to phrasing, not to fix a misroute on the
        # originals, so accuracy alone understates them.
        tids = {r["id"] for r in test}
        tpara = [pp for pp in _PARA if pp["orig_id"] in tids]
        pstable = sum(1 for pp in tpara if p3.route(pp["question"]) == held)
        for sym, pat in saved.items():
            setattr(p3, sym, pat)
        results.append({"held_out_route": held, "n": len(test), "correct": correct,
                        "acc": round(correct / len(test), 4), "patterns_disabled": disabled,
                        "n_para": len(tpara), "para_stable": pstable,
                        "para_rate": round(pstable / len(tpara), 4) if tpara else None})
    mean = sum(r["acc"] for r in results) / len(results)
    print(f"\n=== 3b. leave-one-route-out : mean acc {mean:.3f} ===")
    for r in results:
        print(f"  held out {r['held_out_route']:12} n={r['n']:3d} acc={r['acc']:.3f} "
              f"para_stability={r['para_rate'] if r['para_rate'] is not None else float('nan'):.3f}"
              + (f"  (disabled: {', '.join(r['patterns_disabled'])})" if r["patterns_disabled"] else "  (nothing ablated)"))
    return {"mean_acc": round(mean, 4), "per_route": results}


if __name__ == "__main__":
    main()
