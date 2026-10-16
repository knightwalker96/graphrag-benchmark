"""Assert that every multi-query P3 route actually issued ALL of its hops.

`pipeline3_olympics/pipeline.py` cites this file as the check on Task-1 defect #1: on the SEC
dataset the `bridge` route resolved its first hop, silently never issued the second, and answered
"not supported by the provided context" — scoring 1/20 while the equivalent question answered at
set-F1 1.000 through another route. Nothing in that run would have revealed it, because a route
that half-works looks like a route that works.

So each two-query branch records `hop_log`, and this script fails loudly if any question on a
multi-hop route is missing a hop. It reads benchmark output only — no LLM calls, no graph calls.

Usage:
    python scripts/verify_branch_hops.py results/olympics/bench_olympics_test_held70_run1.json
    python scripts/verify_branch_hops.py results/olympics/bench_olympics_dev_frozen_run1.json
"""
from __future__ import annotations
import argparse
import collections
import json
import sys

# route -> the queries it MUST issue, in order. Single-query routes are listed too, so a branch
# that silently stopped issuing its one query is caught as well.
EXPECTED = {
    "lookup":      ["event_by_title"],
    "aggregation": ["events_in_group"],
    "superlative": ["events_in_group"],
    "temporal":    ["games_before", "events_in_group"],
    "multi_hop":   ["event_by_venue_date", "medallists_of"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", help="bench_*.json from run_olympics_benchmark.py")
    ap.add_argument("--pipeline", default="graphrag")
    a = ap.parse_args()

    rows = json.load(open(a.bench))["pipelines"][a.pipeline]
    bad, counts = [], collections.Counter()
    for r in rows:
        route, hops = r.get("route"), list(r.get("hop_log") or [])
        exp = EXPECTED.get(route)
        counts[(route, tuple(hops))] += 1
        if exp is None:                      # "default" / vector fallback has no fixed contract
            continue
        missing = [q for q in exp if q not in hops]
        if missing:
            bad.append((r.get("id"), route, hops, missing))
        if not r.get("branch_resolved") and r.get("tg_queries", 0) == 0:
            bad.append((r.get("id"), route, hops, ["branch did not resolve and issued no query"]))

    print(f"{a.bench}\n{len(rows)} rows on pipeline '{a.pipeline}'\n")
    print(f"{'route':12} {'hops issued':46} {'n':>4}")
    for (route, hops), n in sorted(counts.items()):
        print(f"{route or '-':12} {('+'.join(hops) or '(none)'):46} {n:>4}")

    multi = {k: v for k, v in EXPECTED.items() if len(v) > 1}
    n_multi = sum(n for (route, _), n in counts.items() if route in multi)
    print(f"\nmulti-hop routes ({', '.join(multi)}): {n_multi} questions")

    if bad:
        print(f"\n{len(bad)} FAILURE(S) — a route did not issue every hop:")
        for qid, route, hops, miss in bad[:25]:
            print(f"  {qid}  route={route}  issued={hops}  missing={miss}")
        sys.exit(1)
    print("PASS — every route issued all of its expected queries")


if __name__ == "__main__":
    main()
