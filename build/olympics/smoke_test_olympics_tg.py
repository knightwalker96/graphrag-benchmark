"""Post-ingest smoke test of the TigerGraph `olympics` graph.

Sibling of `build/kg/smoke_test_tg.py`, which does the same for `finance`.

Every check is a KNOWN-ANSWER FIXTURE taken from the public question set: the expected value is
written here, and the test fails if the graph disagrees. That is stronger than "the query
returned something", which is what a smoke test usually settles for and which would have passed
even while the SEC bridge branch was returning nothing useful for 20 questions.

The headline fixture is the one named in the plan:
    event_by_venue_date("Olympic Weightlifting Gymnasium", "20 September 1988")
    -> Weightlifting at the 1988 Summer Olympics – Men's 60 kg  (Q25239316)
    -> gold: Naim Süleymanoğlu          <- pub-005's gold answer

Uses installed queries only. getVertexCount is unreliable on this Savanna (stale stats), so
nothing here depends on it.

    python "build/olympics/smoke_test_olympics_tg.py"
    python "build/olympics/smoke_test_olympics_tg.py" --skip-vectors
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline3_olympics"))
from dotenv import load_dotenv
load_dotenv(str(ROOT / ".env"))
import pyTigerGraph as tg  # noqa: E402
import schema as S  # noqa: E402

GRAPH = "olympics"


def rows_of(res):
    """Flatten a PRINT <vertexSet> result into attribute dicts."""
    out = []
    for block in res or []:
        for key, val in block.items():
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, dict) and "attributes" in v:
                        out.append(v["attributes"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-vectors", action="store_true")
    a = ap.parse_args()

    conn = tg.TigerGraphConnection(host=os.getenv("TIGERGRAPH_HOST"), graphname=GRAPH,
                                   gsqlSecret=os.getenv("TIGERGRAPH_SECRET") or None,
                                   username=os.getenv("TIGERGRAPH_USERNAME"),
                                   password=os.getenv("TIGERGRAPH_PASSWORD"))
    conn.getToken(os.getenv("TIGERGRAPH_SECRET"))
    failures = []

    def check(label, fn, expect):
        try:
            got = fn()
        except Exception as e:
            failures.append(label)
            print(f"  ✗ {label}\n      ERROR {type(e).__name__}: {str(e)[:160]}")
            return
        ok = got == expect
        if not ok:
            failures.append(label)
        print(f"  {'✓' if ok else '✗'} {label}")
        print(f"      got      {got!r}")
        if not ok:
            print(f"      expected {expect!r}")

    print("\n=== multi_hop branch: event_by_venue_date (the plan's named fixture) ===")
    def mh():
        r = conn.runInstalledQuery("event_by_venue_date", {
            "venue_key": S.norm_venue("Olympic Weightlifting Gymnasium"),
            "date_key": S.norm_date("20 September 1988")})
        return sorted((e["doc_id"], e["gold_medal"]) for e in rows_of(r))
    check("event_by_venue_date(Olympic Weightlifting Gymnasium, 20 September 1988) [pub-005]",
          mh, [("Q25239316", "Naim Süleymanoğlu")])

    print("\n  the ambiguous key must return BOTH events, not an arbitrary winner "
          "(OLYMPICS_DATA_NOTES.md §5.1)")
    def mh_amb():
        r = conn.runInstalledQuery("event_by_venue_date", {
            "venue_key": S.norm_venue("Laura Biathlon & Ski Complex"),
            "date_key": S.norm_date("22 February 2014")})
        return sorted(e["doc_id"] for e in rows_of(r))
    check("event_by_venue_date(Laura Biathlon & Ski Complex, 22 February 2014) [pub-099]",
          mh_amb, ["Q15052014", "Q15055028"])

    print("\n  the date normalizer must bridge '15 to 22 August' / '15–22 August' [eval-001]")
    def mh_date():
        r = conn.runInstalledQuery("event_by_venue_date", {
            "venue_key": S.norm_venue("Olympic Tennis Centre"),
            "date_key": S.norm_date("15 to 22 August 2004")})
        return sorted(e["doc_id"] for e in rows_of(r))
    check("event_by_venue_date(Olympic Tennis Centre, 15 to 22 August 2004)", mh_date, ["Q942805"])

    print("\n=== lookup branch: event_by_title ===")
    def lk():
        r = conn.runInstalledQuery("event_by_title", {
            "title_key": S.norm_text("Sailing at the 2016 Summer Olympics – Women's RS:X")})
        return [(e["doc_id"], e["nations_int"]) for e in rows_of(r)]
    check("event_by_title(Sailing 2016 Women's RS:X).nations == 26 [pub-009]",
          lk, [("Q26254891", 26)])

    print("\n=== aggregation branch: events_in_group ===")
    def agg():
        r = conn.runInstalledQuery("events_in_group",
                                   {"sport_key": S.norm_text("Cycling"), "games": "2000 Summer"})
        evs = rows_of(r)
        return (len(evs), sum(1 for e in evs if e["competitors_int"] > 30))
    check("events_in_group(cycling, 2000 Summer) -> (group size, count>30) [pub-010 gold=4]",
          agg, (10, 4))

    def agg_big():
        r = conn.runInstalledQuery("events_in_group",
                                   {"sport_key": S.norm_text("Athletics"), "games": "1988 Summer"})
        evs = rows_of(r)
        return (len(evs), sum(1 for e in evs if e["competitors_int"] > 41))
    check("events_in_group(athletics, 1988 Summer) -> 38 events, 17 over 41 [pub-087]",
          agg_big, (38, 17))

    print("\n=== superlative branch: argmax over the same group ===")
    def sup():
        r = conn.runInstalledQuery("events_in_group",
                                   {"sport_key": S.norm_text("Shooting"), "games": "2008 Summer"})
        evs = [e for e in rows_of(r) if e["competitors_int"] > 0]
        return max(evs, key=lambda e: e["competitors_int"])["title"]
    check("argmax competitors in (shooting, 2008 Summer) [pub-037]",
          sup, "Shooting at the 2008 Summer Olympics – Men's 50 metre rifle prone")

    print("\n=== temporal branch: games_before, then the group ===")
    def tmp1():
        r = conn.runInstalledQuery("games_before", {"season_name": "Summer", "target_year": 2016})
        return sorted(gm["games_id"] for gm in rows_of(r))
    check("games_before(Summer, 2016) -> 2012 Summer [pub-002/007/013...]", tmp1, ["2012 Summer"])

    def tmp2():
        r = conn.runInstalledQuery("events_in_group",
                                   {"sport_key": S.norm_text("Judo"), "games": "2016 Summer"})
        return [e["gold_medal"] for e in rows_of(r) if e["event_name"] == "Women's 57 kg"]
    check("judo 2016 Women's 57 kg gold == Rafaela Silva [pub-006 hop 2]",
          tmp2, ["Rafaela Silva"])

    print("\n=== provenance: medallists_of ===")
    def med():
        r = conn.runInstalledQuery("medallists_of", {"doc_id": "Q25239316"})
        return sorted(r[0]["medals"]) if r else []
    check("medallists_of(Q25239316)", med,
          ["bronze :: Ye Huanming", "gold :: Naim Süleymanoğlu", "silver :: Stefan Topurov"])

    if not a.skip_vectors:
        print("\n=== vector_search (TigerGraph as the vector DB) ===")
        try:
            from embeddings.embedder import get_embedder
            qv = get_embedder().embed_query(
                "How many nations competed in Sailing at the 2016 Summer Olympics – Women's RS:X?")
            r = conn.runInstalledQuery("vector_search", {"query_vec": qv, "k": 5})
            ids = [v.get("v_id") for row in r for v in (row.get("vs") or [])]
            hit = any(str(i).startswith("Q26254891") for i in ids)
            print(f"  {'✓' if hit else '✗'} top-5 chunk ids: {ids}")
            if not hit:
                failures.append("vector_search")
        except Exception as e:
            failures.append("vector_search")
            print(f"  ✗ vector_search ERROR {str(e)[:160]}")

    print("\n" + ("=" * 66))
    if failures:
        print(f"⚠  {len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print(f"✅ TigerGraph '{GRAPH}' READY — every installed query matched its known answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
