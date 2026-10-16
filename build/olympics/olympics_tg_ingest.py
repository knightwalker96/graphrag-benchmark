"""Olympics TigerGraph Savanna ingest — a SEPARATE `olympics` graph, plus the vector index.

Satisfies the PS deliverable on this dataset too: TigerGraph is BOTH the graph DB and the
vector DB. Modelled on `build/kg/tg_ingest.py` — same connection pattern, same
raw-REST upsert path, same "verify with live queries, never with getVertexCount" discipline.

DOES NOT TOUCH `finance`. Every type name is suffixed `_olympics` and the graph is created
separately, so the SEC corpus graph is untouched and both coexist in the shared workspace.

Schema (graph = `olympics`)
  Event_olympics   (doc_id=QID, title, title_key, sport, event_name, games_id, year, season,
                    venue, venue_key, event_date, date_key, competitors/_int, nations/_int,
                    win_value, win_label, gold/silver/bronze + NOCs, url)
  Games_olympics   (games_id, year, season, n_events)
  Sport_olympics   (sport_name)
  Venue_olympics   (venue_key, venue_name)      PRIMARY_ID is the NORMALIZED key
  Nation_olympics  (noc)
  Athlete_olympics (athlete_name)               RAW medal string, never split — see schema.py
  Chunk_olympics   (chunk_id, doc_id, title, text, emb VECTOR)   <- the vector DB

  event_at_games / event_in_sport / event_at_venue / won_medal / represents /
  medal_for_nation / games_precedes / mentions

WHY ONE EDGE TYPE PER MEDAL
  TigerGraph keys an edge by (from, type, to), so two rows sharing that triple collapse into
  one edge and the later row's attributes overwrite the earlier one's — silently. Measured on
  this graph: 524 (NOC, event) pairs win MORE THAN ONE medal in the same event (USA taking gold
  and silver in one final, say). With `medal` as a plain attribute, 576 medal_for_nation rows
  vanished and the survivors carried whichever medal happened to be written last — a
  plausible-looking wrong value of exactly the kind this project has twice had to revert
  (RESULTS_SEC.md and the reverted events.json feature). DISCRIMINATOR(medal)
  makes the medal part of the edge identity, so all three medals coexist.
  `represents` drops `medal` entirely: that edge means "competed for this nation", and carrying
  a medal on it was meaningless (1,449 of its rows were duplicates of the same athlete-nation
  pair, one per medal they won).

Installed queries — one per P3 branch, so each branch's cost is a real graph round-trip
  event_by_title(title_key)                  lookup
  events_in_group(sport_name, games)         aggregation + superlative + temporal hop 2
  event_by_venue_date(venue_key, date_key)   multi_hop
  games_before(season_name, target_year)     temporal hop 1 — traverses games_precedes
  medallists_of(doc_id)                      medal rows for provenance sentences
  vector_search(query_vec, k)                default route / TG as vector DB

Usage
  DATASET=olympics EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    python "build/olympics/olympics_tg_ingest.py" --fresh
Flags: --fresh, --skip-schema, --skip-vectors, --vectors-only, --queries-only, --limit N

NOTE ON KEYS. venue_key / date_key / title_key are written here with the SAME normalizers the
pipeline will query with (`pipeline3_olympics/schema.py`). Two copies of norm_date() would be
the exact drift bug schema.py exists to prevent, so this module imports them rather than
reimplementing them.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import pyTigerGraph as tg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline3_olympics"))
load_dotenv(str(ROOT / ".env"))
import schema as S  # noqa: E402

TG_HOST = os.getenv("TIGERGRAPH_HOST", "")
TG_USER = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASS = os.getenv("TIGERGRAPH_PASSWORD", "")
TG_SECRET = os.getenv("TIGERGRAPH_SECRET", "")

# Hard-coded, NOT read from TIGERGRAPH_GRAPH: that variable is 'finance' in .env and a typo
# there must never let this script write into the SEC corpus graph.
g = "olympics"

INDEX_DIR = ROOT / "pipeline3_olympics/index"
CORPUS = ROOT / "datasets/olympics/corpus.jsonl"
EMB_DIM = int(os.getenv("EMB_DIM", "384"))

V_EVENT, V_GAMES, V_SPORT = f"Event_{g}", f"Games_{g}", f"Sport_{g}"
V_VENUE, V_NATION, V_ATHLETE, V_CHUNK = f"Venue_{g}", f"Nation_{g}", f"Athlete_{g}", f"Chunk_{g}"
E_AT_GAMES, E_IN_SPORT, E_AT_VENUE = f"event_at_games_{g}", f"event_in_sport_{g}", f"event_at_venue_{g}"
E_REPRESENTS = f"represents_{g}"
# One edge type PER MEDAL COLOUR, mirroring the SEC graph's one-type-per-relation design.
# See "WHY ONE EDGE TYPE PER MEDAL" in the module docstring.
E_WON = {m: f"won_{m}_{g}" for m in ("gold", "silver", "bronze")}
E_MEDAL_NAT = {m: f"{m}_for_nation_{g}" for m in ("gold", "silver", "bronze")}
E_PRECEDES, E_MENTIONS = f"games_precedes_{g}", f"mentions_{g}"

VERT_BATCH, CHUNK_BATCH, EDGE_BATCH, RETRIES = 1000, 400, 1000, 8

FMT = dict(graph=g, ev=V_EVENT, ga=V_GAMES, sp=V_SPORT, ve=V_VENUE, na=V_NATION,
           at=V_ATHLETE, ch=V_CHUNK, e_games=E_AT_GAMES, e_sport=E_IN_SPORT,
           e_venue=E_AT_VENUE, e_rep=E_REPRESENTS,
           e_prec=E_PRECEDES, e_men=E_MENTIONS, dim=EMB_DIM,
           **{f"e_won_{m}": E_WON[m] for m in E_WON},
           **{f"e_{m}_nat": E_MEDAL_NAT[m] for m in E_MEDAL_NAT})

SCHEMA = """USE GLOBAL
CREATE VERTEX {ev} (PRIMARY_ID doc_id STRING, title STRING DEFAULT "", title_key STRING DEFAULT "", sport STRING DEFAULT "", event_name STRING DEFAULT "", games_id STRING DEFAULT "", year INT DEFAULT 0, season STRING DEFAULT "", venue STRING DEFAULT "", venue_key STRING DEFAULT "", event_date STRING DEFAULT "", date_key STRING DEFAULT "", competitors STRING DEFAULT "", competitors_int INT DEFAULT -1, nations STRING DEFAULT "", nations_int INT DEFAULT -1, win_value STRING DEFAULT "", win_label STRING DEFAULT "", gold_medal STRING DEFAULT "", gold_noc STRING DEFAULT "", silver_medal STRING DEFAULT "", silver_noc STRING DEFAULT "", bronze_medal STRING DEFAULT "", bronze_noc STRING DEFAULT "", url STRING DEFAULT "") WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"
CREATE VERTEX {ga} (PRIMARY_ID games_id STRING, year INT DEFAULT 0, season STRING DEFAULT "", n_events INT DEFAULT 0) WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"
CREATE VERTEX {sp} (PRIMARY_ID sport_name STRING, sport_key STRING DEFAULT "") WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"
CREATE VERTEX {ve} (PRIMARY_ID venue_key STRING, venue_name STRING DEFAULT "") WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"
CREATE VERTEX {na} (PRIMARY_ID noc STRING) WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"
CREATE VERTEX {at} (PRIMARY_ID athlete_name STRING) WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"
CREATE VERTEX {ch} (PRIMARY_ID chunk_id STRING, doc_id STRING DEFAULT "", title STRING DEFAULT "", text STRING DEFAULT "") WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"
CREATE DIRECTED EDGE {e_games} (FROM {ev}, TO {ga})
CREATE DIRECTED EDGE {e_sport} (FROM {ev}, TO {sp})
CREATE DIRECTED EDGE {e_venue} (FROM {ev}, TO {ve})
CREATE DIRECTED EDGE {e_won_gold} (FROM {at}, TO {ev})
CREATE DIRECTED EDGE {e_won_silver} (FROM {at}, TO {ev})
CREATE DIRECTED EDGE {e_won_bronze} (FROM {at}, TO {ev})
CREATE DIRECTED EDGE {e_rep} (FROM {at}, TO {na})
CREATE DIRECTED EDGE {e_gold_nat} (FROM {na}, TO {ev})
CREATE DIRECTED EDGE {e_silver_nat} (FROM {na}, TO {ev})
CREATE DIRECTED EDGE {e_bronze_nat} (FROM {na}, TO {ev})
CREATE DIRECTED EDGE {e_prec} (FROM {ga}, TO {ga}, season STRING DEFAULT "")
CREATE UNDIRECTED EDGE {e_men} (FROM {ev}, TO {ch})
CREATE GRAPH {graph}({ev}, {ga}, {sp}, {ve}, {na}, {at}, {ch}, {e_games}, {e_sport}, {e_venue}, {e_won_gold}, {e_won_silver}, {e_won_bronze}, {e_rep}, {e_gold_nat}, {e_silver_nat}, {e_bronze_nat}, {e_prec}, {e_men})
"""

VECTOR_JOB = """USE GLOBAL
DROP JOB add_emb_{graph}
CREATE GLOBAL SCHEMA_CHANGE JOB add_emb_{graph} {{
  ALTER VERTEX {ch} ADD VECTOR ATTRIBUTE emb(DIMENSION={dim}, METRIC="COSINE");
}}
RUN GLOBAL SCHEMA_CHANGE JOB add_emb_{graph}
"""


def connect() -> tg.TigerGraphConnection:
    conn = tg.TigerGraphConnection(host=TG_HOST, graphname=g, gsqlSecret=TG_SECRET or None,
                                   username=TG_USER, password=TG_PASS)
    last = None
    for _ in range(5):
        try:
            conn.getToken(TG_SECRET)
            return conn
        except Exception as e:
            last = e
            time.sleep(3)
    # Fail loud: a suspended Savanna instance must not look like a code bug, and P3 must
    # never silently fall back to a local index (that was a reported defect in the SEC corpus).
    raise RuntimeError(
        f"TigerGraph auth failed after 5 attempts: {last}. If this is a 500 on the token "
        f"endpoint, the Savanna instance is suspended — resume it in the console and re-run.")


def _gsql(conn, stmt: str, what: str = "") -> str:
    for attempt in range(4):
        try:
            return str(conn.gsql(stmt))
        except Exception as e:
            if attempt == 3:
                print(f"  gsql {what} error: {str(e)[:200]}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def graph_exists(conn) -> bool:
    """Is the `olympics` graph present? Checked with LS GLOBAL, which never depends on a USE."""
    out = _gsql(conn, "USE GLOBAL\nLS", "ls")
    return f"Graph {g}(" in out


def drop_graph(conn):
    """Drop ONLY the olympics graph and its own types.

    DANGEROUS-COMMAND GUARD. The first version of this function opened with

        USE GRAPH olympics
        DROP QUERY ALL

    as a single gsql statement. On a fresh instance the graph does not exist yet, so `USE GRAPH
    olympics` FAILED — and gsql went on to execute `DROP QUERY ALL` in the session's fallback
    context, which deleted every installed query on the `finance` graph. The SEC corpus data,
    schema and vector index all survived, but all 13 of its queries had to be reinstalled from
    `tg_ingest.py --queries-only`.

    The lesson generalizes beyond this script: a destructive statement must never ride in the
    same gsql call as the scoping statement that is supposed to constrain it. So `DROP QUERY
    ALL` is now issued only after LS GLOBAL confirms the graph exists, and it is skipped
    entirely otherwise.
    """
    print(f"Dropping graph {g!r} (does not touch 'finance')...")
    steps = []
    if graph_exists(conn):
        steps.append(("queries", f"USE GRAPH {g}\nDROP QUERY ALL"))
    else:
        print(f"  graph {g!r} does not exist — SKIPPING 'DROP QUERY ALL' "
              f"(issuing it without a valid USE would drop another graph's queries)")
    steps += [("job", f"USE GLOBAL\nDROP JOB add_emb_{g}"),
              ("graph", f"USE GLOBAL\nDROP GRAPH {g}")]
    # Edge type names this script has EVER created, not just the ones it creates now. A renamed
    # edge type that is left behind still references Event_olympics/Athlete_olympics, which
    # makes DROP VERTEX fail with "used by another object" — and because _gsql swallows errors,
    # the next CREATE then fails too and the whole rebuild proceeds against a half-dropped
    # schema. Found exactly that way when `won_medal_olympics` was split per medal colour.
    LEGACY_EDGES = [f"won_medal_{g}", f"medal_for_nation_{g}"]
    for e in ([E_AT_GAMES, E_IN_SPORT, E_AT_VENUE, E_REPRESENTS, E_PRECEDES, E_MENTIONS]
              + list(E_WON.values()) + list(E_MEDAL_NAT.values()) + LEGACY_EDGES):
        steps.append((f"e_{e}", f"USE GLOBAL\nDROP EDGE {e}"))
    for v in (V_EVENT, V_GAMES, V_SPORT, V_VENUE, V_NATION, V_ATHLETE, V_CHUNK):
        steps.append((f"v_{v}", f"USE GLOBAL\nDROP VERTEX {v}"))
    for label, stmt in steps:
        _gsql(conn, stmt, f"drop {label}")
    print("  dropped.")


def create_schema(conn):
    print("Creating schema...")
    out = _gsql(conn, SCHEMA.format(**FMT), "schema")
    print("  " + out[:300])
    # Fail loud. A failed CREATE used to be printed and then ignored, so the loaders ran
    # against a half-dropped schema and reported healthy "accepted" counts for data that was
    # going nowhere useful. A broken rebuild must stop here, not three steps later.
    if "Semantic Check Fails" in out or "Failed to create vertex types" in out:
        raise RuntimeError(
            "Schema creation FAILED — see the GSQL output above. The usual cause is a leftover "
            "type from a previous schema version that DROP did not remove; add it to "
            "LEGACY_EDGES in drop_graph(). Refusing to load data into a half-built schema.")
    print("Adding vector attribute...")
    print("  " + _gsql(conn, VECTOR_JOB.format(**FMT), "vector attr")[:200])


# ---------------- queries: one per branch ----------------
def install_queries(conn):
    print("Installing queries...")
    q = {}

    # lookup — one attribute read on one Event, found by its normalized title.
    q["event_by_title"] = """USE GRAPH {graph}
CREATE OR REPLACE QUERY event_by_title(STRING title_key) SYNTAX v3 {{
  evs = SELECT s FROM (s:{ev}) WHERE s.title_key == title_key;
  PRINT evs;
}}
INSTALL QUERY event_by_title"""

    # aggregation + superlative + temporal hop 2 — the (sport, games) group. Traverses
    # event_in_sport so the grouping is done by the graph, not by a scan of an attribute.
    q["events_in_group"] = """USE GRAPH {graph}
CREATE OR REPLACE QUERY events_in_group(STRING sport_key, STRING games) SYNTAX v3 {{
  evs = SELECT s FROM (s:{ev})-[e:{e_sport}]->(t:{sp})
        WHERE t.sport_key == sport_key AND s.games_id == games;
  PRINT evs;
}}
INSTALL QUERY events_in_group"""

    # multi_hop — join two attributes to identify the event. Traverses event_at_venue.
    # Deliberately returns ALL matches: 357 of 1,574 venue+date keys name more than one
    # event, and the branch must see that rather than be handed an arbitrary winner
    # (OLYMPICS_DATA_NOTES.md §5.1).
    q["event_by_venue_date"] = """USE GRAPH {graph}
CREATE OR REPLACE QUERY event_by_venue_date(STRING venue_key, STRING date_key) SYNTAX v3 {{
  evs = SELECT s FROM (s:{ev})-[e:{e_venue}]->(v:{ve})
        WHERE v.venue_key == venue_key AND s.date_key == date_key;
  PRINT evs;
}}
INSTALL QUERY event_by_venue_date"""

    # temporal hop 1 — "the Games held immediately before <year>" is literally one
    # games_precedes edge. This is the query that makes temporal a graph question rather
    # than a date comparison.
    q["games_before"] = """USE GRAPH {graph}
CREATE OR REPLACE QUERY games_before(STRING season_name, INT target_year) SYNTAX v3 {{
  prev = SELECT p FROM (p:{ga})-[e:{e_prec}]->(t:{ga})
         WHERE t.year == target_year AND t.season == season_name;
  PRINT prev;
}}
INSTALL QUERY games_before"""

    # medal rows for provenance sentences, via the Athlete and Nation edges.
    q["medallists_of"] = ("""USE GRAPH {graph}
CREATE OR REPLACE QUERY medallists_of(STRING doc_id) SYNTAX v3 {{
  SetAccum<STRING> @@medals;
  SetAccum<STRING> @@nocs;
""" + "".join(
        f"  a_{m} = SELECT s FROM (s:{{at}})-[e:{E_WON[m]}]->(t:{{ev}}) WHERE t.doc_id == doc_id "
        f'ACCUM @@medals += ("{m} :: " + s.athlete_name);\n'
        f"  n_{m} = SELECT s FROM (s:{{na}})-[e:{E_MEDAL_NAT[m]}]->(t:{{ev}}) WHERE t.doc_id == doc_id "
        f'ACCUM @@nocs += ("{m} :: " + s.noc);\n'
        for m in ("gold", "silver", "bronze")) + """  PRINT @@medals AS medals, @@nocs AS nocs;
}}
INSTALL QUERY medallists_of""")

    # default route / the PS "TigerGraph is also the vector DB" deliverable.
    q["vector_search"] = """USE GRAPH {graph}
CREATE OR REPLACE QUERY vector_search(LIST<float> query_vec, INT k=10) SYNTAX v3 {{
  vs = vectorSearch({{{ch}.emb}}, query_vec, k);
  PRINT vs WITH VECTOR;
}}
INSTALL QUERY vector_search"""

    for name, stmt in q.items():
        r = _gsql(conn, stmt.format(**FMT), f"install {name}")
        ok = any(w in r.lower() for w in ("installed", "finished", "success", "added"))
        print(f"  {name}: {'OK' if ok else r[:160]}")


# ---------------- loading (raw REST; the dict form silently no-ops on this Savanna) --------
def _headers(conn):
    return {"Authorization": f"Bearer {conn.getToken(TG_SECRET)[0]}"}


def _post(url, payload, headers, what):
    for attempt in range(RETRIES):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            if r.status_code == 200 and not r.json().get("error", True):
                res = r.json().get("results", [{}])[0]
                return res.get("accepted_vertices", 0) + res.get("accepted_edges", 0)
            if attempt == RETRIES - 1:
                print(f"    {what} failed: {str(r.text)[:160]}")
        except Exception as e:
            if attempt == RETRIES - 1:
                print(f"    {what} exc: {str(e)[:120]}")
        time.sleep(2 ** attempt + 0.5)
    return 0


def _v(x):
    return {"value": x}


def load_vertices(conn):
    events = json.load(open(INDEX_DIR / "events.json"))
    games = json.load(open(INDEX_DIR / "games.json"))
    url, headers = f"{TG_HOST}/restpp/graph/{g}", _headers(conn)
    total = 0

    print(f"Loading {len(events):,} Event vertices...")
    ids = list(events)
    for i in range(0, len(ids), VERT_BATCH):
        verts = {}
        for did in ids[i:i + VERT_BATCH]:
            e = events[did]
            ib = e["infobox"]
            verts[did] = {
                "title": _v(e["title"]), "title_key": _v(S.norm_text(e["title"])),
                "sport": _v(e["sport"]), "event_name": _v(e["event_name"]),
                "games_id": _v(e["games_id"]), "year": _v(e["year"]), "season": _v(e["season"]),
                "venue": _v(e["venue"]), "venue_key": _v(S.norm_venue(e["venue"])),
                "event_date": _v(e["event_date"]), "date_key": _v(S.norm_date(e["event_date"])),
                "competitors": _v(e["competitors"]), "competitors_int": _v(e["competitors_int"]),
                "nations": _v(e["nations"]), "nations_int": _v(e["nations_int"]),
                "win_value": _v(e["win_value"]), "win_label": _v(e["win_label"]),
                "gold_medal": _v(ib.get("gold", "")), "gold_noc": _v(ib.get("goldNOC", "")),
                "silver_medal": _v(ib.get("silver", "")), "silver_noc": _v(ib.get("silverNOC", "")),
                "bronze_medal": _v(ib.get("bronze", "")), "bronze_noc": _v(ib.get("bronzeNOC", "")),
                "url": _v(e["url"])}
        total += _post(url, {"vertices": {V_EVENT: verts}}, headers, "events")
    print(f"  ✓ {total:,} events")

    print(f"Loading {len(games)} Games vertices...")
    gv = {k: {"year": _v(v["year"]), "season": _v(v["season"]), "n_events": _v(v["n_events"])}
          for k, v in games.items()}
    print(f"  ✓ {_post(url, {'vertices': {V_GAMES: gv}}, headers, 'games'):,} games")

    sports = sorted({e["sport"] for e in events.values()})
    sv = {s: {"sport_key": _v(S.norm_text(s))} for s in sports}
    print(f"  ✓ {_post(url, {'vertices': {V_SPORT: sv}}, headers, 'sports'):,} sports")

    venues = {}
    for e in events.values():
        if e["venue"]:
            venues[S.norm_venue(e["venue"])] = {"venue_name": _v(e["venue"])}
    print(f"  ✓ {_post(url, {'vertices': {V_VENUE: venues}}, headers, 'venues'):,} venues")

    nations = json.load(open(INDEX_DIR / "nations.json"))
    nv = {n: {} for n in nations}
    print(f"  ✓ {_post(url, {'vertices': {V_NATION: nv}}, headers, 'nations'):,} nations")

    edges = json.load(open(INDEX_DIR / "edges_prov.json"))
    athletes = sorted({e["source"] for e in edges if e["relation"] == "won_medal"})
    n = 0
    for i in range(0, len(athletes), VERT_BATCH):
        n += _post(url, {"vertices": {V_ATHLETE: {a: {} for a in athletes[i:i + VERT_BATCH]}}},
                   headers, "athletes")
    print(f"  ✓ {n:,} athletes")


def load_edges(conn):
    edges = json.load(open(INDEX_DIR / "edges_prov.json"))
    url, headers = f"{TG_HOST}/restpp/graph/{g}", _headers(conn)

    # relation -> (edge type, from vertex, to vertex, attribute keys to carry)
    # (edge type, from vertex, to vertex). Medal relations fan out to one type per colour so
    # the colour is part of the EDGE IDENTITY, not an attribute that a later row overwrites.
    spec = {
        "event_at_games":   (E_AT_GAMES, V_EVENT, V_GAMES),
        "event_in_sport":   (E_IN_SPORT, V_EVENT, V_SPORT),
        "event_at_venue":   (E_AT_VENUE, V_EVENT, V_VENUE),
        "represents":       (E_REPRESENTS, V_ATHLETE, V_NATION),
        "games_precedes":   (E_PRECEDES, V_GAMES, V_GAMES),
    }
    for m in E_WON:
        spec[f"won_medal/{m}"] = (E_WON[m], V_ATHLETE, V_EVENT)
        spec[f"medal_for_nation/{m}"] = (E_MEDAL_NAT[m], V_NATION, V_EVENT)

    by_rel: dict = {k: [] for k in spec}
    for e in edges:
        rel = e["relation"]
        key = f"{rel}/{e.get('medal')}" if rel in ("won_medal", "medal_for_nation") else rel
        if key not in spec:
            continue
        tgt = e["target"]
        if rel == "event_at_venue":          # Venue PRIMARY_ID is the normalized key
            tgt = S.norm_venue(tgt)
        by_rel[key].append((e["source"], tgt, {}))

    total = 0
    for rel, (etype, fv, tv) in spec.items():
        triples = by_rel[rel]
        if not triples:
            continue
        sub = 0
        for i in range(0, len(triples), EDGE_BATCH):
            payload: dict = {}
            for s, t, a in triples[i:i + EDGE_BATCH]:
                payload.setdefault(fv, {}).setdefault(s, {}).setdefault(etype, {}) \
                       .setdefault(tv, {})[t] = {k: _v(v) for k, v in a.items()}
            sub += _post(url, {"edges": payload}, headers, etype)
        total += sub
        print(f"    {etype:28s} {sub:,}")
    print(f"  ✓ {total:,} edges accepted")


def load_chunks_and_vectors(conn, limit: int = 0, skip_vectors: bool = False):
    from embeddings.embedder import get_embedder
    chunks = [json.loads(l) for l in open(CORPUS)]
    if limit:
        chunks = chunks[:limit]
    print(f"Loading {len(chunks):,} Chunk vertices" + ("" if skip_vectors else " + vectors") + "...")
    emb = None if skip_vectors else get_embedder()
    url, headers = f"{TG_HOST}/restpp/graph/{g}", _headers(conn)
    total = 0
    men: list = []
    for i in range(0, len(chunks), CHUNK_BATCH):
        b = chunks[i:i + CHUNK_BATCH]
        vecs = emb.embed_documents([c["text"][:1000] for c in b]) if emb else [None] * len(b)
        verts = {}
        for c, v in zip(b, vecs):
            a = {"doc_id": _v(c.get("doc_id", "")), "title": _v(c.get("doc_title", "")),
                 "text": _v(c["text"][:1500])}
            if v is not None:
                a["emb"] = _v(v)
            verts[c["title"]] = a
            men.append((c.get("doc_id", ""), c["title"], {}))
        total += _post(url, {"vertices": {V_CHUNK: verts}}, headers, "chunks")
        if (i // CHUNK_BATCH) % 10 == 0:
            print(f"    {total:,}/{len(chunks):,} chunks...")
    print(f"  ✓ {total:,} chunks")

    print(f"Loading {len(men):,} event→chunk mention edges...")
    events = set(json.load(open(INDEX_DIR / "events.json")))
    men = [m for m in men if m[0] in events]
    n = 0
    for i in range(0, len(men), EDGE_BATCH):
        payload: dict = {}
        for s, t, _a in men[i:i + EDGE_BATCH]:
            payload.setdefault(V_EVENT, {}).setdefault(s, {}).setdefault(E_MENTIONS, {}) \
                   .setdefault(V_CHUNK, {})[t] = {}
        n += _post(url, {"edges": payload}, headers, "mentions")
    print(f"  ✓ {n:,} mention edges")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--skip-schema", action="store_true")
    ap.add_argument("--skip-vectors", action="store_true")
    ap.add_argument("--vectors-only", action="store_true")
    ap.add_argument("--queries-only", action="store_true")
    ap.add_argument("--edges-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    print("=" * 66)
    print(f"Olympics TG ingest — graph={g}  host={TG_HOST}")
    print("=" * 66)
    conn = connect()

    if a.queries_only:
        install_queries(conn); return
    if a.vectors_only:
        load_chunks_and_vectors(conn, a.limit); return
    if a.edges_only:
        load_edges(conn); return
    if a.fresh:
        drop_graph(conn)
    if not a.skip_schema:
        create_schema(conn)
        conn = connect()
    load_vertices(conn)
    load_edges(conn)
    load_chunks_and_vectors(conn, a.limit, a.skip_vectors)
    install_queries(conn)
    print("\nDone. Run the smoke test:")
    print('  python "build/olympics/smoke_test_olympics_tg.py"')


if __name__ == "__main__":
    main()
