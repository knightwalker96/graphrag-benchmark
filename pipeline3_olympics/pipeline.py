"""Pipeline 3 — GraphRAG over the Olympics corpus, backed by TigerGraph Savanna.

STRUCTURE IS COPIED FROM pipeline3_sec/pipeline.py AND MUST STAY THAT WAY.
What is carried over verbatim, because it is the reusable architecture the Olympics corpus is testing:

  * `_tg_query_count` + `_run_query()` — the per-question live-graph counter. `tg_live` is
    DERIVED from it (`_tg_query_count > 0`), never a session-level boolean. The session-level
    version was a real reported defect that overstated live-graph usage 1.00 -> 0.84
    (RESULTS_SEC.md §7). Do not reintroduce it.
  * `route()` and the `P3_ROUTE_FORCE` hook — fixed label or `md5(question)`-seeded random,
    keeping every branch live and degrading ONLY the label. That is the un-confounded
    ablation; `P3_ROUTER_OFF` is kept too but is the confounded one.
  * `graph_doc_ids`, `graph_entity_count`, the token accounting, `preserve_answer`.
  * `_connect()` failing LOUD when `P3_REQUIRE_TG=1`: a suspended Savanna must never look like
    a silently-degraded local run.

What is dataset-specific lives in `lexicon.py` (question patterns) and `schema.py` (vertex /
edge names, the infobox parser, the key normalizers). This module should contain no regex that
knows the word "Olympic" — the branches below parse questions with patterns imported from
`lexicon`, and look up keys with normalizers imported from `schema`.

THE SIX ROUTES, AND THE GRAPH QUERY EACH ONE ISSUES
  lookup       event_by_title            1 query   read `nations` off one Event
  superlative  events_in_group           1 query   rank the (sport, games) group by competitors
  aggregation  events_in_group           1 query   count the group members over a threshold
  temporal     games_before              2 queries hop 1: the Games edge; hop 2: that group
               + events_in_group
  multi_hop    event_by_venue_date       2 queries hop 1: identify the event by (venue, date)
               + medallists_of                     hop 2: read its medal edges
  default      vector_search             1 query   TigerGraph as the vector DB

TWO DEFECTS FROM THE SEC PIPELINE THAT THIS FILE IS WRITTEN TO AVOID
  1. THE BRIDGE BRANCH THAT NEVER MADE ITS SECOND HOP. On SEC, `bridge` scored 1/20 because it
     resolved company -> auditor and then never issued the auditor -> companies query, answering
     "not supported by the provided context" while the aggregation route answered the equivalent
     question at set-F1 1.000 (the SEC study_EXTRA100_RESULTS §5.1). The two multi-query routes here
     (temporal, multi_hop) therefore RECORD their hop count, and
     `scripts/verify_branch_hops.py` asserts on dev that both hops actually fire.
  2. SILENT LOCAL FALLBACK. Every branch tries the graph first and records whether it got there.
     A branch that fell back to the local index reports `tg_queries == 0` for that question and
     is counted honestly as not-live, rather than inheriting the session's status.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pyTigerGraph as tg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(str(ROOT / ".env"))

import config  # noqa: E402
from metrics import MetricsTimer, CallMetrics  # noqa: E402
from llm_client import get_llm_response  # noqa: E402
import schema as S  # noqa: E402
import lexicon as L  # noqa: E402

TG_HOST = os.getenv("TIGERGRAPH_HOST", "")
TG_GRAPH = "olympics"          # hard-coded: TIGERGRAPH_GRAPH is 'finance' in .env
TG_USER = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASS = os.getenv("TIGERGRAPH_PASSWORD", "")
TG_SECRET = os.getenv("TIGERGRAPH_SECRET", "")
INDEX_DIR = ROOT / "pipeline3_olympics/index"
CORPUS = ROOT / "datasets/olympics/corpus.jsonl"

ROUTER_OFF = os.getenv("P3_ROUTER_OFF", "0") != "0"
GRAPH_OFF = os.getenv("P3_GRAPH_OFF", "0") != "0"
COMPRESS_OFF = os.getenv("P3_COMPRESS_OFF", "0") != "0"
REQUIRE_TG = (os.getenv("P3_REQUIRE_TG", "1") != "0") and not GRAPH_OFF
ROUTE_FORCE = os.getenv("P3_ROUTE_FORCE", "").strip().lower()

MAX_CTX_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", str(config.CONTEXT_NUM_CHUNKS)))
MAX_CTX_CHARS = int(os.getenv("MAX_CHUNK_CHARS", str(config.CONTEXT_CHUNK_CHARS)))
VEC_K = int(os.getenv("P3_VEC_K", "10"))
LOOKUP_K = int(os.getenv("P3_LOOKUP_K", "6"))
# Cap on evidence rows emitted by the high-fanout routes. The largest real group is 43 events
# (athletics 2004/2008), so the default must exceed that or aggregation silently undercounts —
# the exact failure mode measured on P2 at step 5 (recall 1.00 and still the wrong count).
GROUP_ROW_CAP = int(os.getenv("P3_GROUP_ROW_CAP", "60"))

ROUTABLE = ["lookup", "aggregation", "superlative", "multi_hop", "temporal"]

# --- shared state (loaded once) ---
_conn: tg.TigerGraphConnection | None = None
_embedder = None
_chunk_text: dict | None = None
_events: dict | None = None
_games: dict | None = None
_sports_by_len: list | None = None
_tg_reachable = True

_tg_query_count = 0
_hop_log: list[str] = []          # which graph queries this question issued, in order


def _load():
    global _chunk_text, _events, _games, _sports_by_len, _embedder
    if _chunk_text is not None:
        return
    _chunk_text = {}
    for line in open(CORPUS):
        c = json.loads(line)
        _chunk_text[c["title"]] = c["text"]
    _events = json.load(open(INDEX_DIR / "events.json"))
    _games = json.load(open(INDEX_DIR / "games.json"))
    # Longest-first so "short-track speed skating" wins over "speed skating", and
    # "cross-country skiing" over any shorter ski sport. Order matters; length is the rule.
    _sports_by_len = sorted({e["sport"] for e in _events.values()}, key=len, reverse=True)
    from embeddings.embedder import get_embedder
    _embedder = get_embedder()
    print(f"[p3-oly] loaded {len(_chunk_text):,} chunks, {len(_events):,} events, "
          f"{len(_games)} games, {len(_sports_by_len)} sports")


def _connect() -> tg.TigerGraphConnection | None:
    global _conn, _tg_reachable
    if not _tg_reachable:
        return None
    if _conn is not None:
        return _conn
    last = None
    for attempt in range(4):
        try:
            conn = tg.TigerGraphConnection(host=TG_HOST, graphname=TG_GRAPH,
                                           gsqlSecret=TG_SECRET or None,
                                           username=TG_USER, password=TG_PASS)
            conn.getToken(TG_SECRET)
            conn.echo()
            _conn = conn
            return _conn
        except Exception as e:
            last = e
            time.sleep(2 ** attempt + 1)
    if REQUIRE_TG:
        raise RuntimeError(
            f"[p3-oly] live TigerGraph REQUIRED but unreachable after retries: {str(last)[:160]}. "
            f"A 500 on the token endpoint means the Savanna instance is suspended — resume it "
            f"rather than letting P3 quietly answer from the local index.")
    _tg_reachable = False
    return None


def _run_query(conn, name, params, retries=3):
    """runInstalledQuery with retry/backoff, counting every SUCCESSFUL live call."""
    global _tg_query_count
    last = None
    for attempt in range(retries):
        try:
            r = conn.runInstalledQuery(name, params)
            _tg_query_count += 1
            _hop_log.append(name)
            return r
        except Exception as e:
            last = e
            if "500" in str(e) or "timeout" in str(e).lower() or "reset" in str(e).lower():
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last


def _rows(res) -> list[dict]:
    """Flatten PRINT <vertexSet> into attribute dicts."""
    out = []
    for block in res or []:
        for val in block.values():
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, dict) and "attributes" in v:
                        out.append(v["attributes"])
    return out


# ---------------- question router ----------------
def route(question: str) -> str:
    """Route label. The MECHANISM is identical to the SEC router; only `lexicon` differs.

    P3_ROUTE_FORCE is the un-confounded ablation: every branch stays live and only the LABEL
    is degraded. P3_ROUTER_OFF is the old confounded arm (it forces "default", which disables
    graph traversal and context compression at the same time) and is kept only so the two can
    be compared on this dataset the way §6 of the final report compared them on SEC.
    """
    if ROUTE_FORCE:
        if ROUTE_FORCE == "random":
            h = int(hashlib.md5(question.encode()).hexdigest(), 16)
            return ROUTABLE[h % len(ROUTABLE)]
        return ROUTE_FORCE
    if ROUTER_OFF:
        return "default"
    return L.route_of(question)


# ---------------- question-side parsing (patterns live in lexicon.py) ----------------
def _sport_in(text: str) -> str | None:
    """Longest sport name occurring in the text, or None."""
    t = S.norm_text(text)
    for sp in _sports_by_len:
        if S.norm_text(sp) in t:
            return sp
    return None


# ---------------- branches ----------------
def _prov(doc_id: str, sentence: str) -> dict:
    """One evidence row. `chunk_id` is the document's FIRST chunk, which is where the infobox
    lives (build_olympics_corpus.py guarantees that), so citations resolve in the evaluator's
    chunk table and retrieval precision/recall@k can split the id on '#'."""
    return {"evidence": sentence, "chunk_id": f"{doc_id}#c0", "doc_id": doc_id}


def _event_sentence(e: dict) -> str:
    bits = [f"{e.get('title')}:"]
    if e.get("nations_int", -1) >= 0:
        bits.append(f"{e['nations_int']} nations competed;")
    if e.get("competitors_int", -1) >= 0:
        bits.append(f"{e['competitors_int']} competitors;")
    if e.get("venue"):
        bits.append(f"held at {e['venue']}")
    if e.get("event_date"):
        bits.append(f"on {e['event_date']};")
    for m in ("gold", "silver", "bronze"):
        if e.get(f"{m}_medal"):
            bits.append(f"{m}: {e[f'{m}_medal']} ({e.get(f'{m}_noc','')});")
    return " ".join(bits)


def _branch_lookup(conn, question):
    m = re.match(r'^How many (?:nations|countries|competitors|athletes) '
                 r'(?:competed|took part|participated) in (.+)\?$', question, re.I)
    if not (m and conn):
        return None
    evs = _rows(_run_query(conn, "event_by_title", {"title_key": S.norm_text(m.group(1))}))
    if not evs:
        return None
    e = evs[0]
    facts = f"{e['title']} — nations: {e['nations']}, competitors: {e['competitors']}."
    return facts, [_prov(e["doc_id"], _event_sentence(e))], [e["doc_id"]]


def _group(conn, sport, games):
    return _rows(_run_query(conn, "events_in_group",
                            {"sport_key": S.norm_text(sport), "games": games}))


def _parse_group_question(question):
    """(sport, games, threshold|None) for the two high-fanout templates."""
    m = re.search(r'how many (.+?) events at the (\d{4}) (Summer|Winter) Olympics had '
                  r'(?:more than|over|greater than|at least|fewer than|less than|under) (\d+)',
                  question, re.I)
    if m:
        return m.group(1), S.games_id(int(m.group(2)), m.group(3)), int(m.group(4))
    m = re.search(r'which (.+?) events? at the (\d{4}) (Summer|Winter) Olympics had the '
                  r'(?:highest|greatest|largest|most|lowest|fewest|smallest)', question, re.I)
    if m:
        return m.group(1), S.games_id(int(m.group(2)), m.group(3)), None
    return None, None, None


def _branch_aggregation(conn, question):
    sport, games, n = _parse_group_question(question)
    if not (sport and games and n is not None and conn):
        return None
    evs = _group(conn, sport, games)
    if not evs:
        return None
    # Non-integer competitors ("23 teams", "32 (16 pairs)") are -1 and are EXCLUDED, not
    # treated as 0 — see schema.as_int. 24 events corpus-wide are affected.
    hits = [e for e in evs if e.get("competitors_int", -1) > n]
    facts = (f"Graph traversal of the ({sport}, {games}) group found {len(evs)} events; "
             f"{len(hits)} of them had more than {n} competitors.")
    prov = [_prov(e["doc_id"], f"{e['title']}: {e['competitors_int']} competitors.")
            for e in sorted(hits, key=lambda x: -x.get("competitors_int", 0))[:GROUP_ROW_CAP]]
    return facts, prov, [e["doc_id"] for e in evs]


def _branch_superlative(conn, question):
    sport, games, n = _parse_group_question(question)
    if not (sport and games and n is None and conn):
        return None
    evs = [e for e in _group(conn, sport, games) if e.get("competitors_int", -1) >= 0]
    if not evs:
        return None
    evs.sort(key=lambda e: -e["competitors_int"])
    # The cheapest route in the system by design (TASK2_OLYMPICS_PLAN §6): (title, count)
    # pairs only, no event descriptions, no medal rows. The model performs the argmax; the
    # branch does not pre-compute the answer, so the evidence is auditable.
    facts = (f"Graph traversal of the ({sport}, {games}) group, ranked by competitors "
             f"(highest first):\n" +
             "\n".join(f"- {e['title']}: {e['competitors_int']}" for e in evs[:GROUP_ROW_CAP]))
    prov = [_prov(evs[0]["doc_id"], f"{evs[0]['title']}: {evs[0]['competitors_int']} competitors.")]
    return facts, prov, [e["doc_id"] for e in evs]


def _branch_temporal(conn, question):
    """TWO HOPS, both of which must issue their own query (the SEC corpus defect #1)."""
    m = re.match(r'^Who won the gold medal in the (.+?) event at the (Summer|Winter) Olympics '
                 r'held immediately before (\d{4})\?$', question, re.I)
    if not (m and conn):
        return None
    desc, season, year = m.group(1), m.group(2).capitalize(), int(m.group(3))

    # HOP 1 — the games_precedes edge. Not a date comparison: the question asks for the Games
    # held immediately before, which is a property of the Games set (OLYMPICS_DATA_NOTES §5.3).
    prev = _rows(_run_query(conn, "games_before",
                            {"season_name": season, "target_year": year}))
    if not prev:
        return None
    prev_games = prev[0]["games_id"]

    # HOP 2 — that Games' (sport, group), then the event within it.
    sport = _sport_in(desc)
    if not sport:
        return None
    evs = _group(conn, sport, prev_games)
    if not evs:
        return None
    want = S.norm_text(desc)
    key = lambda e: S.norm_text(e["event_name"] + " " + e["sport"])
    hit = [e for e in evs if key(e) == want] or \
          [e for e in evs if all(t in key(e) for t in want.split())]
    if not hit:
        return None
    facts = (f"The {season} Olympics held immediately before {year} were {prev_games} "
             f"(graph edge games_precedes).\n" +
             "\n".join(f"- {e['title']} — gold: {e['gold_medal']} ({e.get('gold_noc','')})"
                       for e in hit[:4]))
    return facts, [_prov(e["doc_id"], _event_sentence(e)) for e in hit[:4]], \
        [e["doc_id"] for e in hit]


def _branch_multi_hop(conn, question):
    """TWO HOPS: identify the event by (venue, date), then read its medal edges."""
    m = re.match(r'^Who won the gold medal in the event held at (.+?) on (.+?)'
                 r'(?: at the (\d{4}) (Summer|Winter) Olympics)?\?$', question, re.I)
    if not (m and conn):
        return None
    # HOP 1 — the composite key, via the event_at_venue edge.
    evs = _rows(_run_query(conn, "event_by_venue_date",
                           {"venue_key": S.norm_venue(m.group(1)),
                            "date_key": S.norm_date(m.group(2))}))
    if m.group(3):
        gid = S.games_id(int(m.group(3)), m.group(4))
        evs = [e for e in evs if e.get("games_id") == gid]
    if not evs:
        return None
    # DO NOT TIE-BREAK. 357 of 1,574 (venue, date) keys name more than one event; 2 of the 150
    # questions land on one. Handing the model every match is honest; a rule that resolves
    # pub-099 correctly by construction would be unfalsifiable (OLYMPICS_DATA_NOTES §5.1).
    lines, prov, ids = [], [], []
    for e in evs[:4]:
        # HOP 2 — the medal edges for that event.
        med = _run_query(conn, "medallists_of", {"doc_id": e["doc_id"]})
        medals = sorted(med[0].get("medals", [])) if med else []
        gold = next((x.split(" :: ", 1)[1] for x in medals if x.startswith("gold")),
                    e.get("gold_medal", ""))
        lines.append(f"- {e['title']} (held at {e.get('venue')} on {e.get('event_date')}) "
                     f"— gold: {gold}")
        prov.append(_prov(e["doc_id"], _event_sentence(e)))
        ids.append(e["doc_id"])
    note = ("" if len(evs) == 1 else
            f"\nNOTE: {len(evs)} events share that venue and date; all are listed.")
    return "Events matching that venue and date (graph traversal):\n" + "\n".join(lines) + note, \
        prov, ids


# ---------------- context assembly ----------------
def _dedup(ids):
    seen, out = set(), []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _context_from_chunks(chunk_ids):
    ids = _dedup(chunk_ids)[:MAX_CTX_CHUNKS]
    blocks = [f"[{c}]\n{(_chunk_text.get(c) or '')[:MAX_CTX_CHARS]}"
              for c in ids if _chunk_text.get(c)]
    return "\n\n---\n\n".join(blocks), ids


def _context_from_prov(prov, extra_chunks=None):
    """Compressed context: one evidence sentence per graph row instead of a whole chunk.
    This is the token win, and it is the same mechanism as the SEC pipeline's."""
    blocks, ids = [], []
    for e in prov[:GROUP_ROW_CAP]:
        ev = (e.get("evidence") or "").strip()
        if not ev:
            continue
        cid = e.get("chunk_id") or e.get("doc_id") or "graph"
        blocks.append(f"[{cid}] {ev}")
        ids.append(cid)
    for cid in (extra_chunks or []):
        if cid not in ids and len(ids) < MAX_CTX_CHUNKS:
            txt = (_chunk_text.get(cid) or "")[:MAX_CTX_CHARS]
            if txt:
                blocks.append(f"[{cid}]\n{txt}")
                ids.append(cid)
    return "\n\n---\n\n".join(blocks), ids


def _tg_vector_search(conn, question, k=VEC_K):
    if not conn:
        return []
    qv = _embedder.embed_query(question)
    r = _run_query(conn, "vector_search", {"query_vec": qv, "k": k})
    out = []
    for row in r or []:
        for v in (row.get("vs") or []):
            if v.get("v_id"):
                out.append(v["v_id"])
    return out


def _build_prompt(question, context, graph_facts=""):
    gf = f"Knowledge-graph facts:\n{graph_facts}\n\n" if graph_facts else ""
    return f"{config.SYSTEM_PROMPT}\n{gf}Context:\n{context}\n\nQuestion: {question}\nAnswer:"


BRANCHES = {
    "lookup": _branch_lookup,
    "aggregation": _branch_aggregation,
    "superlative": _branch_superlative,
    "temporal": _branch_temporal,
    "multi_hop": _branch_multi_hop,
}


# ---------------- main entry ----------------
def run_single(question: str, vectorstore=None, num_hops: int = 2) -> CallMetrics:
    _load()
    t0 = time.perf_counter()
    global _tg_query_count, _hop_log
    _tg_query_count = 0
    _hop_log = []
    r = route(question)
    conn = None if GRAPH_OFF else _connect()
    graph_facts, graph_doc_ids, optimizer_tokens = "", [], 0
    resolved = False
    context, ids = "", []

    branch = BRANCHES.get(r)
    if branch and not GRAPH_OFF:
        try:
            out = branch(conn, question)
        except Exception as e:
            print(f"  [p3-oly] {r} branch error: {str(e)[:140]}")
            out = None
        if out:
            graph_facts, prov, doc_ids = out
            graph_doc_ids = [p["chunk_id"] for p in prov]
            if COMPRESS_OFF:
                context, ids = _context_from_chunks([f"{d}#c0" for d in doc_ids])
            else:
                context, ids = _context_from_prov(prov)
            resolved = True

    # default route, and the fallback for any branch that could not resolve. Uses TigerGraph's
    # own vector index, so even the fallback is a live-graph path.
    if not resolved:
        k = LOOKUP_K if r == "lookup" else VEC_K
        vids = _tg_vector_search(conn, question, k=k)
        context, ids = _context_from_chunks(vids)
        graph_doc_ids = ids

    prompt = _build_prompt(question, context, graph_facts)
    with MetricsTimer("Pipeline 3: GraphRAG (Olympics)", question) as timer:
        answer, raw = get_llm_response(prompt)
    accounted_context = (graph_facts + "\n\n" + context) if graph_facts else context
    m = timer.record(answer, raw.get("usage", {}), question=question,
                     context_text=accounted_context, retrieved_chunks=len(ids),
                     retrieved_ids=ids, optimizer_tokens=optimizer_tokens, num_llm_calls=1,
                     preserve_answer=(config.DATASET in ("sec", "olympics")))
    m.latency_seconds = round(time.perf_counter() - t0, 3)
    m.route = r
    # tg_live DERIVED from the per-question counter, never a session boolean.
    m.tg_queries = _tg_query_count
    m.tg_live = (_tg_query_count > 0) and not GRAPH_OFF
    m.tg_host = TG_HOST if m.tg_live else ""
    m.graph_doc_ids = graph_doc_ids
    m.graph_entity_count = len(graph_doc_ids)
    m.hop_log = list(_hop_log)      # which queries fired, in order — the defect-1 guard
    m.branch_resolved = resolved
    return m


def run_batch(questions, vectorstore=None, num_hops: int = 2, verbose: bool = False):
    from tqdm import tqdm
    _load()
    _connect()
    out = []
    for q in tqdm(questions, desc="Pipeline 3 (Olympics GraphRAG)"):
        m = run_single(q)
        out.append(m)
        if verbose:
            print(f"  [{m.route}] {q[:58]} -> {m.answer[:60]} "
                  f"({m.total_inference_tokens} tok, {m.tg_queries} q)")
    return out
