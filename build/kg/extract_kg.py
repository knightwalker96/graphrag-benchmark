"""Knowledge-graph extraction — FULL-CORPUS, every chunk, via Gemini.

Design (per user decision: max recall — extract from ALL 61k chunks, skip nothing):

  BACKBONE (deterministic, no LLM), from manifest + filenames:
    Company{ticker,name,sector} -[in_sector]-> Sector
    Company -[filed]-> Filing{form,date}
  (Sector is a structured CSV column — no reason to spend LLM calls on it.)

  GEMINI EXTRACTION over EVERY chunk of every filing. One typed schema, prompted so
  the model returns BOTH the query-critical financial/governance relations AND open
  narrative entities/relations:
    audited_by, has_director, has_ceo        (from proxy text)
    competitor_of, uses_supplier             (named orgs only)
    discusses_risk (canonical RISK_TOPIC)    (climate / GLP-1 / AI / cyber / ...)
    reports_event -> Event{type,amount,date,person}   (8-K events)
    company financial facts                  (revenue / net_income / rnd / fy_end)
    open (entity)-[relation]->(entity)       (everything else, for narrative Qs)
  Chunk-level extraction means every edge already knows its chunk_id -> exact
  provenance (source sentence + citation target) for free.

  CANONICALIZATION collapses org/auditor name variants to one node so aggregation
  questions (EQ25/26/44) count each company/firm once.

Resumable: chunks already in done_chunks.json are skipped, so the long run can be
interrupted and restarted.

Outputs (pipeline3_sec/index/): entity_to_docs, entity_types, relationships,
edges_prov, company_facts, events, doc_content(filing-level), ingestion_tokens.

Env: GEMINI_MODEL, EXTRACT_WORKERS (default 10), BATCH_CHUNKS (chunks/call, default 6),
     LIMIT_CHUNKS (debug cap; 0=all), CHECKPOINT_EVERY (default 500).
"""
from __future__ import annotations
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, ".")

DATA = Path("datasets/sec")
OUT = Path(os.getenv("GEMINI_INDEX_DIR", "pipeline3_sec/index"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
WORKERS = int(os.getenv("EXTRACT_WORKERS", "10"))
BATCH_CHUNKS = int(os.getenv("BATCH_CHUNKS", "6"))
LIMIT_CHUNKS = int(os.getenv("LIMIT_CHUNKS", "0"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "500"))
PER_CALL_RETRIES = 4
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


# ---------------- entity canonicalization ----------------
CANON_ALIASES = {
    "taiwan semiconductor manufacturing company limited": "TSMC",
    "taiwan semiconductor manufacturing company": "TSMC",
    "taiwan semiconductor": "TSMC", "tsmc": "TSMC",
    "intel corporation": "Intel", "intel": "Intel",
    "nvidia corporation": "NVIDIA", "nvidia": "NVIDIA",
    "advanced micro devices, inc.": "AMD", "advanced micro devices": "AMD", "amd": "AMD",
    "qualcomm incorporated": "Qualcomm", "qualcomm": "Qualcomm",
    "broadcom corporation": "Broadcom", "broadcom inc.": "Broadcom", "broadcom": "Broadcom",
    "samsung electronics co., ltd.": "Samsung", "samsung electronics": "Samsung", "samsung": "Samsung",
    "globalfoundries inc.": "GlobalFoundries", "global foundries": "GlobalFoundries",
    "united microelectronics corporation": "UMC", "umc": "UMC",
    "microsoft corporation": "Microsoft", "microsoft": "Microsoft",
    "amazon, inc.": "Amazon", "amazon.com, inc.": "Amazon", "amazon": "Amazon",
    "amazon web services": "Amazon Web Services", "aws": "Amazon Web Services",
    "alphabet inc.": "Alphabet", "google": "Alphabet",
    "cisco systems, inc.": "Cisco", "cisco": "Cisco",
    "apple inc.": "Apple", "apple": "Apple", "netflix, inc.": "Netflix", "netflix": "Netflix",
    "ernst & young llp": "Ernst & Young LLP", "ernst & young": "Ernst & Young LLP",
    "ernst and young llp": "Ernst & Young LLP", "ey": "Ernst & Young LLP",
    "pricewaterhousecoopers llp": "PricewaterhouseCoopers LLP",
    "pricewaterhousecoopers": "PricewaterhouseCoopers LLP", "pwc": "PricewaterhouseCoopers LLP",
    "deloitte & touche llp": "Deloitte & Touche LLP", "deloitte and touche llp": "Deloitte & Touche LLP",
    "deloitte": "Deloitte & Touche LLP", "deloitte & touche": "Deloitte & Touche LLP",
    "kpmg llp": "KPMG LLP", "kpmg": "KPMG LLP",
}
_SUFFIX_RE = re.compile(
    r",?\s+(inc\.?|incorporated|corporation|corp\.?|company|co\.?|ltd\.?|limited|llc|"
    r"plc|n\.v\.|s\.a\.|l\.p\.|group|holdings|technologies|technology)\b\.?", re.I)

# Canonical risk-topic labels the eval set cares about; map free-text to these.
RISK_CANON = {
    "climate": "climate change", "climate change": "climate change",
    "global warming": "climate change", "environmental": "climate change",
    "glp-1": "GLP-1 / weight-loss drugs", "glp 1": "GLP-1 / weight-loss drugs",
    "weight loss": "GLP-1 / weight-loss drugs", "weight-loss": "GLP-1 / weight-loss drugs",
    "obesity": "GLP-1 / weight-loss drugs", "anti-obesity": "GLP-1 / weight-loss drugs",
    "ai": "artificial intelligence", "artificial intelligence": "artificial intelligence",
    "machine learning": "artificial intelligence", "generative ai": "artificial intelligence",
    "cyber": "cybersecurity", "cybersecurity": "cybersecurity", "cyber security": "cybersecurity",
    "supply chain": "supply chain", "interest rate": "interest rates", "interest rates": "interest rates",
    "regulation": "regulation", "regulatory": "regulation", "competition": "competition",
    "pandemic": "pandemic", "covid": "pandemic",
}


def canon(name: str) -> str:
    n = (name or "").strip()
    low = n.lower().strip().rstrip(".")
    if low in CANON_ALIASES:
        return CANON_ALIASES[low]
    stripped = _SUFFIX_RE.sub("", n).strip().rstrip(",.")
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped or n


# Generic non-entities that occasionally leak into competitor/supplier slots despite
# the "named proper nouns only" instruction — reject them so aggregation stays clean.
GENERIC_ORG_REJECT = {
    "competitors", "competitor", "companies", "company", "others", "other companies",
    "suppliers", "supplier", "customers", "customer", "vendors", "third parties",
    "certain competitors", "our competitors", "many competitors", "new competitors",
    "foundries", "manufacturers", "contract manufacturers", "peers", "market",
    "counterparties", "partners", "clients", "some competitors",
}


def _valid_org(name: str) -> bool:
    n = (name or "").strip()
    if len(n) < 2 or n.lower() in GENERIC_ORG_REJECT:
        return False
    # reject run-on strings the model sometimes returns instead of a clean name
    # (a real company name is short: few words, no clause punctuation).
    if len(n) > 60 or n.count(",") >= 2 or " who " in n.lower() or len(n.split()) > 7:
        return False
    # must contain an uppercase letter (proper noun) and not be a bare common word
    return any(ch.isupper() for ch in n)


def canon_risk(label: str) -> str | None:
    t = (label or "").lower().strip()
    for k, v in RISK_CANON.items():
        if k in t:
            return v
    return t if len(t) > 2 else None


def _key() -> str:
    k = os.getenv("GOOGLE_API_KEY", "")
    if not k:
        from dotenv import load_dotenv
        load_dotenv()
        k = os.getenv("GOOGLE_API_KEY", "")
    if not k:
        raise EnvironmentError("GOOGLE_API_KEY not set")
    return k


KEY = _key()
_tok_lock = threading.Lock()
_ingest_tokens = {"prompt": 0, "output": 0, "calls": 0}


# ---------------- extraction prompt (one schema for all chunk types) ----------------
PROMPT = (
    "You extract a financial knowledge graph from a chunk of a US SEC filing (10-K annual "
    "report, 8-K current report, or DEF 14A proxy statement). Each chunk is labeled with its "
    "filing and section. Extract ONLY facts stated in the text. Return STRICT JSON.\n\n"
    "For the FILING'S OWN COMPANY (the subject company of the filing), extract any of these "
    "that appear in the chunk:\n"
    "- auditor: the independent registered public accounting firm (proxy statements). One of "
    "'Ernst & Young LLP','PricewaterhouseCoopers LLP','Deloitte & Touche LLP','KPMG LLP' or the "
    "exact firm named.\n"
    "- directors: full names of board members / director nominees.\n"
    "- ceo: full name of the Chief Executive Officer if identified.\n"
    "- competitors: SPECIFIC NAMED competitor companies (proper nouns only, e.g. 'Intel', "
    "'NVIDIA'). Never generic words like 'competitors'/'others'.\n"
    "- suppliers: SPECIFIC NAMED suppliers / foundry / manufacturing partners (e.g. 'TSMC').\n"
    "- risk_topics: major business-risk themes discussed (short labels, e.g. 'climate change', "
    "'weight-loss drugs', 'artificial intelligence', 'cybersecurity', 'supply chain').\n"
    "- events: for 8-K content, corporate events. type in {dividend, buyback, "
    "officer_appointment, officer_departure, acquisition, restructuring, earnings, "
    "credit_facility, joint_venture, other}; include amount (with units), person, role, date.\n"
    "- financials: total revenue/net sales, net income, R&D expense (keep units & $), and "
    "fiscal-year-end date, if in this chunk.\n"
    "Also extract OPEN facts for narrative/governance text:\n"
    "- entities: notable named entities as {name, type} (type in PERSON, ORG, GPE, PRODUCT, "
    "EVENT, MONEY, DATE, RISK, OTHER).\n"
    "- relations: {source, relation, target} triples where BOTH ends are named entities and the "
    "relation is a short verb phrase. Only if clearly stated.\n"
    "For EVERY extracted item, include 'evidence': the single verbatim sentence (<=240 chars) "
    "supporting it. Omit fields with nothing to report (use null / empty list).\n"
)

CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "auditor": {"type": "object", "properties": {"name": {"type": "string"}, "evidence": {"type": "string"}}},
        "directors": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "evidence": {"type": "string"}}}},
        "ceo": {"type": "object", "properties": {"name": {"type": "string"}, "evidence": {"type": "string"}}},
        "competitors": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "evidence": {"type": "string"}}}},
        "suppliers": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "evidence": {"type": "string"}}}},
        "risk_topics": {"type": "array", "items": {"type": "object", "properties": {"label": {"type": "string"}, "evidence": {"type": "string"}}}},
        "events": {"type": "array", "items": {"type": "object", "properties": {
            "type": {"type": "string"}, "amount": {"type": "string"}, "person": {"type": "string"},
            "role": {"type": "string"}, "date": {"type": "string"}, "evidence": {"type": "string"}}}},
        "financials": {"type": "object", "properties": {
            "revenue": {"type": "string"}, "net_income": {"type": "string"},
            "rnd": {"type": "string"}, "fy_end": {"type": "string"}, "evidence": {"type": "string"}}},
        "entities": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "type": {"type": "string"}}}},
        "relations": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "relation": {"type": "string"}, "target": {"type": "string"},
            "evidence": {"type": "string"}}}},
    },
}

BATCH_INSTR = (
    "\n\nYou will receive MULTIPLE chunks, each starting with a line '### CHUNK <i> | "
    "<company> (<ticker>) | <form> | <section>'. For EACH chunk independently, extract as "
    "described. Return a JSON array; each element {\"chunk_index\": <i>, ...same fields...}.\n\n"
)
BATCH_SCHEMA = {
    "type": "array",
    "items": {"type": "object", "properties": {
        "chunk_index": {"type": "integer"}, **CHUNK_SCHEMA["properties"]}},
}


def _call_batch(payload_text: str) -> list | None:
    body = {"contents": [{"parts": [{"text": payload_text}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": BATCH_SCHEMA}}
    for attempt in range(PER_CALL_RETRIES):
        try:
            r = requests.post(f"{API_URL}?key={KEY}", json=body, timeout=180)
            if r.status_code == 200:
                j = r.json()
                usage = j.get("usageMetadata", {})
                with _tok_lock:
                    _ingest_tokens["prompt"] += usage.get("promptTokenCount", 0)
                    _ingest_tokens["output"] += usage.get("candidatesTokenCount", 0)
                    _ingest_tokens["calls"] += 1
                return json.loads(j["candidates"][0]["content"]["parts"][0]["text"])
            if r.status_code in (429, 500, 503):
                time.sleep(2 ** attempt + 0.5)
                continue
            return None
        except Exception:
            time.sleep(2 ** attempt + 0.5)
    return None


# ---------------- shared index (thread-safe) ----------------
class Index:
    def __init__(self):
        self.lock = threading.Lock()
        self.entity_to_docs: dict[str, set] = defaultdict(set)
        self.entity_types: dict[str, str] = {}
        self.rel_counts: dict[tuple, int] = defaultdict(int)
        self.edges_prov: list[dict] = []
        self.company_facts: dict[str, dict] = {}
        self.events: list[dict] = []
        self.doc_content: dict[str, str] = {}
        self.done_chunks: set[str] = set()

    def ent(self, name: str, etype: str, doc_id: str):
        name = (name or "").strip()
        if len(name) < 2:
            return
        k = name.lower()
        self.entity_to_docs[k].add(doc_id)
        self.entity_types.setdefault(k, etype)

    def edge(self, s: str, rel: str, t: str, doc_id: str, chunk_id: str, evidence: str):
        s, t = (s or "").strip(), (t or "").strip()
        if len(s) < 2 or len(t) < 2:
            return
        sl, tl = s.lower(), t.lower()
        self.rel_counts[(sl, rel, tl)] += 1
        self.edges_prov.append({"source": sl, "relation": rel, "target": tl,
                                "evidence": (evidence or "")[:240], "chunk_id": chunk_id, "doc_id": doc_id})


def _fold(res: dict, chunk: dict, idx: Index) -> None:
    """Fold one chunk's extraction into the index. Caller holds idx.lock."""
    ticker, company = chunk["ticker"], chunk["company"]
    doc_id, chunk_id, sector = chunk["doc_id"], chunk["title"], chunk.get("sector", "")
    ck = f"{company} ({ticker})"

    def ev(o):  # evidence sentence -> attribute to THIS chunk (chunk-level = exact)
        return (o.get("evidence") if isinstance(o, dict) else "") or ""

    aud = res.get("auditor")
    if aud and aud.get("name"):
        a = canon(aud["name"])
        idx.ent(a, "AUDITOR", doc_id); idx.edge(ck, "audited_by", a, doc_id, chunk_id, ev(aud))
    for d in res.get("directors") or []:
        if d.get("name"):
            idx.ent(d["name"], "PERSON", doc_id); idx.edge(ck, "has_director", d["name"], doc_id, chunk_id, ev(d))
    ceo = res.get("ceo")
    if ceo and ceo.get("name"):
        idx.ent(ceo["name"], "PERSON", doc_id); idx.edge(ck, "has_ceo", ceo["name"], doc_id, chunk_id, ev(ceo))
    for c in res.get("competitors") or []:
        if c.get("name") and _valid_org(c["name"]):
            cc = canon(c["name"])
            if cc.lower() != ck.lower() and cc.lower() != ticker.lower():
                idx.ent(cc, "ORG", doc_id); idx.edge(ck, "competitor_of", cc, doc_id, chunk_id, ev(c))
    for s in res.get("suppliers") or []:
        if s.get("name") and _valid_org(s["name"]):
            sc = canon(s["name"])
            idx.ent(sc, "ORG", doc_id); idx.edge(ck, "uses_supplier", sc, doc_id, chunk_id, ev(s))
    for rt in res.get("risk_topics") or []:
        lab = canon_risk(rt.get("label", ""))
        if lab:
            idx.ent(lab, "RISK_TOPIC", doc_id); idx.edge(ck, "discusses_risk", lab, doc_id, chunk_id, ev(rt))
    for e in res.get("events") or []:
        etype = e.get("type") or "other"
        node = f"{etype} [{ticker} {chunk.get('filing_date','')}]"
        idx.ent(node, "EVENT", doc_id); idx.edge(ck, "reports_event", node, doc_id, chunk_id, ev(e))
        idx.events.append({"ticker": ticker, "doc_id": doc_id, "chunk_id": chunk_id, "type": etype,
                           "date": e.get("date") or chunk.get("filing_date", ""), "amount": e.get("amount"),
                           "person": e.get("person"), "role": e.get("role"), "evidence": ev(e)[:240]})
        if e.get("person"):
            idx.ent(e["person"], "PERSON", doc_id)
            idx.edge(node, "involves_person", e["person"], doc_id, chunk_id, ev(e))
    fin = res.get("financials")
    if fin:
        cf = idx.company_facts.setdefault(ticker, {"company": company, "sector": sector,
                                                   "revenue": None, "net_income": None, "rnd": None, "fy_end": None})
        for k in ("revenue", "net_income", "rnd", "fy_end"):
            if fin.get(k) and not cf.get(k):
                cf[k] = fin[k]
    # open entities/relations (narrative)
    for e in res.get("entities") or []:
        if e.get("name"):
            idx.ent(e["name"], (e.get("type") or "OTHER").upper(), doc_id)
    for r in res.get("relations") or []:
        s, rel, t = r.get("source"), (r.get("relation") or "related_to"), r.get("target")
        if s and t:
            idx.edge(s, rel.lower().replace(" ", "_"), t, doc_id, chunk_id, ev(r))


def _payload(batch: list[dict]) -> str:
    parts = [PROMPT + BATCH_INSTR]
    for i, c in enumerate(batch):
        parts.append(f"### CHUNK {i} | {c['company']} ({c['ticker']}) | {c['form']} | "
                     f"{c.get('section','')}\n{c['text']}\n\n")
    return "".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = Index()

    # resume
    cp = OUT / "done_chunks.json"
    if cp.exists():
        idx.done_chunks = set(json.load(open(cp)))
        for fn, attr in [("entity_to_docs.json", None), ("entity_types.json", "entity_types"),
                         ("relationships.json", None), ("edges_prov.json", "edges_prov"),
                         ("company_facts.json", "company_facts"), ("events.json", "events")]:
            p = OUT / fn
            if not p.exists():
                continue
            data = json.load(open(p))
            if fn == "entity_to_docs.json":
                for k, v in data.items():
                    idx.entity_to_docs[k] = set(v)
            elif fn == "relationships.json":
                for r in data:
                    idx.rel_counts[(r["source"], r["relation"], r["target"])] = r["count"]
            elif attr:
                setattr(idx, attr, data)
        it = OUT / "ingestion_tokens.json"
        if it.exists():
            _ingest_tokens.update(json.load(open(it)))
        print(f"[extract] resuming — {len(idx.done_chunks):,} chunks already done")

    chunks = [json.loads(l) for l in open(DATA / "corpus.jsonl")]
    # backbone: sector + filed edges (deterministic; cheap; do once)
    docs = [json.loads(l) for l in open(DATA / "documents.jsonl")]
    for d in docs:
        idx.doc_content[d["doc_id"]] = d["text"]
        ck = f"{d['company']} ({d['ticker']})"
        idx.ent(ck, "COMPANY", d["doc_id"])
        if d.get("sector"):
            idx.ent(d["sector"], "SECTOR", d["doc_id"])
            idx.rel_counts[(ck.lower(), "in_sector", d["sector"].lower())] += 1
        idx.rel_counts[(ck.lower(), "filed", f"{d['form']} {d['filing_date']} [{d['ticker']}]".lower())] += 1

    todo = [c for c in chunks if c["title"] not in idx.done_chunks and c["text"].strip()]
    if LIMIT_CHUNKS:
        todo = todo[:LIMIT_CHUNKS]
    batches = [todo[i:i + BATCH_CHUNKS] for i in range(0, len(todo), BATCH_CHUNKS)]
    print(f"[extract] model={GEMINI_MODEL} workers={WORKERS} chunks_todo={len(todo):,} "
          f"batches={len(batches):,} (batch={BATCH_CHUNKS})")

    n_done = 0

    def save():
        json.dump({k: sorted(v) for k, v in idx.entity_to_docs.items()}, open(OUT / "entity_to_docs.json", "w"))
        json.dump(idx.entity_types, open(OUT / "entity_types.json", "w"))
        json.dump([{"source": s, "relation": r, "target": t, "count": c}
                   for (s, r, t), c in idx.rel_counts.items()], open(OUT / "relationships.json", "w"))
        json.dump(idx.edges_prov, open(OUT / "edges_prov.json", "w"))
        json.dump(idx.company_facts, open(OUT / "company_facts.json", "w"), indent=2)
        json.dump(idx.events, open(OUT / "events.json", "w"), indent=2)
        json.dump(idx.doc_content, open(OUT / "doc_content.json", "w"))
        json.dump(sorted(idx.done_chunks), open(OUT / "done_chunks.json", "w"))
        json.dump(_ingest_tokens, open(OUT / "ingestion_tokens.json", "w"), indent=2)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_call_batch, _payload(b)): b for b in batches}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Gemini KG (full corpus)"):
            batch = futs[fut]
            arr = fut.result() or []
            by_i = {el.get("chunk_index"): el for el in arr if isinstance(el, dict)}
            with idx.lock:
                for i, c in enumerate(batch):
                    res = by_i.get(i)
                    if res:
                        _fold(res, c, idx)
                    idx.done_chunks.add(c["title"])
                    n_done += 1
                if n_done % CHECKPOINT_EVERY < BATCH_CHUNKS:
                    save()

    with idx.lock:
        save()
    it = _ingest_tokens
    print(f"\n[extract] done. entities={len(idx.entity_to_docs):,} "
          f"edges(distinct)={len(idx.rel_counts):,} prov_edges={len(idx.edges_prov):,} events={len(idx.events):,}")
    print(f"[extract] INGESTION TOKENS (one-time): prompt={it['prompt']:,} output={it['output']:,} calls={it['calls']:,}")
    print(f"[extract] output -> {OUT}/")


if __name__ == "__main__":
    main()
