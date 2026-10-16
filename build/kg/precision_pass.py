"""Targeted high-precision extraction pass — layered ON TOP of the full-corpus KG.

The full-corpus pass maximised recall/narrative coverage but is imprecise on the exact
facts the eval set aggregates over (auditor, foundry, competitors). This pass extracts
those from the RIGHT section of the RIGHT document, one focused Gemini call each, with
strict validation, then MERGES clean edges into pipeline3_sec/index/.

Per company:
  auditor        <- the auditor paragraph of its DEF 14A            (1 call)   -> audited_by
  competitors    <- the 'Competition' subsection of its 10-K        (1 call)   -> competitor_of
  foundries      <- the manufacturing/foundry text of its 10-K      (same call)-> uses_supplier

Emits edges with fresh, verified provenance and REPLACES the noisy full-corpus versions
of these three relations (audited_by / competitor_of / uses_supplier) so aggregation is clean.
discusses_risk + events + open edges from the full-corpus pass are kept as-is (recall).

Output: rewrites entity_to_docs / entity_types / relationships / edges_prov in place,
and writes precision_edges.json (audit trail). Ingestion tokens are added to the running total.

Env: PRECISION_WORKERS (default 8), LIMIT_COMPANIES (debug; 0=all).
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
OUT = Path("pipeline3_sec/index")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
WORKERS = int(os.getenv("PRECISION_WORKERS", "8"))
LIMIT = int(os.getenv("LIMIT_COMPANIES", "0"))
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# reuse canonicalization from the extractor
import importlib.util
_spec = importlib.util.spec_from_file_location("ekg", "build/kg/extract_kg.py")
_ekg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ekg)
canon, _valid_org = _ekg.canon, _ekg._valid_org


def _key() -> str:
    k = os.getenv("GOOGLE_API_KEY", "")
    if not k:
        from dotenv import load_dotenv
        load_dotenv()
        k = os.getenv("GOOGLE_API_KEY", "")
    return k


KEY = _key()
_tok_lock = threading.Lock()
_tokens = {"prompt": 0, "output": 0, "calls": 0}


def _call(prompt: str, schema: dict) -> dict | None:
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": schema}}
    for attempt in range(4):
        try:
            r = requests.post(f"{API_URL}?key={KEY}", json=body, timeout=90)
            if r.status_code == 200:
                j = r.json()
                u = j.get("usageMetadata", {})
                with _tok_lock:
                    _tokens["prompt"] += u.get("promptTokenCount", 0)
                    _tokens["output"] += u.get("candidatesTokenCount", 0)
                    _tokens["calls"] += 1
                return json.loads(j["candidates"][0]["content"]["parts"][0]["text"])
            if r.status_code in (429, 500, 503):
                time.sleep(2 ** attempt + 0.5); continue
            return None
        except Exception:
            time.sleep(2 ** attempt + 0.5)
    return None


AUDITOR_SCHEMA = {"type": "object", "properties": {
    "auditor": {"type": "string"}, "evidence": {"type": "string"}}}
TENK_SCHEMA = {"type": "object", "properties": {
    "competitors": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "evidence": {"type": "string"}}}},
    "foundries": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "evidence": {"type": "string"}}}}}}

AUDITOR_PROMPT = (
    "From this proxy-statement (DEF 14A) text for {company}, identify the company's INDEPENDENT "
    "REGISTERED PUBLIC ACCOUNTING FIRM (its external auditor) — one of 'Ernst & Young LLP', "
    "'PricewaterhouseCoopers LLP', 'Deloitte & Touche LLP', 'KPMG LLP', or the exact firm named. "
    "Return the firm name and the single sentence naming it as auditor. If no auditor is named "
    "in this text, return an empty string.\n\nTEXT:\n"
)
TENK_PROMPT = (
    "From this 10-K text for {company}, extract ONLY:\n"
    "1. competitors: SPECIFIC NAMED competitor companies this company says it competes against "
    "(proper nouns like 'Intel', 'NVIDIA'; NEVER generic words).\n"
    "2. foundries: SPECIFIC NAMED foundry / contract-manufacturing partners it says IT USES to "
    "make its products (e.g. 'TSMC', 'GlobalFoundries'). Only firms it actually relies on for "
    "manufacturing — NOT companies merely mentioned in passing, held as investments, or listed "
    "as risks.\nFor each, give the exact supporting sentence.\n\nTEXT:\n"
)


def _find_section(chunks: list[dict], keywords: tuple, max_chars: int = 9000) -> str:
    """Concatenate the chunks whose text is densest in the keywords (the target section)."""
    def score(c):
        t = c["text"].lower()
        return sum(t.count(k) for k in keywords)
    ranked = sorted([c for c in chunks if score(c) > 0], key=score, reverse=True)
    out, n = [], 0
    for c in ranked:
        out.append(c["text"]); n += len(c["text"])
        if n >= max_chars:
            break
    return "\n\n".join(out)[:max_chars]


def process_company(ticker: str, company: str, docs: dict, chunks_by_doc: dict, result: dict):
    ck = f"{company} ({ticker})"
    # --- auditor from DEF 14A ---
    proxy = [d for d in docs if d["ticker"] == ticker and d["form"] == "DEF 14A"]
    if proxy:
        pid = proxy[0]["doc_id"]
        sec = _find_section(chunks_by_doc.get(pid, []),
                            ("ernst & young", "deloitte", "pricewaterhouse", "kpmg",
                             "independent registered", "ratif", "audit committee"))
        if sec:
            res = _call(AUDITOR_PROMPT.format(company=company) + sec, AUDITOR_SCHEMA)
            if res and res.get("auditor"):
                aud = canon(res["auditor"])
                if any(x in aud.lower() for x in ("llp", "ernst", "deloitte", "kpmg", "pricewaterhouse")):
                    with result["lock"]:
                        result["audited_by"].append((ck, aud, res.get("evidence", ""), pid))
    # --- competitors + foundries from 10-K ---
    tenk = [d for d in docs if d["ticker"] == ticker and d["form"] == "10-K"]
    if tenk:
        tid = tenk[0]["doc_id"]
        sec = _find_section(chunks_by_doc.get(tid, []),
                            ("we compete", "competitor", "competition", "foundr", "tsmc",
                             "we utilize", "wafer", "manufactur"))
        if sec:
            res = _call(TENK_PROMPT.format(company=company) + sec, TENK_SCHEMA)
            if res:
                with result["lock"]:
                    for c in res.get("competitors", []) or []:
                        if c.get("name") and _valid_org(c["name"]):
                            cc = canon(c["name"])
                            if cc.lower() not in (ck.lower(), ticker.lower()):
                                result["competitor_of"].append((ck, cc, c.get("evidence", ""), tid))
                    for f in res.get("foundries", []) or []:
                        if f.get("name") and _valid_org(f["name"]):
                            result["uses_supplier"].append((ck, canon(f["name"]), f.get("evidence", ""), tid))


def main():
    docs = [json.loads(l) for l in open(DATA / "documents.jsonl")]
    chunks_by_doc = defaultdict(list)
    for l in open(DATA / "corpus.jsonl"):
        c = json.loads(l); chunks_by_doc[c["doc_id"]].append(c)
    companies = {}
    for d in docs:
        companies.setdefault(d["ticker"], d["company"])
    items = list(companies.items())
    if LIMIT:
        items = items[:LIMIT]
    print(f"[precision] {len(items)} companies, model={GEMINI_MODEL} workers={WORKERS}")

    result = {"lock": threading.Lock(), "audited_by": [], "competitor_of": [], "uses_supplier": []}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(process_company, t, c, docs, chunks_by_doc, result) for t, c in items]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="precision pass"):
            pass

    print(f"[precision] audited_by={len(result['audited_by'])} "
          f"competitor_of={len(result['competitor_of'])} uses_supplier={len(result['uses_supplier'])}")

    # ---- merge into the KG: REPLACE these 3 relations with the clean edges ----
    rels = json.load(open(OUT / "relationships.json"))
    prov = json.load(open(OUT / "edges_prov.json"))
    e2d = json.load(open(OUT / "entity_to_docs.json"))
    etypes = json.load(open(OUT / "entity_types.json"))

    REPLACE = {"audited_by", "competitor_of", "uses_supplier"}
    rels = [r for r in rels if r["relation"] not in REPLACE]
    prov = [p for p in prov if p["relation"] not in REPLACE]

    rc = defaultdict(int)
    new_prov, precision_edges = [], []
    etype_for = {"audited_by": "AUDITOR", "competitor_of": "ORG", "uses_supplier": "ORG"}
    for rel in REPLACE:
        for (s, t, ev, doc) in result[rel]:
            sl, tl = s.lower(), t.lower()
            rc[(sl, rel, tl)] += 1
            new_prov.append({"source": sl, "relation": rel, "target": tl,
                             "evidence": (ev or "")[:240], "chunk_id": doc, "doc_id": doc})
            precision_edges.append({"source": sl, "relation": rel, "target": tl, "evidence": ev})
            e2d.setdefault(tl, [])
            if doc not in e2d[tl]:
                e2d[tl].append(doc)
            etypes.setdefault(tl, etype_for[rel])

    for (s, rel, t), c in rc.items():
        rels.append({"source": s, "relation": rel, "target": t, "count": c})
    prov.extend(new_prov)

    json.dump(rels, open(OUT / "relationships.json", "w"))
    json.dump(prov, open(OUT / "edges_prov.json", "w"))
    json.dump({k: sorted(set(v)) for k, v in e2d.items()}, open(OUT / "entity_to_docs.json", "w"))
    json.dump(etypes, open(OUT / "entity_types.json", "w"))
    json.dump(precision_edges, open(OUT / "precision_edges.json", "w"), indent=2)

    # add tokens to the running ingestion total
    it_path = OUT / "ingestion_tokens.json"
    it = json.load(open(it_path)) if it_path.exists() else {"prompt": 0, "output": 0, "calls": 0}
    for k in ("prompt", "output", "calls"):
        it[k] = it.get(k, 0) + _tokens[k]
    json.dump(it, open(it_path, "w"), indent=2)

    # report distinct-company coverage
    aud_comps = {s for (s, r, t) in rc if r == "audited_by"}
    print(f"[precision] merged. audited_by covers {len(aud_comps)} companies. "
          f"precision-pass tokens: prompt={_tokens['prompt']:,} output={_tokens['output']:,} calls={_tokens['calls']}")


if __name__ == "__main__":
    main()
