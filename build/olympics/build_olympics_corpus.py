"""Chunk the Olympics corpus and build the P2 vector index.

IN   data/new_dataset/corpus/corpus.jsonl
     2,951 docs · 5,466,414 approx_tokens · doc_id (= Wikidata QID) · title · url ·
     wikidata_qid · wikipedia_pageid · approx_tokens · text

OUT  datasets/olympics/corpus.jsonl   one line per CHUNK, `title` = the chunk id
     datasets/olympics/manifest.json  doc_id -> {title, url, sport, games, is_event, ...}
     chroma_db_olympics/main          collection `olympics_docs`, BAAI/bge-small-en-v1.5

Run:
    DATASET=olympics EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
      python "build/olympics/build_olympics_corpus.py"

    --no-chroma   write the JSONL + manifest only (fast; for inspecting the chunking)

--------------------------------------------------------------------------------------------
THE CHUNKING RULE THAT MATTERS, AND WHY
--------------------------------------------------------------------------------------------
Every fact any of the 150 questions asks for lives in the `[Infobox Olympic event]` block:
`nations`, `competitors`, `gold`, `venue`, `date`. The block is ~15 short lines — median 88
approx-tokens, p95 117, max 200 (OLYMPICS_DATA_NOTES.md §2).

So the chunker **keeps the whole infobox region in chunk 0 of its document, never split**. If a
window boundary landed inside the infobox, P2 could retrieve a chunk holding `competitors` but
not `gold`, and lose questions for a reason that is an artifact of our chunker rather than a
property of vector retrieval. The SEC round was criticised for exactly that class of unfairness,
so the baseline's job is made easy on purpose.

`split_head()` ends the infobox region at the first blank line that is NOT followed by another
infobox header. That second condition is load-bearing: 212 documents stack a second infobox
directly onto the first (`[Infobox Olympic event]` then `[Infobox ski area]`, no blank line
between), and a blank-line-only rule would cut the region in the wrong place.

CHUNK_CHARS / CHUNK_OVERLAP are copied from `build/data_prep/build_sec_corpus.py` unchanged, so
the only chunking difference between the two datasets is the infobox rule above. (Chunking is
listed as dataset-specific in the comparability contract, but keeping the window parameters
identical removes one more confound for free.)

--------------------------------------------------------------------------------------------
CHUNK IDS
--------------------------------------------------------------------------------------------
`<doc_id>#c<i>`, e.g. `Q303623#c0`. The doc_id is the Wikidata QID, which is also what
`gold_doc_ids` names, so retrieval precision/recall@k is a string split away:
    chunk_id.split("#")[0] in set(gold_doc_ids)
`title` carries the chunk id because `evaluate.load_chunk_text()` keys on `title`.
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path

SRC = Path("data/new_dataset/corpus/corpus.jsonl")
OUT_DIR = Path("datasets/olympics")

CHUNK_CHARS = 1800        # identical to build_sec_corpus.py
CHUNK_OVERLAP = 250       # identical to build_sec_corpus.py
MIN_CHUNK_CHARS = 80

INFOBOX_HDR = re.compile(r'^\[[Ii]nfobox ([^\]]+)\]\s*$')
FIELD = re.compile(r'^  ([A-Za-z_][A-Za-z0-9_ ]*?): ?(.*)$')
TITLE_RE = re.compile(r'^(.*?) at the (\d{4}) (Summer|Winter) Olympics – (.*)$')


def split_head(text: str) -> tuple[str, str]:
    """(infobox_region, remaining_body). Empty head if the doc opens with no infobox."""
    lines = text.split("\n")
    if not lines or not INFOBOX_HDR.match(lines[0]):
        return "", text
    i = end = 0
    while i < len(lines):
        ln = lines[i]
        if INFOBOX_HDR.match(ln) or FIELD.match(ln):
            i += 1
            end = i
            continue
        if ln.strip() == "":
            j = i
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and INFOBOX_HDR.match(lines[j]):
                i = j            # blank line(s) then another infobox — region continues
                continue
            break                 # blank line ends the region
        break                     # a prose line ends the region
    return "\n".join(lines[:end]), "\n".join(lines[end:]).lstrip("\n")


def window(text: str, first_budget: int) -> list[str]:
    """Slide a CHUNK_CHARS window with overlap, breaking near a paragraph/sentence boundary.
    `first_budget` is the size of the FIRST window only, shrunk by however many characters the
    infobox head will occupy, so chunk 0 stays within CHUNK_CHARS once the head is prepended."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks, start, budget = [], 0, max(first_budget, 300)
    while start < len(text):
        end = start + budget
        if end < len(text):
            tail = text[end - 300:end]
            for sep in ("\n\n", ". ", "\n"):
                pos = tail.rfind(sep)
                if pos != -1:
                    end = (end - 300) + pos + len(sep)
                    break
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
        budget = CHUNK_CHARS
    return chunks


def parse_infoboxes(text: str) -> dict:
    out, cur = {}, None
    for ln in text.split("\n"):
        m = INFOBOX_HDR.match(ln)
        if m:
            cur = m.group(1).strip()
            out.setdefault(cur, {})
            continue
        if cur is None:
            continue
        if ln.strip() == "":
            cur = None
            continue
        f = FIELD.match(ln)
        if f:
            out[cur][f.group(1)] = f.group(2).strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-chroma", action="store_true", help="write JSONL + manifest only")
    args = ap.parse_args()

    docs = [json.loads(l) for l in open(SRC) if l.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest, records = {}, []
    per_doc_chunks = Counter()
    n_infobox_whole = n_with_infobox = n_head_overflow = 0

    for d in docs:
        text = d["text"]
        head, body = split_head(text)
        boxes = parse_infoboxes(head) if head else {}
        is_event = "Olympic event" in boxes
        t = TITLE_RE.match(d["title"]) if is_event else None

        meta = {
            "doc_id": d["doc_id"], "doc_title": d["title"], "url": d.get("url", ""),
            "is_event": is_event, "has_infobox": bool(head),
            "infobox_kinds": list(boxes),
            "sport": t.group(1) if t else "",
            "year": int(t.group(2)) if t else 0,
            "season": t.group(3) if t else "",
            "games": f"{t.group(2)} {t.group(3)}" if t else "",
            "event_name": t.group(4) if t else "",
            "approx_tokens": d.get("approx_tokens", 0),
        }
        manifest[d["doc_id"]] = {k: meta[k] for k in
                                 ("doc_title", "url", "is_event", "sport", "games",
                                  "year", "season", "event_name", "approx_tokens")}

        head_len = len(head) + 2 if head else 0
        if head:
            n_with_infobox += 1
            if head_len > CHUNK_CHARS:
                n_head_overflow += 1
        pieces = window(body, first_budget=CHUNK_CHARS - head_len)
        if head:
            pieces = [(head + "\n\n" + pieces[0]) if pieces else head] + pieces[1:]
        if not pieces:
            pieces = [text.strip()[:CHUNK_CHARS]] if text.strip() else []

        # the whole infobox region is inside chunk 0, uncut
        if head and pieces and head in pieces[0]:
            n_infobox_whole += 1

        for ci, piece in enumerate(pieces):
            records.append({
                "title": f"{d['doc_id']}#c{ci}", "text": piece,
                "chunk_ix": ci, **{k: v for k, v in meta.items() if k != "infobox_kinds"},
            })
        per_doc_chunks[len(pieces)] += 1

    with open(OUT_DIR / "corpus.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(manifest, open(OUT_DIR / "manifest.json", "w"), ensure_ascii=False, indent=1)

    n_ev = sum(1 for m in manifest.values() if m["is_event"])
    print(f"[corpus] docs={len(docs)}  event docs={n_ev}  distractors={len(docs)-n_ev}")
    print(f"[corpus] chunks={len(records)}  avg chunks/doc={len(records)/len(docs):.2f}")
    print(f"[corpus] chunks-per-doc histogram: "
          f"{dict(sorted(per_doc_chunks.items()))}")
    print(f"[corpus] docs with an infobox: {n_with_infobox}  "
          f"infobox kept WHOLE in chunk 0: {n_infobox_whole}  "
          f"({n_infobox_whole}/{n_with_infobox})")
    print(f"[corpus] infobox regions larger than CHUNK_CHARS={CHUNK_CHARS}: {n_head_overflow}")
    cl = sorted(len(r["text"]) for r in records)
    print(f"[corpus] chunk chars  median={cl[len(cl)//2]}  p95={cl[int(.95*len(cl))]}  max={max(cl)}")
    print(f"[corpus] wrote {OUT_DIR/'corpus.jsonl'} and {OUT_DIR/'manifest.json'}")

    if args.no_chroma:
        print("[corpus] --no-chroma: skipping the vector index")
        return

    import os
    import sys
    sys.path.insert(0, ".")
    os.environ.setdefault("DATASET", "olympics")
    os.environ["SEC_PRECHUNKED"] = "1"     # these chunks are final; do not re-split
    import config
    assert config.DATASET == "olympics", \
        f"DATASET is {config.DATASET!r} — run with DATASET=olympics or the index lands in the wrong place"
    assert config.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5", \
        f"EMBEDDING_MODEL is {config.EMBEDDING_MODEL!r} — set EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 " \
        f"explicitly; the default is 768-dim and Chroma will reject the 384-dim vectors later"
    from langchain_core.documents import Document
    from chroma_store import build_vectorstore
    lc_docs = [Document(page_content=r["text"],
                        metadata={k: v for k, v in r.items() if k != "text"})
               for r in records]
    print(f"[corpus] building Chroma at {config.CHROMA_PERSIST_DIR} "
          f"(collection {config.CHROMA_COLLECTION_NAME}, {config.EMBEDDING_MODEL})")
    build_vectorstore(lc_docs)


if __name__ == "__main__":
    main()
