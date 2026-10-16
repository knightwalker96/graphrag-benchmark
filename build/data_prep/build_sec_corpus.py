"""Build the SEC corpus: 400 SEC filings -> section-aware, citation-ready chunks.

Why section-aware: SEC forms are structured by 'Item N' headers (10-K: Item 1
Business, Item 1A Risk Factors, Item 7 MD&A, Item 8 Financials; 8-K: Item 5.02
officer changes, Item 8.01 other events; DEF 14A: auditor/board sections). We
split on those headers first, then window each section into ~CHUNK_CHARS pieces
with overlap. Every chunk gets a stable, human-readable ID that doubles as a
CITATION target, e.g.  AAPL/10-K_2025-10-31#item7-c2.

Manifest join: data/manifest.csv gives ticker, company name, sector, form,
filing_date per doc — attached to every chunk as metadata so graph/vector
retrieval and the answer prompt can cite "AAPL (Apple Inc., Information
Technology) 10-K".  (The manifest's local_path column points at an old
'sp100_dataset/...' layout; we ignore it and walk data/<TICKER>/ directly.)

Outputs (datasets/sec/):
  corpus.jsonl   — one line per CHUNK: {title(=chunk_id), text, ticker, company,
                   sector, form, filing_date, section, doc_id}
  documents.jsonl— one line per DOC (full text) for the KG extractor: {doc_id,
                   text, ticker, company, sector, form, filing_date}
  manifest_clean.json — {ticker: {company, sector}} for downstream schema use
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

DATA_DIR = Path("data/data")
MANIFEST = DATA_DIR / "manifest.csv"
OUT_DIR = Path("datasets/sec")

CHUNK_CHARS = 1800       # target chunk size (chars) — ~450 tokens
CHUNK_OVERLAP = 250      # overlap between windows within a section
MIN_CHUNK_CHARS = 80     # drop tiny fragments
MAX_SECTION_HEAD = 8000  # for the KG extractor, cap per-doc text? no — keep full in documents.jsonl

# SEC item header at start of a line: "Item 7." / "Item 1A. Risk Factors" / "ITEM 9.01"
ITEM_RE = re.compile(r"^\s*(ITEM|Item)\s+(\d+[A-C]?(?:\.\d+)?)\.?\s*(.*)$")


def load_manifest() -> dict[str, dict]:
    """Return {ticker: {company, sector}} using proper CSV parsing (company names
    contain commas, which naive splitting breaks)."""
    out: dict[str, dict] = {}
    with open(MANIFEST, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("ticker") or "").strip()
            if not t:
                continue
            out.setdefault(t, {
                "company": (row.get("name") or "").strip(),
                "sector": (row.get("sector") or "").strip(),
            })
    return out


def form_from_filename(fname: str) -> str:
    # '10-K_2025-10-31.txt' -> '10-K' ; '8-K_2026-04-20_2.txt' -> '8-K' ; 'DEF14A_...' -> 'DEF 14A'
    stem = fname.rsplit(".", 1)[0]
    head = stem.split("_", 1)[0]
    return "DEF 14A" if head.upper() == "DEF14A" else head.upper()


def date_from_filename(fname: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    return m.group(1) if m else ""


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split into (section_label, section_text). Section label is normalised like
    'item7' / 'item1a' / 'item5.02'. The leading preamble (before the first real
    item body) becomes 'header'. We use the LAST occurrence of each item header as
    the real section start (the first occurrence is the table-of-contents line)."""
    lines = text.split("\n")
    # find header lines with substantive following content (title present, or long section)
    marks: list[tuple[int, str]] = []  # (line_idx, label)
    for i, ln in enumerate(lines):
        m = ITEM_RE.match(ln)
        if m:
            num = m.group(2).lower()
            title = (m.group(3) or "").strip()
            # Real section starts carry a title OR are followed by body text (not another item within 1 line).
            label = f"item{num}"
            marks.append((i, label))
    if not marks:
        return [("body", text)]

    # Keep the LAST mark for each distinct label — collapses TOC duplicates to the real section.
    last_for_label: dict[str, int] = {}
    for idx, label in marks:
        last_for_label[label] = idx
    starts = sorted(last_for_label.items(), key=lambda x: x[1])  # [(label, line_idx)] by position

    sections: list[tuple[str, str]] = []
    preamble = "\n".join(lines[: starts[0][1]]).strip()
    if len(preamble) >= MIN_CHUNK_CHARS:
        sections.append(("header", preamble))
    for j, (label, start_idx) in enumerate(starts):
        end_idx = starts[j + 1][1] if j + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start_idx:end_idx]).strip()
        if len(body) >= MIN_CHUNK_CHARS:
            sections.append((label, body))
    return sections


def window(text: str) -> list[str]:
    """Slide a CHUNK_CHARS window with overlap over a section, breaking near
    paragraph/sentence boundaries where possible."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= CHUNK_CHARS:
        return [text] if len(text) >= MIN_CHUNK_CHARS else []
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        if end < len(text):
            # try to end on a paragraph or sentence boundary within the last 300 chars
            window_tail = text[end - 300:end]
            for sep in ("\n\n", ". ", "\n"):
                pos = window_tail.rfind(sep)
                if pos != -1:
                    end = (end - 300) + pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def main() -> None:
    manifest = load_manifest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    corpus_f = open(OUT_DIR / "corpus.jsonl", "w")
    docs_f = open(OUT_DIR / "documents.jsonl", "w")

    n_docs = n_chunks = 0
    unknown_tickers = []
    per_form: dict[str, int] = {}

    for ticker in tickers:
        meta = manifest.get(ticker, {"company": ticker, "sector": ""})
        if ticker not in manifest:
            unknown_tickers.append(ticker)
        for fpath in sorted((DATA_DIR / ticker).glob("*.txt")):
            fname = fpath.name
            form = form_from_filename(fname)
            fdate = date_from_filename(fname)
            doc_id = f"{ticker}/{fname.rsplit('.', 1)[0]}"   # e.g. AAPL/10-K_2025-10-31
            text = fpath.read_text(errors="ignore")
            if not text.strip():
                continue
            n_docs += 1
            per_form[form] = per_form.get(form, 0) + 1

            # full-doc record for the KG extractor
            docs_f.write(json.dumps({
                "doc_id": doc_id, "text": text, "ticker": ticker,
                "company": meta["company"], "sector": meta["sector"],
                "form": form, "filing_date": fdate,
            }, ensure_ascii=False) + "\n")

            # chunked records for vector retrieval
            for label, sec_text in split_sections(text):
                pieces = window(sec_text)
                for ci, piece in enumerate(pieces):
                    chunk_id = f"{doc_id}#{label}-c{ci}"
                    corpus_f.write(json.dumps({
                        "title": chunk_id, "text": piece,
                        "doc_id": doc_id, "ticker": ticker,
                        "company": meta["company"], "sector": meta["sector"],
                        "form": form, "filing_date": fdate, "section": label,
                    }, ensure_ascii=False) + "\n")
                    n_chunks += 1

    corpus_f.close()
    docs_f.close()
    json.dump(manifest, open(OUT_DIR / "manifest_clean.json", "w"), indent=2)

    print(f"[corpus] docs={n_docs}  chunks={n_chunks}  avg_chunks/doc={n_chunks / max(n_docs,1):.1f}")
    print(f"[corpus] forms: {dict(sorted(per_form.items()))}")
    print(f"[corpus] tickers={len(tickers)}  manifest_covered={len(manifest)}")
    if unknown_tickers:
        print(f"[corpus] WARNING tickers missing from manifest: {unknown_tickers}")
    print(f"[corpus] wrote corpus.jsonl, documents.jsonl, manifest_clean.json -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
