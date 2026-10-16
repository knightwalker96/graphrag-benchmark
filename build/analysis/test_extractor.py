#!/usr/bin/env python3
"""Unit tests for the optimization shared extractors. No LLM, no TigerGraph, no network.

WHAT IS BEING ASSERTED. For a question naming a company and a financial metric, the extractor
must return the ONE statement row carrying the answer. The test checks that the gold figure
literally appears in the extracted row, which is an objective pass/fail rather than a judgement
call. Gold answers are used HERE, offline, as test fixtures only — the pipeline itself never
reads them (see the no-gold-at-inference rule in the optimization brief).

Also asserted: the relation gate fires on every ordinary English phrasing of an auditor
question (the initial implementation defect), and the curated-edge path returns a real evidence sentence.

Usage:  python "build/analysis/test_extractor.py"
"""
from __future__ import annotations
import importlib.util, json, os, re, sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
os.environ.setdefault("DATASET", "sec")
os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("P3_REQUIRE_TG", "0")      # these tests never touch the graph

spec = importlib.util.spec_from_file_location("p3t", "pipeline3_sec/pipeline.py")
p3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p3)
p3._load()

ROWS = {r["id"]: r for r in
        (json.loads(l) for l in open("datasets/sec/combined150_set.jsonl"))}


def digits(s: str) -> list[str]:
    """Comparable numeric tokens: '$88,309 million' -> ['88309']; drops years and small ints
    so a match cannot be scored on a '2025' or a '6 %' that appears in every row."""
    out = []
    for t in re.findall(r"\d[\d,]*(?:\.\d+)?", s or ""):
        v = t.replace(",", "")
        if len(v.split(".")[0]) >= 4 and not re.fullmatch(r"(19|20)\d{2}", v):
            out.append(v)
    return out


def gold_values(ans: str) -> list[float]:
    """Gold figures as numbers, unit-normalized to millions.

    A string compare is the wrong test here. Gold sometimes states a rounded figure in a
    different unit from the filing — '$104.8 billion' against the row's '104,780' (millions) —
    so the check is numeric with a relative tolerance, not substring membership."""
    out = [float(v) for v in digits(ans)]
    for t in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*billion", ans or "", re.I):
        out.append(float(t.replace(",", "")) * 1000.0)
    for t in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*thousand", ans or "", re.I):
        out.append(float(t.replace(",", "")) / 1000.0)
    return out


def matches(ans: str, block: str, rel_tol: float = 0.002) -> bool:
    """True if any gold figure appears in the extracted row within rel_tol (0.2%), which
    absorbs the gold's own rounding without admitting a different line item."""
    g, got = gold_values(ans), [float(v) for v in digits(block)]
    return any(abs(a - b) <= rel_tol * max(abs(a), abs(b)) for a in g for b in got)


passes, fails = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (passes if ok else fails).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if detail else ""))


# =====================================================================================
print("\n=== A. metric-row extraction: gold figure must appear in the extracted row ===")
# 10 hand-checkable cases: the extra100 single-company revenue lookups.
metric_cases = [q for q in sorted(ROWS) if ROWS[q].get("shape") == "lookup_revenue"]
for qid in metric_cases:
    r = ROWS[qid]
    labels = p3._metric_labels_for(r["question"])
    seed = p3._lookup_company_seed(r["question"])
    block, cids = p3._metric_block(seed, labels) if seed else ("", [])
    ok = matches(r["answer"], block)
    check(f"{qid} {r['question'][:52]:54} gold={r['answer'][:18]}",
          ok, "" if ok else f"seed={seed!r} labels={labels[:3]} block={block[:150]!r}")

# =====================================================================================
print("\n=== B. metric-row extraction on the official held-out Tier A numerics ===")
for qid in ("EQ01", "EQ02", "EQ03", "EQ04", "EQ05"):
    r = ROWS[qid]
    labels = p3._metric_labels_for(r["question"])
    seed = p3._lookup_company_seed(r["question"])
    block, _ = p3._metric_block(seed, labels) if seed else ("", [])
    ok = matches(r["answer"], block)
    check(f"{qid} {r['question'][:52]:54} gold={r['answer'][:18]}",
          ok, "" if ok else f"seed={seed!r} labels={labels[:3]} block={block[:150]!r}")

# =====================================================================================
print("\n=== C. compare: BOTH companies' rows must be recovered ===")
cmp_cases = [q for q in sorted(ROWS) if (ROWS[q].get("shape") or "").startswith("compare")][:6]
for qid in cmp_cases:
    r = ROWS[qid]
    labels = p3._metric_labels_for(r["question"])
    comps = p3._compare_companies(r["question"])
    blocks = [p3._metric_block(c, labels)[0] for c in comps]
    blocks = [b for b in blocks if b]
    ok = len(comps) >= 2 and len(blocks) >= 2
    check(f"{qid} {r['question'][:52]:54} companies={len(comps)} rows={len(blocks)}",
          ok, "" if ok else f"comps={comps} labels={labels[:2]}")

# =====================================================================================
print("\n=== D. the initial implementation gate defect: auditor relation must fire on every phrasing ===")
phrasings = ["Who audits Altria?", "Who is Altria's auditor?", "Which firm audits Altria?",
             "Altria is audited by which firm?", "What accounting firm audits Altria?",
             "Who is the independent auditor of Altria?",
             "Which accounting firm signs Altria's financial statements?"]
for q in phrasings:
    rels = p3._wanted_relations(q)
    check(f"audited_by detected in: {q:58}", "audited_by" in rels, f"got {rels}")
check("the initial implementation gate (_wants_curated_facts) MISSED 'Who audits Altria?' — regression guard",
      not p3._wants_curated_facts("Who audits Altria?"))

# =====================================================================================
print("\n=== E. numeric questions must NOT pull curated relations (no token waste) ===")
for q in ["What were Boeing's total revenues for fiscal year 2025?",
          "Apple's total net sales for fiscal year 2025?",
          "Which company reported higher net income, Qualcomm or Salesforce?"]:
    rels = p3._wanted_relations(q)
    check(f"no curated relations for: {q[:58]:60}", rels == [], f"got {rels}")

# =====================================================================================
print("\n=== F. curated-edge evidence: a real sentence + chunk id, for the asked relation only ===")
for qid in ("SQ061", "SQ064", "SQ068"):
    r = ROWS[qid]
    seed = p3._lookup_company_seed(r["question"])
    rels = p3._wanted_relations(r["question"])
    prov, lines = p3._curated_prov(seed, rels) if seed else ([], [])
    has_ev = any((e.get("evidence") or "").strip() and e.get("chunk_id") for e in prov)
    only_asked = all(e["relation"] in rels for e in prov)
    check(f"{qid} {r['question'][:34]:36} rels={rels} edges={len(prov)} evidence={has_ev}",
          bool(prov) and has_ev and only_asked,
          f"lines={lines[:2]}")

# =====================================================================================
print("\n=== G. extractor returns nothing when it should (safe fallback) ===")
check("no metric labels for a non-financial question",
      p3._metric_labels_for("Who audits Altria?") == [])
check("empty block for a company with no matching row",
      p3._metric_block("altria (mo)", ["total gizmos sold"])[0] == "")

# =====================================================================================
print("\n=== H. header recovery: the row must arrive with its year/units context ===")
rows, hdr = p3._extract_metric_rows("TMUS/10-K_2026-02-11#item7-c16", ["total revenues"])
check("TMUS item7-c16 row is the Total revenues line",
      bool(rows) and rows[0].lower().startswith("total revenues"), f"rows={rows}")
check("TMUS item7-c16 header carries >=2 year columns",
      len(re.findall(r"\b(?:19|20)\d{2}\b", hdr)) >= 2, f"hdr={hdr!r}")
rows2, hdr2 = p3._extract_metric_rows("AAPL/10-K_2025-10-31#item7-c3", ["total net sales"])
check("AAPL item7-c3 row is the Total net sales line",
      bool(rows2) and "416,161" in rows2[0], f"rows={rows2}")
check("AAPL item7-c3 header carries units or years",
      bool(hdr2) and ("millions" in hdr2.lower() or
                      len(re.findall(r"\b(?:19|20)\d{2}\b", hdr2)) >= 2), f"hdr={hdr2!r}")

print(f"\n{'='*86}\nPASSED {len(passes)}   FAILED {len(fails)}")
if fails:
    print("failing:", fails)
sys.exit(1 if fails else 0)
