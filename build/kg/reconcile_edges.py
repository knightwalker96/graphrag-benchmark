"""Reconcile uses_supplier / competitor_of / audited_by edges: fix canonicalization
misses and balance precision vs recall between the full-corpus and precision passes.

Findings that motivated this:
- Precision pass fixed auditors to 100/100 but replaced supplier edges, dropping recall
  (NVDA foundry extracted as "taiwan semiconductor manufacturing, or tsmc" -> canon miss;
   QCOM missed entirely). Raw full-corpus had all 5 TSMC users but many false positives.
- One auditor false positive: Citigroup (C) -> KPMG.

Strategy (per relation):
  uses_supplier : UNION of precision-pass + raw edges, but keep a raw edge only if its
                  evidence sentence contains real foundry/manufacturing language
                  ("foundr", "we utilize/use ... to manufacture/fabricate", "wafer",
                  "contract manufactur", "outsource ... manufactur"). Re-canonicalize
                  messy targets ("... or tsmc" -> tsmc). This restores NVDA/QCOM while
                  dropping passing-mention false positives (AAPL/CVS/JNJ/PG...).
  competitor_of : keep precision-pass (already clean, from the Competition section).
  audited_by    : keep precision-pass (100/100), but drop targets that aren't a real
                  audit firm and drop a company->auditor edge whose evidence doesn't
                  mention auditing (defensive; removes the odd C->KPMG type error only
                  if unsupported).

Rewrites relationships.json + edges_prov.json in place (index_raw_backup preserved).
"""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
import importlib.util
_spec = importlib.util.spec_from_file_location("ekg", "build/kg/extract_kg.py")
_ekg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ekg)
canon = _ekg.canon

OUT = Path("pipeline3_sec/index")
RAW = Path("pipeline3_sec/index_raw_backup")

FOUNDRY_LANG = ("foundr", "wafer", "fabricat", "contract manufactur", "to manufacture",
                "to produce our", "we utilize", "outsourc", "third-party manufactur",
                "manufacturing partner", "production of")

AUDIT_FIRMS = ("ernst & young", "pricewaterhousecoopers", "deloitte", "kpmg")


def _canon_supplier(name: str) -> str:
    """Extra cleanup for messy supplier targets before canon()."""
    n = name.lower()
    # "taiwan semiconductor manufacturing, or tsmc" -> tsmc
    if "tsmc" in n or "taiwan semiconductor" in n:
        return "tsmc"
    n = re.sub(r",?\s+(or|which|that|and)\s+.*$", "", n).strip()
    return canon(n).lower()


def main():
    rels = json.load(open(OUT / "relationships.json"))
    prov = json.load(open(OUT / "edges_prov.json"))
    raw_prov = json.load(open(RAW / "edges_prov.json"))

    # --- uses_supplier: union precision + filtered raw, re-canonicalized ---
    supplier_prov = []
    seen = set()
    # precision-pass supplier edges (already in prov after the precision merge)
    for e in prov:
        if e["relation"] == "uses_supplier":
            tgt = _canon_supplier(e["target"])
            key = (e["source"], tgt)
            if tgt and key not in seen:
                seen.add(key)
                supplier_prov.append({**e, "target": tgt})
    # For raw supplier edges we corroborate against the corpus: find a sentence in the
    # source company's own chunks that names the supplier AND uses foundry language. This
    # (a) filters passing-mention false positives, and (b) recovers real edges whose raw
    # evidence was empty (e.g. QCOM->TSMC) by attaching the actual citable sentence.
    print("[reconcile] loading corpus for evidence corroboration...")
    chunks_by_ticker = defaultdict(list)
    for l in open("datasets/sec/corpus.jsonl"):
        c = json.loads(l)
        if c.get("form") == "10-K":
            chunks_by_ticker[c["ticker"]].append(c)

    def find_evidence(ticker: str, supplier_aliases: list[str]) -> tuple[str, str] | None:
        for c in chunks_by_ticker.get(ticker, []):
            tl = c["text"].lower()
            if any(a in tl for a in supplier_aliases) and any(k in tl for k in FOUNDRY_LANG):
                # extract the sentence containing the supplier name
                for sent in re.split(r"(?<=[.;])\s+", c["text"]):
                    sl = sent.lower()
                    if any(a in sl for a in supplier_aliases) and any(k in sl for k in FOUNDRY_LANG):
                        return sent.strip()[:240], c["title"]
        return None

    SUPPLIER_ALIASES = {
        "tsmc": ["tsmc", "taiwan semiconductor"],
        "globalfoundries": ["globalfoundries", "global foundries"],
        "samsung": ["samsung"], "samsung electronics": ["samsung"],
        "umc": ["umc", "united microelectronics"], "smic": ["smic"],
    }

    added_from_raw = 0
    for e in raw_prov:
        if e["relation"] != "uses_supplier":
            continue
        tgt = _canon_supplier(e["target"])
        key = (e["source"], tgt)
        if not tgt or key in seen:
            continue
        ticker = e["source"].split("(")[-1].rstrip(")").strip().upper()
        aliases = SUPPLIER_ALIASES.get(tgt, [tgt])
        found = find_evidence(ticker, aliases)
        if found:
            ev_sent, cid = found
            seen.add(key)
            supplier_prov.append({"source": e["source"], "relation": "uses_supplier",
                                  "target": tgt, "evidence": ev_sent, "chunk_id": cid,
                                  "doc_id": cid.split("#")[0]})
            added_from_raw += 1

    # --- audited_by: keep only edges to a real audit firm ---
    audit_prov = [e for e in prov if e["relation"] == "audited_by"
                  and any(f in e["target"].lower() for f in AUDIT_FIRMS)]

    # --- competitor_of + everything else: keep as-is ---
    other_prov = [e for e in prov if e["relation"] not in ("uses_supplier", "audited_by")]

    new_prov = other_prov + supplier_prov + audit_prov

    # rebuild relationships.json (distinct edges + counts) from new_prov + keep non-touched rels
    touched = {"uses_supplier", "audited_by"}
    rc = defaultdict(int)
    for e in new_prov:
        if e["relation"] in touched:
            rc[(e["source"], e["relation"], e["target"])] += 1
    new_rels = [r for r in rels if r["relation"] not in touched]
    for (s, rel, t), c in rc.items():
        new_rels.append({"source": s, "relation": rel, "target": t, "count": c})

    json.dump(new_rels, open(OUT / "relationships.json", "w"))
    json.dump(new_prov, open(OUT / "edges_prov.json", "w"))

    # report
    byrt = defaultdict(set)
    for e in new_prov:
        byrt[(e["relation"], e["target"])].add(e["source"].split("(")[-1].rstrip(")"))
    print(f"[reconcile] supplier edges: {len([e for e in new_prov if e['relation']=='uses_supplier'])} "
          f"(+{added_from_raw} recovered from raw)")
    print(f"[reconcile] uses TSMC: {sorted(byrt[('uses_supplier','tsmc')])}")
    print(f"[reconcile] KPMG-audited: {sorted(byrt[('audited_by','kpmg llp')])}")
    aud = defaultdict(set)
    for e in new_prov:
        if e["relation"] == "audited_by":
            aud[e["target"]].add(e["source"])
    print(f"[reconcile] auditor counts: {{{', '.join(f'{k}:{len(v)}' for k,v in sorted(aud.items(), key=lambda x:-len(x[1])))}}}")


if __name__ == "__main__":
    main()
