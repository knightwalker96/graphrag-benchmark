"""Set-F1 for the set-valued questions in extra100 (SQ001-SQ100).

Why this metric exists. Strict pass is a single PASS/FAIL from the judge, so a 32-member
aggregation answer that gets 30 companies right scores exactly the same as one that gets none.
That is the right call for a headline correctness number but it throws away the signal that
matters most on list questions, so set-F1 is reported ALONGSIDE strict/graded/BERTScore,
never instead of them.

Scope. `gold_members` is polymorphic across the five routes:

  aggregation (20)  company-name sets, 2-32 members   -> set-F1 meaningful
  bridge      (20)  company-name sets, 1-11 members   -> set-F1 meaningful
  compare     (20)  ONE company (the winner)          -> set-F1 computable, degenerates to
                                                         exact match; reported separately
  lookup      (20)  one auditor name or dollar figure -> NOT a set, skipped
  temporal    (20)  two dates                         -> NOT a company set, skipped

So set-F1 is computed only where every gold member resolves to a company in
`datasets/sec/manifest_clean.json` (verified: 60/60 rows across the first three routes,
0/40 across the last two).

Membership detection. The answer space is CLOSED — exactly the 100 companies in the manifest —
so membership is detected by matching each company's surface forms against the answer text
rather than by splitting on commas, which would break on prose answers. Matching is:

  * case-sensitive (company names are proper nouns; this stops "the target company" from
    scoring as Target Corporation),
  * word-boundary anchored,
  * longest-surface-first with matched spans blanked out, so "Intuitive Surgical" is never
    also counted as Intuit (the one containment pair among the 100 names).

Known limit, stated rather than hidden: a company named in passing ("unlike Apple, …") counts
as a predicted member. That inflates recall-side noise slightly for every pipeline equally.
"""
from __future__ import annotations
import json
import re
from functools import lru_cache

MANIFEST = "datasets/sec/manifest_clean.json"

# corporate-suffix stripping, applied longest-first
_SUFFIXES = [", Inc.", " Corporation", " Company", " Incorporated", " Inc.", " plc", " Co.",
             " Group", " LLP", " N.A."]
_PARENS = re.compile(r"\s*\([^)]*\)")


def _surfaces(name: str, ticker: str) -> list[str]:
    """Surface forms a generator might plausibly use for one company.

    The ticker is kept whatever its length. 19 of the 100 tickers are 1-2 characters (T=AT&T,
    V=Visa Inc., C=Citigroup, KO, MO, SO, …) and P3 in particular answers list questions in
    tickers, so dropping them silently cost real recall: an answer naming "T, V, C" scored those
    three companies as missing. Short tickers are safe here because matching is case-SENSITIVE,
    uppercase and word-boundary anchored — prose "so"/"Ms."/"the" cannot collide with SO/MS/T.
    Name-derived surfaces still require 2+ characters.
    """
    bare = _PARENS.sub("", name).strip()          # "Alphabet Inc. (Class A)" -> "Alphabet Inc."
    out = {name, bare}
    for s in (name, bare):
        for suf in _SUFFIXES:
            if s.endswith(suf):
                out.add(s[: -len(suf)].strip())
    # "Coca-Cola Company (The)" -> bare "Coca-Cola Company" -> "Coca-Cola"
    out = {s for s in out if len(s) >= 2}
    out.add(ticker)
    return sorted(out, key=len, reverse=True)


@lru_cache(maxsize=1)
def _alias_table() -> list[tuple[str, str]]:
    """[(surface, canonical_company_name)] sorted longest surface first."""
    man = json.load(open(MANIFEST))
    pairs: list[tuple[str, str]] = []
    for ticker, v in man.items():
        for s in _surfaces(v["company"], ticker):
            pairs.append((s, v["company"]))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


@lru_cache(maxsize=1)
def _canonical_names() -> frozenset[str]:
    return frozenset(v["company"] for v in json.load(open(MANIFEST)).values())


def is_company_set(gold_members: list) -> bool:
    """True when every gold member is a manifest company — i.e. set-F1 applies."""
    return bool(gold_members) and all(g in _canonical_names() for g in gold_members)


def extract_members(text: str) -> set[str]:
    """Companies named in `text`, as canonical manifest names."""
    if not text:
        return set()
    buf = text
    found: set[str] = set()
    for surface, canonical in _alias_table():
        if canonical in found:
            continue
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(surface) + r"(?![A-Za-z0-9])")
        m = pat.search(buf)
        if m:
            found.add(canonical)
            buf = buf[: m.start()] + " " * (m.end() - m.start()) + buf[m.end():]
    return found


def set_f1(answer: str, gold_members: list) -> dict | None:
    """Precision/recall/F1 of the companies named in `answer` against `gold_members`.

    Returns None when the question is not a company-set question (so the caller can leave the
    field null rather than recording a misleading 0.0).
    """
    if not is_company_set(gold_members):
        return None
    gold = set(gold_members)
    pred = extract_members(answer)
    tp = len(pred & gold)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"set_precision": round(prec, 4), "set_recall": round(rec, 4), "set_f1": round(f1, 4),
            "n_gold": len(gold), "n_pred": len(pred),
            "missing": sorted(gold - pred), "spurious": sorted(pred - gold)}


if __name__ == "__main__":
    # Self-test: extracting from each question's OWN gold answer string must recover
    # gold_members exactly. Any F1 < 1.0 here is a normalizer bug, not a pipeline result.
    import collections
    rows = [json.loads(l) for l in open("datasets/sec/extra100_set.jsonl")]
    scope = collections.Counter()
    bad = []
    for r in rows:
        res = set_f1(r["answer"], r["gold_members"])
        if res is None:
            scope[f"{r['route_gold']}: skipped (not a company set)"] += 1
            continue
        scope[f"{r['route_gold']}: scored"] += 1
        if res["set_f1"] < 1.0:
            bad.append((r["id"], r["route_gold"], res))
    for k in sorted(scope):
        print(f"  {k}: {scope[k]}")
    print(f"\nround-trip failures: {len(bad)}")
    for i, rt, res in bad[:10]:
        print(f"  {i} {rt} f1={res['set_f1']} missing={res['missing'][:4]} spurious={res['spurious'][:4]}")
