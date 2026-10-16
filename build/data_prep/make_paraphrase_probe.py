#!/usr/bin/env python3
"""Mechanical paraphrase-robustness suite for the deterministic router (the optimization §6.3).

WHY. The single documented router defect in this project is a phrasing failure, not a
vocabulary failure: all 20 `compare` questions in extra100 misrouted because the pattern had
been fitted to the held-out set's wording ("Which HAD the higher X: A or B?") and broke on
"Which company reported higher X, A or B?" — same domain, same vocabulary. A pattern set
fitted to 150 specific strings will do that again unless it is tested for it.

WHAT. Each rule is a structural, meaning-preserving rewrite (voice change, clause reorder,
interrogative <-> imperative, possessive <-> of-phrase, synonym of a FUNCTION word). A
paraphrase inherits its original's gold route by construction, because none of these rewrites
changes what is being asked. The probe then asserts route(paraphrase) == route(original).

RULES ARE WRITTEN BEFORE ANY TUNING and are never edited to flatter a result. They are test
data, not pipeline code — the §0.1 ban on question-identifying lexicons applies to the
pipeline, not to the probe that attacks it.

Usage:  python "build/data_prep/make_paraphrase_probe.py"
Writes: datasets/sec/paraphrase_probe.jsonl
"""
from __future__ import annotations
import json, re
from pathlib import Path

SRC = Path("datasets/sec/combined150_set.jsonl")
OUT = Path("datasets/sec/paraphrase_probe.jsonl")

# Each rule: (name, compiled pattern, template or callable). A rule fires only when its
# pattern matches; most questions therefore get 2-4 paraphrases and every question gets >=1.
RULES: list[tuple[str, re.Pattern, object]] = []


def _poss(name: str) -> str:
    """English possessive, respecting names that already end in s."""
    n = name.strip()
    return n + ("'" if n.lower().endswith("s") else "'s")


def rule(name: str, pattern: str, repl, flags=re.I):
    RULES.append((name, re.compile(pattern, flags), repl))


# ---- aggregation: auditor ("Which companies are audited by <FIRM>?") ----
rule("agg_auditor.passive2active",
     r"^which companies are audited by (?P<firm>.+?)\??$",
     lambda m: f"Which companies does {m['firm']} audit?")
rule("agg_auditor.imperative",
     r"^which companies are audited by (?P<firm>.+?)\??$",
     lambda m: f"List all companies audited by {m['firm']}.")
rule("agg_auditor.name_the",
     r"^which companies are audited by (?P<firm>.+?)\??$",
     lambda m: f"Name the companies that {m['firm']} audits.")

# ---- aggregation: sector ("Which companies are in the <SECTOR> sector?") ----
rule("agg_sector.classified",
     r"^which companies are in the (?P<sec>.+?) sector\??$",
     lambda m: f"Which companies are classified in the {m['sec']} sector?")
rule("agg_sector.imperative",
     r"^which companies are in the (?P<sec>.+?) sector\??$",
     lambda m: f"List the companies in the {m['sec']} sector.")
rule("agg_sector.belong",
     r"^which companies are in the (?P<sec>.+?) sector\??$",
     lambda m: f"Which companies belong to the {m['sec']} sector?")

# ---- aggregation: sector x auditor ("Which <SECTOR> companies are audited by <FIRM>?") ----
rule("agg_sector_auditor.expand",
     r"^which (?P<sec>.+?) companies are audited by (?P<firm>.+?)\??$",
     lambda m: f"Which companies in the {m['sec']} sector are audited by {m['firm']}?")
rule("agg_sector_auditor.among",
     r"^which (?P<sec>.+?) companies are audited by (?P<firm>.+?)\??$",
     lambda m: f"Among {m['sec']} companies, which ones does {m['firm']} audit?")

# ---- aggregation: supplier ("How many companies name <X> as a supplier, and which ...?") ----
rule("agg_supplier.reorder",
     r"^how many companies name (?P<sup>.+?) as a supplier, and which companies are they\??$",
     lambda m: f"Which companies name {m['sup']} as a supplier, and how many are there?")
rule("agg_supplier.imperative",
     r"^how many companies name (?P<sup>.+?) as a supplier, and which companies are they\??$",
     lambda m: f"Name the companies that list {m['sup']} as a supplier, and give the count.")
rule("agg_supplier.listform",
     r"^how many companies name (?P<sup>.+?) as a supplier, and which companies are they\??$",
     lambda m: f"How many companies list {m['sup']} as a supplier, and which are they?")

# ---- bridge: shared auditor ("The company that audits <X> — which others does it audit?") ----
rule("bridge_auditor.possessive",
     r"^the company that audits (?P<co>.+?) [—–-]+ which others does it audit\??$",
     lambda m: f"{m['co']}'s auditor also audits which other companies?")
rule("bridge_auditor.share",
     r"^the company that audits (?P<co>.+?) [—–-]+ which others does it audit\??$",
     lambda m: f"Which other companies share the same auditor as {m['co']}?")
rule("bridge_auditor.clause_reorder",
     r"^the company that audits (?P<co>.+?) [—–-]+ which others does it audit\??$",
     lambda m: f"Which others are audited by the company that audits {m['co']}?")

# ---- bridge: shared director ----
rule("bridge_director.person_first",
     r"^the company whose board includes (?P<p>.+?) [—–-]+ on which other boards does that person serve\??$",
     lambda m: f"{m['p']} sits on which other company boards?")
rule("bridge_director.besides",
     r"^the company whose board includes (?P<p>.+?) [—–-]+ on which other boards does that person serve\??$",
     lambda m: f"Besides that company, on which other boards does {m['p']} serve?")

# ---- bridge: shared supplier ----
rule("bridge_supplier.reorder",
     r"^the company whose foundry partner is named by (?P<co>.+?) [—–-]+ which other firms name that same supplier\??$",
     lambda m: f"Which other firms name the same foundry partner that {m['co']} names?")
rule("bridge_supplier.possessive",
     r"^the company whose foundry partner is named by (?P<co>.+?) [—–-]+ which other firms name that same supplier\??$",
     lambda m: f"{m['co']}'s foundry partner is also named by which other firms?")

# ---- compare: the documented defect. "reported higher X, A or B" <-> "had the higher X: A or B"
rule("compare.reported2had",
     r"^which company reported higher (?P<metric>.+?), (?P<a>.+?) or (?P<b>.+?)\??$",
     lambda m: f"Which had the higher {m['metric']}: {m['a']} or {m['b']}?")
rule("compare.between",
     r"^which company reported higher (?P<metric>.+?), (?P<a>.+?) or (?P<b>.+?)\??$",
     lambda m: f"Between {m['a']} and {m['b']}, which reported greater {m['metric']}?")
rule("compare.whose",
     r"^which company reported higher (?P<metric>.+?), (?P<a>.+?) or (?P<b>.+?)\??$",
     lambda m: f"Whose {m['metric']} was larger, {_poss(m['a'])} or {_poss(m['b'])}?")
rule("compare.posted",
     r"^which company reported higher (?P<metric>.+?), (?P<a>.+?) or (?P<b>.+?)\??$",
     lambda m: f"Which firm posted the higher {m['metric']}, {m['a']} or {m['b']}?")
# the reverse direction, so held-out-style phrasings are attacked too
rule("compare.had2reported",
     r"^which had the higher (?P<metric>.+?): (?P<a>.+?) or (?P<b>.+?)\??$",
     lambda m: f"Which company reported higher {m['metric']}, {m['a']} or {m['b']}?")

# ---- lookup: auditor ("Who audits <X>?") ----
rule("lookup_auditor.possessive",
     r"^who audits (?P<co>.+?)\??$",
     lambda m: f"Who is {m['co']}'s auditor?")
rule("lookup_auditor.firm",
     r"^who audits (?P<co>.+?)\??$",
     lambda m: f"Which accounting firm audits {m['co']}?")
rule("lookup_auditor.passive",
     r"^who audits (?P<co>.+?)\??$",
     lambda m: f"{m['co']} is audited by which firm?")
rule("lookup_auditor.independent",
     r"^who audits (?P<co>.+?)\??$",
     lambda m: f"Who is the independent auditor of {m['co']}?")

# ---- lookup: revenue ("What were <X>'s total revenues for fiscal year <Y>?") ----
rule("lookup_revenue.terse",
     r"^what were (?P<co>.+?)'s (?P<metric>[\w &/-]+?) for (?P<period>.+?)\??$",
     lambda m: f"{m['co']}'s {m['metric']} for {m['period']}?")
rule("lookup_revenue.howmuch",
     r"^what were (?P<co>.+?)'s (?P<metric>[\w &/-]+?) for (?P<period>.+?)\??$",
     lambda m: f"How much did {m['co']} report in {m['metric']} for {m['period']}?")
rule("lookup_revenue.imperative",
     r"^what were (?P<co>.+?)'s (?P<metric>[\w &/-]+?) for (?P<period>.+?)\??$",
     lambda m: f"Report the {m['metric']} of {m['co']} for {m['period']}.")

# ---- temporal: ordering two events ----
rule("temporal.whichfirst",
     r"^put these two (?P<co>.+?) events in chronological order: the (?P<a>.+?) and the (?P<b>.+?)\.?$",
     lambda m: f"For {m['co']}, which came first: the {m['a']} or the {m['b']}?")
rule("temporal.imperative",
     r"^put these two (?P<co>.+?) events in chronological order: the (?P<a>.+?) and the (?P<b>.+?)\.?$",
     lambda m: f"Order these two {m['co']} events chronologically: the {m['a']} and the {m['b']}.")
rule("temporal.sequence",
     r"^put these two (?P<co>.+?) events in chronological order: the (?P<a>.+?) and the (?P<b>.+?)\.?$",
     lambda m: f"In what sequence did {m['co']}'s {m['a']} and {m['b']} occur?")
rule("temporal.generic_order",
     r"^put these (?P<rest>.+?) in chronological order: (?P<items>.+?)\.?$",
     lambda m: f"Arrange these {m['rest']} chronologically: {m['items']}.")

# ---- generic rules, mostly for the hand-written held-out 50 ----
# case-SENSITIVE (flags=0): this rule keys on a leading Capitalized company name, and under
# re.I the [A-Z] classes would match lowercase words too ("What were <Co>'s ..." would be
# captured as the company). Everything after the possessive stays case-insensitive by nature.
rule("generic.possessive2of",
     r"^(?P<co>[A-Z][\w.&'-]*(?: [A-Z][\w.&'-]*){0,3})'s (?P<attr>.+?)\?$",
     # skip compound questions ("<Co>'s X is whom, and who is ...?"): fronting the company
     # leaves the second clause dangling and the result is not a faithful paraphrase.
     lambda m: (None if re.search(r"\b(is|are|was|were|does|do)\b", m["attr"])
                else f"For {m['co']}, what was the {m['attr']}?"), flags=0)
rule("generic.which2list",
     r"^which companies (?P<rest>.+?)\?$",
     lambda m: f"List the companies that {m['rest']}.")
rule("generic.whichtwo_reorder",
     r"^two companies (?P<rest>.+?) [—–-]+ which two\??$",
     lambda m: f"Which two companies {m['rest']}?")
rule("generic.serves_reorder",
     r"^(?P<p>.+?) serves on the boards of which two (?P<what>.+?)\??$",
     lambda m: f"On the boards of which two {m['what']} does {m['p']} serve?")
rule("generic.listall2give",
     r"^list all (?P<rest>.+?)\.?$",
     lambda m: f"Give me every one of the {m['rest']}.")
rule("generic.fy_expand",
     r"\bFY(?P<y>\d{4})\b",
     lambda m: None)   # handled specially below (substitution, not a full rewrite)
rule("generic.besides_reorder",
     r"^besides (?P<x>.+?), which company (?P<rest>.+?)\?$",
     lambda m: f"Which company other than {m['x']} {m['rest']}?")
rule("generic.dash_clause_swap",
     r"^the company that (?P<clause>.+?) [—–-]+ (?P<ask>.+?)\?$",
     lambda m: f"{m['ask'][0].upper()}{m['ask'][1:]}, for the company that {m['clause']}?")


def paraphrases(q: str) -> list[tuple[str, str]]:
    """All (rule_name, paraphrase) pairs for a question. Deterministic, order-stable."""
    out: list[tuple[str, str]] = []
    for name, pat, repl in RULES:
        if name == "generic.fy_expand":
            if pat.search(q):
                out.append((name, pat.sub(lambda m: f"fiscal year {m['y']}", q)))
            continue
        m = pat.match(q.strip())
        if not m:
            continue
        try:
            p = repl(m) if callable(repl) else repl
        except Exception:
            continue
        if not p:
            continue
        p = re.sub(r"\s+", " ", p).strip()
        if p.lower() != q.strip().lower() and all(p.lower() != o.lower() for _, o in out):
            out.append((name, p))
    # last-resort generic: FY <-> fiscal year the other direction, so every question gets >=1
    if not out:
        if re.search(r"\bfiscal year (\d{4})\b", q, re.I):
            out.append(("generic.fy_abbrev",
                        re.sub(r"\bfiscal year (\d{4})\b", r"FY\1", q, flags=re.I)))
        elif re.match(r"^the company that (?P<c>.+?) [—–-]+ (?P<a>.+?)\.?$", q.strip(), re.I):
            mm = re.match(r"^the company that (?P<c>.+?) [—–-]+ (?P<a>.+?)\.?$", q.strip(), re.I)
            out.append(("generic.dash_clause_period",
                        f"For the company that {mm['c']}, {mm['a']}."))
        elif q.strip().endswith("?"):
            out.append(("generic.tell_me", "Tell me: " + q.strip()[0].lower() + q.strip()[1:]))
    return out


def main() -> None:
    rows = [json.loads(l) for l in open(SRC)]
    probe, no_para = [], []
    for r in rows:
        ps = paraphrases(r["question"])
        if not ps:
            no_para.append(r["id"])
        for i, (rule_name, p) in enumerate(ps, 1):
            probe.append({"id": f"{r['id']}#p{i}", "orig_id": r["id"], "rule": rule_name,
                          "question": p, "orig_question": r["question"],
                          "route_gold": r.get("route_gold", ""), "shape": r.get("shape", ""),
                          "tier": r.get("tier", ""), "set_name": r.get("set_name", "")})
    with open(OUT, "w") as f:
        for p in probe:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {OUT}  n={len(probe)} paraphrases of {len(rows)} questions "
          f"({len(probe)/len(rows):.1f} per question)")
    if no_para:
        print(f"  !! no paraphrase generated for {len(no_para)}: {no_para}")
    from collections import Counter
    c = Counter(p["rule"] for p in probe)
    print(f"\n{'rule':34}{'n':>4}")
    for k, v in sorted(c.items()):
        print(f"{k:34}{v:4d}")


if __name__ == "__main__":
    main()
