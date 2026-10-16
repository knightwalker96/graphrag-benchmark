"""Olympics router lexicon — the question-side patterns, and NOTHING else.

The router *mechanism* — the priority chain, the `P3_ROUTE_FORCE` ablation hook, the fallback —
lives in `pipeline.py` and is copied structurally from `pipeline3_sec/pipeline.py`. Only this
file knows the word "Olympic".

WHY THE SEPARATION MATTERS
    `results/sec/router_generalization.json`: the SEC router on 150 Olympics questions labels
    100% of them `lookup` — 26/150 = 0.173, total collapse to the fallback branch.
    `results/sec/router_indomain_extra100.json`: the same router in domain scores 80/100 = 0.800.
    The un-confounded ablation (RESULTS_SEC.md §7) shows correct labels are worth +0.160
    strict and −1,152 tokens, so the mechanism is load-bearing and portable while the lexicon is
    not portable at all. Quarantining the lexicon here means a third dataset needs one new file.

--------------------------------------------------------------------------------------------
HOW THIS WAS BUILT, AND THE HONEST LIMIT ON WHAT ITS ACCURACY MEANS
--------------------------------------------------------------------------------------------
Every pattern below was written against the 30 questions in `datasets/olympics/dev_set.jsonl`
and nothing else. The held-out 70 and the hidden 50 were not measured against during
development; `scripts/router_lexicon_eval.py` scores the held-out set exactly once, after this
file is frozen.

One exposure, stated rather than glossed: step 1 of the plan instructed reading all 150
questions, so the author HAS read the held-out and hidden question strings. What dev-only
discipline can honestly claim here is that no pattern was *iterated against a measurement* on
those sets — which is the leak that actually inflates a score — not that the strings were never
seen. The weaker claim is the true one.

**And the accuracy this file achieves is a WEAK result, not a strong one.** All 150 questions in
this dataset are generated from exactly five rigid templates (OLYMPICS_DATA_NOTES.md §1), so a
near-perfect router here demonstrates far less than the SEC router's 0.800 did on genuinely
varied phrasing. The number belongs in the 2x2 as the in-domain control for the 0.173 collapse,
and nowhere near a claim that regex routing generalizes. The interesting cell of that table is
still the SEC lexicon's 0.173, not this one.

--------------------------------------------------------------------------------------------
PRIORITY ORDER: "NARROWEST DISCRIMINATOR FIRST, WITHIN EACH SHARED-PREFIX FAMILY"
--------------------------------------------------------------------------------------------
MEASURED FIRST, THEN ARGUED. The order below is **inert on this dataset**: exactly one pattern
matches each question (verified on all 30 dev questions and on all 50 hidden question strings —
a pattern-overlap check that uses no labels and scores nothing), and **all 120 possible
orderings of the chain score 30/30 on dev**. `scripts/router_lexicon_eval.py --probe-ordering`
reproduces both facts.

So the ordering is NOT what makes this lexicon correct — mutual exclusivity is. That is worth
stating plainly, because an earlier draft of this docstring claimed the order was load-bearing
("TEMPORAL FIRST, otherwise multi_hop siphons every temporal question") and the exhaustive
permutation test showed that claim was false for the patterns as actually written.

It is also the right property to have aimed for. The SEC router's correctness DID depend on its
ordering, and that dependence broke it twice: `AGG_RE` before `TEMPORAL_RE` was a hand-fix worth
Tier D +0.17 (RESULTS_SEC.md), and `AGG_RE` before `COMPARE_RE` then cost all 20
`compare` questions in eval_extra100 (§8.3). Both failures have one shape: two routes share a
sentence opening, and the route tested first had the *broader* discriminator. A chain whose
patterns are mutually exclusive cannot fail that way at all.

The order is therefore fixed as defence in depth, under a rule chosen so that loosening any
pattern later cannot silently reverse an outcome:

    When two routes share a question prefix, test the one whose DISCRIMINATING PHRASE IS
    NARROWER first.

The three shared-prefix families in this corpus, and how the rule places them:

  "Who won the gold medal in the ..."      temporal  before  multi_hop
        temporal's discriminator is "held immediately before <year>"; multi_hop's is
        "event held at <venue> on <date>". Both contain "held", so a multi_hop pattern loosened
        to key on "held at" alone would start swallowing temporal questions. Today it does not.

  "According to the provided corpus, ... competitors?"   superlative  before  aggregation
        superlative's is "the highest number of"; aggregation's is "more than <N>". Both name a
        (sport, games) group and end in "competitors?", so an aggregation pattern loosened to
        key on "competitors" would swallow superlatives. This is the closest structural analogue
        to the SEC compare defect, which is why superlative is placed ahead.

  "How many ..."                            aggregation  before  lookup
        aggregation's is "events ... more than <N> competitors"; lookup's is
        "nations competed in". lookup is also the FALLBACK, so it must be last on principle: a
        fallback that is tested early is not a fallback.

The patterns are deliberately a little wider than the dev strings require (SUPERLATIVE_RE
accepts "most"/"largest"/"fewest"/"lowest", AGGREGATION_RE accepts "fewer than" and "at least").
Those alternatives are NOT evidenced in dev — they are defensive generalization against
rephrasing, not fitting. `scripts/router_lexicon_eval.py --ablate-widening` strips every one of
them back to the literal dev wording and confirms dev accuracy is unchanged at 30/30, so the
widening cannot be quietly doing the work.
"""
from __future__ import annotations

import re

# --- 1. TEMPORAL ----------------------------------------------------------------------------
# dev: "Who won the gold medal in the <event> event at the Summer Olympics held immediately
#       before 2016?"  (7/7 dev temporal questions, identical shape)
# The narrow literal is the relative-time reference, not "who won".
TEMPORAL_RE = re.compile(
    r"\b(?:immediately|directly)\s+(?:before|after|preceding|following)\b"
    r"|\bgames?\s+(?:held\s+)?(?:before|after|preceding|following)\s+\d{4}\b"
    r"|\b(?:previous|preceding|next|following)\s+(?:summer|winter)\s+olympics\b",
    re.IGNORECASE)

# --- 2. MULTI_HOP ---------------------------------------------------------------------------
# dev: "Who won the gold medal in the event held at <venue> on <date>?"  (8/8 dev)
# Two attributes joined to identify ONE event, then a third attribute read off it. Requires BOTH
# the venue clause and the date clause, so it cannot fire on a temporal question.
MULTIHOP_RE = re.compile(
    r"\bevent\s+(?:held|staged|contested)\s+at\b.*\bon\b"
    r"|\bheld\s+at\b.+?\bon\s+(?:\d|[A-Z][a-z]+\s+\d)",
    re.IGNORECASE | re.DOTALL)

# --- 3. SUPERLATIVE -------------------------------------------------------------------------
# dev: "According to the provided corpus, which <sport> event at the <games> Olympics had the
#       highest number of competitors?"  (3/3 dev; all three say "highest")
# Rank a numeric attribute within a (sport, games) group and take the argmax.
SUPERLATIVE_RE = re.compile(
    r"\bwhich\b.*\b(?:had|has|with)\b.*\bthe\s+(?:highest|greatest|largest|most|lowest|fewest|smallest)\b"
    r"|\bthe\s+(?:highest|greatest|largest|lowest|fewest|smallest)\s+number\s+of\b"
    r"|\bwhich\b.*\bhad\s+the\s+most\b",
    re.IGNORECASE | re.DOTALL)

# --- 4. AGGREGATION -------------------------------------------------------------------------
# dev: "According to the provided corpus, how many <sport> events at the <games> Olympics had
#       more than <N> competitors?"  (6/6 dev)
# Count/filter members of a (sport, games) group. Requires the plural "events" AND a numeric
# threshold, which is what keeps it off the lookup template.
AGGREGATION_RE = re.compile(
    r"\bhow\s+many\b.*\bevents\b.*\b(?:more|fewer|less|greater|at\s+least|at\s+most|over|under)\b.*\b\d+\b"
    r"|\bhow\s+many\b.*\bevents\b.*\bcompetitors\b",
    re.IGNORECASE | re.DOTALL)

# --- 5. LOOKUP ------------------------------------------------------------------------------
# dev: "How many nations competed in <full event title>?"  (6/6 dev)
# One attribute read off ONE event that the question names in full. This is also the fallback
# route, exactly as in the SEC router.
LOOKUP_RE = re.compile(
    r"\bhow\s+many\s+(?:nations|countries|competitors|athletes)\s+(?:competed|took\s+part|participated)\b"
    r"|\bhow\s+many\s+(?:nations|countries|competitors|athletes)\b(?!.*\bevents\b)",
    re.IGNORECASE | re.DOTALL)

# The chain, in the order argued above. (regex, route) — first match wins.
ROUTE_CHAIN: list[tuple[re.Pattern, str]] = [
    (TEMPORAL_RE,    "temporal"),
    (MULTIHOP_RE,    "multi_hop"),
    (SUPERLATIVE_RE, "superlative"),
    (AGGREGATION_RE, "aggregation"),
    (LOOKUP_RE,      "lookup"),
]

FALLBACK = "lookup"

# The widened alternatives that are NOT evidenced in the dev set. Asserted to be inert on dev by
# scripts/router_lexicon_eval.py --ablate-widening, so they cannot be quietly doing the work.
WIDENED_ALTERNATIVES = [
    "SUPERLATIVE_RE: greatest|largest|most|lowest|fewest|smallest (dev shows only 'highest')",
    "AGGREGATION_RE: fewer|less|greater|at least|at most|over|under (dev shows only 'more than')",
    "LOOKUP_RE: countries|competitors|athletes, took part|participated (dev shows only "
    "'nations competed')",
    "TEMPORAL_RE: directly|after|preceding|following, 'previous Summer Olympics' "
    "(dev shows only 'immediately before')",
    "MULTIHOP_RE: staged|contested (dev shows only 'held at')",
]


def route_of(question: str) -> str:
    """The route label for a question. Pure function of the string: no corpus, no LLM, no tokens.

    Kept here next to the patterns so `scripts/router_lexicon_eval.py` can score the lexicon
    without importing the pipeline (which loads the index and opens a TigerGraph connection).
    `pipeline.py` calls this and owns the P3_ROUTE_FORCE ablation hook on top of it.
    """
    for pattern, name in ROUTE_CHAIN:
        if pattern.search(question or ""):
            return name
    return FALLBACK
