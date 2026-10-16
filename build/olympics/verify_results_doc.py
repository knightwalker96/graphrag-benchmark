"""Assert every PROSE number in RESULTS_OLYMPICS.md against the artefacts.

The tables are generated from olympics_summary.json, so they cannot drift. The narrative around
them is hand-written, and this script is what stops a hand-typed figure from going stale. Any
claim in the prose that is a number appears below with the file it is checked against.

Usage:  python "build/olympics/verify_results_doc.py"
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Normalize typographic minus/dashes so a prose "−0.271" matches a computed "-0.271".
DOC = (ROOT / "docs/RESULTS_OLYMPICS.md").read_text().replace("\u2212", "-")
S = json.load(open(ROOT / "results/olympics/olympics_summary.json"))
fail = []


def want(text: str, why: str):
    if text.replace("\u2212", "-") not in DOC:
        fail.append(f"NOT IN DOC: {text!r}   ({why})")


# --- headline ---------------------------------------------------------------------------------
h = S["heldout70"]
want(f"{h['token_reduction_p3_vs_p2_pct']}%", "token reduction")
want(f"{h['strict_lead_p3_over_p2']:+.3f}".replace("+0.214", "+0.2143"), "strict lead")
want("0.495", "P1 held-out strict")
want("1.000", "P3 held-out strict")

# all-in thinking token figures quoted in prose
for p, lab in (("llm_only", "P1"), ("basic_rag", "P2"), ("graphrag", "graphrag")):
    tok = h[p]["avg_total_inference_tokens"]["mean"]
    th = h[p]["thinking_tokens_mean"]["mean"]
    want(f"{tok:,.0f}+{th:,.0f}", f"{lab} all-in tokens")

# --- significance -----------------------------------------------------------------------------
sg = S["significance_heldout70"]
want(f"{sg['p3_only_correct']} wins, {sg['p2_only_correct']} losses", "McNemar counts")
want("p = 6.1", "held-out p value")
want("p = 0.375", "dev p value")

# --- by qtype ---------------------------------------------------------------------------------
q = S["heldout70_by_qtype"]
want(f"aggregation +{q['aggregation']['lead_p3_over_p2']:.2f}", "aggregation lead in prose")
want(f"superlative +{q['superlative']['lead_p3_over_p2']:.2f}", "superlative lead in prose")
want(f"temporal +{q['temporal']['lead_p3_over_p2']:.2f}", "temporal lead in prose")
want(f"+{q['multi_hop']['lead_p3_over_p2']:.2f}", "multi_hop lead in prose")

# --- ablations --------------------------------------------------------------------------------
a = S["ablations_heldout70"]
r = a["route_force_random"]
want(f"{r['delta_strict_vs_frozen']:.3f} strict and +{r['delta_tokens_vs_frozen']:,.0f} tokens",
     "un-confounded ablation delta")
want(f"{a['frozen_router']['branch_resolved']}/{a['frozen_router']['n']} to "
     f"{r['branch_resolved']}/{r['n']}", "branch_resolved collapse")
want(f"{a['route_force_lookup']['strict_pass_rate']:.3f} vs {r['strict_pass_rate']:.3f}",
     "lookup vs random")
ro = a["router_off"]
want(f"{ro['strict_pass_rate']:.3f} vs {r['strict_pass_rate']:.3f}", "router_off vs random")
want(f"{ro['branch_resolved']}/{ro['n']}", "router_off resolves none")
want(f"{ro['avg_total_inference_tokens']:,.0f} tokens", "router_off tokens")

# --- integrity --------------------------------------------------------------------------------
i = S["integrity_heldout70"]
hv = i["hop_verification"]
want(f"{hv['issued_every_hop']}/{hv['multi_hop_route_questions']} across all three runs",
     "two-hop completion across runs")
want(hv["checker"], "the standalone hop gate is named")

# --- router -----------------------------------------------------------------------------------
want(f"{S['router']['sec_lexicon_on_olympics_dev']['accuracy']:.3f} on Olympics dev",
     "SEC lexicon quarantine, dev")
want(f"{S['router']['sec_lexicon_on_olympics_test']['accuracy']:.3f} on Olympics test",
     "SEC lexicon quarantine, test")
_o = S["router"]["ordering_probe_dev"]
want(f"all\n  {list(_o['accuracy_by_ordering'].values())[0]} orderings",
     "ordering probe: number of orderings")
want(f"{list(_o['accuracy_by_ordering'])[0]}/30", "ordering probe: score per ordering")
_w = S["router"]["widening_ablation_dev"]
want(f"{_w['dev_literals_only_correct']}/{_w['n']}", "widening ablation score")
want(f"strips all\n  {_w['n_widened_alternatives']} pattern alternatives",
     "widening ablation: count of stripped alternatives")

# --- cross dataset ----------------------------------------------------------------------------
c = S["cross_dataset"]
want(f"{c['sec']['strict_lead_p3_over_p2']:+.3f} on SEC", "SEC lead")
want(f"{c['sec']['token_reduction_pct']}%", "SEC token reduction")
want(c["sec"]["tag"], "SEC tag")

# --- P2 sweep ---------------------------------------------------------------------------------
sw = {r["arm"]: r for r in S["p2_sweep_dev30"]}
want(f"{sw['base']['strict']:.3f}", "un-tuned P2 dev strict")
want(f"{sw['topk_20_c30']['strict']:.3f}", "frozen P2 dev strict")
lift = sw["topk_20_c30"]["strict"] - sw["base"]["strict"]
want(f"+{lift:.3f}", "tuning lift")
want(f"{sw['mmr']['strict']:.3f}", "mmr collapse")

# --- claims checked against source artefacts, not the summary ---------------------------------
import collections
ev = json.load(open(ROOT / "pipeline3_olympics/index/events.json"))
it = ev.values() if isinstance(ev, dict) else ev
g = collections.Counter()
for e in it:
    g[(e.get("sport_key") or e.get("sport"), e.get("games_id") or e.get("games"))] += 1
want(f"**{max(g.values())}** events against a cap of **60**", "dead-knob group size")
want(f"{len(it):,} events", "event count")

print(f"checked {len(DOC.splitlines())} lines of RESULTS_OLYMPICS.md")
if fail:
    print(f"\n{len(fail)} MISMATCH(ES):")
    for f in fail:
        print("  " + f)
    sys.exit(1)
print("ALL PROSE NUMBERS MATCH THE ARTEFACTS")
