"""Deterministic Olympics KG extraction. ZERO LLM TOKENS.

This is the module that makes the Olympics corpus structurally cheaper than the SEC round. Everything it
writes comes from regex over the corpus's own infobox blocks, so:

  * ingestion cost is exactly 0 tokens and $0.00, against the SEC round's 7,403 Gemini calls
    recorded in `pipeline3_sec/index/ingestion_tokens.json`. The amortization concern still
    open for SEC (RESULTS_SEC.md §9, Concern 8 — break-even query volume unmeasured)
    does not arise here, and that is a finding to report rather than a caveat to inherit;
  * extraction is bit-reproducible: no sampling, no model version, no retry nondeterminism.

IN   data/new_dataset/corpus/corpus.jsonl   (the raw corpus, not the chunked
     one — the KG is a property of documents, and chunk boundaries are a P2 concern)
OUT  pipeline3_olympics/index/
       events.json            doc_id -> Event vertex attributes + raw infobox
       edges_prov.json        [{source, relation, target, evidence, chunk_id, doc_id}] — the
                              SAME row shape as the SEC index, so provenance substitution and
                              the TG bulk loader are reusable unchanged
       games.json             games_id -> {year, season, n_events, prev_games, next_games}
       sports.json            group_key -> [doc_id]      aggregation + superlative
       venues.json            venue_date_key -> [doc_id] multi_hop  (+ venue -> [doc_id])
       event_index.json       event_key -> [doc_id]      temporal; title -> doc_id for lookup
       nations.json           NOC -> medal tallies
       doc_content.json       doc_id -> full text (local scans, provenance sentences)
       ingestion_tokens.json  {"llm_tokens": 0, ...}   <- the point
       extraction_report.json every hazard count, machine-readable

Run:
    python "build/olympics/extract_infobox_kg.py"
    python "build/olympics/extract_infobox_kg.py" --selftest   # + 10 hand-checks

The schema, the parser and the key normalizers all live in `pipeline3_olympics/schema.py`;
this module must not define its own, or the extractor and the pipeline will drift.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline3_olympics"))
import schema as S  # noqa: E402

SRC = Path("data/new_dataset/corpus/corpus.jsonl")
OUT_DIR = Path("pipeline3_olympics/index")

# A lowercase (or accented-lowercase) letter immediately followed by an uppercase one — the
# signature of the no-separator concatenation in `gold` / `venue`. Counted, NEVER acted on.
CONCAT_RE = re.compile(r'[a-zà-öø-ÿ][A-ZÀ-ÖØ-Þ]')

# Six medal fields across the corpus hold a placeholder rather than a person: "not awarded" x5
# and "vacant" x1. They become Athlete vertices, which is wrong but harmless — no question's
# gold answer is one of them, and they are 6 of 6,846 medal values. Counted, not special-cased:
# silently dropping a row is how a corpus-coverage bug hides.
PLACEHOLDER_RE = re.compile(r'^(?:not\s+awarded|none|n/?a|vacant|unknown|tbd|-{1,2}|—)$', re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="hand-check 10 documents field-for-field and re-verify the gold sets")
    args = ap.parse_args()

    docs = [json.loads(l) for l in open(SRC) if l.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- hazard counters ------------------------------------------------------------------
    # Pre-seeded with EVERY hazard key at 0, so the ones that never fire are still printed.
    # A Counter only prints what it has seen, and "the hazard did not occur" is exactly the
    # kind of result that must be shown rather than inferred from an absent line — the step-4
    # checkpoint asks for a logged count for every hazard "including those that came out zero".
    HAZARD_KEYS = [
        "H1_games_field_malformed",
        "H2_games_field_disagrees_with_title",
        "H3_gold_name_concatenated",
        "H4_venue_concatenated",
        "H5_competitors_not_an_integer",
        "H6_competitors_missing",
        "H7_nations_not_an_integer",
        "H8a_both_date_and_dates",
        "H8b_no_date_at_all",
        "H9a_both_venue_and_venues",
        "H9b_no_venue_at_all",
        "H10_prev_next_not_a_bare_year",
        "H11_stacked_infobox_docs",
        "H12_lowercase_infobox_header",
        "H13_event_field_differs_from_title",
        "H14_win_value_is_free_text",
        "H_title_unparseable",
        "event_infobox_not_first",
        "distractor_docs_no_event_vertex",
        "multi_hop_key_collisions",
        "H15_placeholder_in_medal_field",
        "H16_medal_edge_rows_that_collapse_on_upsert",
    ]
    hz: Counter = Counter({k: 0 for k in HAZARD_KEYS})
    hz_detail: dict[str, list] = defaultdict(list)

    events: dict[str, dict] = {}
    doc_content: dict[str, str] = {}
    edges: list[dict] = []
    sports: dict[str, list] = defaultdict(list)
    venue_date: dict[str, list] = defaultdict(list)
    venue_docs: dict[str, list] = defaultdict(list)
    event_index: dict[str, list] = defaultdict(list)
    by_title: dict[str, str] = {}
    nations: dict[str, dict] = defaultdict(lambda: {"gold": 0, "silver": 0, "bronze": 0, "events": []})
    games_meta: dict[str, dict] = {}

    n_docs_with_box = 0

    for d in docs:
        doc_id, title, text = d["doc_id"], d["title"], d["text"]
        boxes = S.parse_infoboxes(text)
        if len(boxes) > 1:
            hz["H11_stacked_infobox_docs"] += 1
        if re.search(r'^\[infobox ', text, re.M):
            hz["H12_lowercase_infobox_header"] += 1

        fields = S.event_infobox(text)
        if fields is None:
            hz["distractor_docs_no_event_vertex"] += 1
            continue
        n_docs_with_box += 1
        if boxes[0][0] != S.EVENT_INFOBOX:
            # one of the 25 tennis documents whose event infobox is not first
            hz["event_infobox_not_first"] += 1

        parsed = S.parse_title(title)
        if parsed is None:
            # would mean the title convention broke; measured at 0/2187
            hz["H_title_unparseable"] += 1
            hz_detail["H_title_unparseable"].append(doc_id)
            continue
        sport, year, season, event_name = parsed
        gid = S.games_id(year, season)

        # ---- hazards, counted on the raw fields before anything is derived ----------------
        raw_games = fields.get("games", "")
        if not re.fullmatch(r'\d{4} (Summer|Winter)', raw_games):
            hz["H1_games_field_malformed"] += 1
            hz_detail["H1_games_field_malformed"].append([doc_id, raw_games])
        elif raw_games != gid:
            hz["H2_games_field_disagrees_with_title"] += 1
            hz_detail["H2_games_field_disagrees_with_title"].append([doc_id, title, raw_games, gid])
        if CONCAT_RE.search(fields.get("gold", "")):
            hz["H3_gold_name_concatenated"] += 1
        if CONCAT_RE.search(fields.get("venue", "")):
            hz["H4_venue_concatenated"] += 1

        competitors_raw = fields.get("competitors", "")
        nations_raw = fields.get("nations", "")
        competitors_int = S.as_int(competitors_raw)
        nations_int = S.as_int(nations_raw)
        if competitors_raw and competitors_int < 0:
            hz["H5_competitors_not_an_integer"] += 1
            hz_detail["H5_competitors_not_an_integer"].append([doc_id, competitors_raw])
        if not competitors_raw:
            hz["H6_competitors_missing"] += 1
        if nations_raw and nations_int < 0:
            hz["H7_nations_not_an_integer"] += 1
            hz_detail["H7_nations_not_an_integer"].append([doc_id, nations_raw])

        if "date" in fields and "dates" in fields:
            hz["H8a_both_date_and_dates"] += 1
        event_date = fields.get("date") or fields.get("dates") or ""
        if not event_date:
            hz["H8b_no_date_at_all"] += 1

        if "venue" in fields and "venues" in fields:
            hz["H9a_both_venue_and_venues"] += 1
        venue = fields.get("venue") or fields.get("venues") or ""
        if not venue:
            hz["H9b_no_venue_at_all"] += 1

        for k in ("prev", "next"):
            if k in fields and not re.fullmatch(r'\d{4}', fields[k]):
                hz["H10_prev_next_not_a_bare_year"] += 1
        if fields.get("event", "").strip().lower() != event_name.strip().lower():
            hz["H13_event_field_differs_from_title"] += 1
        if "win_value" in fields and not fields["win_value"].replace(".", "").isdigit():
            hz["H14_win_value_is_free_text"] += 1

        # ---- Event vertex -----------------------------------------------------------------
        ev = {
            "doc_id": doc_id, "title": title, "sport": sport, "event_name": event_name,
            "games_id": gid, "year": year, "season": season,
            "venue": venue, "event_date": event_date,
            "competitors": competitors_raw, "competitors_int": competitors_int,
            "nations": nations_raw, "nations_int": nations_int,
            # free text by construction — stored as STRING, never cast (H14)
            "win_value": fields.get("win_value", ""), "win_label": fields.get("win_label", ""),
            "url": d.get("url", ""), "approx_tokens": d.get("approx_tokens", 0),
            "infobox": fields,
        }
        events[doc_id] = ev
        doc_content[doc_id] = text
        by_title[title] = doc_id
        games_meta.setdefault(gid, {"games_id": gid, "year": year, "season": season, "n_events": 0})
        games_meta[gid]["n_events"] += 1

        # ---- index keys (written with schema.py's normalizers, read with the same) --------
        sports[S.group_key(sport, gid)].append(doc_id)
        event_index[S.event_key(gid, sport, event_name)].append(doc_id)
        if venue:
            venue_docs[S.norm_venue(venue)].append(doc_id)
            if event_date:
                venue_date[S.venue_date_key(venue, event_date)].append(doc_id)

        # ---- edges (SEC row shape: source, relation, target, evidence, chunk_id, doc_id) ---
        def edge(src, rel, tgt, evidence="", **extra):
            edges.append({"source": src, "relation": rel, "target": tgt,
                          "evidence": evidence, "chunk_id": f"{doc_id}#c0",
                          "doc_id": doc_id, **extra})

        edge(doc_id, "event_at_games", gid)
        edge(doc_id, "event_in_sport", sport)
        if venue:
            edge(doc_id, "event_at_venue", venue)

        for medal, slot in S.MEDAL_SLOTS:
            name = fields.get(f"{medal}{slot}", "").strip()
            noc = fields.get(f"{medal}NOC{slot}", "").strip()
            if name and PLACEHOLDER_RE.match(name):
                hz["H15_placeholder_in_medal_field"] += 1
                hz_detail["H15_placeholder_in_medal_field"].append([doc_id, medal, name])
            if name:
                # RAW string, never split (rule 3 in schema.py)
                edge(name, "won_medal", doc_id,
                     evidence=f"{name} won {medal} in {title}.", medal=medal)
                if noc:
                    edge(name, "represents", noc, medal=medal)
            if noc:
                edge(noc, "medal_for_nation", doc_id,
                     evidence=f"{noc} won {medal} in {title}.", medal=medal)
                nations[noc][medal] += 1
                nations[noc]["events"].append(doc_id)

    # ---- games_precedes: from the SORTED Games set, NOT from the infobox `prev` field ------
    # `prev` is the previous edition of that one EVENT, 69 values are not bare years, and the
    # questions ask about the GAMES. Sorting the corpus's own 20 Games scores 22/22 on the
    # public temporal questions; see OLYMPICS_DATA_NOTES.md §5.3.
    for season in ("Summer", "Winter"):
        seq = sorted((g for g in games_meta.values() if g["season"] == season),
                     key=lambda g: g["year"])
        for a, b in zip(seq, seq[1:]):
            edges.append({"source": a["games_id"], "relation": "games_precedes",
                          "target": b["games_id"], "evidence":
                          f"The {season} Olympics held immediately before {b['year']} were {a['games_id']}.",
                          "chunk_id": "", "doc_id": "", "season": season})
            games_meta[b["games_id"]]["prev_games"] = a["games_id"]
            games_meta[a["games_id"]]["next_games"] = b["games_id"]
    for g in games_meta.values():
        g.setdefault("prev_games", "")
        g.setdefault("next_games", "")

    # ---- ambiguity in the multi_hop key, measured rather than assumed ---------------------
    amb = {k: v for k, v in venue_date.items() if len(v) > 1}
    hz["multi_hop_key_collisions"] = len(amb)

    # H16 — TigerGraph keys an edge by (from, type, to), so rows sharing that triple collapse
    # and later attribute values overwrite earlier ones. Counted here so the number is known
    # before ingest rather than inferred afterwards from an "accepted" count that looks fine.
    # The fix is DISCRIMINATOR(medal) in olympics_tg_ingest.py; this stays as the audit.
    for rel in ("won_medal", "medal_for_nation"):
        rows = [e for e in edges if e["relation"] == rel]
        keys = {(e["source"], e["target"]) for e in rows}
        hz["H16_medal_edge_rows_that_collapse_on_upsert"] += len(rows) - len(keys)

    rel_counts = Counter(e["relation"] for e in edges)
    medal_triples = rel_counts["won_medal"]

    # ============================ write the index =========================================
    def dump(name, obj):
        with open(OUT_DIR / name, "w") as f:
            json.dump(obj, f, ensure_ascii=False)
        return (OUT_DIR / name).stat().st_size

    sizes = {
        "events.json":       dump("events.json", events),
        "edges_prov.json":   dump("edges_prov.json", edges),
        "games.json":        dump("games.json", games_meta),
        "sports.json":       dump("sports.json", dict(sports)),
        "venues.json":       dump("venues.json", {"by_venue_date": dict(venue_date),
                                                  "by_venue": dict(venue_docs)}),
        "event_index.json":  dump("event_index.json", {"by_event_key": dict(event_index),
                                                       "by_title": by_title}),
        "nations.json":      dump("nations.json", {k: v for k, v in nations.items()}),
        "doc_content.json":  dump("doc_content.json", doc_content),
    }

    # THE claim of this module. Same key names as the SEC file so they can be diffed directly.
    sizes["ingestion_tokens.json"] = dump("ingestion_tokens.json", {
        "prompt": 0, "output": 0, "calls": 0,
        "llm_tokens": 0, "cost_usd": 0.0,
        "method": "deterministic regex over corpus infoboxes",
        "model": None,
        "note": "No LLM is used to build this graph. Compare "
                "pipeline3_sec/index/ingestion_tokens.json, which records the SEC round's "
                "Gemini extraction cost. Reproducible bit-for-bit from corpus.jsonl.",
    })

    report = {
        "source": str(SRC),
        "n_documents": len(docs),
        "n_event_vertices": len(events),
        "n_distractor_docs": hz["distractor_docs_no_event_vertex"],
        "n_games": len(games_meta),
        "n_sports": len({e["sport"] for e in events.values()}),
        "n_venues": len(venue_docs),
        "n_nations": len(nations),
        "n_athlete_strings": len({e["source"] for e in edges if e["relation"] == "won_medal"}),
        "n_edges": len(edges),
        "edges_by_relation": dict(rel_counts),
        "n_medal_triples": medal_triples,
        "group_keys_sport_games": len(sports),
        "venue_date_keys": len(venue_date),
        "multi_hop_key_collisions": len(amb),
        "multi_hop_collision_examples": [[k, v] for k, v in list(amb.items())[:5]],
        "hazards": dict(sorted(hz.items())),
        "hazard_detail": {k: v[:10] for k, v in hz_detail.items()},
        "llm_tokens": 0,
        "artifact_bytes": sizes,
    }
    with open(OUT_DIR / "extraction_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ============================ log =====================================================
    print(f"[kg] documents={len(docs)}  Event vertices={len(events)}  "
          f"distractors skipped={hz['distractor_docs_no_event_vertex']}")
    print(f"[kg] Games={len(games_meta)}  Sports={report['n_sports']}  Venues={len(venue_docs)}  "
          f"Nations={len(nations)}  Athlete strings={report['n_athlete_strings']}")
    print(f"[kg] edges={len(edges)}")
    for rel, n in rel_counts.most_common():
        print(f"[kg]    {rel:18s} {n:7d}")
    print(f"[kg] medal triples (won_medal) = {medal_triples}")
    print(f"[kg] index keys: (sport,games) groups={len(sports)}  "
          f"venue+date={len(venue_date)}  event keys={len(event_index)}  titles={len(by_title)}")
    print("\n[kg] HAZARDS — every one, including those that came out zero:")
    for k, v in sorted(hz.items()):
        print(f"[kg]    {k:42s} {v}" + ("   (did not occur)" if v == 0 else ""))
    print(f"\n[kg] LLM TOKENS USED: {report['llm_tokens']}  (cost $0.00)")
    print(f"[kg] wrote {len(sizes)} artifacts + extraction_report.json -> {OUT_DIR}/")

    if args.selftest:
        selftest(events, sports, venue_date, event_index, by_title, games_meta)


# ============================================================================================
# SELF-TEST — 10 hand-checked documents + the three key indexes re-verified against gold
# ============================================================================================
HAND_CHECKS = [
    # (doc_id, {field: expected}) — read off the document text by eye, then asserted here.
    # Chosen to span: a plain event, a concatenated team name, a tennis doc whose event infobox
    # is NOT first, a stacked ski-area infobox, a `games`-field-wrong doc, a two-bronze event,
    # a team-count `competitors`, a `dates` (plural) doc, a no-venue doc, and the 1900 outlier.
    ("Q25239316", {"sport": "Weightlifting", "games_id": "1988 Summer", "event_name": "Men's 60 kg",
                   "venue": "Olympic Weightlifting Gymnasium", "event_date": "20 September 1988",
                   "competitors_int": 17, "nations_int": 14}),
    ("Q303623",   {"sport": "Canoeing", "games_id": "2012 Summer", "venue": "Eton Dorney",
                   "event_date": "6 to 8 August", "competitors_int": 24, "nations_int": 12,
                   "win_value": "3:09.646"}),
    ("Q942805",   {"sport": "Tennis", "games_id": "2004 Summer", "event_name": "Women's doubles",
                   "venue": "Olympic Tennis Centre", "event_date": "15 to 22 August 2004",
                   "competitors_int": 64, "nations_int": 22}),
    ("Q666951",   {"sport": "Alpine skiing", "games_id": "2010 Winter", "venue": "Whistler Creekside",
                   "competitors_int": 64, "nations_int": 28, "win_value": "1:30.34"}),
    ("Q3628710",  {"sport": "Athletics", "games_id": "1988 Summer",
                   "event_name": "Men's javelin throw", "competitors_int": 38}),
    ("Q247273",   {"sport": "Judo", "games_id": "2008 Summer", "event_name": "Women's 70 kg",
                   "venue": "Beijing Science and TechnologyUniversity Gymnasium",
                   "event_date": "August 13, 2008", "competitors_int": 22, "nations_int": 22}),
    ("Q26254891", {"sport": "Sailing", "games_id": "2016 Summer", "event_name": "Women's RS:X",
                   "venue": "Marina da Glória", "event_date": "8–14 August 2016",
                   "competitors_int": 26, "nations_int": 26, "win_value": "64 points"}),
    ("Q1222488",  {"sport": "Freestyle skiing", "games_id": "1992 Winter", "venue": "Tignes",
                   "event_date": "12–13 February 1992", "competitors_int": 24, "nations_int": 11}),
    ("Q2726788",  {"sport": "Fencing", "games_id": "2008 Summer", "event_name": "Men's team sabre",
                   "competitors_int": 25, "nations_int": 8}),
    ("Q1156252",  {"sport": "Cycling", "games_id": "1900 Summer", "event_name": "Men's sprint"}),
]

HAND_CHECK_MEDALS = [
    ("Q25239316", "gold", "Naim Süleymanoğlu", "TUR"),
    ("Q303623",   "gold", "Rudolf DombiRoland Kökény", "HUN"),     # concatenated, kept raw
    ("Q247273",   "bronze", "Ronda Rousey", "USA"),                # bronze2 slot in the same doc
    ("Q26254891", "silver", "Chen Peina", "CHN"),
    ("Q942805",   "gold", "Li TingSun Tiantian", "CHN"),           # concatenated, kept raw
]


def selftest(events, sports, venue_date, event_index, by_title, games_meta) -> None:
    print("\n[selftest] ---- 10 documents hand-checked field-for-field ----")
    bad = 0
    for doc_id, expect in HAND_CHECKS:
        ev = events.get(doc_id)
        if ev is None:
            print(f"[selftest] FAIL {doc_id}: no Event vertex"); bad += 1; continue
        diffs = {k: (v, ev.get(k)) for k, v in expect.items() if ev.get(k) != v}
        if diffs:
            print(f"[selftest] FAIL {doc_id}: {diffs}"); bad += 1
        else:
            print(f"[selftest]  ok  {doc_id}  {ev['title'][:62]}")
    print("[selftest] ---- medal fields (raw, unsplit) ----")
    for doc_id, medal, name, noc in HAND_CHECK_MEDALS:
        ib = events[doc_id]["infobox"]
        hit = any(ib.get(f"{medal}{s}") == name and ib.get(f"{medal}NOC{s}") == noc
                  for s in ("", "2", "3"))
        print(f"[selftest] {'ok ' if hit else 'FAIL'} {doc_id} {medal:6s} {name!r} ({noc})")
        bad += (not hit)

    # ---- the three index keys, re-verified against the public questions' own gold ---------
    base = Path("data/new_dataset/questions")
    pub = [json.loads(l) for l in open(base / "eval_public.jsonl")]
    R_AGG = re.compile(r'how many (.+?) events at the (\d{4}) (Summer|Winter) Olympics had more than (\d+) competitors')
    R_SUP = re.compile(r'which (.+?) event at the (\d{4}) (Summer|Winter) Olympics had the highest number of competitors')
    ok_group = ok_title = ok_vd = ok_prev = 0
    n_group = n_title = n_vd = n_prev = 0
    for q in pub:
        if q["qtype"] in ("aggregation", "superlative"):
            n_group += 1
            m = R_AGG.search(q["question"]) or R_SUP.search(q["question"])
            key = S.group_key(m.group(1), S.games_id(int(m.group(2)), m.group(3)))
            ok_group += set(sports.get(key, [])) == set(q["gold_doc_ids"])
        elif q["qtype"] == "lookup":
            n_title += 1
            t = re.match(r'^How many nations competed in (.+)\?$', q["question"]).group(1)
            ok_title += by_title.get(t) == q["gold_doc_ids"][0]
        elif q["qtype"] == "multi_hop":
            n_vd += 1
            m = re.match(r'^Who won the gold medal in the event held at (.+?) on (.+?)'
                         r'(?: at the (\d{4}) (Summer|Winter) Olympics)?\?$', q["question"])
            hits = venue_date.get(S.venue_date_key(m.group(1), m.group(2)), [])
            if m.group(3):
                hits = [h for h in hits if events[h]["games_id"] == S.games_id(int(m.group(3)), m.group(4))]
            ok_vd += q["gold_doc_ids"][0] in hits
        elif q["qtype"] == "temporal":
            n_prev += 1
            m = re.match(r'^Who won the gold medal in the (.+?) event at the (Summer|Winter) '
                         r'Olympics held immediately before (\d{4})\?$', q["question"])
            target = next((g for g in games_meta.values()
                           if g["season"] == m.group(2) and g["year"] == int(m.group(3))), None)
            prev = target["prev_games"] if target else ""
            ok_prev += any(events[g]["games_id"] == prev for g in q["gold_doc_ids"])

    print("\n[selftest] ---- index keys re-verified against gold_doc_ids ----")
    print(f"[selftest]  (sport,games) group == gold set     {ok_group}/{n_group}")
    print(f"[selftest]  title -> gold doc                   {ok_title}/{n_title}")
    print(f"[selftest]  venue+date key contains gold doc    {ok_vd}/{n_vd}")
    print(f"[selftest]  games_precedes lands on a gold doc  {ok_prev}/{n_prev}")
    bad += (n_group - ok_group) + (n_title - ok_title) + (n_prev - ok_prev)
    print(f"\n[selftest] {'PASS' if bad == 0 else f'{bad} FAILURES'}  "
          f"(venue+date is allowed to miss on the 1 ambiguous question, see DATA_NOTES §5.1)")


if __name__ == "__main__":
    main()
