"""Step-1 data probe for the Olympics corpus — measurement only, no pipeline code.

Every number in `docs/design/OLYMPICS_DATA_NOTES.md` is printed by this script, so the notes
can be regenerated from the files rather than trusted.

    python "build/olympics/data_probe.py"

It also runs a DETERMINISTIC ORACLE: it answers all 100 public questions by regex over the
infoboxes alone, with no LLM and no retrieval. That number is the ceiling a perfectly-routed
graph pipeline could reach on this corpus, and it is the evidence behind the zero-token
ingestion claim (TASK2_OLYMPICS_PLAN.md §1.2a).

The oracle uses only question text + corpus. It never reads `answer` except to score itself,
and it is run over the PUBLIC 100 only; the hidden 50 are checked for *structural
resolvability* (does the lookup key resolve to exactly one event?), never for correctness,
because they have no answers.
"""
import json, re, collections, statistics, sys, os

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "new_dataset")
BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../data/new_dataset"))

INFOBOX_HDR = re.compile(r'^\[[Ii]nfobox ([^\]]+)\]\s*$')
FIELD       = re.compile(r'^  ([A-Za-z_][A-Za-z0-9_ ]*?): ?(.*)$')
TITLE_RE    = re.compile(r'^(.*?) at the (\d{4}) (Summer|Winter) Olympics – (.*)$')


def parse_infoboxes(text):
    """All infobox blocks in a document, in order.

    A block ends at a blank line OR at the next '[Infobox ...]' header. The second
    condition matters: 32 documents stack a second infobox directly onto the first with no
    blank line between them, and a blank-line-only parser silently merges the second
    infobox's keys into the first one's field map.
    """
    out, cur = [], None
    for ln in text.split("\n"):
        m = INFOBOX_HDR.match(ln)
        if m:
            cur = (m.group(1).strip(), {}); out.append(cur); continue
        if cur is None:
            continue
        if ln.strip() == "":
            cur = None; continue
        f = FIELD.match(ln)
        if f:
            cur[1][f.group(1)] = f.group(2).strip()
    return out


def load_events(docs):
    """Events keyed by doc_id. Indexed on 'contains an Olympic event infobox', not 'opens
    with one' — 25 tennis documents put [Infobox tennis tournament event] first."""
    ev = {}
    for d in docs:
        boxes = dict(parse_infoboxes(d["text"]))
        if "Olympic event" not in boxes:
            continue
        f = dict(boxes["Olympic event"])
        t = TITLE_RE.match(d["title"])
        f.update(id=d["doc_id"], title=d["title"], sport=t.group(1),
                 year=int(t.group(2)), season=t.group(3), event_name=t.group(4))
        f["event_date"] = f.get("date") or f.get("dates") or ""
        f["venue_s"]    = f.get("venue") or f.get("venues") or ""
        ev[d["doc_id"]] = f
    return ev


def norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def dnorm(s):
    """Date normalizer. Questions quote the infobox date verbatim 27/28 times, but
    eval-001 says '15 to 22 August 2004' where another document says '15–22 August 2004',
    so 'to' / en-dash / em-dash / comma / parens are folded together."""
    s = norm(s).replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r'\s*(?:to|through|-)\s*', '-', s)
    s = re.sub(r'[,()]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


R_LOOKUP = re.compile(r'^How many nations competed in (.+)\?$')
R_SUPER  = re.compile(r'which (.+?) event at the (\d{4}) (Summer|Winter) Olympics had the highest number of competitors')
R_AGG    = re.compile(r'how many (.+?) events at the (\d{4}) (Summer|Winter) Olympics had more than (\d+) competitors')
R_MULTI  = re.compile(r'^Who won the gold medal in the event held at (.+?) on (.+?)(?: at the (\d{4}) (Summer|Winter) Olympics)?\?$')
R_TEMP   = re.compile(r'^Who won the gold medal in the (.+?) event at the (Summer|Winter) Olympics held immediately before (\d{4})\?$')


def oracle(q, EV, by_title):
    """Deterministic answer + a note. Returns (answer_or_None, note)."""
    qt, qq = q["qtype"], q["question"]

    if qt == "lookup":
        m = R_LOOKUP.match(qq)
        f = by_title.get(m.group(1)) if m else None
        return (f.get("nations") if f else None), ("title resolved" if f else "TITLE NOT FOUND")

    if qt == "superlative":
        m = R_SUPER.search(qq)
        sp, y, se = norm(m.group(1)), int(m.group(2)), m.group(3)
        g = [f for f in EV.values() if norm(f["sport"]) == sp and f["year"] == y
             and f["season"] == se and f.get("competitors", "").isdigit()]
        return (max(g, key=lambda f: int(f["competitors"]))["title"] if g else None), f"group={len(g)}"

    if qt == "aggregation":
        m = R_AGG.search(qq)
        sp, y, se, n = norm(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        g = [f for f in EV.values() if norm(f["sport"]) == sp and f["year"] == y and f["season"] == se]
        hit = sum(1 for f in g if f.get("competitors", "").isdigit() and int(f["competitors"]) > n)
        return str(hit), f"group={len(g)}"

    if qt == "multi_hop":
        m = R_MULTI.match(qq)
        v, dt = norm(m.group(1)), dnorm(m.group(2))
        g = [f for f in EV.values() if norm(f["venue_s"]) == v and dnorm(f["event_date"]) == dt]
        if m.group(3):
            g = [f for f in g if f["year"] == int(m.group(3)) and f["season"] == m.group(4)]
        if len(g) == 1:
            return g[0].get("gold"), "unique"
        return None, f"AMBIGUOUS matches={len(g)} {[f['title'] for f in g]}"

    if qt == "temporal":
        m = R_TEMP.match(qq)
        ev, se, yr = norm(m.group(1)), m.group(2), int(m.group(3))
        # "the Games held immediately before Y" is a GAMES-level hop over the corpus's own
        # games set — NOT the infobox `prev` field, which is the previous edition of that
        # one EVENT and is a bare year that may not exist in the corpus at all.
        years = sorted({f["year"] for f in EV.values() if f["season"] == se and f["year"] < yr})
        if not years:
            return None, "no prior games in corpus"
        py = years[-1]
        cand = [f for f in EV.values() if f["year"] == py and f["season"] == se]
        key = lambda f: norm(f["event_name"] + " " + f["sport"])
        hit = [f for f in cand if key(f) == ev] or \
              [f for f in cand if all(t in key(f) for t in ev.split())]
        if len(hit) == 1:
            return hit[0].get("gold"), f"prev_games={py}"
        return None, f"prev_games={py} candidates={len(hit)} {[f['title'] for f in hit][:3]}"

    return None, "unknown qtype"


def main():
    docs = [json.loads(l) for l in open(os.path.join(BASE, "corpus/corpus.jsonl"))]
    by_id = {d["doc_id"]: d for d in docs}
    pub = [json.loads(l) for l in open(os.path.join(BASE, "questions/eval_public.jsonl"))]
    hid = [json.loads(l) for l in open(os.path.join(BASE, "questions/eval_hidden.jsonl"))]
    EV = load_events(docs)
    by_title = {f["title"]: f for f in EV.values()}

    print("=" * 78)
    print("1. CORPUS SHAPE")
    print("=" * 78)
    print(f"  documents                         {len(docs)}")
    print(f"  tokens (sum of approx_tokens)     {sum(d['approx_tokens'] for d in docs)}")
    print(f"  Olympic-event docs (contains)     {len(EV)}")
    first = sum(1 for d in docs if (parse_infoboxes(d['text']) or [('', {})])[0][0] == 'Olympic event')
    print(f"    ... of which infobox is FIRST   {first}")
    print(f"    ... preceded by another infobox {len(EV) - first}   (all [Infobox tennis tournament event])")
    print(f"  true distractors                  {len(docs) - len(EV)}")
    tok = sorted(d["approx_tokens"] for d in docs)
    et = sorted(f and by_id[f]["approx_tokens"] for f in EV)
    print(f"  all docs  median/p90/max tokens   {statistics.median(tok)} / {tok[int(.9*len(tok))]} / {max(tok)}")
    print(f"  event docs median/mean/max tokens {statistics.median(et)} / {round(sum(et)/len(et))} / {max(et)}")
    ib = [len(by_id[i]["text"].split("\n\n")[0]) // 4 for i in EV]
    print(f"  infobox block tokens med/p95/max  {statistics.median(ib)} / {sorted(ib)[int(.95*len(ib))]} / {max(ib)}")
    print(f"  titles matching the canonical pattern  {sum(1 for f in EV.values() if TITLE_RE.match(f['title']))}/{len(EV)}")
    print(f"  distinct games / sports / venues / NOCs  "
          f"{len({(f['year'],f['season']) for f in EV.values()})} / "
          f"{len({f['sport'] for f in EV.values()})} / "
          f"{len({f['venue_s'] for f in EV.values() if f['venue_s']})} / "
          f"{len({f[k] for f in EV.values() for k in ('goldNOC','silverNOC','bronzeNOC') if k in f})}")

    print("\n  distractor infobox kinds:")
    kinds = collections.Counter()
    for d in docs:
        ks = [k for k, _ in parse_infoboxes(d["text"])]
        if "Olympic event" in ks:
            continue
        kinds[ks[0] if ks else "(no infobox)"] += 1
    print("   ", dict(kinds.most_common()))

    print("\n  field coverage over the %d event docs:" % len(EV))
    cov = collections.Counter(k for f in EV.values() for k in f
                              if k not in ("id", "title", "sport", "year", "season",
                                           "event_name", "event_date", "venue_s"))
    print("   ", dict(cov.most_common()))

    print("\n" + "=" * 78)
    print("2. HAZARDS — every one counted, including the ones that came out zero")
    print("=" * 78)
    mal = [f["games"] for f in EV.values() if not re.fullmatch(r'\d{4} (Summer|Winter)', f.get("games", ""))]
    print(f"  H1 malformed `games` inside event infoboxes        {len(mal)}  {collections.Counter(mal)}")
    g30 = [d["doc_id"] for d in docs if re.search(r'^  games: *30\s*$', d["text"], re.M)]
    print(f"     (`games: 30` found in the corpus at all:        {len(g30)} — {g30}, both")
    print( "      [Infobox international ice hockey competition], meaning 'matches played')")
    dis = [(f["id"], f["title"], f["games"]) for f in EV.values()
           if f.get("games") != f"{f['year']} {f['season']}"]
    print(f"  H2 `games` field disagrees with the TITLE          {len(dis)}")
    for x in dis:
        print(f"       {x[0]}  {x[1]!r}  games={x[2]!r}")
    print(f"  H3 `gold` with a lowercase->uppercase boundary     "
          f"{sum(1 for f in EV.values() if re.search(r'[a-zà-ÿ][A-ZÀ-Þ]', f.get('gold','')))}")
    print(f"  H4 `venue` with the same concatenation pattern     "
          f"{sum(1 for f in EV.values() if re.search(r'[a-zà-ÿ][A-ZÀ-Þ]', f.get('venue','')))}")
    ni = [f["competitors"] for f in EV.values() if "competitors" in f and not f["competitors"].isdigit()]
    print(f"  H5 non-integer `competitors`                       {len(ni)}  e.g. {ni[:5]}")
    print(f"  H6 missing `competitors` entirely                  {sum(1 for f in EV.values() if 'competitors' not in f)}")
    print(f"  H7 non-integer `nations`                           "
          f"{sum(1 for f in EV.values() if 'nations' in f and not f['nations'].isdigit())}")
    print(f"  H8 docs with BOTH `date` and `dates`               {sum(1 for f in EV.values() if 'date' in f and 'dates' in f)}")
    print(f"     docs with NEITHER                               {sum(1 for f in EV.values() if 'date' not in f and 'dates' not in f)}")
    print(f"  H9 docs with no `venue` and no `venues`            {sum(1 for f in EV.values() if 'venue' not in f and 'venues' not in f)}")
    pn = [f[k] for f in EV.values() for k in ("prev", "next") if k in f and not re.fullmatch(r'\d{4}', f[k])]
    print(f"  H10 `prev`/`next` not a bare 4-digit year          {len(pn)}  e.g. {collections.Counter(pn).most_common(4)}")
    print(f"  H11 stacked infobox with no blank line between     "
          f"{sum(1 for d in docs if len(parse_infoboxes(d['text'])) > 1)}")
    print(f"  H12 lowercase `[infobox` headers                   "
          f"{sum(1 for d in docs if re.search(r'^\[infobox ', d['text'], re.M))} (none are Olympic event)")
    mm = sum(1 for f in EV.values() if f.get("event", "").strip().lower() != f["event_name"].strip().lower())
    print(f"  H13 infobox `event` field != title event-name      {mm} ({mm/len(EV):.0%})")

    print("\n" + "=" * 78)
    print("3. gold_doc_ids SEMANTICS")
    print("=" * 78)
    refs = {g for q in pub for g in q["gold_doc_ids"]}
    print(f"  references {sum(len(q['gold_doc_ids']) for q in pub)} · distinct {len(refs)} · "
          f"resolve {sum(1 for r in refs if r in by_id)} · non-event {len(refs - set(EV))}")
    exact = 0
    for q in pub:
        if q["qtype"] not in ("aggregation", "superlative"):
            continue
        m = R_AGG.search(q["question"]) or R_SUPER.search(q["question"])
        sp, y, se = norm(m.group(1)), int(m.group(2)), m.group(3)
        grp = {f["id"] for f in EV.values() if norm(f["sport"]) == sp and f["year"] == y and f["season"] == se}
        exact += (grp == set(q["gold_doc_ids"]))
    print(f"  agg+superlative: title-derived (sport, games) group == gold_doc_ids  {exact}/31")
    fan = collections.defaultdict(list)
    for q in pub:
        fan[q["qtype"]].append(len(q["gold_doc_ids"]))
    for k in ("lookup", "multi_hop", "temporal", "aggregation", "superlative"):
        v = fan[k]
        print(f"    {k:12s} n={len(v):3d}  gold docs min/median/max = {min(v)}/{statistics.median(v):g}/{max(v)}")

    print("\n" + "=" * 78)
    print("4. DETERMINISTIC ORACLE — public 100, regex over infoboxes, ZERO LLM tokens")
    print("=" * 78)
    res, fails = collections.Counter(), []
    for q in pub:
        got, note = oracle(q, EV, by_title)
        ok = got is not None and str(got).strip() == str(q["answer"][0]).strip()
        res[(q["qtype"], ok)] += 1
        if not ok:
            fails.append((q["qid"], q["qtype"], q["question"], q["answer"][0], got, note))
    total = 0
    for qt in ("lookup", "multi_hop", "temporal", "aggregation", "superlative"):
        n = res[(qt, True)] + res[(qt, False)]
        total += res[(qt, True)]
        print(f"    {qt:12s} {res[(qt,True)]:3d}/{n:3d}")
    print(f"    {'TOTAL':12s} {total:3d}/100")
    for f in fails:
        print(f"\n    MISS [{f[0]}] {f[1]} — {f[5]}\n      Q: {f[2]}\n      gold={f[3]!r}  oracle={f[4]!r}")

    print("\n" + "=" * 78)
    print("5. HIDDEN 50 — structural resolvability only (no answers exist; nothing is scored)")
    print("=" * 78)
    r, notes = collections.Counter(), []
    for q in hid:
        got, note = oracle(q, EV, by_title)
        r[(q["qtype"], got is not None)] += 1
        if got is None:
            notes.append(f"    [{q['qid']}] {q['qtype']}: {note}\n      {q['question']}")
    for qt in ("lookup", "multi_hop", "temporal", "aggregation", "superlative"):
        n = r[(qt, True)] + r[(qt, False)]
        print(f"    {qt:12s} {r[(qt,True)]:2d}/{n:2d} resolve to exactly one lookup key")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
