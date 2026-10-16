# Olympics corpus — data notes

**Date:** 2026-10-05 · **Scope:** read-only measurement. No pipeline code written yet.
**Source of every number below:** `build/olympics/data_probe.py`, run against
`data/new_dataset/`. Re-run it rather than trusting this file.

Documents read in full or in part while writing this: the gold documents behind `pub-005` and
`pub-009`; one document of every distinct infobox kind in the corpus (25 kinds); the three
stacked-infobox ski-area documents; five tennis documents; eight randomly sampled event
documents (seed 7); and the two documents behind each of the two ambiguous `multi_hop`
questions. Every other claim is a whole-corpus count from the probe.

---

## 1. Headline: the whole question set is answerable by regex, with no LLM

The probe includes a **deterministic oracle** that answers each public question by pattern-
matching the question against infobox fields. No retrieval, no generation, no tokens.

| qtype | oracle | why |
|---|:-:|---|
| lookup | **19/19** | the question quotes the document title verbatim; read `nations` |
| multi_hop | **27/28** | `(venue, date)` → event → read `gold` |
| temporal | **22/22** | previous Games of that season → that event → read `gold` |
| aggregation | **21/21** | count `(sport, games)` group members with `competitors > N` |
| superlative | **10/10** | argmax `competitors` within the `(sport, games)` group |
| **total** | **99/100** | |

The single miss is a genuinely ambiguous question, not an extraction failure (§5.1).

Three things follow, and they shape everything downstream:

1. **The knowledge graph costs zero LLM tokens**, confirming `TASK2_OLYMPICS_PLAN.md` §1.2(a).
   `extract_text_kg.py` has nothing to add: all 518 distinct gold documents are Olympic-event
   documents, and every field any question needs is in the infobox. **Recommendation: delete
   `extract_text_kg.py`** rather than implement it.
2. **99/100 is the P3 ceiling, not a prediction of the P3 score.** The oracle is handed the
   question *already routed and already parsed*. A live pipeline must additionally pick the right
   route from the string, issue the right query, and have a generator write the answer from
   returned evidence. Each of those loses points.
3. **The five gold `qtype` labels are five syntactic templates**, not five fuzzy intents. Every
   one of the 150 questions matches one of five regexes with no exceptions. The router's job on
   this corpus is therefore much easier than on SEC — which must be said out loud, because it
   makes a high Olympics router accuracy a *weaker* result than the SEC 0.800, not a stronger one.

---

## 2. Corpus shape — measured, with three corrections to the brief

| property | brief says | **measured** |
|---|:-:|:-:|
| documents | 2,951 | **2,951** ✓ |
| tokens (`approx_tokens` summed) | 5,466,414 | **5,466,414** ✓ |
| Olympic-event docs | 2,162 | **2,187** ✗ |
| distractors | 789 | **764** ✗ |
| games covered | 22 | **20** ✗ |
| `gold_doc_ids` refs / distinct / resolving | 547 / 518 / 100% | **547 / 518 / 518** ✓ |
| gold docs inside distractors | 0 | **0** ✓ |
| event docs median tokens | 800 | **802** ✓ |

### Correction 1 — there are 2,187 event documents, not 2,162

2,162 documents *open* with `[Infobox Olympic event]`. Another **25 carry it as their second
infobox**, behind `[Infobox tennis tournament event]`. They are Olympic events in every respect:
canonical titles, full medal fields, and they are the gold documents for tennis questions.

This is not cosmetic. **`eval-001` ("Olympic Tennis Centre on 15 to 22 August 2004") is answerable
only from one of those 25** — `Q942805`, Tennis at the 2004 Summer Olympics – Women's doubles. An
extractor that indexes on "opens with" silently drops it, and the failure would look like a
retrieval miss rather than an indexing bug.

**Rule for step 4: index on "contains an Olympic event infobox", never on "opens with".**

### Correction 2 — 764 distractors, and the `games: 30` rows are not event documents

The 25 tennis documents the brief counts as distractors are events. True distractor kinds:

```
film 549 · officeholder 73 · person 67 · company 16 · (no infobox) 11 · writer 9
international football competition 8 · scientist 6 · handball 4 · aircraft occurrence 2
military conflict 2 · airline 2 · website 2 · ice hockey 2 · field hockey 2
Olympic water polo 2 · plus 6 singletons (venue, Olympic games, Song Contest, television,
media franchise, civilian attack)
```

The brief's hazard "2 rows carry a malformed `games: 30`" resolves: both are
`[Infobox international ice hockey competition]` (`Q644408`, `Q3672792`), where `games: 30` means
*matches played in the tournament*. **Inside Olympic-event infoboxes there are zero malformed
`games` values.** Logged as a hazard that came out zero, with its cause named.

It also explains the brief's field-coverage table (`games 2,189 · event 2,187 · gold 2,187`):
those were corpus-wide counts — 2,187 events plus the 2 ice-hockey rows — paired with an
opens-with document count. Both are now consistent at 2,187.

### Correction 3 — 20 distinct Games, not 22

`1900 Summer` (1 document, `Q1156252`), then 1988 → 2022 continuously. Event counts per Games:

```
1900 Su 1 · 1988 Su 165 · 1988 Wi 34 · 1992 Su 177 · 1992 Wi 41 · 1994 Wi 54 · 1996 Su 169
1998 Wi 52 · 2000 Su 209 · 2002 Wi 65 · 2004 Su 193 · 2006 Wi 67 · 2008 Su 209 · 2010 Wi 75
2012 Su 243 · 2014 Wi 73 · 2016 Su 273 · 2018 Wi 24 · 2020 Su 44 · 2022 Wi 19
```

Also measured: 42 sports · 325 distinct venues · 133 NOC codes · infobox block median 88 tokens
(p95 117, max 200) · event docs median 802 / mean 1,005 / max 6,585 tokens.

---

## 3. Fields a question needs — the extraction surface

All five templates are satisfied by **six** pieces of information. Coverage over the 2,187 events:

| needed for | field | coverage | notes |
|---|---|:-:|---|
| lookup | `nations` | 2,152 | always an integer where present (0 exceptions) |
| aggregation, superlative | `competitors` | 2,154 | **not always an integer** — see H5 |
| multi_hop, temporal | `gold` | **2,187** | the only field present on every event |
| multi_hop | `venue` / `venues` | 2,106 / 19 | 62 events have neither |
| multi_hop | `date` / `dates` | 1,254 / 910 | mutually exclusive; 23 have neither |
| grouping (all routes) | **the document title** | **2,187** | see §4 — the title, not `games`, is authoritative |

Also present and useful for provenance sentences, though no question needs them:
`silver` 2,180 · `bronze` 2,180 · `goldNOC` 2,178 · `silverNOC` 2,171 · `bronzeNOC` 2,169 ·
`next` 2,136 · `prev` 2,045 · `win_value` 1,384 · `win_label` 465 · `longnames` 373 ·
`bronze2` 284 · `bronzeNOC2` 282 · `teams` 78 · `event` 2,187.

### `gold_doc_ids` is retrieval ground truth, and it is exact

For all **31/31** aggregation + superlative questions, the title-derived `(sport, games)` group is
**exactly** the gold document set — same members, no difference in either direction. So
`gold_doc_ids` names the enumeration set, not the answer set (`pub-001` lists 11 documents and the
answer is 5). This is what makes retrieval precision/recall@k measurable per `TASK2_OLYMPICS_PLAN.md`
§5, and it independently validates `(sport, games)` as the right grouping key.

---

## 4. The title is authoritative; `games` and `event` are the untrustworthy structured fields

This corpus's instance of the SEC `events.json` hazard (`RESULTS_SEC.md` §4.5) is **not**
the athlete-name concatenation the brief anticipated. It is the `games` and `event` infobox fields.

- **All 2,187 titles** match `"<Sport> at the <YYYY> <Season> Olympics – <Event>"` exactly. Zero
  exceptions. Sport, year, season and event name are all recoverable from the title with one regex.
- **The `games` field disagrees with the title on 4 documents:**

  | doc | title | `games` field |
  |---|---|---|
  | `Q7400295` | Sailing at the **1988** Summer Olympics – Women's 470 | `1984 Summer` |
  | `Q7400301` | Sailing at the **1988** Summer Olympics – Star | `1984 Summer` |
  | `Q1133965` | Badminton at the **1992** Summer Olympics – Women's singles | `1996 Summer` |
  | `Q3628710` | Athletics at the **1988** Summer Olympics – Men's javelin throw | `1992 Summer` |

  This is not hypothetical. `pub-087` ("athletics events at the 1988 Summer Olympics") has 38 gold
  documents; grouping by the `games` field yields 37 and drops `Q3628710`. The *count* answer
  happens to survive (that event had 38 competitors, below the question's threshold of 41), but
  **retrieval recall against `gold_doc_ids` would be wrong**, and the next question of this shape
  might not be so lucky.

- **The `event` field differs from the title's event name on 416 of 2,187 documents (19%)** —
  e.g. `Q303623` title says "Men's K-2 1000 metres", the field says "Men's canoe sprint K-2 1,000
  metres"; `Q915058` title "Women's pursuit", field "Women's biathlon pursuit". The temporal
  template matches against the title's event name and scores 22/22; matching against the `event`
  field would need fuzzy logic for a fifth of the corpus.

**Design rule for step 4: derive `sport`, `year`, `season` and `event_name` from the title. Store
the `games` and `event` fields, but never key on them.** This is the step-1 decision most likely
to be questioned, so the 4 rows and the 416 rows are named above rather than summarised.

---

## 5. Every hazard, with a count — including the ones that came out zero

| # | hazard | count | handling |
|:-:|---|:-:|---|
| H1 | malformed `games` inside event infoboxes | **0** | none needed; cause of the brief's "2 rows" named in §2 |
| H2 | `games` field disagrees with the title | **4** | key on the title (§4) |
| H3 | `gold` with a lowercase→uppercase boundary (team concatenation) | **268** | **do not split** — see §5.2 |
| H4 | `venue` with the same concatenation pattern | **143** | **do not split** — questions quote it verbatim |
| H5 | non-integer `competitors` (`"23 teams"`, `"32 (16 pairs)"`) | **24** | store as STRING, parse to int only when it is all digits; never cast |
| H6 | `competitors` missing entirely | **33** | excluded from counts and argmax |
| H7 | non-integer `nations` | **0** | logged zero |
| H8 | both `date` and `dates` on one event | **0** | so `event_date = date or dates` is unambiguous; **23** events have neither |
| H9 | no `venue` and no `venues` | **62** | those events are unreachable by the multi_hop key; acceptable, none is a gold doc |
| H10 | `prev`/`next` not a bare year (`"2008 Beijing"`, `"Albertville 1992"`) | **69** | do not build the temporal hop from these at all — see §5.3 |
| H11 | documents with a stacked second infobox | **212** | parser must end a block at the next `[Infobox` header, not only at a blank line |
| H12 | lowercase `[infobox` headers | **7** | none is an Olympic event; match case-insensitively anyway |
| H13 | `event` field ≠ title event name | **416** | key on the title (§4) |

H11 is worth emphasising because it fails *silently*: 32 event documents stack
`[Infobox ski area]` directly onto `[Infobox Olympic event]` with no blank line, and a
blank-line-only parser merges `name`, `location`, `vertical`, `top_elevation`, `base_elevation`
into the event's field map. Nothing errors; the event just acquires five fields that are not its own.

### 5.1 Two questions are irreducibly ambiguous — one in each set

`pub-099`: *"the event held at Laura Biathlon & Ski Complex on 22 February 2014"* matches **two**
events — Cross-country skiing Women's 30 km freestyle (`Q15055028`) and Biathlon Men's relay
(`Q15052014`). Gold is the biathlon one. The only textual difference between the two documents is
that one uses the key `date` and the other `dates`, which is not a semantic signal; tie-breaking on
it would be fitting to one question. `eval-032` in the hidden set has the same shape (Pacific
Coliseum, 20 February 2010, two short-track events).

**Decision: do not tie-break.** When the `(venue, date)` key matches more than one event, the
multi_hop branch will put *all* matching events' medal rows in the context and let the generator
choose. That is ~50% on these, versus a rule that is 100% on `pub-099` by construction and
unfalsifiable on the hidden set. It caps the multi_hop oracle at **27/28 (0.964)** on public.

### 5.2 Do not split concatenated athlete names

The brief (§4.2.3) proposes splitting `gold: Rudolf DombiRoland Kökény` on a lowercase→uppercase
boundary as best effort. **The measurement says do not split at all.** The gold *answers* carry the
concatenation verbatim:

- `pub-015` → `'Dani KingLaura TrottJoanna Rowsell'`
- `pub-099` → `'Erik LesserDaniel BöhmArnd PeifferSimon Schempp'`

Splitting would introduce a mismatch between the context we supply and the reference the judge
grades against, on the 268 events where it fires. The `Athlete` vertex from
`TASK2_OLYMPICS_PLAN.md` §4 should therefore store the **raw `gold` string**, and `medal_for_nation`
(NOC codes, 133 clean values) carries any aggregation that must be exact — which is what the plan
already recommended for a different reason.

Same for `venue`: 143 venues are concatenations (`"Xiaohaituo Bobsleigh and Luge TrackBeijing"`,
`"Beijing Science and TechnologyUniversity Gymnasium"`) and **the questions quote them verbatim**,
so exact match on the raw string is correct and splitting would break it.

### 5.3 The temporal hop is Games-level, not `prev`-level — a correction to the plan

`TASK2_OLYMPICS_PLAN.md` §4 proposes `games_precedes` edges built from the infobox `prev`/`next`
fields. **That is the wrong mechanism for these questions.**

- `prev` is the previous edition of *that event*, which is not the previous *Games* when an event
  was not contested (and 69 values are not bare years anyway).
- The question asks for "the Summer Olympics held immediately before 2016", which is a property of
  the Games set, not of any event.
- Resolving it as `max(year) < target, same season` over the corpus's own 20 Games scores **22/22**.

**Build `games_precedes` by sorting the Games vertices**, and keep `prev`/`next` as event
attributes only. One consequence to note: the corpus jumps 1900 → 1988, so a question asking for
the Games before 1988 Summer would resolve to 1900 Summer. No question in either set does
(the earliest is "immediately before 1992"), but the branch should say so rather than answer.

---

## 6. Date matching needs one normalizer

27 of 28 public `multi_hop` questions quote the infobox date **byte-for-byte**, so exact match
scores 27/28. `eval-001` does not: it says `15 to 22 August 2004` where the infobox says
`15–22 August 2004`. Folding `to` / en-dash / em-dash / commas / parentheses into one form fixes it
and changes no other answer. Formats in the corpus include `20 September 1988`, `August 13, 2008`,
`6–13 August 2016`, `6 to 8 August`, `18 February`, and
`22 September 2000 (heats)25 September 2000 (final)` — the last being another no-separator
concatenation that the questions again quote verbatim.

---

## 7. What this implies for the pipelines — written before measuring

- **lookup will be easy for P2 and the prediction of "little or no gain" looks safe.** The question
  *is* the document title. Dense retrieval over 2,951 documents with an exact title string should
  put the right chunk first nearly always.
- **multi_hop is less certain than its name suggests.** 28 of 100, single gold document, and the
  key is a highly distinctive literal (`"Royal Artillery Barracks"` + a date). This is the
  prediction I would least trust; §8.1.4 of the plan already flagged it.
- **aggregation and superlative are where the graph must win.** 31 questions needing 8–43 documents
  each; at k≈10 chunks P2 structurally cannot enumerate them. Both are also exactly the shape that
  should be *cheap* for P3 — a superlative needs only `(title, competitors)` pairs for ≤43 events,
  which is well under a thousand tokens.
- **The high-fanout share is 31%, against 40% on the SEC held-out 50.** With the saving again
  concentrated there, the token-reduction headline should land below SEC's 34.2%, as predicted.
- **A new validity risk this corpus has: the judge knows the Olympics.** `gpt-4o-mini` can name
  Olympic gold medallists from parametric memory. The judge prompt must say to grade against the
  supplied reference only. `guess_baseline` is 0.0 for all 100 questions, which should be verified
  empirically at step 5 via P1 — and P1's score is the measurement that tells us whether the
  corpus is genuinely required here.

---

## 8. Open decisions for the next steps

1. `extract_text_kg.py` — **delete**, per §1. (Confirming with you before removing the file.)
2. Reference string for the judge: every public `answer` is a **singleton list** (100/100), so
   `answer[0]` is the reference with no join rule needed. Recorded here so the decision is not
   re-made silently at step 2.
3. Ambiguous `multi_hop` — do not tie-break (§5.1).
4. Athlete names — do not split (§5.2).
5. `games_precedes` — build from sorted Games, not from `prev` (§5.3).
