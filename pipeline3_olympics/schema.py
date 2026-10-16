"""Olympics knowledge-graph schema — the single source of truth.

Declares the vertex types, edge types, attribute names and the *pure* parsing/normalizing
functions that turn a corpus document into those vertices, in ONE place, so that
`build/olympics/extract_infobox_kg.py`, `olympics_tg_ingest.py` and
`pipeline3_olympics/pipeline.py` cannot drift apart.

DEVIATION FROM THE SCAFFOLD CONTRACT, AND WHY
    The stub said "Nothing here performs IO or extraction." This module still performs no IO —
    it opens no file and makes no network call — but it DOES hold the infobox parser and the
    title/date/text normalizers. They live here because they are precisely the things that must
    not drift: the extractor writes keys with `norm_venue()`, and the pipeline looks them up with
    `norm_venue()`. Two copies of that function in two files is the bug this module exists to
    prevent. Everything here is a pure function of its arguments.

--------------------------------------------------------------------------------------------
WHAT THE CORPUS ACTUALLY LOOKS LIKE  (measured — build/olympics/data_probe.py)
--------------------------------------------------------------------------------------------
2,187 of 2,951 documents carry an `[Infobox Olympic event]` block of two-space-indented
`key: value` lines. Field coverage over those 2,187:

    event 2187 · games 2187 · gold 2187 · silver 2180 · bronze 2180 · goldNOC 2178
    silverNOC 2171 · bronzeNOC 2169 · competitors 2154 · nations 2152 · next 2136
    venue 2106 · prev 2045 · win_value 1384 · date 1254 · dates 910 · win_label 465
    longnames 373 · bronze2 284 · bronzeNOC2 282 · teams 78 · venues 19 · gold2 7 ...

The other 764 documents are distractors (549 film, 73 officeholder, 67 person, 16 company,
plus wars, air disasters, websites). They get no Event vertex.

--------------------------------------------------------------------------------------------
THREE RULES THAT ARE NOT OBVIOUS FROM THE FIELD NAMES
--------------------------------------------------------------------------------------------
1.  INDEX ON "CONTAINS", NOT "OPENS WITH". 2,162 documents open with the Olympic-event
    infobox; 25 more carry it behind `[Infobox tennis tournament event]`. Those 25 are real
    events and one of them (`Q942805`) is the only document that answers `eval-001`.

2.  THE TITLE IS AUTHORITATIVE; `games` AND `event` ARE NOT. All 2,187 titles match
    `"<Sport> at the <YYYY> <Season> Olympics – <Event>"` exactly. The `games` FIELD contradicts
    the title on 4 documents (e.g. `Q3628710`, a 1988 Summer event tagged `games: 1992 Summer`),
    and the `event` FIELD differs from the title's event name on 416 (19%). Grouping by the
    `games` field drops `Q3628710` from the 1988 athletics group, which `pub-087`'s
    `gold_doc_ids` says contains it. So sport/year/season/event_name all come from the TITLE.
    This is this corpus's instance of the SEC `events.json` hazard (RESULTS_SEC.md):
    a field that looks structured and is not.

3.  DO NOT SPLIT CONCATENATED NAMES. 268 `gold` values run team members together with no
    separator (`"Rudolf DombiRoland Kökény"`) and 143 `venue` values do the same. The GOLD
    ANSWERS carry the concatenation verbatim (`pub-015` -> `"Dani KingLaura TrottJoanna
    Rowsell"`) and the QUESTIONS quote the concatenated venue verbatim. Splitting would make
    our context disagree with the graded reference. Athlete vertices therefore store the RAW
    string, and `medal_for_nation` (133 clean NOC codes) carries anything that must be exact.
"""
from __future__ import annotations

import re

# ============================================================================================
# VERTEX TYPES
# ============================================================================================
# Games     PRIMARY_ID games_id STRING     "2012 Summer" — 20 distinct (1900, then 1988-2022)
#           year INT, season STRING, n_events INT
# Event     PRIMARY_ID doc_id STRING       the Wikidata QID, e.g. "Q303623"
#           title, sport, event_name, games_id, venue, event_date, win_value, win_label STRING
#           competitors, nations STRING  (raw — 24 competitors values are "23 teams" etc.)
#           competitors_int, nations_int INT  (-1 when the raw value is not all digits)
# Venue     PRIMARY_ID venue_name STRING   325 distinct, raw (concatenations preserved)
# Nation    PRIMARY_ID noc STRING          133 three-letter NOC codes
# Athlete   PRIMARY_ID athlete_name STRING RAW medal-field string, never split (see rule 3)
# Sport     PRIMARY_ID sport_name STRING   42 distinct, from the title
VERTEX_TYPES = {
    "Games":   ["games_id", "year", "season", "n_events"],
    "Event":   ["doc_id", "title", "sport", "event_name", "games_id", "venue", "event_date",
                "competitors", "competitors_int", "nations", "nations_int",
                "win_value", "win_label", "url"],
    "Venue":   ["venue_name"],
    "Nation":  ["noc"],
    "Athlete": ["athlete_name"],
    "Sport":   ["sport_name"],
}

# ============================================================================================
# EDGE TYPES
# ============================================================================================
EDGE_TYPES = {
    "event_at_games":   ("Event",   "Games",  []),
    "event_in_sport":   ("Event",   "Sport",  []),
    "event_at_venue":   ("Event",   "Venue",  []),
    "won_medal":        ("Athlete", "Event",  ["medal"]),      # medal in {gold,silver,bronze}
    "represents":       ("Athlete", "Nation", []),
    "medal_for_nation": ("Nation",  "Event",  ["medal"]),      # NOC-level, unambiguous
    "games_precedes":   ("Games",   "Games",  ["season"]),     # sorted Games, NOT infobox `prev`
}

MEDALS = ("gold", "silver", "bronze")
# Secondary medal slots. 284 events award two bronzes (judo, boxing, wrestling, taekwondo);
# a handful award two golds or two/three silvers after a tie.
MEDAL_SLOTS = [(m, s) for m in MEDALS for s in ("", "2", "3")]

ROUTES = ("lookup", "aggregation", "superlative", "multi_hop", "temporal", "default")

# ============================================================================================
# PARSING — pure functions, no IO
# ============================================================================================
INFOBOX_HDR = re.compile(r'^\[[Ii]nfobox ([^\]]+)\]\s*$')
FIELD = re.compile(r'^  ([A-Za-z_][A-Za-z0-9_ ]*?): ?(.*)$')
TITLE_RE = re.compile(r'^(.*?) at the (\d{4}) (Summer|Winter) Olympics – (.*)$')
EVENT_INFOBOX = "Olympic event"


def parse_infoboxes(text: str) -> list[tuple[str, dict]]:
    """Every infobox block in a document, in order, as (kind, {field: value}).

    A block ends at a blank line OR at the next `[Infobox ...]` header. The second condition
    is load-bearing: 212 documents stack a second infobox straight onto the first with no blank
    line between them, and a blank-line-only parser merges the second box's keys into the
    first's field map — silently, with no error.
    """
    out: list[tuple[str, dict]] = []
    cur: tuple[str, dict] | None = None
    for ln in text.split("\n"):
        m = INFOBOX_HDR.match(ln)
        if m:
            cur = (m.group(1).strip(), {})
            out.append(cur)
            continue
        if cur is None:
            continue
        if ln.strip() == "":
            cur = None
            continue
        f = FIELD.match(ln)
        if f:
            cur[1][f.group(1)] = f.group(2).strip()
    return out


def event_infobox(text: str) -> dict | None:
    """The Olympic-event infobox if the document has one ANYWHERE, else None (see rule 1)."""
    for kind, fields in parse_infoboxes(text):
        if kind == EVENT_INFOBOX:
            return fields
    return None


def parse_title(title: str) -> tuple[str, int, str, str] | None:
    """(sport, year, season, event_name) from the title — authoritative (see rule 2)."""
    m = TITLE_RE.match(title)
    return (m.group(1), int(m.group(2)), m.group(3), m.group(4)) if m else None


def as_int(value: str | None) -> int:
    """Integer value, or -1 when the field is absent or not all digits.

    NEVER cast blindly: 24 `competitors` values are team counts (`"23 teams"`,
    `"32 (16 pairs)"`) and `win_value` is free text (`"3:09.646"`, `"6:14.60 speed skating"`).
    -1 means "not a comparable count" and is excluded from counts and argmax, not treated as 0.
    """
    v = (value or "").strip()
    return int(v) if v.isdigit() else -1


# ============================================================================================
# NORMALIZERS — the extractor writes keys with these; the pipeline looks them up with these
# ============================================================================================
def norm_text(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or "").strip().lower())


def norm_venue(s: str) -> str:
    """Venue strings are matched VERBATIM modulo case/whitespace. 143 contain no-separator
    concatenations (`"Xiaohaituo Bobsleigh and Luge TrackBeijing"`) and the questions quote
    them exactly as the infobox writes them, so no splitting or cleanup is applied."""
    return norm_text(s)


def norm_date(s: str) -> str:
    """Fold the date spellings the questions and infoboxes disagree on.

    27 of 28 public multi_hop questions quote the infobox date byte-for-byte; `eval-001` says
    `"15 to 22 August 2004"` where the infobox says `"15–22 August 2004"`. Folding
    `to` / en-dash / em-dash / minus / commas / parentheses into one form fixes that one and
    changes no other match.
    """
    s = norm_text(s).replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r'\s*(?:to|through|-)\s*', '-', s)
    s = re.sub(r'[,()]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def games_id(year: int, season: str) -> str:
    return f"{year} {season}"


def group_key(sport: str, games: str) -> str:
    """The (sport, games) grouping key behind aggregation and superlative. Measured to equal
    `gold_doc_ids` exactly on all 31 public questions of those two types."""
    return f"{norm_text(sport)}||{norm_text(games)}"


def venue_date_key(venue: str, date: str) -> str:
    """The multi_hop lookup key. Not unique: 2 of 150 questions name a (venue, date) pair that
    two events share. The branch returns BOTH rather than tie-breaking — see
    OLYMPICS_DATA_NOTES.md §5.1."""
    return f"{norm_venue(venue)}||{norm_date(date)}"


def event_key(games: str, sport: str, event_name: str) -> str:
    """The temporal lookup key: which event, at which Games. Built from the TITLE's event name,
    not the `event` field, which differs on 19% of documents."""
    return f"{norm_text(games)}||{norm_text(sport)}||{norm_text(event_name)}"
