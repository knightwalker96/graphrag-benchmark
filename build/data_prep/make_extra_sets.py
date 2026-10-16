"""Build the Task-1 question splits from eval_extra100.jsonl + the official held-out 50.

Writes two JSONL splits under datasets/sec/, leaving eval_set.jsonl untouched:

  extra100_set.jsonl     SQ001-SQ100 — the 100 new SEC questions, route/shape-labelled
  combined150_set.jsonl  SQ001-SQ100 then EQ01-EQ50 — the combined set

The two source schemas differ (the new set has route/shape/gold_members/answer_from and NO
tier; the official set has tier/hop/fanout and no route label), so rows are normalized to a
common superset here rather than inside the benchmark harness. Every field either source
carries is preserved; fields the other source lacks are written as "" / [] so downstream
slicing never has to guess whether a key is missing or genuinely empty.

`set_name` marks which source a row came from, so the three reporting views (SQ100 alone,
EQ50 alone, combined 150) are sliceable without parsing id prefixes.

Usage:  python "build/data_prep/make_extra_sets.py"
"""
from __future__ import annotations
import json
from pathlib import Path

EXTRA = Path("data/Eval Questions/eval_extra100.jsonl")
OFFICIAL = Path("datasets/sec/eval_set.jsonl")
OUT_DIR = Path("datasets/sec")

FIELDS = ["id", "question", "answer", "split", "set_name",
          "route_gold", "shape", "gold_members", "answer_from", "evidence",
          "tier", "hop", "fanout", "source_docs"]


def _norm(row: dict, set_name: str) -> dict:
    """Project a source row onto the common superset schema."""
    ev = row.get("evidence", "")
    if not isinstance(ev, str):
        ev = json.dumps(ev, ensure_ascii=False)   # 20 new rows carry a dict here
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row.get("answer", ""),
        "split": row.get("split", "test"),
        "set_name": set_name,
        # new-set fields
        "route_gold": row.get("route", ""),
        "shape": row.get("shape", ""),
        "gold_members": row.get("gold_members", []),
        "answer_from": row.get("answer_from", ""),
        "evidence": ev,
        # official-set fields
        "tier": row.get("tier", ""),
        "hop": row.get("hop", ""),
        "fanout": row.get("fanout", ""),
        "source_docs": row.get("source_docs", ""),
    }


def main() -> None:
    extra = [_norm(json.loads(l), "extra100") for l in open(EXTRA)]
    official = [_norm(json.loads(l), "heldout50") for l in open(OFFICIAL)]

    ids = [r["id"] for r in extra] + [r["id"] for r in official]
    assert len(ids) == len(set(ids)), "duplicate question id across the two sets"
    assert len(extra) == 100 and len(official) == 50, f"expected 100+50, got {len(extra)}+{len(official)}"

    for name, rows in [("extra100_set.jsonl", extra),
                       ("combined150_set.jsonl", extra + official)]:
        out = OUT_DIR / name
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[sets] wrote {out}  n={len(rows)}")

    # sanity summary so a wrong file is obvious immediately
    import collections
    print("[sets] extra100 route_gold:", dict(collections.Counter(r["route_gold"] for r in extra)))
    print("[sets] extra100 answer_from:", dict(collections.Counter(r["answer_from"] for r in extra)))
    print("[sets] heldout50 tier:", dict(collections.Counter(r["tier"] for r in official)))


if __name__ == "__main__":
    main()
