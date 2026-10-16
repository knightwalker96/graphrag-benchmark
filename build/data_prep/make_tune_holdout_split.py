#!/usr/bin/env python3
"""Stratified tune-120 / holdout-30 split of the 150 SEC questions (Phase 1 of the optimization study).

WHY THIS EXISTS. The optimization work tunes the router and the context builders against questions we
have already measured. Doing that on all 150 leaves no held-out set and makes every
post-tuning metric in-sample. This script carves out a 30-question holdout that is touched
exactly ONCE, at the Phase-1 freeze, so there is one genuine generalization number before
anything becomes in-sample.

STRATIFICATION uses only labels that already existed in the data — never anything we derived:
  * extra100 (SQ001-SQ100): the `shape` field (12 shapes)
  * official held-out 50 (EQ01-EQ50): the organiser's `tier` field (A/B/C/D)
Proportional allocation at 20% with largest-remainder rounding to land on exactly 30.

Deterministic: SEED is fixed and the sort is by question id before shuffling, so the split is
reproducible from this file alone.

Usage:  python "build/data_prep/make_tune_holdout_split.py" [--seed 20261005] [--holdout 30]
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

SRC = Path("datasets/sec/combined150_set.jsonl")
OUT_DIR = Path("datasets/sec")
SEED = 20261005


def stratum_of(row: dict) -> str:
    """The pre-existing label this row is stratified on."""
    if row.get("set_name") == "extra100":
        return "shape:" + (row.get("shape") or "unlabelled")
    return "tier:" + (row.get("tier") or "unlabelled")


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate `total` holdout slots across strata proportionally to size."""
    n = sum(counts.values())
    exact = {k: v * total / n for k, v in counts.items()}
    base = {k: int(v) for k, v in exact.items()}
    short = total - sum(base.values())
    # hand the remaining slots to the largest fractional parts; ties broken by stratum name
    order = sorted(counts, key=lambda k: (-(exact[k] - base[k]), k))
    for k in order[:short]:
        base[k] += 1
    # never take a whole stratum: leave at least one row for tuning
    for k in base:
        if base[k] >= counts[k] and counts[k] > 0:
            base[k] = counts[k] - 1
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--holdout", type=int, default=30)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(SRC)]
    assert len(rows) == 150, f"expected 150 questions, got {len(rows)}"

    strata: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        strata[stratum_of(r)].append(r)
    for k in strata:
        strata[k].sort(key=lambda r: r["id"])            # stable before shuffling

    counts = {k: len(v) for k, v in strata.items()}
    quota = largest_remainder(counts, a.holdout)

    rng = random.Random(a.seed)
    holdout, tune = [], []
    for k in sorted(strata):
        pool = list(strata[k])
        rng.shuffle(pool)
        holdout += pool[:quota[k]]
        tune += pool[quota[k]:]
    holdout.sort(key=lambda r: r["id"])
    tune.sort(key=lambda r: r["id"])

    # --- invariants: disjoint, exhaustive, right size ---
    hid, tid = {r["id"] for r in holdout}, {r["id"] for r in tune}
    assert not (hid & tid), f"split is NOT disjoint: {sorted(hid & tid)[:5]}"
    assert hid | tid == {r["id"] for r in rows}, "split does not cover all 150"
    assert len(holdout) == a.holdout, f"holdout is {len(holdout)}, expected {a.holdout}"

    for name, part in (("tune120", tune), ("holdout30", holdout)):
        p = OUT_DIR / f"split_{name}.jsonl"
        with open(p, "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {p}  n={len(part)}")

    meta = {"seed": a.seed, "source": str(SRC), "n_total": len(rows),
            "n_tune": len(tune), "n_holdout": len(holdout),
            "stratification": "extra100 -> `shape`; held-out 50 -> organiser `tier`",
            "allocation": "proportional 20% with largest-remainder rounding",
            "strata": {k: {"n": counts[k], "holdout": quota[k], "tune": counts[k] - quota[k]}
                       for k in sorted(counts)},
            "holdout_ids": sorted(hid), "tune_ids": sorted(tid),
            "disjoint_verified": True}
    mp = OUT_DIR / "split_tune_holdout_meta.json"
    json.dump(meta, open(mp, "w"), indent=1)
    print(f"wrote {mp}")

    print(f"\n{'stratum':30}{'n':>4}{'tune':>6}{'holdout':>9}")
    print("-" * 49)
    for k in sorted(counts):
        print(f"{k:30}{counts[k]:4d}{counts[k]-quota[k]:6d}{quota[k]:9d}")
    print("-" * 49)
    print(f"{'TOTAL':30}{sum(counts.values()):4d}{len(tune):6d}{len(holdout):9d}")


if __name__ == "__main__":
    main()
