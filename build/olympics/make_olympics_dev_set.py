"""Split the Olympics public 100 into dev 30 / test 70, stratified by qtype.

Mirrors `build/data_prep/make_dev_set.py` in role, not in method: the SEC dev set was
hand-authored on different companies so no held-out fact could leak into tuning. Here the
organisers supplied exactly 100 answered questions, so the only honest option is a stratified
random split of that one set, with the seed recorded.

IN   data/new_dataset/questions/eval_public.jsonl   (100 Q, with answers)
OUT  datasets/olympics/dev_set.jsonl        30 Q — the ONLY set tuning may touch
     datasets/olympics/test_set.jsonl       70 Q — run once per frozen config
     datasets/olympics/split_manifest.json  seed, per-qtype counts, sha256 of each split

`eval_hidden.jsonl` (50 Q) is NOT split or copied here. It carries `qid`, `question`, `qtype`
and no answers, so it can never be scored. Its one legitimate use is router-label accuracy
against the gold `qtype`; see scripts/router_lexicon_eval.py.

--------------------------------------------------------------------------------------------
THE REFERENCE-STRING DECISION, MADE ONCE, HERE
--------------------------------------------------------------------------------------------
`answer` in the source file is a LIST. Measured: all 100 are singletons (data_probe.py §3), so
the reference is `answer[0]` verbatim — no join rule, no separator convention, nothing for the
evaluator or the judge to re-decide. The assertion below fails loudly if that ever stops being
true, rather than silently joining with ", " and changing what the judge grades against.

The raw list is preserved as `answer_list` so nothing is lost.

--------------------------------------------------------------------------------------------
HARNESS SCHEMA
--------------------------------------------------------------------------------------------
`build/bench/run_benchmark.py:_row()` reads: id, tier, hop, question, answer,
source_docs, route_gold, shape, gold_members, answer_from, set_name. We populate:

    id          <- qid                      ("pub-005")
    question    <- question
    answer      <- answer[0]                judge + BERTScore reference
    route_gold  <- qtype                    GOLD ROUTER LABEL — the harness already carries
                                            this field, so router accuracy is scorable on the
                                            same rows as the answers, with no extra plumbing
    shape       <- qtype
    tier        <- fanout class (see TIER below) so slice_runs.py can group by it unchanged
    hop         <- gold-doc count as a string
    fanout      <- "<n> docs · High|Low"    matching the SEC file's cosmetic convention
    source_docs <- the gold QIDs, comma-joined (human-readable provenance in the bench file)
    gold_doc_ids<- the list, for retrieval precision/recall@k (TASK2_OLYMPICS_PLAN.md §5)

`gold_members` stays empty: no Olympics question is set-valued (aggregation answers are counts,
superlative answers are a single event title), so set-F1 does not apply to this dataset and
`build/eval/set_f1.py` is NOT carried over. Stated here so its absence is a decision.
"""
from __future__ import annotations
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("data/new_dataset/questions/eval_public.jsonl")
OUT_DIR = Path("datasets/olympics")

# Fixed and recorded. Chosen once, before looking at any split's contents, and never changed:
# re-rolling a seed until the split looks favourable is tuning on the test set by another name.
SEED = 20261005
N_DEV = 30

# Fanout class, from the measured gold-doc counts (OLYMPICS_DATA_NOTES.md §3). This is the
# Olympics analogue of the SEC A/B/C/D tiers and exists so `slice_runs.py --by tier` works
# unchanged. It is derived from gold_doc_ids, not from qtype, so it stays honest if a question
# of some type turns out to need a different number of documents.
def tier_for(n_gold: int) -> str:
    if n_gold == 1:
        return "S"    # Single-doc   — lookup, multi_hop
    if n_gold <= 3:
        return "P"    # Paired       — temporal
    return "H"        # High-fanout  — aggregation, superlative (8-43 docs)


def stratified_counts(by_type: dict[str, list], n_dev: int) -> dict[str, int]:
    """Largest-remainder apportionment, so the dev proportions match the source mix as closely
    as integers allow and the total is exactly n_dev."""
    total = sum(len(v) for v in by_type.values())
    exact = {k: len(v) * n_dev / total for k, v in by_type.items()}
    take = {k: int(v) for k, v in exact.items()}
    short = n_dev - sum(take.values())
    for k, _ in sorted(exact.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))[:short]:
        take[k] += 1
    return take


def to_row(q: dict, split: str) -> dict:
    ans = q["answer"]
    assert isinstance(ans, list) and len(ans) == 1, \
        f"{q['qid']}: answer is not a singleton list ({ans!r}) — the reference-string rule " \
        f"documented in this module's docstring no longer holds; decide it again explicitly."
    n_gold = len(q["gold_doc_ids"])
    return {
        "id": q["qid"],
        "question": q["question"],
        "answer": ans[0],
        "answer_list": ans,
        "qtype": q["qtype"],
        "route_gold": q["qtype"],
        "shape": q["qtype"],
        "tier": tier_for(n_gold),
        "hop": str(n_gold),
        "fanout": f"{n_gold} docs · {'High' if n_gold > 3 else 'Low'}",
        "gold_doc_ids": q["gold_doc_ids"],
        "source_docs": ", ".join(q["gold_doc_ids"]),
        "gold_members": [],
        "answer_from": "wikipedia infobox",
        "guess_baseline": q.get("guess_baseline"),
        "answer_verified": q.get("answer_verified"),
        "set_name": "olympics_public",
        "split": split,
        "split_seed": SEED,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    questions = [json.loads(l) for l in open(SRC) if l.strip()]
    assert len(questions) == 100, f"expected 100 public questions, got {len(questions)}"

    by_type: dict[str, list] = defaultdict(list)
    for q in questions:
        by_type[q["qtype"]].append(q)
    for v in by_type.values():                      # deterministic order before shuffling
        v.sort(key=lambda q: q["qid"])

    take = stratified_counts(by_type, N_DEV)
    rng = random.Random(SEED)

    dev, test = [], []
    for qtype in sorted(by_type):
        pool = list(by_type[qtype])
        rng.shuffle(pool)
        dev.extend(pool[: take[qtype]])
        test.extend(pool[take[qtype]:])
    dev.sort(key=lambda q: q["qid"])
    test.sort(key=lambda q: q["qid"])

    # --- assertions before anything is written ---
    dev_ids, test_ids = {q["qid"] for q in dev}, {q["qid"] for q in test}
    all_ids = {q["qid"] for q in questions}
    assert len(dev) == N_DEV, f"dev has {len(dev)}, expected {N_DEV}"
    assert len(test) == 100 - N_DEV, f"test has {len(test)}"
    assert not (dev_ids & test_ids), f"splits overlap: {sorted(dev_ids & test_ids)}"
    assert dev_ids | test_ids == all_ids, "splits do not cover the 100"
    for qtype in by_type:
        assert sum(1 for q in dev if q["qtype"] == qtype) >= 1, \
            f"qtype {qtype!r} absent from dev — stratification failed"
        assert sum(1 for q in test if q["qtype"] == qtype) >= 1, \
            f"qtype {qtype!r} absent from test — stratification failed"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rows in (("dev_set", dev), ("test_set", test)):
        p = OUT_DIR / f"{name}.jsonl"
        split = "dev" if name == "dev_set" else "test"
        with open(p, "w") as f:
            for q in rows:
                f.write(json.dumps(to_row(q, split), ensure_ascii=False) + "\n")
        paths[name] = p

    src_mix = Counter(q["qtype"] for q in questions)
    dev_mix = Counter(q["qtype"] for q in dev)
    test_mix = Counter(q["qtype"] for q in test)

    manifest = {
        "source": str(SRC),
        "seed": SEED,
        "n_total": len(questions), "n_dev": len(dev), "n_test": len(test),
        "apportionment": "largest remainder, stratified by qtype",
        "reference_string_rule": "answer[0] — all 100 answers are singleton lists",
        "qtype_counts": {
            qt: {"total": src_mix[qt], "dev": dev_mix[qt], "test": test_mix[qt],
                 "source_pct": round(100 * src_mix[qt] / len(questions), 1),
                 "dev_pct": round(100 * dev_mix[qt] / len(dev), 1),
                 "test_pct": round(100 * test_mix[qt] / len(test), 1)}
            for qt in sorted(src_mix)
        },
        "tier_counts": {
            "dev": dict(Counter(tier_for(len(q["gold_doc_ids"])) for q in dev)),
            "test": dict(Counter(tier_for(len(q["gold_doc_ids"])) for q in test)),
        },
        "dev_ids": sorted(dev_ids), "test_ids": sorted(test_ids),
        "sha256": {k: sha256(v) for k, v in paths.items()},
    }
    json.dump(manifest, open(OUT_DIR / "split_manifest.json", "w"), indent=2)

    print(f"[split] seed={SEED}  dev={len(dev)}  test={len(test)}  (disjoint, union = all 100)")
    print(f"[split] {'qtype':14s} {'total':>6s} {'dev':>6s} {'test':>6s}   {'src%':>6s} {'dev%':>6s} {'test%':>6s}")
    for qt in sorted(src_mix):
        c = manifest["qtype_counts"][qt]
        print(f"[split] {qt:14s} {c['total']:6d} {c['dev']:6d} {c['test']:6d}   "
              f"{c['source_pct']:6.1f} {c['dev_pct']:6.1f} {c['test_pct']:6.1f}")
    print(f"[split] tier (by gold-doc fanout) dev={manifest['tier_counts']['dev']} "
          f"test={manifest['tier_counts']['test']}")
    for k, v in paths.items():
        print(f"[split] wrote {v}  sha256={manifest['sha256'][k][:16]}…")
    print(f"[split] wrote {OUT_DIR / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
