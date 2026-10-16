# Olympics — build pipeline

Every script here is re-runnable and deterministic unless it calls an LLM. Run in this order.
No script in this directory writes to any SEC/Task-1 file; the only SEC artefact touched is
`results/sec/summary.json`, **read-only**, for the cross-dataset table.

| # | script | what it does | output |
|---|---|---|---|
| 1 | `data_probe.py` | measures every figure in OLYMPICS_DATA_NOTES, incl. the 99/100 regex oracle | stdout |
| 2 | `make_olympics_dev_set.py` | stratified dev 30 / test 70 split, `SEED=20261005` | `datasets/olympics/{dev,test}_set.jsonl` |
| 3 | `build_olympics_corpus.py` | chunking (1800/250), infobox kept whole in chunk 0 | `datasets/olympics/corpus.jsonl` |
| 4 | `extract_infobox_kg.py` | **zero-LLM-token** regex KG; `--selftest` hand-checks 10 docs | `pipeline3_olympics/index/*` |
| 5 | `olympics_tg_ingest.py` | schema + 6 GSQL queries + vectors on the `olympics` graph | TigerGraph |
| 6 | `smoke_test_olympics_tg.py` | 11 known-answer fixtures; **set `EMBEDDING_MODEL`** | stdout |
| 7 | `probe_branch.py` | one branch's graph work, no LLM call (branch ceiling) | stdout |
| 8 | `p2_sweep_olympics.py` | the baseline optimization sweep, dev only | `results/olympics/p2_sweep/` |
| 9 | `p3_tune_olympics.py --check-inert` | proves P3's knobs are inert *from files* | stdout |
| 10 | `run_olympics_benchmark.py` | the frozen env + harness wrapper | `results/olympics/bench_*.json` |
| 11 | `evaluate_olympics.py` | judge + BERTScore + retrieval P/R | `results/olympics/eval_*.json` |
| 12 | `make_olympics_summary.py` | **the single source of every reported number** | `results/olympics/olympics_summary.json` |
| 13 | `make_olympics_tables.py` | renders all report tables from that JSON | `results/olympics/olympics_tables.md` |
| 14 | `verify_results_doc.py` | asserts every hand-written prose number in the report | exit 1 on drift |
| 15 | `../../../scripts/verify_branch_hops.py` | gate: every multi-query route issued **all** its hops | exit 1 on a missing hop |
| — | `report_dev.py` | dev-only convenience view during tuning | stdout |

## Reproducing the reported results end to end

```bash
# 0. the graph must be up; this also proves the SEC corpus's `finance` graph is untouched
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 python "build/olympics/smoke_test_olympics_tg.py"
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 python "build/kg/smoke_test_tg.py"

# 1. held-out: 3 runs, all three pipelines, ONE batch
python "build/olympics/run_olympics_benchmark.py" \
    --split test --runs 3 --pipelines 1,2,3 --tag held70 --out-dir results/olympics
for r in 1 2 3; do
  python "build/olympics/evaluate_olympics.py" \
     "results/olympics/bench_olympics_test_held70_run${r}.json" \
     --out "results/olympics/eval_held70_run${r}.json"
done

# 2. router ablations (P3 only). Each arm is independent — one Savanna 500 must not abort the batch.
for a in "routeforce_random P3_ROUTE_FORCE=random" \
         "routeforce_lookup P3_ROUTE_FORCE=lookup" \
         "router_off P3_ROUTER_OFF=1"; do
  set -- $a
  env $2 python "build/olympics/run_olympics_benchmark.py" \
      --split test --runs 1 --pipelines 3 --tag "abl_$1" --out-dir results/olympics/ablations
  python "build/olympics/evaluate_olympics.py" \
      "results/olympics/ablations/bench_olympics_test_abl_$1_run1.json" \
      --out "results/olympics/ablations/eval_abl_$1.json"
done

# 3. router accuracy + the two self-claims + the quarantine 2x2 (no LLM calls at all)
for s in dev test hidden; do
  python scripts/router_lexicon_eval.py --split $s --lexicon olympics \
      --out "results/olympics/router_lexicon_$s.json"
done
for s in dev test; do        # the SEC lexicon must FAIL here — that is the quarantine check
  python scripts/router_lexicon_eval.py --split $s --lexicon sec \
      --out "results/olympics/router_sec_lexicon_$s.json"
done
python scripts/router_lexicon_eval.py --split dev --lexicon olympics --ablate-widening \
    --out results/olympics/router_ablate_widening_dev.json
python scripts/router_lexicon_eval.py --split dev --lexicon olympics --probe-ordering \
    --out results/olympics/router_probe_ordering_dev.json

# 3b. prove every multi-query route issued all its hops, on every frozen run
for f in results/olympics/bench_olympics_test_held70_run{1,2,3}.json; do
  python scripts/verify_branch_hops.py "$f"
done

# 4. summary -> tables -> verify
python "build/olympics/make_olympics_summary.py"
python "build/olympics/make_olympics_tables.py" --out results/olympics/olympics_tables.md
python "build/olympics/verify_results_doc.py"
```

## Traps that cost real time on this dataset

- **`EMBEDDING_MODEL` must be set for any script that embeds.** The default is
  `all-mpnet-base-v2` (768-dim); the Olympics vector index is 384-dim. Both smoke tests fail with
  a dimension error if you forget — which is a *harness* failure, not a graph failure. The
  benchmark wrapper sets it; the smoke tests do not.
- **Never put a destructive GSQL statement in the same call as its scoping statement.**
  `USE GRAPH x\nDROP QUERY ALL` as one call executed `DROP QUERY ALL` against the *session's*
  fallback graph when `USE GRAPH` failed, and dropped the SEC corpus's installed queries.
  `olympics_tg_ingest.py` now guards with `graph_exists()` first.
- **Gemini 2.5 thinking tokens count against `maxOutputTokens`** but report separately from
  `candidatesTokenCount`. A truncation audit that tests `output_tokens` against the cap cannot
  see this. Check `finish_reason`, not token counts.
- **One edge type per medal colour.** `DISCRIMINATOR(medal)` did not survive the raw-REST upsert
  path and silently lost 576 of 6,846 medal rows (TigerGraph keys edges by `(from, type, to)`).
- **Savanna auto-suspends.** A 500 on `/gsql/v1/tokens` means suspended, not broken. `P3_REQUIRE_TG=1`
  makes P3 fail loudly instead of answering from the local index.
