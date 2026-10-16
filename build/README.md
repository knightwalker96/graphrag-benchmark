# this project — build/

All executable code for this project, grouped by what it does. **Every script is run from the repository
root**, not from this folder — they call `sys.path.insert(0, ".")` and resolve data paths relative
to the root, so `cd`-ing in here will break them.

```bash
# correct
python "build/bench/run_benchmark.py" --split dev --runs 1 --pipelines 1,2,3
```

---

## Pipeline order

The folders are listed in the order you would actually use them on a fresh corpus.

### `data_prep/` — build the corpus and the question sets

| Script | Does |
|---|---|
| `build_sec_corpus.py` | chunk `dataset/data/` into `datasets/sec/corpus.jsonl` (61,172 chunks) |
| `parse_eval_docx.py` | extract EQ01–EQ50 from the organisers' `eval_questions.docx` |
| `make_dev_set.py` | the 24-question dev split |
| `make_extra_sets.py` | SQ001–SQ100 + the combined 150 |
| `make_tune_holdout_split.py` | stratified tune-120 / holdout-30 split, fixed seed |
| `make_paraphrase_probe.py` | 408 mechanical paraphrases for router-stability testing |

### `kg/` — extract the knowledge graph and load it into TigerGraph

| Script | Does |
|---|---|
| `extract_kg.py` | LLM entity/relation extraction → `pipeline3_sec/index/` |
| `precision_pass.py` | second-pass precision edges — loads `kg/extract_kg.py` dynamically |
| `reconcile_edges.py` | merge/canonicalise edges — also loads `kg/extract_kg.py` |
| `tg_ingest.py` | create schema, install GSQL queries, bulk-load into Savanna (`--fresh`) |
| `smoke_test_tg.py` | is the live graph reachable and answering? |

### `bench/` — run the pipelines

| Script | Does |
|---|---|
| `run_benchmark.py` | **the main entry point.** `--split` / `--runs` / `--pipelines` / `--tag` / `--out-dir` |
| `run_ablations.sh` | the graph-off / router-off / compress-off ablation arms |
| `p2_ablation.py` | P2 retrieval-mode sweep — subprocesses `bench/run_benchmark.py` and `eval/evaluate.py`. **Paid: needs `--run`.** Bare invocation prints a dry plan; `--summarize-only` rebuilds the table for free |
| `dryrun_context.py` | offline context/route/token dry run, no LLM spend |

### `eval/` — judge the answers

| Script | Does |
|---|---|
| `evaluate.py` | strict pass + graded quality (LLM judge), BERTScore, citation quality |
| `set_f1.py` | set-F1 over company-set answers; imported by both evaluators |

### `analysis/` — turn runs into numbers and documents

| Script | Does |
|---|---|
| `aggregate_runs.py` | mean ± std across runs; subprocesses `eval/evaluate.py` |
| `slice_runs.py` | slice a combined-150 batch into extra100 / held-out 50 / combined views |
| `refresh_summary.py` | regenerate `results/sec/initial_summary.json` |
| `summarize_results.py` | regenerate `results/sec/summary.json` |
| `summarize_results.py` | regenerate every table in `RESULTS_SEC.md` |
| `make_dashboard.py` | build `dashboards/dashboard.html` |
| `router_probe.py` | routing accuracy, paraphrase stability, k-fold CV — free, no LLM |
| `verify_claims.py` | **re-checks every narrative figure in `docs/results/` against the raw artifacts; exits non-zero on drift. Free.** |
| `test_extractor.py` | unit checks on the optimization metric-row extractor |
| `apply_compare_fix.py` | idempotent patcher for the baseline compare-route fix (historical) |

### `olympics/` — the Olympics study

Same shape as above for the Olympics corpus: `build_olympics_corpus.py`,
`extract_infobox_kg.py`, `olympics_tg_ingest.py`, `run_olympics_benchmark.py`,
`evaluate_olympics.py`, `p2_sweep_olympics.py`, `p3_tune_olympics.py`, plus probes and reporters.

`verify_results_doc.py` is the Olympics study counterpart to `analysis/verify_claims.py`: it re-checks
every prose number in `docs/results/RESULTS_OLYMPICS.md` against the artifacts and fails
loudly on drift. Both are free. Between them, every figure quoted in either task's results
document is machine-checked:

```bash
python "build/analysis/verify_claims.py"          # the SEC study
python "build/olympics/verify_results_doc.py"     # the Olympics study
```

---

## Cross-script dependencies

Six places load a sibling script at runtime rather than importing it normally. If you move a script,
these are what break:

| In | Loads |
|---|---|
| `eval/evaluate.py` | `sys.path.insert(0, "build/eval")` → `import set_f1` |
| `kg/precision_pass.py` | `build/kg/extract_kg.py` by file path |
| `kg/reconcile_edges.py` | `build/kg/extract_kg.py` by file path |
| `bench/p2_ablation.py` | subprocesses `bench/run_benchmark.py` + `eval/evaluate.py` |
| `analysis/aggregate_runs.py` | subprocesses `eval/evaluate.py` |
| `olympics/evaluate_olympics.py` | loads `build/eval/evaluate.py` and puts `build/eval` on `sys.path` |

`olympics/p2_sweep_olympics.py` and `olympics/p3_tune_olympics.py` resolve their siblings through
`ROOT / "build/olympics/..."`, which is why `olympics/` was kept intact as one unit.

---

## Typical full sequence

```bash
# 1. corpus + question sets
python "build/data_prep/build_sec_corpus.py"
python "build/data_prep/make_extra_sets.py"

# 2. knowledge graph → TigerGraph
python "build/kg/extract_kg.py"
python "build/kg/tg_ingest.py" --fresh
python "build/kg/smoke_test_tg.py"

# 3. free checks before spending anything
python "build/analysis/router_probe.py" --tag mytag
python "build/bench/dryrun_context.py"

# 4. the paid batch
DATASET=sec LLM_PROVIDER=gemini GEMINI_MODEL=gemini-2.5-flash \
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 P2_ADAPTIVE_K=1 P3_REQUIRE_TG=1 \
python "build/bench/run_benchmark.py" \
  --split combined150 --runs 3 --pipelines 1,2,3 --tag mytag --out-dir results/sec

# 5. judge each run
for i in 1 2 3; do
  JUDGE_MODEL=gpt-4o-mini python "build/eval/evaluate.py" \
    results/sec/bench_combined150_mytag_run$i.json \
    --ref-file datasets/sec/combined150_set.jsonl \
    --out results/sec/eval_mytag_run$i.json
done

# 6. summarise
python "build/analysis/slice_runs.py" --tag mytag --runs 3
python "build/analysis/summarize_results.py"

# 7. re-verify the documented figures against the new artifacts (free)
python "build/analysis/verify_claims.py"
```

## Paid vs free

Anything that calls Gemini or OpenAI costs money and takes time. The free scripts can be run
freely while iterating:

| Free (JSON/regex/embeddings only) | Paid (model calls) |
|---|---|
| `analysis/verify_claims.py`, `analysis/router_probe.py`, `analysis/summarize_results.py`, `analysis/slice_runs.py`, `analysis/refresh_summary.py`, `analysis/summarize_results.py`, `analysis/make_dashboard.py`, `analysis/test_extractor.py`, `bench/dryrun_context.py`, `bench/p2_ablation.py --summarize-only`, `kg/smoke_test_tg.py` | `bench/run_benchmark.py`, `eval/evaluate.py`, `bench/p2_ablation.py --run`, `bench/run_ablations.sh`, `kg/extract_kg.py`, `kg/precision_pass.py`, and the `olympics/` runners/judges |
