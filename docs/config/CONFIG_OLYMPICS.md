# Frozen pipeline configuration — Olympics corpus

**Status: FROZEN on the 30-question dev set. The held-out 70 has not been run.**

Everything below was selected on `datasets/olympics/dev_set.jsonl` only. No number in this
document comes from `test_set.jsonl`, and no configuration decision may be revisited after the
held-out run — that is the whole point of the split.

Tuning artefacts: `results/olympics/p2_sweep/` (13 arms, stage-1 single-run + stage-2 3-run),
`results/olympics/p3_tune/`. Sweep driver: `build/olympics/p2_sweep_olympics.py`.

---

## 1. Headline: what the sweep did to the Task-2 claim

| | un-tuned P2 (k=10) | **tuned P2 (frozen)** | P3 GraphRAG |
|---|:-:|:-:|:-:|
| strict pass (dev 30) | 0.633 | **0.867** | 0.967 |
| avg inference tokens | 6,135 | **4,997** | 496 |
| retrieval recall@k | 0.731 | 0.861 | 0.674 |

Tuning the baseline moved it **+0.234 strict**, which cut P3's dev lead from **+0.333 to
+0.100**. The token reduction barely moved (91.9% → **90.1%**) because the winning P2 config is
*also* cheaper than the one it replaced.

**The remaining +0.100 is not statistically significant on dev.** Exact McNemar on the paired
per-question outcomes: 4 questions P3 alone gets right, 1 question P2 alone gets right, 5
discordant pairs, **p = 0.375**. Three questions on n=30 is not evidence. The accuracy claim
stands or falls on the held-out 70; the token claim (a 10× difference) does not need a test.

---

## 2. Shared across all three pipelines (the comparability contract)

- **Generator**: Gemini 2.5 Flash (`gemini-2.5-flash`), temperature 0, `MAX_TOKENS=2048`
- **System prompt**: `config._OLYMPICS_PROMPT`, verbatim identical for P1/P2/P3
- **Judge**: OpenAI `gpt-4o-mini`, blind to pipeline, with the declared anti-parametric clause
- **Embedder** (P2 + P3): `BAAI/bge-small-en-v1.5`, 384-dim, cosine, same `get_embedder()`
- **BERTScore**: `bert-score==0.3.13`, roberta-large, lang=en, rescale_with_baseline=True, idf=False
- **Corpus**: 15,884 chunks, 1,800-char target / 250 overlap, shared by P2 and P3

### `MAX_TOKENS=2048` is not a free parameter — it is a bug fix
Gemini 2.5's *thinking* tokens count against `maxOutputTokens` but are reported separately from
`candidatesTokenCount`. At the inherited 256 cap, answers were being truncated mid-title while
`output_tokens` read 11, so truncation was invisible to the SEC-study audit (which tested
`output_tokens` against the cap). Thinking was observed up to 1,964 tokens. At 2048, **90/90 dev
rows across all three pipelines finish with `finish_reason = STOP`.** Applied identically to all
three pipelines, so it biases no comparison.

---

## 3. Pipeline 1 — LLM-only
No retrieval, no knobs. Dev strict **0.300–0.333** (two runs; P1 is the one pipeline here whose
answers are not reproducible run to run — see §4.3).

Note for the report: every question carries `guess_baseline: 0.0`, and that is empirically
false — P1 answers 0.86 of temporal and 0.67 of superlative questions from parametric knowledge
alone. This corpus is real Wikipedia and the judge model knows the Olympics.

---

## 4. Pipeline 2 — Basic RAG (the baseline), **tuned**

```
P2_SEARCH=similarity        CONTEXT_NUM_CHUNKS=20      CONTEXT_CHUNK_CHARS=2000
P2_COMPRESS=1               P2_COMPRESS_SENTS=30       P2_ADAPTIVE_K=0
```

### 4.1 Three SEC-era defects fixed *before* tuning
The baseline was inheriting SEC-study decisions that were wrong for this corpus. Each would have
made P3 look better by making P2 worse, so each was fixed first.

| # | Defect | Effect if left |
|---|---|---|
| 1 | adaptive-k borrowed the **SEC** router | 24/30 dev questions mislabelled `lookup`; adaptive-k crippled. Now uses `pipeline3_olympics/lexicon.py` → **30/30** |
| 2 | adaptive-k was **min-capped** at the global k | could only ever retrieve *fewer* docs, while aggregation/superlative questions have 11–12 gold docs each |
| 3 | the **SEC ticker filter** was live | measured inert (fires on 0 of all 150 dev+test+hidden questions), now dataset-gated anyway |

SEC regression: across all 74 SEC questions, old and new `pipeline2_basic_rag.py` give **0
differences** in ticker-filter and adaptive-k decisions; route table, search default, compression
defaults and `build_prompt` all unchanged. the SEC study is untouched.

### 4.2 The sweep (dev 30, single run per arm)

| arm | strict | graded | bert | tokens | recall@k |
|---|:-:|:-:|:-:|:-:|:-:|
| **topk_20_c30  ← FROZEN** | **0.867** | 0.867 | 0.779 | **4,997** | 0.861 |
| topk_20 | 0.833 | 0.833 | 0.779 | 12,099 | 0.861 |
| topk_30 | 0.833 | 0.833 | 0.730 | 17,932 | 0.884 |
| topk_20_chunk1000 | 0.833 | 0.833 | 0.772 | 6,973 | 0.861 |
| topk_15 | 0.800 | 0.800 | 0.776 | 9,092 | 0.834 |
| topk_30_chunk1000 | 0.800 | 0.800 | 0.760 | 10,342 | 0.884 |
| adapt_wide | 0.767 | 0.767 | 0.652 | 8,074 | 0.784 |
| base (k=10) | 0.633 | 0.633 | 0.675 | 6,135 | 0.731 |
| chunk_1000 | 0.633 | 0.633 | 0.675 | 3,605 | 0.731 |
| chunk_3000 | 0.633 | 0.633 | 0.668 | 6,133 | 0.731 |
| mmr | 0.500 | 0.500 | 0.523 | 5,976 | 0.595 |
| mmr_rerank | 0.500 | 0.500 | 0.467 | 5,922 | 0.614 |
| topk_5 | 0.500 | 0.500 | 0.575 | 3,167 | 0.600 |

**What decided it.** `k` is the dominant knob: 10 → 20 is **+0.200 strict**, and the gain is
entirely in aggregation (0.17 → 0.83) and multi_hop (0.38 → 0.62) — the two types whose gold
answers span many documents. It saturates at 20 (k=30 adds recall but no strict pass and 48%
more tokens). Extractive compression on top is close to free accuracy: it cut tokens 59%
(12,099 → 4,997) while strict went 0.833 → 0.867. MMR and the cross-encoder reranker both
*hurt* badly (0.500) — same finding as the SEC study, independently reproduced.

`adapt_wide` (per-route k: 30 for aggregation/superlative, 4–6 elsewhere) **lost** to flat k=20.
Adaptive-k is therefore **not** adopted on Olympics. the SEC study adopted it; this is a genuine
cross-dataset difference, and the contract explicitly forbade carrying the SEC values over.

### 4.3 Stage 2 — three runs each, and why the spread is zero

| arm | strict mean | sd | spread | runs |
|---|:-:|:-:|:-:|:-:|
| topk_20_c30 | 0.867 | 0.000 | 0.000 | 3 |
| topk_20 | 0.833 | 0.000 | 0.000 | 3 |
| base | 0.633 | 0.000 | 0.000 | 3 |

This is **not** the same situation as SEC (±0.019–0.025 strict over 3 runs), but the zero is
narrower than it looks, and an initial reading of it was wrong. **The noise floor is
pipeline-dependent, not a property of the dataset:**

| pipeline | identical answer text across independent runs | strict flips |
|---|:-:|:-:|
| P3 GraphRAG | **30/30** | 0 |
| P2 Basic RAG (fixed config, 3 runs) | 30/30 `base`, 29/30 `topk_20_c30` | 0 |
| P1 LLM-only | **21/30** | 1 (0.333 ↔ 0.300) |

The two **grounded** pipelines are effectively deterministic: they copy short factoid answers
out of retrieved context, so there is little wording to vary. **P1 is not** — it generates from
parametric memory and its answers drift. The one flip is a transliteration variant,
`Yevgeny Dementyev` vs `Evgeniy Dementyev`, passed in one run and failed in the next.

So P1's dev strict is **0.300–0.333 (±0.033 over two runs)**, while the P2-vs-P3 comparison that
the submission actually rests on is between two stable pipelines. A first pass at this section
claimed "the generator is deterministic on this dataset"; the P1 measurement falsifies that, and
the claim is withdrawn.

Read the zero correctly: **run-to-run noise ≈ 0 for P2 and P3, but dev-set sampling error on
n=30 is large.** Zero variance across repeated runs says nothing about which 30 questions landed
in dev. That is why §1 reports a significance test rather than pointing at the zero sd.

### 4.4 A property of this corpus worth recording
Olympic infoboxes contain no sentence-ending punctuation, so the extractive compressor's
sentence splitter treats an entire infobox as **one 500–1,100-char "sentence"**. This is
favourable to P2 — the answer-bearing unit is never cut in half — but it means
`P2_COMPRESS_SENTS=30` is a far coarser budget than the name suggests. Only 3 of 15,884 chunks
exceed 2,000 chars, which is also why `chunk_3000` reproduced `base` exactly and served as an
unplanned noise control.

---

## 5. Pipeline 3 — GraphRAG: **frozen at as-built defaults, not tuned**

```
P3_GROUP_ROW_CAP=60   P3_VEC_K=10   P3_LOOKUP_K=6
MAX_CONTEXT_CHUNKS=10 MAX_CHUNK_CHARS=2000
```

P3 scores 29/30 on dev and the single miss (`pub-099`) is the deliberately ambiguous
(venue, date) key the multi_hop branch is designed *not* to tie-break. There is no accuracy
headroom, and tuning toward that one question would mean tuning against the one case the design
intentionally declines to answer.

The remaining knobs are **provably inert**, from files rather than from runs
(`p3_tune_olympics.py --check-inert`):

- **`P3_GROUP_ROW_CAP=60` is a dead knob.** The largest (sport, games) group in the entire
  corpus has **43** events, so the cap never binds — 0 of 265 groups are truncated. Lowering it
  to 30 would truncate 6 groups and to 20 would truncate 14, and for the superlative branch
  truncation can drop the argmax row itself. It would buy a few tokens by risking the answer.
- **`P3_VEC_K` / `P3_LOOKUP_K` drive the vector fallback, which fires on 0/30 dev questions**
  (`branch_resolved` is True 30/30). They are untestable on dev, so they are left at their
  as-built values rather than tuned on no evidence. They may matter on the held-out 70 if a
  branch fails to resolve; that is a declared risk, not a tuned choice.

So **P3 was not tuned on dev at all.** Its dev score is, in effect, already an out-of-sample
number for every knob in this section.

### 5.1 The context-budget asymmetry, declared
The contract said P2 and P3 share one ceiling (`CONTEXT_NUM_CHUNKS=10`, 2,000 chars). Raising P2
to k=20 breaks that. Because P3 defaults `MAX_CTX_CHUNKS`/`MAX_CTX_CHARS` off the same shared
config keys, the change would silently have doubled P3's chunk ceiling too — so P3's caps are
now **pinned explicitly** in `run_olympics_benchmark.py` to the values it was measured at.

Net effect: **the baseline is allowed a larger retrieval budget than the system under test.**
That asymmetry favours P2. It is deliberate, it follows from "optimize P2 irrespective of P3",
and it is declared here so no reader has to discover it.

---

## 6. Predictions, scored

Recorded before any Olympics pipeline was run. Three of four are wrong.

| prediction | outcome |
|---|---|
| token reduction lands **below** SEC's 34.2% | **WRONG** — 90.1%, nearly triple SEC |
| large gains on aggregation, superlative, temporal | **MOSTLY WRONG** vs a tuned baseline: superlative +0.00, temporal +0.00, aggregation +0.17 |
| little or no gain on lookup | **CORRECT** — +0.00, both at 1.00 |
| uncertain on multi_hop | the **largest** gain (+0.25) — the least confident prediction was the real effect |

The step-8 reading ("the lead is concentrated in aggregation and multi_hop") identified the right
*where* but the wrong *magnitude*: most of the apparent superlative and temporal advantage was
the baseline under-retrieving, not the graph helping.

---

## 7. Exact reproduction

```bash
# the frozen env is baked into the wrapper and printed on every run
python "build/olympics/run_olympics_benchmark.py" --split dev --runs 1 --pipelines 1,2,3 --tag frozen
python "build/olympics/evaluate_olympics.py" \
    results/olympics/bench_olympics_dev_frozen_run1.json --out results/olympics/eval_frozen_run1.json

# the sweep that chose it
python "build/olympics/p2_sweep_olympics.py" --stage 1
python "build/olympics/p2_sweep_olympics.py" --stage 2 --arms topk_20_c30,topk_20,base --runs 3
python "build/olympics/p3_tune_olympics.py" --check-inert
```

## 7a. Freeze verification (run after freezing, before any held-out run)

`results/olympics/eval_frozen_run1.json`, all three pipelines in one batch under the frozen env:

| pipeline | strict | tokens | recall@k | precision@k | matches |
|---|:-:|:-:|:-:|:-:|---|
| P1 | 0.300 | 283 | — | — | 0.333 in dev3 (±1 question, §4.3) |
| P2 | 0.8667 | 4,997 | 0.861 | 0.207 | sweep arm `topk_20_c30` exactly |
| P3 | 0.9667 | 497 | 0.674 | 0.983 | dev3 (496 tokens) — **pin held** |

The P3 row is the one that mattered: it confirms raising P2 to k=20 did **not** leak into P3
through the shared `CONTEXT_NUM_CHUNKS` config key.

## 8. What step 10 must do, now that this is frozen
Held-out 70, 3 runs, all three pipelines in one batch; the `P3_ROUTE_FORCE` un-confounded router
ablation; `router_lexicon_eval.py` on test 70 and on the hidden 50 (router accuracy only — the
hidden set has no answers and must never be scored for accuracy).
