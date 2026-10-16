# GraphRAG on SEC filings — context optimization across every query route

**Date:** 2026-10-06 · **Config tag:** `final` (supersedes `the intermediate build`, which is retained as a
labelled intermediate) · **Generator:** Gemini 2.5 Flash (temp 0) · **Judge:** OpenAI
`gpt-4o-mini` (blind, different model family) · **Embedder:** `BAAI/bge-small-en-v1.5` (384-dim,
shared by P2 and P3) · **Graph:** TigerGraph Savanna · **Runs:** 3 × 3 pipelines on 150 questions
in one batch, plus three single-run arms.

Supersedes [RESULTS_SEC.md](RESULTS_SEC.md) for Pipeline 3's context
assembly and router. Companions: [RESULTS_SEC.md](RESULTS_SEC.md) ·
[RESULTS_SEC.md](RESULTS_SEC.md) · [RESULTS_SEC.md](RESULTS_SEC.md)

Machine-readable: `results/sec/summary.json` · raw:
`bench_combined150_final_run{1,2,3}.json`, `eval_final_run{1,2,3}.json`

---

## 0. The one-paragraph summary

The initial implementation's token win came from one mechanism — substituting a graph edge's evidence *sentence* for the
whole chunk it lives in — and that mechanism fired on **35 of 150 questions**. On those 35 it cost
652 tokens against Basic RAG's 4,061 (−84%) and scored 0.83 strict against 0.14. On the other 115
it cost **4,406 tokens, 4% MORE than the baseline**, for 0.31 against 0.28. The mechanism was never
the problem; its coverage was. The optimization extends it to every route — a metric-row extractor for financial
lookups and comparisons, a curated-edge evidence path for relation questions, an SEC item-code
timeline for temporal questions, and evidence substitution for bridge. **Result: 1,026 tokens
against the baseline's 4,187 (−75.5%) at 0.487 strict against 0.253, with the official held-out 50
holding exactly the initial implementation's accuracy (0.553) on half the tokens.** Pipeline 3 now runs **31% below the
compressed baseline's budget at 2.5× its accuracy**, which settles the equal-budget objection by
construction rather than by argument.

The claim that got *smaller* this round is the set of routes that work: `temporal_order` sits at
0.00 and `bridge` at 0.05, unchanged. The optimization made them cheap, not correct.

---

## 0.1 What the token figures in this document do and do not include

Every `total_inference_tokens` figure here is **prompt + completion**. Gemini 2.5 Flash also
emits *reasoning* ("thinking") tokens, which the API bills as output but reports in a separate
field. `metrics.py` records them as `thinking_tokens` and **excludes them from
`total_inference_tokens`**, so the headline reduction understates what was actually paid for.

Both accountings, combined 150, 3 runs:

| | reported (prompt+completion) | thinking | true billed | thinking share |
|---|:-:|:-:|:-:|:-:|
| P1 LLM-only | 281.6 | +279.9 | 561.5 | 49.8% |
| P2 Basic RAG | 4,186.9 | +392.7 | 4,579.6 | 8.6% |
| **P3 GraphRAG** | **1,026.1** | **+269.4** | **1,295.5** | **20.8%** |

**Token reduction P3 vs P2: 75.5% as reported · 71.7% on true billed cost.**

The reduction is smaller under the honest accounting but the conclusion does not change, and
neither does the equal-budget result in §5: P3's true 1,295.5 against the compressed baseline's
true 1,852.1 is **30.1% below budget** (31.3% on reported tokens) at **2.52× the accuracy**.

Thinking tokens hurt P1 most in relative terms (half its spend) because its prompts are tiny, and
P2 least (8.6%) because its 4k-token contexts dominate. Reporting prompt+completion is therefore
mildly *unfavourable* to P3 in the P1 comparison and mildly favourable in the P2 comparison; the
table above lets a reader use whichever basis they prefer.

`build/analysis/verify_claims.py` recomputes this table and asserts the exclusion is real.

---

## 1. The diagnosis — one structural asymmetry explained the whole token table

> **Scope: this whole section describes the initial implementation, not the final system.** Every figure below is measured from
> `bench_combined150_initial_run{1,2,3}.json` — the *pre-optimization* baseline. It is the motivation for
> the optimization, and auditing it against `final` artifacts will not reproduce it. For what the final system actually
> does to these same questions, see §3. (After the optimization, the "full chunks" group below falls from 4,406
> tokens to 1,139.9.)

Pipeline 3 had two context builders:

- `_context_from_prov` — one evidence sentence per graph edge. **The token win.**
- `_context_from_chunks` — whole chunks, capped at 2,000 chars (~500 tokens) each, up to 10.

In the initial implementation, `_context_from_prov` was called from **two call sites, both inside the `aggregation`
branch**. Every other route called `_context_from_chunks`. Splitting all 150 questions by which
builder actually ran:

| group | n | P2 tok | P3 tok | saving | P3 ctx tok | P3 strict | P2 strict |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| evidence substitution fired | 35 | 4,061 | **652** | **+83.9%** | 303 | **0.829** | 0.143 |
| full chunks | 115 | 4,226 | 4,406 | **−4.3%** | 3,540 | 0.310 | 0.281 |

This is a sharper statement than "the aggregation route is cheap", and it corrects that framing:
the aggregation *branch* also has two chunk fall-throughs, and **29 of 64 aggregation-routed
questions took them** (20 misrouted `compare`, 4 Tier B, 3 Tier C, 2 `agg_supplier`). Route label
did not predict cost; builder did.

### 1.1 Four defects found while building the fix

Each was measured, not inferred:

| # | Defect | Evidence |
|:-:|---|---|
| 1 | **The relation gate never fired on the questions it existed for.** `_wants_curated_facts` tested a noun list containing `auditor` but not the verb `audits`. All 10 `lookup_auditor` questions read "Who audits X?" and failed the gate, so the `audited_by` edge that answers them outright was never consulted. Across all 36 lookup-routed questions the gate fired on 6 and a company seed resolved on 3. | `lookup_auditor` cost 3,256 tokens to Basic RAG's 2,500 and merely **tied** it at 0.90 strict |
| 2 | **Financial retrieval was looking in the wrong sections.** `_company_financial_chunks` filtered to `item7`/`item8`, but the chunker assigns statements inconsistently per filer: `header` for GE, `item15` for JPMorgan, `item16` for ExxonMobil, `item6`/`item7a` for AIG, and `body` for Citigroup — **which has zero item7/item8 chunks**. The answer row was unreachable for those companies at any phrasing. | gold figure absent from every candidate chunk for 5 of 15 metric test cases |
| 3 | **Company-name resolution dropped punctuation and ate real apostrophes.** The alias index registered `comp.lower().split()[0]`, so "Netflix, Inc." indexed as `netflix,` **with the comma** and the common name never resolved. Separately the possessive stripper turned `Lowe's` into `lowe`, failing a company sitting in the index under its exact name. | 5/100 companies unresolvable in common form; SQ063 (3M) and SQ078 (Netflix) resolved no seed at all |
| 4 | **All 20 `compare` questions misrouted to aggregation.** Documented in RESULTS_SEC.md; the initial implementation shipped the fix behind `P3_COMPARE_FIX`, off by default. | router accuracy 0.800 |

---

## 2. What the optimization changed

All in `pipeline3_sec/pipeline.py`. Every item is deterministic string or graph work — no LLM
call, so `optimizer_tokens` stays 0 and the token comparison stays apples-to-apples.

| # | Change | Symbols |
|:-:|---|---|
| 1 | **Metric-row extractor.** Finds the one income-statement row answering a financial question, with the year/units header that disambiguates its columns. Label-anchored scoring (the metric label must *prefix* the text before the first figure), a ≥2-numeric-token gate, a dollar-magnitude test, and an adjacent-chunk header fallback for tables straddling a chunk boundary. | `_extract_metric_rows`, `_row_score`, `_metric_block`, `_chunk_lines` |
| 2 | **The question's own metric phrase as the top-priority label.** A fixed lexicon cannot be specific enough: ExxonMobil's consolidated line is `Total revenues and other income 332,238` while its *segment* table reads `Total revenues 111,193`, which matched the lexicon label exactly and won. Taking the phrasing from the question fixes that and also recovers Coca-Cola's `Net Operating Revenues`, a label no lexicon held. | `_question_metric_phrase`, `_METRIC_LEX` |
| 3 | **Broadened candidate pool.** All 10-K chunks for the ticker, any section (defect 2). Safe because the row scorer is itself highly selective, so this trades a cheap local scan for recall rather than noise. | `_fin_pool` |
| 4 | **Relation-gated curated-edge path.** Which curated relations a question asks about, keyed on ordinary English forms of the relation rather than a noun list, then the live graph queried for the targets and the local provenance index for each evidence sentence. | `_RELATION_PATTERNS`, `_wanted_relations`, `_graph_curated_targets`, `_prov_for_targets`, `_curated_prov` |
| 5 | **SEC item-code 8-K timeline** for temporal questions: one dated line per filing, built from the filing date and filer-assigned item code in the chunk id. **Not** from `events.json`, whose `type` field this project already distrusts and over which an earlier implementation feature was deliberately reverted. | `_company_8k_timeline`, `_SEC_ITEM_TITLES` |
| 6 | **Name-resolution rebuild** (defect 3): punctuation-stripped and suffix-stripped aliases, and the raw name tried *before* the possessive stripper. | `_company_key_for` |
| 7 | **Router: binary-comparison guard promoted to default**, plus structural broadening of the bridge, temporal and comparison patterns. | `BINARY_COMPARE_RE`, `COMPARE_METRIC_RE`, `BRIDGE_SHARED_RE`, `BRIDGE_REL_RE`, `TEMPORAL_EXT_RE` |
| 8 | **Bridge context assembly** switched to evidence substitution with a reduced vector backstop. Hop resolution untouched — this is a token change and must not be read as a bridge-accuracy fix. | `_bridge_graph_facts` |
| 9 | **Compound-question guard (the final system)** — see §4. | `_COMPOUND_RE`, `_asks_multiple`, `COMPOUND_K` |

---

## 3. Headline — combined 150, 3 runs, one batch (mean ± population std)

| metric | P1 LLM-only | P2 Basic RAG | **P3 GraphRAG** |
|---|:-:|:-:|:-:|
| strict pass | 0.093 | 0.253 | **0.487 ± 0.005** |
| graded quality | 0.209 | 0.341 | **0.527** |
| avg inference tokens | 282 | 4,187 | **1,026** |
| strict pass / 1k tokens | 0.330 | 0.061 | **0.474** |

**Token reduction 75.5% · strict lead +0.234 · efficiency 7.84×.**

### 3.1 By evaluation set

| set | n | P1 | P2 | P3 initial | **P3 final** | reduction | efficiency |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| extra100 (route-balanced) | 100 | 0.083 | 0.227 @ 4,227 | 0.370 @ 3,944 | **0.453 ± 0.005 @ 848** | **+79.9%** | 9.97× |
| official held-out 50 | 50 | 0.113 | 0.307 @ 4,106 | 0.553 @ 2,702 | **0.553 ± 0.009 @ 1,382** | **+66.3%** | 5.36× |
| combined 150 | 150 | 0.093 | 0.253 @ 4,187 | 0.431 @ 3,530 | **0.487 ± 0.005 @ 1,026** | **+75.5%** | 7.84× |
| tune-120 | 120 | 0.108 | 0.258 @ 4,186 | 0.414 @ 3,592 | 0.469 ± 0.004 @ 1,023 | +75.6% | 7.43× |
| holdout-30 | 30 | 0.033 | 0.233 @ 4,189 | 0.500 @ 3,279 | 0.556 ± 0.016 @ 1,037 | +75.2% | 9.62× |
| holdout-23 (untouched) | 23 | 0.043 | 0.130 @ 4,173 | 0.478 @ 3,106 | 0.551 ± 0.021 @ 1,191 | +71.5% | 14.8× |

**Read the held-out 50 row carefully, because it is the honest one.** The final system scores *exactly* the initial implementation's
0.553 — not better — on **half the tokens**. The accuracy gains are on extra100; the official set
buys efficiency only. The route-balanced extra100 is where the token thesis was weakest in the initial implementation
(+6.7%) and is now strongest (+79.9%), which is the reverse of the initial implementation picture and the single most
load-bearing change in this study.

### 3.2 By gold route, extra100

| gold route | n | P2 | P3 initial | **P3 final** | P2 tok | initial tok | **final tok** | reduction |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| aggregation | 20 | 0.10 | 0.85 | 0.85 | 3,926 | 860 | **857** | +78.2% |
| lookup | 20 | 0.85 | 0.75 | **0.95** | 2,975 | 4,058 | **510** | +82.9% |
| compare | 20 | 0.10 | 0.15 | **0.42** | 5,067 | 5,232 | **486** | +90.4% |
| bridge | 20 | 0.05 | 0.05 | 0.05 | 4,323 | 5,015 | **1,767** | +59.1% |
| temporal | 20 | 0.03 | 0.05 | **0.00** | 4,844 | 4,552 | **620** | +87.2% |

**`aggregation` is byte-identical by design** — the route the optimization did not touch, reported as a control.
**`lookup` reverses the initial implementation's one outright loss to Basic RAG** (0.75 vs 0.85 → 0.95 vs 0.85) at 83%
fewer tokens. **`compare` nearly triples** (0.15 → 0.42). **`bridge` and `temporal` got cheap, not
correct** — see §7.

### 3.3 By tier, official held-out 50

| tier | n | P2 | P3 initial | **P3 final** | P2 tok | initial tok | **final tok** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A — single-hop | 8 | **0.88** | 0.75 | 0.75 | 3,653 | 4,558 | **1,534** |
| B — bridge / multi-hop | 16 | 0.21 | 0.35 | **0.38** | 4,109 | 3,622 | **1,940** |
| C — aggregation | 20 | 0.15 | 0.60 | 0.60 | 4,252 | 672 | 672 |
| D — temporal | 6 | 0.33 | **0.67** | 0.61 | 4,218 | 4,540 | **2,060** |

Tier A remains Basic RAG's win (0.88 vs 0.75) — the optimization made it 58% cheaper without closing it. Tier D's
0.67 → 0.61 is one question on n=6 and sits inside the run spread; it is *not* evidence the temporal
work helped, and §7 says so.

### 3.4 Per-shape token table, before and after

| shape | n | P2 tok | initial tok | **final tok** | reduction | initial ctx | **final ctx** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `agg_sector` | 5 | 4,259 | 367 | 367 | +91.4% | 72 | 72 |
| `compare_revenue` | 7 | 5,420 | 5,286 | **485** | **+91.0%** | 4,155 | **139** |
| `compare_net_income` | 13 | 4,877 | 5,203 | **486** | **+90.0%** | 4,130 | **130** |
| `lookup_revenue` | 10 | 3,456 | 4,861 | **396** | **+88.5%** | 3,632 | **69** |
| `temporal_order` | 20 | 4,844 | 4,552 | **620** | **+87.2%** | 3,769 | **288** |
| `agg_sector_auditor` | 8 | 3,792 | 617 | 617 | +83.7% | 274 | 274 |
| `lookup_auditor` | 10 | 2,494 | 3,256 | **624** | **+75.0%** | 2,705 | **321** |
| `agg_auditor` | 4 | 3,711 | 1,176 | 1,176 | +68.3% | 720 | 720 |
| `bridge_shared_auditor` | 12 | 4,203 | 4,828 | **1,589** | **+62.2%** | 4,067 | **1,163** |
| `bridge_shared_supplier` | 2 | 4,512 | 4,687 | **1,964** | +56.5% | 4,049 | **1,507** |
| `bridge_shared_director` | 6 | 4,498 | 5,499 | **2,058** | +54.3% | 4,563 | **1,563** |
| `agg_supplier` | 3 | 4,018 | 1,909 | 1,888 | +53.0% | 1,422 | 1,400 |
| Tier C | 20 | 4,252 | 672 | 672 | +84.2% | 325 | 325 |
| Tier A | 8 | 3,653 | 4,558 | **1,534** | +58.0% | 3,457 | **1,099** |
| Tier B | 16 | 4,109 | 3,622 | **1,940** | +52.8% | 2,887 | **1,441** |
| Tier D | 6 | 4,218 | 4,540 | **2,060** | +51.2% | 3,644 | **1,486** |

**Every shape is now token-positive against the baseline.** In the initial implementation, **eight of sixteen** were
negative: `lookup_revenue` (−40.6%), `lookup_auditor` (−30.2%), `tierA` (−24.8%),
`bridge_shared_director` (−22.6%), `bridge_shared_auditor` (−14.8%), `tierD` (−7.7%),
`compare_net_income` (−6.7%), `bridge_shared_supplier` (−3.9%). Verified by
`build/analysis/verify_claims.py`.

### 3.5 Set-F1, extra100

| shape | n | P1 | P2 | the initial implementation | **final** |
|---|:-:|:-:|:-:|:-:|:-:|
| `agg_auditor` | 4 | 0.095 | 0.478 | 1.000 | 1.000 |
| `agg_sector` | 5 | 0.606 | 0.335 | 1.000 | 1.000 |
| `agg_sector_auditor` | 8 | 0.611 | 0.484 | 1.000 | 1.000 |
| `agg_supplier` | 3 | 0.583 | 0.556 | **0.250** | **0.250** |
| `compare_net_income` | 13 | 0.641 | 0.120 | 0.350 | **0.675** |
| `compare_revenue` | 7 | **1.000** | 0.286 | 0.143 | **0.603** |
| `bridge_shared_auditor` | 12 | 0.054 | 0.000 | 0.000 | 0.000 |
| `bridge_shared_director` | 6 | 0.185 | 0.250 | 0.167 | 0.167 |
| `bridge_shared_supplier` | 2 | 0.500 | 0.000 | 0.667 | 0.667 |

The aggregation 1.000s are the circularity RESULTS_SEC.md quantified — those golds *are* this graph's
contents — and must be read as "recovers its own graph", not "is perfect". The real movement is
`compare`, which nearly doubles on net income and quadruples on revenue. `agg_supplier` is
unchanged and still **below Basic RAG** — see §7.

---

## 4. The final system — the regression the optimization caused, and the fix

`the intermediate build` is reported here as a labelled intermediate rather than hidden, because its failure is
the most instructive result in the study.

The optimization's organising rule was *when a deterministic artifact answers the question, feed that and nothing
else*. That is correct for a single-fact question and **wrong for a compound one**. Measured on
`the intermediate build` run 1:

> **EQ49** — "Cisco's May 2026 restructuring 8-K funds which growth areas, **and** who is Cisco's
> auditor?" the initial implementation answered both halves from 4,460 context tokens. The optimization matched the auditor relation,
> resolved that one edge, fed **90 context tokens**, and refused.

**EQ45** failed identically (two CFO transition dates dropped by the timeline snippet). The damage
landed exactly where compound questions live:

| set / tier | the initial implementation | **intermediate** | **final** |
|---|:-:|:-:|:-:|
| official held-out 50 | 0.553 | **0.480** | **0.553** |
| Tier B (n=16) | 0.35 | 0.25 | **0.38** |
| Tier D (n=6) | 0.67 | 0.33 | 0.61 |
| Tier A (n=8) | 0.75 | 0.75 | 0.75 |
| Tier C (n=20) | 0.60 | 0.60 | 0.60 |
| combined 150 | 0.431 | 0.460 | **0.487** |
| combined 150 tokens | 3,530 | 981 | 1,026 |

**The fix** is one detector keyed on coordination structure — a comma-joined `and` clause, or `and`
followed by a second interrogative — not on any question's wording. When it fires, the resolved
artifact is **augmented** with `P3_COMPOUND_K=4` retrieved chunks instead of replacing them. It
fires on the three compound held-out questions and on none of eight single-ask controls
(`Who audits Altria?`, the revenue lookups, the comparisons, the chronological-order questions, the
auditor enumerations).

Its measured footprint is confined to the shapes that regressed — **Tier D +562 and Tier B +187
context tokens, ±0 on every other shape** — and costs 45 tokens on the combined-150 average
(981 → 1,026). EQ45 and EQ49 both flip back to correct. EQ47 still fails, as it did in the initial implementation.

---

## 5. The equal-budget comparison — against BOTH baseline arms

Once Pipeline 3 gets cheap, comparing it only to the 4,187-token baseline flatters it. The
compressed baseline from The initial implementation §5.2 is re-run here on the same 150 questions.

| arm | strict | graded | tokens | strict / 1k |
|---|:-:|:-:|:-:|:-:|
| P2 adaptive-k (frozen baseline, 3 runs) | 0.253 | 0.341 | 4,187 | 0.061 |
| P2 + extractive compression (1 run) | 0.193 | 0.270 | 1,493 | 0.130 |
| **P3 GraphRAG (3 runs)** | **0.487** | **0.527** | **1,026** | **0.474** |

**Pipeline 3 runs 31.3% BELOW the compressed baseline's token budget at 2.52× its strict pass.**
Efficiency is **3.66×** against the compressed arm and 7.84× against the frozen one; the 3.66× is
the conservative figure and the one to quote.

This is a stronger form of the answer than the initial implementation could give. In the initial implementation the compressed baseline was
*cheaper* than Pipeline 3 (1,550 vs 2,703) and the argument had to be "it is still less than half as
accurate on a smaller budget". Now Pipeline 3 is both cheaper and more accurate, so the budget
confound is gone rather than argued around.

---

## 6. Router — accuracy, paraphrase stability, and cross-validation

`route()` is a pure function of the question string and costs zero tokens, so every number here is
free and re-runnable: `python "build/analysis/router_probe.py" --tag <tag>`.

| metric | frozen initial | **final** |
|---|:-:|:-:|
| routing accuracy (n=100 gold-labelled) | 0.800 | **1.000** |
| **paraphrase stability (n=408)** | **0.493** | **1.000** |
| stratified 5-fold CV | 0.802 | 1.000 |
| leave-one-route-out | 0.800 | 0.800 |

### 6.1 Paraphrase robustness — the safeguard that mattered most

`datasets/sec/paraphrase_probe.jsonl` mechanically rewrites all 150 questions into **408
paraphrases** via 42 structural rules (voice change, clause reorder, interrogative ↔ imperative,
possessive ↔ of-phrase, function-word synonyms), then asserts the router assigns each paraphrase its
original's route. It needs no gold label at all, which is why it is the strongest generalization
signal available here. It was built **before** any pattern was edited.

**The frozen initial router scored 0.493 — barely better than a coin flip — which its 0.800 routing
accuracy completely hid:**

| shape | initial stability | cause |
|---|:-:|---|
| bridge (all 3 shapes) | **0/64 = 0.000** | `BRIDGE_RE` required the literal stock phrase "the company that/whose/which" |
| `temporal_order` | 20/80 = 0.250 | `TEMPORAL_RE` keyed on the literal "order"; `\bchronologic\b` does not even match "chronologically" |
| compare | 1/80 = 0.013 | the documented misroute |
| `agg_*`, `lookup_*`, Tier A | 1.000 | — though `lookup` is the fallback route, so its stability is free |

The final system reaches **408/408**, and on both splits separately (tune-120 326/326, holdout-30 82/82).

Two honesty notes. **Stability is two-sided**: some the initial implementation instability ran in the *good* direction — a
paraphrase of a misrouted compare question routed correctly — so stability measures consistency, not
correctness, and routing accuracy is reported beside it. And a pattern that fires on an original but
not its paraphrase was rewritten structurally or dropped; none was retained.

### 6.2 Cross-validation, and why the k-fold number is worthless here

The 5-fold CV reports 1.000, and that figure should be ignored: **no pattern is ever ablated**,
because every newly added pattern's supporting questions span all five folds, so each fold's "training" folds
justify every pattern and the CV just reproduces in-sample accuracy. `router_probe.py` prints this
warning in its own output rather than letting the number stand.

**Leave-one-route-out** has real bite. A whole route is held out, every pattern whose justification
lies entirely within it is neutralized, and the route is re-scored:

| held-out route | routing accuracy | paraphrase stability | patterns ablated |
|---|:-:|:-:|---|
| aggregation | 1.000 | 1.000 | none (support is pre-existing) |
| bridge | 1.000 | **0.000** | `BRIDGE_SHARED_RE`, `BRIDGE_REL_RE` |
| compare | **0.000** | 0.738 | `BINARY_COMPARE_RE`, `COMPARE_METRIC_RE` |
| lookup | 1.000 | 1.000 | none |
| temporal | 1.000 | **0.250** | `TEMPORAL_EXT_RE` |
| **mean** | **0.800** | — | |

**The honest reading: the optimization's router patterns buy robustness, not in-sample accuracy** — except
`compare`, where they buy both. Bridge and temporal originals are still routed correctly by patterns
that predate this work; what collapses without the optimization additions is their *paraphrase* stability
(0.000 and 0.250). `TEMPORAL_RE` was deliberately kept separate from `TEMPORAL_EXT_RE` precisely so
this ablation could not credit the optimization with pre-existing work.

**Limitation:** LORO can only ablate patterns added during the optimization. Patterns from earlier iterations are tagged
`pre150` and are never removed, because no fold of these 150 questions is what motivated them. The
0.800 mean is therefore an upper bound on what this method can test.

---

## 7. Refreshed router ablation — routing vs branch machinery

`P3_ROUTE_FORCE` keeps every branch live and corrupts only the route *label*, so this isolates
routing from the branch machinery. Single run each, combined 150.

| arm | strict | graded | tokens | tg_live | GSQL queries | strict/1k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **router ON** (the final system run 1) | **0.493** | **0.533** | **1,025** | 0.95 | 182 | **0.481** |
| router RANDOM (`md5`-seeded) | 0.200 | 0.257 | 2,118 | 0.97 | 169 | 0.094 |
| router FIXED = lookup | 0.247 | 0.326 | 2,160 | 1.00 | 154 | 0.114 |

| component | Δ strict | Δ tokens |
|---|:-:|:-:|
| correct route labels (RANDOM → ON) | **+0.293** | **−1,093** |
| correct route labels (FIXED → ON) | +0.247 | −1,135 |

Correct routing is worth **+0.293 strict and 1,093 tokens**, against the initial implementation's +0.160 and 1,152. The
router became substantially more load-bearing for accuracy, because the branches it selects now do
much more work than they did in the initial implementation. These are single-run arms against a ±0.019–0.025 spread, so the
token figures are solid and the accuracy deltas — being 10×+ the noise floor — are safe, but should
not be quoted to three decimals.

---

## 8. Fairness invariants — each checked, with how

| invariant | how checked | result |
|---|---|---|
| Frozen generator / judge | `meta` in each bench file; `SYSTEM_PROMPT` sha256 recorded at session start and re-verified at the end | `gemini-2.5-flash` temp 0, `MAX_TOKENS=512`; prompt sha `3b47e5aab820c496` **unchanged** |
| Judge blind + different family | `gpt-4o-mini`, no pipeline identity in the prompt; SEC judge prompt rendered and compared byte-for-byte against the frozen submission literal | **byte-identical** |
| Shared embedder | both pipelines call `get_embedder()` | `BAAI/bge-small-en-v1.5`, 384-dim, one singleton |
| `optimizer_tokens == 0` | asserted across all 450 P3 rows | **true** |
| P2 untouched | P2's retrieval replayed over all 150 questions and compared to the batch; P2 rows compared across the intermediate build and the final system batches | `retrieved_ids` **identical 150/150**; `retrieved_ids` and `context_tokens` identical across both batches; P1 tokens 282.0 in both |
| One batch | P1, P2, P3 in a single `run_benchmark.py` invocation per run | 3 runs, tag `final` |
| Live graph | `P3_REQUIRE_TG=1`; `tg_live` derived from the per-question `tg_queries` counter | **0.947** combined / 1.00 extra100 / 0.84 held-out — **identical to the initial implementation**; 546 live GSQL queries |
| Equal ceiling | `MAX_CTX_CHUNKS` / `MAX_CTX_CHARS` still derive from config | 10 / 2,000, unchanged |
| No question-id leakage | `grep -nE '\b(EQ[0-9]{2}\|SQ[0-9]{3})\b'` | 4 matches, **all inside explanatory comments**, none in logic |
| No gold at inference | grep for `gold_members`, `correct_answer`, `answer`, `source_docs`, `answer_from`, and every eval-set path | **none appear anywhere in Pipeline 3** |
| No LLM in router/assembly | single `get_llm_response` call site | one, inside the generation block |
| Output not truncated | max `output_tokens` across 450 P3 rows | 106 of 512; **zero** rows at cap |
| §10 regression reproduces | re-evaluated `bench_test_v7final_run1.json` | **zero** per-item BERTScore differences, **zero** token differences, all three pipelines |

### 8.1 On `tg_live`, and a trade that was reversed

A first implementation of the optimization answered `compare`, `temporal` and `lookup_revenue` from local
artifacts with **zero** graph calls, which would have dropped `tg_live` from 0.947 to **0.55** while
the pipeline still called itself GraphRAG — exactly the provenance overstatement The initial implementation §6 was written
to end. That was reversed: TigerGraph's vector index now supplies the row candidates on those paths,
restoring `tg_live` to the initial implementation's figure **at no token cost**, and `_ROW_SOURCE_STATS` records whether the
graph's candidates or the local pool supplied the winning row, so the graph's contribution there is
reported rather than assumed.

The held-out 50 remains at 0.84 for the reason The initial implementation §6 enumerated: eight aggregation questions resolve
against the local index or a corpus scan by design. The honest framing is unchanged — **a live-graph
+ local-index hybrid** whose live share depends on question shape.

---

## 9. Methodology — what is in-sample, stated plainly

**The regexes and extractors were tuned against questions we had already measured. After Phase 2,
no held-out set remains for Pipeline 3, and every metric on the 150 is an in-sample number.** That
was a deliberate, accepted decision. It is labelled here rather than hidden, and three mechanisms
were used to supply generalization evidence by other means.

**Phase 1 / Phase 2.** A stratified 80/20 split was built first
(`make_tune_holdout_split.py`, seed `20261005`, stratified on `shape` for extra100 and the
organiser's `tier` for the official 50; disjointness, exhaustiveness and size asserted in code).
Everything was built against tune-120.

**The holdout-30 is partly contaminated, and by exactly 7 questions.** While debugging the
extractor I inspected EQ01, EQ02, EQ20, SQ018, SQ043, SQ074 and SQ075, all of which fell in the
holdout. `split_holdout23_clean.jsonl` holds the untouched 23 and
`results/sec/holdout30_contamination.json` records the full audit. Both are reported:

| split | n | P2 | **P3 final** | reduction | note |
|---|:-:|:-:|:-:|:-:|---|
| holdout-30 | 30 | 0.233 | **0.556 ± 0.016** | +75.2% | 7 questions inspected during development |
| **holdout-23 (untouched)** | 23 | 0.130 | **0.551 ± 0.021** | **+71.5%** | the honest generalization estimate |
| tune-120 | 120 | 0.258 | 0.469 ± 0.004 | +75.6% | in-sample |

The holdout-23 figure is the cleanest number in this document, and the gap between it and tune-120
is *negative* (0.551 vs 0.469) — Pipeline 3 does slightly **better** on data it was not tuned on,
which is weak evidence against overfitting. n=23 is small, so read it as reassurance, not proof.

**Phase 1 and Phase 2 collapsed into one batch.** The brief anticipated running the holdout once,
then re-tuning and re-running. No re-tuning happened after the freeze — the router was already at
1.000/1.000 and the extractor at 41/41 — so the holdout figures above are slices of the *same*
frozen `final` batch rather than a separate run. That is strictly better methodology than a
splice across batches, and it is why no number here mixes runs.

**One exception to the phasing, declared.** The compound-question defect (§4) was diagnosed from
**held-out** Tier D and EQ49 failures in `the intermediate build`. Folding its fix into `final` therefore makes
the held-out-50 figure in §3 in-sample. The initial implementation's `P3_COMPARE_FIX` was handled the same way and shipped
behind a flag; here it is on by default because the brief's Phase 2 explicitly authorizes folding
all 150 in with in-sample labelling. The `the intermediate build` numbers are retained above so the effect of that
one decision is visible rather than absorbed.

**The binary-comparison guard is not holdout-informed.** Its pattern is justified by the 16 compare
questions inside tune-120 alone; the 4 in the holdout add nothing to it.

---

## 10. Honest limitations

- **Two routes are cheap but still broken.** `temporal_order` is **0.00** and `bridge` **0.05**,
  both unchanged from the initial implementation. The optimization cut their cost by 87% and 59%; it did not make them work. Any summary
  saying "the optimization improved every route" would be false.
- **Temporal quality is not demonstrated.** The 20 `temporal_order` golds derive from
  `events.json dates`, a field this project documents as untrustworthy, so a score measured against
  it is weak evidence in either direction. The optimization timeline deliberately uses SEC item codes instead,
  which means it is *not* measured against its own source — but it also does not match that gold.
  Judged on Tier D instead, as the brief directs, the figure is 0.61 against the initial implementation's 0.67 on **n=6**:
  one question, inside the run spread, and **not** a quality gain.
- **`agg_supplier` still loses to Basic RAG** (set-F1 0.250 vs 0.556, n=3). Diagnosed and
  deliberately not fixed: two of the three questions name suppliers absent from `SUPPLIER_LEX`, and
  the only way to fix them within this route would be to add those literal supplier names — which
  §0.1 of the brief forbids as question-specific. The third over-returns 5 companies against a gold
  of 3, a graph-precision issue on the ingestion side.
- **Tier A remains Basic RAG's win** (0.88 vs 0.75). A tuned dense retriever is still the better
  tool for a single numeric fact; the optimization made that tier 58% cheaper without closing the gap.
- **extra100's gold overlaps this graph** on 70 of 100 questions (RESULTS_SEC.md). Its 0.453 is not an
  independent confirmation, and the aggregation set-F1 1.000s are circular by construction.
- **The ablation arms and the compressed-P2 arm are single-run** against a ±0.019–0.025 spread.
  Their token figures are reliable; their accuracy figures are indicative.
- **LORO cannot test pre-optimization patterns** (§6.2), so 0.800 is an upper bound on what it measures.
- **Router portability is still untested here.** In-domain routing is 1.000 and paraphrase
  stability 1.000, but that is one corpus. The SEC lexicon remains SEC-specific by construction;
  the 0.173 cross-domain figure in RESULTS_SEC.md is the relevant counterweight.
- **Ingestion cost is still not amortized** into per-question tokens (the initial implementation concern #8, open).
- **A shared-file collision was survived, not avoided.** A parallel session edited `config.py`,
  `metrics.py` and `pipeline2_basic_rag.py` during this work, and briefly made the TigerGraph
  `finance` queries unavailable mid-session. Both were verified harmless after the fact (§8) rather
  than prevented, and that verification is the only reason these numbers can be trusted.

---

## 11. Trend across configs (combined 150 unless noted)

| config | P2 strict | P3 strict | P3 tokens | reduction | P3 strict/1k |
|---|:-:|:-:|:-:|:-:|:-:|
| the initial implementation (held-out 50 only) | 0.300 | 0.533 | 2,703 | 34.2% | 0.197 |
| initial | 0.249 | 0.431 | 3,530 | 15.7% | 0.122 |
| the intermediate build | 0.251 | 0.456 | 981 | 76.6% | 0.465 |
| **the final system (`final`)** | **0.253** | **0.487** | **1,026** | **75.5%** | **0.474** |

The token-reduction headline went 50.2% → 34.2% → 15.7% → **75.5%**, and the direction reversed for
a reportable reason rather than a presentational one: 15.7% was the cost of a mechanism that covered
23% of questions, and 75.5% is that same mechanism covering all of them. Quality-per-token is
**3.9× initial's**, which is the first time in this project's history that a change moved the
efficiency *frontier* rather than the position on it — an earlier implementation, by contrast, was explicitly flat
(0.2049 vs 0.2063).

---

## 12. Reproducing

```bash
# splits + probes (free — no API, no corpus scan)
python "build/data_prep/make_tune_holdout_split.py"
python "build/data_prep/make_paraphrase_probe.py"
python "build/analysis/router_probe.py" --tag the intermediate build
python "build/analysis/test_extractor.py"

# offline context-size preview (free; reproduces measured context tokens exactly)
python "build/bench/dryrun_context.py" --split combined150

# the batch (3 runs, all three pipelines, one invocation each)
DATASET=sec LLM_PROVIDER=gemini GEMINI_MODEL=gemini-2.5-flash \
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 P2_ADAPTIVE_K=1 P3_REQUIRE_TG=1 \
TIGERGRAPH_GRAPH=finance MAX_TOKENS=512 CONTEXT_NUM_CHUNKS=10 \
CONTEXT_CHUNK_CHARS=2000 TEMPERATURE=0 \
python "build/bench/run_benchmark.py" --split combined150 --runs 3 \
  --pipelines 1,2,3 --tag final --out-dir results/sec

for i in 1 2 3; do
  JUDGE_MODEL=gpt-4o-mini DATASET=sec EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
  python "build/eval/evaluate.py" \
    results/sec/bench_combined150_final_run$i.json \
    --ref-file datasets/sec/combined150_set.jsonl \
    --out results/sec/eval_final_run$i.json
done

# compressed-baseline arm
... P2_COMPRESS=1 P2_COMPRESS_SENTS=12 --runs 1 --pipelines 2 --tag abl_compressed_baseline

# un-confounded router ablation
... P3_ROUTE_FORCE=random --runs 1 --pipelines 3 --tag abl_router_random
... P3_ROUTE_FORCE=lookup --runs 1 --pipelines 3 --tag abl_router_lookup

# tables
python "build/analysis/summarize_results.py" --tag final --baseline-tag initial

# re-verify every narrative figure in this document against the raw artifacts (free, no API calls)
python "build/analysis/verify_claims.py"
```

Every table in this document is generated by `summarize_results.py` or `router_probe.py` from the JSON
artifacts named above.

The figures that appear in **prose** rather than in a generated table — the §1 builder split, the
§1 fall-through breakdown, the §3.4 sign counts, the §5 equal-budget comparison and the
thinking-token accounting in §0.1 — are checked by `build/analysis/verify_claims.py`, which
recomputes each one from `bench_*`/`eval_*` and exits non-zero if any has drifted from what this
document asserts. It writes `results/sec/claims_verification.json`. Current status: **5 checks,
0 failing assertions.**

One caveat it records rather than hides: the §1 split is detected by a citation-prefix proxy, and
on `initial` one question (`EQ42`) flips that proxy between runs (sizes 35/35/34) while its
`context_tokens` stay at 213 in all three — the builder fired every time and only the citation
label moved. The proxy is the imprecise part, not the pipeline.
