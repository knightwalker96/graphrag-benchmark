# GraphRAG explained from zero — a complete walkthrough

**What this document is.** A first-principles walkthrough of the this project work in this repo, written
for someone who has never seen it. Every term is defined before it is used. Every number is traced
to a file, and where a number in the repo could not be reproduced, that is said rather than
smoothed over.

**Date written:** 2026-10-05 · **Updated:** 2026-10-11

> ### Update — the gaps this document flagged have since closed
>
> Both of the "we cannot know this yet" caveats below were resolved a few hours after writing:
>
> 1. **`RESULTS_SEC.md` now exists** — see
>    [RESULTS_SEC.md](RESULTS_SEC.md), which supersedes this document's the optimization
>    sourcing and is tagged `final`.
> 2. **the final system has been judged**, all three runs (`results/sec/eval_final_run{1,2,3}.json`).
>    The open question in §D3 — whether the compound-question guard restores the official held-out
>    50 — is answered: **strict pass 0.56 / 0.54 / 0.56, mean 0.553**, exactly recovering the initial implementation's
>    0.553 from the optimization's 0.480. The §D3 arithmetic guessed 0.560; the measured value is 0.553.
>
> **Added 2026-10-11, two corrections to this document:**
>
> 3. The "could not reproduce 35 / 652 / 4,406" note in Part B was **my error and is retracted** —
>    those figures reproduce exactly. The corrected analysis, including why the definition of
>    "fired" mattered, is in place below.
> 4. A **thinking-token** section has been added (§A5a). Every token figure in this document, and
>    in every other this project document, excludes billed reasoning tokens; the true token reduction
>    is **71.7%**, not 75.5%.
>
> Everything else below stands as measured. Part D's the optimization-vs-the initial implementation analysis remains correct for
> `the intermediate build`; read it as the intermediate step it was, with the final system as the landing point.

**A note on sources (as of writing, 2026-10-05).** `RESULTS_SEC.md` did not exist yet —
The optimization had been run and summarised but not written up. So the optimization numbers here come from
`results/sec/summary.json` (3 runs, tag `the intermediate build`) and from the raw bench files, exactly as
instructed. The final system had **two** bench runs and **no judge run**, so the final system appears below with token
numbers and answer text but no accuracy scores. That gap is flagged everywhere it matters — and is
now closed, per the update above.

Sources read in full:

| # | File | What it gave |
|:-:|---|---|
| 1 | `results/sec/summary.json` | all optimization numbers (study doc absent) |
| 2 | `docs/results/RESULTS_SEC.md` | the initial batch, the circularity finding |
| 3 | `docs/results/RESULTS_SEC.md` | baseline tuning, router ablation, `tg_live` fix |
| 4 | `docs/results/RESULTS_SEC.md` | §4.3 tokens by tier, §4.5 the eight an earlier implementation fixes |
| 5 | `docs/results/RESULTS_SEC.md` | defect-by-defect history |
| 6 | `pipeline3_sec/pipeline.py` | all 1,800 lines of Pipeline 3 |
| — | `pipeline2_basic_rag.py`, `metrics.py`, `config.py`, raw `results/sec/*.json` | definitions and re-measurement |

---

# Part A — The setup

## A1. Three pipelines, and why compare them

A **pipeline** here means: a program that takes one question in plain English and returns one answer
in plain English. All three use the same **generator** — the model that writes the final answer —
which is Google's `gemini-2.5-flash` at temperature 0 (temperature 0 means "be as deterministic as
you can"). The only thing that differs between the three is **what text the pipeline puts in front
of the model before asking it the question**.

That text is called the **context**. Everything in this project is about the context.

| | Name | What it feeds the model | Purpose in the comparison |
|---|---|---|---|
| **P1** | LLM-only | nothing but the question | the floor. Shows what the model already knows without help. |
| **P2** | Basic RAG | the top-*k* most similar document chunks, found by vector search | the baseline. The standard, widely-deployed approach. |
| **P3** | GraphRAG | evidence selected by traversing a knowledge graph | the proposal. The thing being argued for. |

**RAG** stands for *retrieval-augmented generation*: retrieve some documents, paste them into the
prompt, let the model read them and answer. **GraphRAG** means the retrieval step walks a graph of
facts instead of (or as well as) ranking documents by similarity.

**Why compare at all?** Three reasons, and each needs its own pipeline:

1. **P1 proves the corpus is necessary.** If LLM-only scored as well as the retrieval pipelines,
   the benchmark would be measuring the model's memory, not retrieval. It does not: on the
   combined 150 questions, P1 strict pass is **0.098** against P2's **0.251** and P3's **0.456**
   (`summary.json → sets.combined150`). The corpus is genuinely required. The dataset is
   deliberately synthetic and future-dated (filings dated 2026) precisely to defeat memorisation.
2. **P2 is the thing P3 has to beat.** Nobody is impressed by beating no baseline. If GraphRAG
   doesn't beat plain vector search, GraphRAG isn't worth its extra machinery.
3. **P3 is the claim.** Specifically: *the same or better answers using fewer tokens.*

## A2. The SEC corpus

**The numbers, verified by counting `datasets/sec/corpus.jsonl` directly:**

- **61,172 chunks** (line count of the file)
- **100 companies** (distinct `ticker` values)
- Filings split: **10-K 33,565 chunks · DEF 14A 26,506 · 8-K 1,101**

### What is a "chunk"?

An SEC filing is enormous — a 10-K can run hundreds of pages. You cannot paste one into a prompt.
So the filings are cut into pieces called **chunks**. Here a chunk is roughly 2,000 characters of
text, which works out to about **350 tokens on average** (measured: mean 352.3 tokens over the
first 3,000 chunks, using the `cl100k_base` tokenizer, at the 2,000-character cap the pipelines
actually apply).

A **token** is the unit a language model bills and thinks in — roughly ¾ of an English word. "Token
count" is the currency of this entire project.

Each chunk carries an ID that doubles as a citation, for example:

```
ABT/DEF14A_2026-03-13#item5-c123
└┬┘ └──┬──┘ └───┬────┘ └──┬──┘ └┬┘
 │     │        │         │     └─ chunk 123 of that section
 │     │        │         └─────── section: Item 5 of the filing
 │     │        └───────────────── filing date
 │     └────────────────────────── form type
 └──────────────────────────────── ticker (Abbott Laboratories)
```

This ID shape matters later: the optimization's temporal feature reads the **date** and the **item code** straight
out of the ID, which is why that evidence is trustworthy (see D1 and E6).

### The three filing types

| Form | Full name | What it contains | Why this project cares |
|---|---|---|---|
| **10-K** | Annual report | The full yearly disclosure: business description, risk factors, management discussion, and the audited financial statements (revenue, net income, assets) | Every financial number question. 55% of the corpus. |
| **DEF 14A** | Definitive proxy statement | The document mailed to shareholders before the annual meeting: the board of directors, executive pay, and **who the auditor is** | Auditor, director and board questions. 43% of the corpus. |
| **8-K** | Current report | A one-off announcement of a material event, filed within days of it happening: a CEO leaving, an acquisition, a buyback, a restructuring. Each is tagged by the filer with an **item code** (e.g. Item 5.02 = officer departure/election) | Every "when did X happen" and event question. Only 1,101 chunks, but they carry all the dates. |

## A3. The knowledge graph

A **knowledge graph** is a set of facts written as triples: *subject — relation → object*. Drawn
out, the subjects and objects are **vertices** (dots) and the relations are **edges** (arrows).

Here is one real edge, read directly from `pipeline3_sec/index/edges_prov.json`:

```json
{
  "source":   "abbott laboratories (abt)",
  "relation": "audited_by",
  "target":   "ernst & young llp",
  "evidence": "In October 2025, the Audit Committee appointed Ernst & Young LLP to act as
               auditors for 2026.",
  "chunk_id": "ABT/DEF14A_2026-03-13#item5-c123",
  "doc_id":   "ABT/DEF14A_2026-03-13"
}
```

**Read the `evidence` field again — it is the single most important field in this project.** Every
edge carries the one sentence from the filing that proves it, plus the chunk that sentence came
from. That pairing is what makes the whole token saving possible (see B3).

**Vertices** are companies (`abbott laboratories (abt)`), audit firms (`ernst & young llp`), people
(directors, CEOs), sectors, suppliers, risk topics, and events.

**Edges** — the relation types Pipeline 3 actually uses, with their row counts in the local index:

| Relation | Means | Local rows |
|---|---|:-:|
| `audited_by` | company → its external audit firm | 100 |
| `in_sector` | company → its GICS sector | **0** (see note) |
| `uses_supplier` | company → a named supplier/foundry | 178 |
| `has_director` | company → a board member | 14,496 |
| `has_ceo` / `is_ceo_of` | company → its chief executive | 547 |
| `discusses_risk` | company → a risk topic it flags | 50,420 |
| `competitor_of` | company → a named rival | 110 |
| `reports_event` | company → an 8-K event | 10,643 |
| `filed` | company → a filing | 193 |

Total: **126,504 provenance edges**, across many relation types (most of them noisy open-extraction
leftovers like `related_to`, which the pipeline ignores).

> **The `in_sector` = 0 oddity is a real, documented bug, now fixed.** Sector is a deterministic
> fact (ticker → sector, straight from the manifest, no AI involved), so it was never written into
> the local evidence index — it lives only in TigerGraph and in chunk metadata. The code read
> `_rel_index["in_sector"]`, got an empty list, and **silently dropped sector from the facts it
> showed the model**. The model then filled the hole with a plausible guess *and attached a
> citation to it* — Costco labelled "Retail" instead of "Consumer Staples". This was an earlier implementation change #4,
> fixed by reading sector from chunk metadata instead (`_sector_of`, pipeline.py:654). It is worth
> dwelling on because it is the nastiest failure mode in the whole report: **a wrong answer that
> looks sourced.**

### Where the graph lives — two places, on purpose

1. **TigerGraph Savanna** — a hosted cloud graph database. Pipeline 3 connects to it and runs
   pre-installed queries written in GSQL (TigerGraph's query language): `agg_audited_by`,
   `typed_neighbors`, `vector_search`, `aggregate_by_relation`. Savanna is *both* the graph store
   *and* the vector store for P3 — it holds an embedding on every chunk.
2. **A local provenance index** — the `pipeline3_sec/index/*.json` files, including
   `edges_prov.json` above. This is where the **evidence sentences** live.

The division of labour is deliberate and is stated plainly in the code (pipeline.py:1411–1420):
**the live graph says *what* the fact is; the local index supplies the *sentence that proves it*.**

This hybrid is why one metric (`tg_live`) is below 1.0 — see E7.

## A4. What this project scores: quality per token

this project's scoring criterion is **quality per token**. Not "best answer". Not "cheapest answer".
Quality divided by cost.

That single decision changes the whole shape of the engineering problem. Under ordinary benchmark
rules you would add context until accuracy stops improving. Under *quality per token* you pay for
every word you add, so the question becomes: **what is the smallest piece of text that answers this
question?**

This is why token count is a first-class metric here rather than a footnote, and why the summary
file reports `quality_per_1k_P2` and `quality_per_1k_P3` as headline numbers alongside accuracy.

**On the combined 150, the optimization (`summary.json`):**

| | P2 Basic RAG | P3 GraphRAG |
|---|:-:|:-:|
| graded quality | 0.3404 | 0.4971 |
| avg tokens | 4,186.3 | 980.7 |
| **quality per 1k tokens** | **0.0600** | **0.4646** |

Efficiency ratio **7.75×**. That is the number this project's criterion rewards.

## A5. Every metric, defined plainly

The judge is OpenAI's `gpt-4o-mini` — a **different model family** from the Gemini generator, so
the grader is not grading its own work. It sees the question, the reference answer and the system
answer, and is **blind to which pipeline produced it** (`build/eval/evaluate.py:52–60`).

### Accuracy metrics

**Strict pass** — a single PASS or FAIL from the judge. The exact instruction is: *"PASS only if the
system answer is COMPLETE, factually correct, and fully consistent with the reference (all required
companies/figures/names present; for list questions, the set must match with no major omissions or
wrong additions). Otherwise FAIL."* Reported as a rate: 0.456 means 45.6% of answers passed.

It is deliberately harsh. Get 30 of 32 companies right and you FAIL.

**Graded quality** — the same judge, four levels, mapped to numbers
(`evaluate.py:98`):

| Grade | Means | Score |
|---|---|:-:|
| FULLY | equivalent to the reference | 1.00 |
| MOSTLY | core correct, minor omission | 0.66 |
| PARTIAL | some correct, significant gaps or errors | 0.33 |
| INCORRECT | wrong, unsupported, or refused | 0.00 |

Graded quality moves in smaller steps than strict pass, which makes it **the more sensitive
signal** when sample sizes are small. RESULTS_SEC.md relies on exactly this property.

**BERTScore F1** — a *wording-similarity* metric, not a correctness metric. It embeds the answer
and the reference with `roberta-large` and compares them (`rescale_with_baseline=True`, pinned per
the problem statement). It can go **negative** after rescaling — SQ001 scores **−0.0906** while
also being a strict PASS, because answering "AAPL, ABBV, ABT…" against a reference reading "Apple
Inc., AbbVie, Abbott Laboratories…" is factually perfect and lexically unlike it.

> Treat BERTScore as a **formatting signal**. The final report says so directly (§9): low BERTScore
> here is "a verbosity/format artifact (`$215,938 million for FY2026` vs `$215,938M`), not an
> accuracy signal."

**Citation quality** — a separate judge call. The judge is shown the question, the answer, and
*the evidence passages the pipeline actually supplied*, then asked: on 0–100, does this evidence
support this answer, and are the claims traceable to it? Divided by 100.

Note what this measures: not "did it cite", but "was what it cited actually enough". A pipeline that
supplies one sentence and answers from it can score higher than one that supplies ten chunks.

**Set-F1** — added for the extra100 set because strict pass throws away partial credit on list
questions. It resolves the answer text and the gold answer to canonical company names against
`manifest_clean.json`, then computes precision, recall and F1 over the two sets.

Scope is limited and stated in `build/eval/set_f1.py`: it is computed only for routes whose gold
is a company set — aggregation (20), bridge (20), compare (20, degenerate: one company). Lookup
(dollar figures) and temporal (dates) are **excluded by construction**, so 60 of 100 questions are
set-scored.

Two warnings travel with it, both from RESULTS_SEC.md: it is **length-sensitive** (a terse 2-company
answer scores 0.585 against a 3–10 member gold and collapses to 0.095 against a 12–32 member one),
and on `compare` it is **not a quality metric at all** — both candidate companies are named in the
question, so P1 with no retrieval scores 1.000 on `compare_revenue` while its strict pass is 0.00.

### Token metrics

This is where precision matters most. From `metrics.py`:

| Metric | Exactly what it is |
|---|---|
| **`context_tokens`** | just the retrieved evidence you pasted into the prompt. This is the number the optimisation work moves. |
| **`prompt_tokens`** | the **whole** input the API billed: system prompt + question + context. From the provider's own usage report, not an estimate (`token_source: "actual"`). |
| **`output_tokens`** | the answer the model generated. |
| **`total_inference_tokens`** | `prompt_tokens + output_tokens + optimizer_tokens`. **The headline number.** Everything spent to answer one question. |
| **`optimizer_tokens`** | tokens spent on *any* model call made to decide retrieval — a routing LLM, an LLM reranker, a compression model. |
| **`tg_queries`** | live GSQL queries this one question actually issued to Savanna. A counter reset at the top of `run_single()`. |
| **`tg_live`** | derived: `tg_queries > 0 and not GRAPH_OFF`. "Did the live cloud graph answer *this* question." |

> **`optimizer_tokens` is the fairness guarantee of the whole comparison, and it is 0 everywhere.**
> Verified: `summary.json → invariants.optimizer_tokens_all_zero: true`.
>
> Here is why it matters. Imagine P3 used a small LLM to pick its route. That LLM would spend real
> tokens, and if those were left out of the accounting, P3 could claim a saving it had not made —
> it would simply have moved the spend somewhere the metric wasn't looking. Keeping
> `optimizer_tokens` at exactly 0 means **nothing is hiding off-ledger**. The router is a regex. The
> row extractor is string matching. P2's compression scores sentences with the embedder *already
> loaded for retrieval*. No side-channel spend, in either pipeline.

Two integrity checks from the same `invariants` block: `max_output_tokens: 106` and
`rows_at_output_cap: 0`. The cap is `MAX_TOKENS = 512`, so no answer was truncated — a real hazard
when gold answers are 32-company lists.

### A5a. The one thing the headline token number leaves out

There is a sixth token type, and it is the most important caveat in this entire document.

**What a "thinking token" is.** Gemini 2.5 Flash is a *reasoning* model. Before writing its answer
it generates a hidden internal draft — working out, which you never see. Those are called
**thinking tokens** (or reasoning tokens). Google **bills them as output tokens**: you pay for
them at the output rate, exactly like the visible answer.

**Why they are not in the headline.** The API reports them in a separate field, and `metrics.py`
stores them separately as `thinking_tokens`. `total_inference_tokens` is `prompt + completion`
only — so **every token figure in every this project document understates what was actually paid.**

Measured on the combined 150, 3 runs:

| pipeline | headline (`total_inference_tokens`) | thinking | true billed | thinking share |
|---|:-:|:-:|:-:|:-:|
| P1 LLM-only | 281.6 | +279.9 | 561.5 | **49.8%** |
| P2 Basic RAG | 4,186.9 | +392.7 | 4,579.6 | 8.6% |
| **P3 GraphRAG** | **1,026.1** | **+269.4** | **1,295.5** | **20.8%** |

**So the real headline is 71.7%, not 75.5%.** That is the single most important correction in this
document, and it is the number to quote if someone asks what GraphRAG actually saved.

Three things worth understanding about that table:

1. **P1 is half thinking.** Its prompts are tiny — just the question — so the reasoning draft
   dominates its bill. A pipeline that retrieves nothing still is not free.
2. **P2 is least affected (8.6%)** because its 4,000-token contexts swamp everything else.
3. **Therefore the choice of accounting is not uniformly self-serving.** Excluding thinking makes
   P3's win over P2 look *bigger* (75.5% vs 71.7%), but makes P3 look *worse* against P1. It is
   not a thumb on the scale in one direction; it is simply incomplete, which is why both
   accountings are published.

**Does any conclusion change?** No. The token reduction shrinks from 75.5% to 71.7%. The
equal-budget result in Part E survives too: P3's true 1,295.5 against the compressed baseline's
true 1,852.1 is still **30.1% below budget at 2.5× the accuracy**. The direction, the mechanism
and the margin all hold; only the magnitude moves, and it moves by about four points.

> **Why this was easy to miss, and worth saying plainly.** `RESULTS_OLYMPICS.md` disclosed
> thinking tokens from the start; the SEC study documents did not, until this was caught in a
> file-by-file audit on 2026-10-11. Two documents in one submission were accounting for cost
> differently. The fix was to disclose both everywhere rather than to quietly pick the flattering
> basis — and `build/analysis/verify_claims.py` now asserts the exclusion is real, so the claim
> cannot rot.

---

# Part B — How Pipeline 3 works, end to end

Pipeline 3's whole structure is: **classify the question, then use a retrieval strategy built for
that class.** Five classes. One function, `run_single()` (pipeline.py:1515), with five branches.

## B1. The deterministic regex router

**What it is.** A pure function from question string to one of five labels
(`route()`, pipeline.py:280):

```python
def route(question: str) -> str:
    q = question.lower()
    if BINARY_COMPARE_RE.search(q) and BINARY_OR_RE.search(q) and COMPARE_METRIC_RE.search(q):
        return "compare"
    if BRIDGE_SHARED_RE.search(q) and BRIDGE_REL_RE.search(q):
        return "bridge"
    if AGG_RE.search(q):              return "aggregation"
    if TEMPORAL_RE.search(q) or TEMPORAL_EXT_RE.search(q): return "temporal"
    if COMPARE_RE.search(q):          return "compare"
    if BRIDGE_RE.search(question):    return "bridge"
    return "lookup"
```

A **regex** (regular expression) is a text pattern. `AGG_RE` matches things like *"which … companies"*,
*"list all … companies"*, *"how many … audit"*. No model is involved. No network call. It is a
handful of string comparisons.

**Why zero tokens, and why that is the point.** Because no model runs, `optimizer_tokens` stays 0.
If the router were an LLM classifier it would spend maybe 100–200 tokens per question, and P3's
claimed saving would have to absorb that. Concretely, P3 averages **980.7** tokens on the combined
150; a 200-token LLM router would be **20% of the entire budget**. The token comparison with P2
stays honest only because the router is free.

Note the symmetry: **P2 is allowed to borrow the same router.** `P2_ADAPTIVE_K=1` makes Basic RAG
call P3's `route()` and pick its own per-question `k` from it (`pipeline2_basic_rag.py:238–241`).
Adaptive budgeting is explicitly *not* treated as a graph-exclusive trick. More on this in E3.

**Order is load-bearing.** Three of the ordering decisions are themselves documented fixes:

- `AGG_RE` **before** `TEMPORAL_RE` — an earlier implementation change #6, worth **+0.17 on Tier D**. "Which companies
  filed an 8-K reporting a CEO transition" matches both patterns, but it needs the enumerate-across-
  100-documents branch, not the single-vector-call branch.
- `BINARY_COMPARE_RE` **before** `AGG_RE` — the optimization. Without it, "Which **company** reported higher net
  income, X or Y?" is claimed by the company-scan pattern. This was the defect that misrouted all
  20 compare questions (D1, defect 4).
- `BRIDGE_SHARED_RE` **before** `AGG_RE` — the optimization. "Which **other companies** share the same auditor as
  Accenture?" would otherwise be claimed as a plain enumeration.

## B2. The five routes

### Route 1 — `aggregation` (the one that always worked)

**Question shape:** scan many companies. *"Which companies are audited by Ernst & Young LLP?"*,
*"How many does each Big Four firm audit?"*, *"Which Health-Care companies audited by Deloitte
discuss obesity risk?"*

**What the branch does** (pipeline.py:1522–1600):

1. `_detect_aggregation_constraints()` maps the question to **every** matching
   `(relation, target)` pair — it deliberately does not stop at the first.
2. Each constraint runs its own graph aggregation via `_tg_aggregate()` — typically a live GSQL
   call to `agg_audited_by`, `agg_in_sector`, etc.
3. The resulting company sets are **intersected**, normalised to tickers first (because one query
   returns `cisco (csco)` and another returns `CSCO`, and those would never intersect as strings).
4. Context is built from the **provenance edges** of the surviving companies — one evidence
   sentence each — plus a synthetic citable line carrying the company list itself.

Special cases it handles:

- `__groupby_auditor__` — "how many does **each** Big Four firm audit" is a *group-by*, not an
  intersection. Treating it as an intersection gives the empty set, because no company has two
  auditors. An earlier implementation change #7; measured on EQ44: **4,137 → 657 tokens** and a real four-firm breakdown.
- `__section__` — a corpus scan over SEC item codes (e.g. `item5.02`), used for "which companies
  filed an 8-K reporting an officer transition".
- `has_director` / `has_ceo` — matched by **surname** across the local index, because the same
  person appears as "indra nooyi", "indra k. nooyi", "mrs. nooyi" and exact-match GSQL misses the
  variants.
- `competitor_of` — corroborated against the corpus rather than trusted from extraction, because
  the extracted edges are noisy.

**Worked example (SQ001, verified in `bench_combined150_the intermediate build_run1.json`):**

> **Q:** "Which companies are audited by Ernst & Young LLP?"
> **Route:** aggregation · **context: 762 tokens** · total: 1,230 · 1 live GSQL query
> **Answer:** `AAPL, ABBV, ABT, AMD, AMGN, AMZN, COF, COP, CRM, CVS, DHR, FDX, GILD, GM, GOOGL,
> INTC, INTU, KO, LLY, LMT, MCD, META, NFLX, ORCL, PLTR, SPG, T, TGT, TXN, USB, VZ, WMT`
> — 32 companies, strict PASS. **P2 on the same question: 3,406 tokens.**

### Route 2 — `lookup` (the fallback, and the biggest the optimization target)

**Question shape:** one fact about one named company. *"Who audits Altria?"*, *"Coca-Cola's net
operating revenues for FY2025?"* It is also the **default** — anything no other pattern claims
lands here.

**What the branch does in the optimization** (pipeline.py:1649 onward), in strict priority order:

1. **Curated-edge path.** If the question asks about a relation the graph holds *and* a company
   resolves, ask the live graph for that edge's targets (`_graph_curated_targets`), then cite each
   target's evidence sentence from the local index. Feed **only that**.
2. **Metric-row path.** If the question names a financial metric, extract the **single statement
   row** that answers it and feed only that row plus its column header.
3. **Timeline path** (temporal only, see Route 5).
4. **Unresolved fallback.** Ordinary vector retrieval, `k=8`. Unchanged from the initial implementation.

**Worked example (SQ061):**

> **Q:** "Who audits Altria?"
> **the initial implementation:** context **2,939 tokens** → 3,543 total
> **the optimization:** context **68 tokens** → 358 total. Answer: "PricewaterhouseCoopers LLP" (gold match).
> **P2 on the same question: 2,508 tokens.**

That is the whole thesis in one row — 68 tokens instead of 2,939, same correct answer.

### Route 3 — `compare` (two numbers, nothing else)

**Question shape:** binary comparison on a metric. *"Which company reported higher net income,
Qualcomm or Salesforce?"*

**What the branch does** (pipeline.py:1626–1648): resolve the two companies, resolve the metric to
candidate row labels, then call `_metric_block()` once per company to extract that company's
statement row — **ticker-scoped**, so one company's figures can never be attributed to the other.
If fewer than two companies yield a row, it falls back to the initial implementation's chunk retrieval rather than answer a
comparison with one side missing.

**Worked example (SQ041):**

> **Q:** "Which company reported higher net income, Qualcomm or Salesforce?"
> **the initial implementation:** route *aggregation* (misrouted!) · context **3,779** → 4,515 total
> **the optimization:** route *compare* · context **135** → 448 total
> **the optimization answer:** "Salesforce $7,457 million / Qualcomm $7,232" · **gold:** "Salesforce — $7,457
> million versus $5,541 million"

The winner is right and Salesforce's figure is exact. Qualcomm's is not — a row-selection
imprecision, which is exactly the known limit of the extractor.

### Route 4 — `bridge` (two hops; still the weak one)

**Question shape:** find the entity related to X, then tell me something about *it*. *"The company
that audits General Dynamics — which others does it audit?"*

**What the branch does** (pipeline.py:1601–1625):

1. Vector-search the **bridge clause only** (text before the dash or comma) to find the linking doc.
2. Identify the company: from graph seeds via `typed_neighbors`, **and** from the companies that own
   those chunks (which recovers the hop when the person isn't a graph node at all).
3. Pull that company's curated facts — the answer usually lives there.
4. The optimization change: feed those resolved edges as **one evidence sentence each**, with a reduced vector
   backstop, instead of ten whole chunks.

> **Read the code comment on this branch carefully** (pipeline.py:1612–1619): *"the optimization changes ONLY
> step 4's context assembly… the hop-resolution logic above is deliberately untouched — the
> documented second-hop defect is out of scope for this pass, so this is a token change and must
> not be read as a bridge-accuracy fix."*
>
> The measurements agree exactly. From `summary.json → by_gold_route.bridge`: tokens
> **5,015.2 → 1,787.0** (−64%), strict **0.05 → 0.05** (unchanged). The branch got cheaper. It did
> not get better.

**Worked example (SQ021):**

> **Q:** "The company that audits General Dynamics — which others does it audit?"
> **the initial implementation:** context 3,892 → 4,644 total. **the optimization:** context 951 → 1,323 total.
> **Both answers:** *"The answer is not supported by the provided context passages and
> knowledge-graph facts."*

**Why it fails** (RESULTS_SEC.md, and the diagnosis is specific): the branch resolves
company → auditor, and then **never issues the auditor → companies query**. The data is right
there — the aggregation route answers the equivalent question *"Which companies are audited by
KPMG LLP?"* at set-F1 **1.000**. Hop 1 fires, hop 2 does not.

### Route 5 — `temporal` (ordering events)

**Question shape:** put events in chronological order; when did X happen.

**What the branch does in the optimization** (pipeline.py:1690–1711): `_company_8k_timeline()` builds a dated
timeline from the company's 8-K chunks — one line per filing, sorted by date, each line carrying
the filing date, the SEC item code, the item's official title, and a ~220-character snippet.
Replaces ten full chunks with about six dated lines.

**Worked example (SQ081):**

> **Q:** "Put these two Adobe Inc. events in chronological order: the officer appointment and the
> officer departure."
> **the initial implementation:** context 3,042 → 3,569 total. **the optimization:** context **493** → 895 total.
> **the optimization answer:** "Officer departure: June 8, 2026. The context does not provide a date for an
> officer appointment."
> **Gold:** "officer appointment (June 11, 2026) first, then officer departure (June 15, 2026)"

Cheaper by 84%, and wrong. `by_gold_route.temporal`: tokens 4,552.2 → 619.8 (−86%), strict
**0.05 → 0.00**. But see E5 — this gold is derived from a file the project itself distrusts, so this
particular score is weak evidence in either direction.

## B3. The central idea — context optimization

Everything above is scaffolding for one mechanism. Here it is.

### The two context builders

Pipeline 3 has exactly two functions that turn retrieval results into prompt text.

**`_context_from_chunks()`** (pipeline.py:1475) — feed whole chunks.

```python
def _context_from_chunks(chunk_ids):
    ids = _dedup(chunk_ids)[:MAX_CTX_CHUNKS]        # up to 10
    for cid in ids:
        txt = _chunk_text[cid][:MAX_CTX_CHARS]      # 2,000 chars each
        blocks.append(f"[{cid}]\n{txt}")
```

Up to 10 items × 2,000 characters ≈ **up to ~4,000 tokens of context.**

**`_context_from_prov()`** (pipeline.py:1485) — feed one evidence sentence per graph edge.

```python
def _context_from_prov(prov, extra_chunks=None):
    for e in prov[:MAX_CTX_CHUNKS]:
        ev = e["evidence"].strip()
        cid = e["chunk_id"] or e["doc_id"]
        blocks.append(f"[{cid}] {ev}")              # ONE SENTENCE
```

The code's own comment: *"This is the token win — a sentence instead of a chunk."*

### Why sentence substitution is the entire token win

Measured on the Abbott `audited_by` edge, with the real tokenizer:

| What you feed | Tokens |
|---|:-:|
| `[ABT/DEF14A_2026-03-13#item5-c123] In October 2025, the Audit Committee appointed Ernst & Young LLP to act as auditors for 2026.` | **44** |
| `[ABT/DEF14A_2026-03-13#item5-c123]` + the whole 2,000-character chunk | **419** |

**9.5× cheaper, identical citation, and the answer is fully present in both.** The 375 extra tokens
in the chunk version are the rest of the proxy statement's Item 5 — audit-fee tables, committee
charters, boilerplate. The model does not need them to answer "who audits Abbott", and it has to
read past them to find the sentence that does.

Multiply by ten chunks and this is the entire story: ~440 tokens of evidence versus ~4,000 tokens of
documents.

And note the second-order benefit, visible in SQ041: feeding less is not just cheaper, it is often
*more accurate*, because the model no longer has to locate one row inside several pages of tables.

### The initial implementation defect: a mechanism that worked, applied almost nowhere

Here is the whole the optimization story in one sentence: **sentence substitution was never the problem — its
coverage was.** In the initial implementation, `_context_from_prov` was called from exactly two places, **both inside the
aggregation branch**. Every other route called `_context_from_chunks`.

The code comment at pipeline.py:928–935 states the measurement:

> *provenance substitution fired on 35 of 150 questions and cost 652 tokens against Basic RAG's
> 4,061 (−84%) at 0.83 strict vs 0.14; the other 115 questions paid 4,406 tokens (+4% MORE than the
> baseline) for 0.31 strict.*

> ### ✅ Verified: these figures reproduce exactly
>
> An earlier revision of this document claimed these numbers could not be reproduced and offered
> a table of its own (43 / 683.9 / 0.674 and 107 / 4,673.4 / +11.0%). **That was my error and it
> is retracted.** Two mistakes caused it: I classified "fired" by scanning `retrieved_ids`
> instead of `citations`, and I partly audited against `final` artifacts when this measurement
> is of **the initial implementation**.
>
> Re-measured correctly over `bench_combined150_initial_run{1,2,3}.json`, classifying "fired" as
> *the row carries a `graph_aggregation:` entry in `citations`*, every figure in the code comment
> and in The optimization §1 reproduces:
>
> | group | n | P2 tokens | P3 tokens | P3 ctx | P3 strict | P2 strict | saving |
> |---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
> | provenance fired | **35** | 4,060.7 | **651.8** | 303.1 | **0.829** | 0.143 | **−83.9%** |
> | full chunks | **115** | 4,225.6 | **4,405.6** | 3,540.1 | 0.310 | 0.281 | **+4.3% MORE** |
>
> Why the definition matters, since it is the whole substance of my mistake:
>
> | cut | n | mean ctx tokens | strict |
> |---|:-:|:-:|:-:|
> | `citations` carries `graph_aggregation:` — **what the doc uses** | 35 | 303.1 | 0.829 |
> | `retrieved_ids` carries it — what I wrongly used | 43 | 334.0 | 0.674 |
> | `context_tokens < 600` — a different 35 (only 29 shared) | 35 | 238.5 | 0.686 |
>
> `retrieved_ids` is a superset of `citations`: it records everything the builder put into the
> context, including evidence the model then declined to cite. Either is a defensible proxy, but
> they are not interchangeable, and the doc's three figures all come from the **same consistent
> cut** — my parenthetical claiming they did not was also wrong.
>
> All of this is now checked automatically by `build/analysis/verify_claims.py`, which recomputes
> the table above and fails if it drifts.

Either way, the diagnosis is the same and it is structural. From
`summary.json → by_shape`, the initial implementation's context tokens by question shape:

| shape | the initial implementation context tokens | builder used |
|---|:-:|---|
| `agg_sector` | 72 | prov (sentences) |
| `agg_sector_auditor` | 274 | prov |
| `tierC` (aggregation) | 325 | prov |
| `agg_auditor` | 720 | prov |
| `agg_supplier` | 1,422 | prov |
| `lookup_auditor` | **2,705** | chunks |
| `lookup_revenue` | **3,632** | chunks |
| `temporal_order` | **3,769** | chunks |
| `bridge_shared_auditor` | **4,067** | chunks |
| `compare_net_income` | **4,130** | chunks |
| `bridge_shared_director` | **4,563** | chunks |

The correlation is perfect. Every shape under ~1,500 tokens is a sentence-substituted aggregation
shape. Every shape above ~2,700 is a full-chunk route. **There is no overlap.**

And `lookup` was worse than merely un-optimised: it **appended**. The initial implementation line was
`_context_from_chunks(fact_chunks + ids)` — so a question whose answer is one `audited_by` edge got
the graph fact *and* all eight retrieved chunks. It paid 2,705 context tokens to merely **tie** P2
(0.90 vs 0.90 on `lookup_auditor`), and on `lookup_revenue` it paid 3,632 and **lost** (0.60 vs
0.80).

### What the optimization added

Four mechanisms, all deterministic string work, all with `optimizer_tokens` = 0.

**1. The metric-row extractor** (`_metric_block`, `_extract_metric_rows`, `_row_score` —
pipeline.py:933–1240).

An income statement row is *a label followed by figures*: `Total revenues 88,309 81,400 78,558`.
The extractor finds that one line and feeds it instead of the pages around it.

- `_METRIC_LEX` maps a metric the question names ("net income", "r&d", "total revenue") to the row
  labels filings actually use, most specific first.
- `_question_metric_phrase()` adds **the question's own wording** as the highest-specificity label.
  This exists because a fixed lexicon cannot be specific enough: ExxonMobil's consolidated line is
  *"Total revenues and other income 332,238"* while its **segment** table says *"Total revenues
  111,193"* — against the lexicon label "total revenues" the segment row matches exactly and the
  consolidated row only by prefix, so the **wrong figure won**. Coca-Cola labels its line "Net
  Operating Revenues", which the lexicon didn't have at all.
- `_row_score()` requires the label to be a **prefix of the text before the first figure**, not
  merely present. This is what rejects prose: NVDA's sentence *"For fiscal year 2025, sales to one
  direct customer represented 12% of total revenue…"* contains "total revenue" and two numbers, and
  a containment test scored it **above** the real `Revenue $ 215,938` row.
- It also carries a dollar-magnitude test, because Lowe's states net sales twice — once as
  `Net sales $ 86,286 100.00%` and once in a percentage-of-sales table as
  `Net sales 100.00% 100.00% 100.00%`. Same head, tied exactly; only the figures tell them apart.
- And it pulls the **column header** along, because `Total revenues 88,309 81,400 78,558` is
  ambiguous without the `2025 2024 2023` line above it. If the table straddles a chunk boundary it
  addresses the predecessor chunk deterministically (ids are `<doc>#<section>-c<N>`).

Measured effect, `by_shape`: `lookup_revenue` context **3,631.6 → 69.3 tokens**;
`compare_revenue` **4,155.2 → 138.9**; `compare_net_income` **4,129.9 → 129.9**.

**2. The curated-edge evidence path** (`_RELATION_PATTERNS`, `_curated_prov`,
`_graph_curated_targets` — pipeline.py:1241–1300, 1411).

Replaces the initial implementation's noun-list gate with patterns that match a relation **in any of its ordinary English
forms** — `audit|audits|audited|auditor|auditors|accounting firm|…` — and then feeds the resolved
edges through `_context_from_prov`. Measured: `lookup_auditor` context **2,704.6 → 320.7 tokens**.

**3. The SEC item-code 8-K timeline** (`_company_8k_timeline`, `_SEC_ITEM_TITLES` —
pipeline.py:1301–1409).

Builds the timeline from **the chunk id**: `AMGN/8-K_2026-05-19#item5.02-c0` gives the ticker, the
filing date, and the filer-assigned item code. Both the ordering evidence and the event label are
therefore deterministic metadata, not AI output. `_ITEM_NOISE` drops item 9.01 and bare
`header`/`body` chunks, which appear on nearly every filing and would crowd out real events.
Measured: `temporal_order` context **3,769.0 → 288.2 tokens**.

**4. Bridge evidence substitution** — described in Route 4 above. Context
**4,067.0 → 1,173.8** on `bridge_shared_auditor`. Token change only; accuracy flat by design.

---

# Part C — The history, and why each number changed

| Version | What changed | Headline token reduction | P3 strict | Measured on |
|---|---|:-:|:-:|---|
| an earlier configuration FINAL | — | 49.6% | 0.447 | held-out 50 |
| an earlier configuration | — | 48.5% | 0.487 | held-out 50 |
| **an earlier configuration** | frozen config behind the README | **50.2%** | 0.487 | held-out 50, **spliced batch** |
| **an earlier implementation** | eight P3 defect fixes | **43.0%** | **0.553** | held-out 50, P3 re-run / P2 reused |
| **the initial implementation** | P2 tuned; router ablation un-confounded; `tg_live` fixed | **34.2%** | 0.533 | held-out 50, **one batch** |
| **initial** | +100 route-balanced questions | **15.7%** (combined) / **6.7%** (extra100) | 0.431 / 0.370 | combined 150 |
| **the optimization** | coverage of context optimization + 4 defect fixes | **76.6%** (combined) | 0.456 / 0.443 | combined 150 |
| the final system | compound-question guard | **75.5%** (combined, 2 runs) | *not yet judged* | combined 150 |

an earlier configuration–the initial implementation are from `initial_summary.json → trend_heldout_strict_and_tokens`; initial from RESULTS_SEC.md; the optimization from
`summary.json`; the final system computed by me from the two the final system bench files.

## C1. An earlier configuration → an earlier implementation: fixed eight defects, accuracy up, headline down

An earlier implementation touched **Pipeline 3 only**. Eight defects, five of which were producing **silently wrong
answers that still carried citations** — the failure mode most likely to survive review.

The eight (RESULTS_SEC.md):

| # | Change | The defect it fixed |
|:-:|---|---|
| 1 | Generic-role stoplist on entity seeds | Job titles ("Chief Executive Officer") matched as graph entity seeds, because `entity_types.json` misclassifies them as PERSON/ORG — so a type filter alone doesn't catch them |
| 2 | Possessive + article resolution | `_company_key_for` never stripped `'s`, so **"Home Depot's" / "Amazon's" never resolved to a company at all** |
| 3 | Entity-linked curated facts on `lookup` | Auditor/sector/CEO questions depended on vector search happening to surface one sentence out of 61k chunks |
| 4 | **Sector backbone fix** | `in_sector` has **zero local rows**, so sector silently vanished and the model emitted wrong GICS labels **with citations attached** (Costco → "Retail") |
| 5 | Item 5.02 `__section__` scan | "Which companies filed an 8-K reporting a CEO transition" is a corpus-wide *scan* being answered by one k=12 vector call, which structurally cannot enumerate across ~100 documents |
| 6 | Router priority: `AGG_RE` before `TEMPORAL_RE` | Company-scan questions containing "transition" were siphoned to `temporal` before aggregation was considered — **worth Tier D +0.17** |
| 7 | `__groupby_auditor__` | "how many does **each** Big Four firm audit" was treated as an *intersection* — structurally empty, since no company has two auditors. EQ44: **4,137 → 657 tokens** |
| 8 | `lookup` k 6→8 | k=6 starved some single-fact lookups into flat refusals — at about **+500 tokens per lookup question** |

Plus one **deliberate revert**, which is important for E6: an event-type group-scan built on
`events.json`'s `type` field. The naive version returned 84 of 100 companies; a tightened version
returned 32 that excluded **both** correct answers. Replaced with change #5 (SEC item codes).

**Why the number moved:** strict pass **0.487 → 0.553**, tokens **2,359 → 2,701 (+14.5%)**, so
reduction vs P2 fell **50.2% → 43.0%**. The honest reading, in An earlier implementation §3.2's own words: *"an earlier implementation did not
make the pipeline more efficient. It made it more accurate at proportionally more tokens."*
Accuracy-per-1k-tokens was **flat** (0.2063 → 0.2049) — a different point on the *same* efficiency
frontier.

Most of the extra tokens landed on Tier A, from change #8. Tier C was the **only** tier where
accuracy rose *and* tokens fell (786 → 673, −14%).

## C2. An earlier implementation → the initial implementation: tuned the baseline, and the headline fell again

An earlier implementation changed P3 and reused P1/P2 numbers from an earlier batch. The initial implementation re-ran **all three pipelines in one
batch** and fixed three methodological problems:

1. **The baseline was optimised.** Six P2 arms swept on the 24-question dev set, best frozen before
   held-out was touched. Winner: `adaptive-k` — "a free 15% token cut", identical strict pass
   (0.250) at 741 fewer tokens. Chosen *because it is the strongest honest baseline*.
2. **The router ablation was un-confounded.** See C2a.
3. **`tg_live` was corrected.** See E7.

**Why the number moved: 43.0% → 34.2%.** Two causes, both making the claim smaller:

- P2 now spends fewer tokens (adaptive-k), so there is less to save against. 43.0% was measured
  against an **untuned** baseline.
- an earlier configuration's 50.2% was additionally an **undisclosed splice** — P3 rows from the `v51final` batch paired
  with P2 rows from the `p2cosine_final` batch, and the P2 artifacts behind it were never shipped.
  v7final is one coherent batch.

**an earlier implementation 0.553 → the initial implementation 0.533 is not a regression.** P3's answer logic is byte-identical between them; the
only the initial implementation P3 change was instrumentation. 0.020 sits inside the ±0.019–0.025 three-run spread. The initial implementation's
number is the one to quote because all three pipelines ran together.

### C2a. The router ablation, as an example of what "un-confounded" means

An **ablation** is: turn one thing off, see what breaks. The old arm, `ROUTER_OFF=1`, forced the
label `"default"` — which **no graph branch matches**. So it disabled routing *and* graph traversal
*and* compression, and reported all three as one number.

The fix (`P3_ROUTE_FORCE`) keeps every branch fully live and corrupts **only the label** — either a
fixed route, or `md5(question)`-seeded random so the arm is reproducible.

| Arm | strict | avg tokens |
|---|:-:|:-:|
| router ON | **0.520** | **2,708** |
| router RANDOM | 0.360 | 3,855 |
| router FIXED = lookup | 0.360 | 3,889 |
| router OFF (old, confounded) | 0.280 | 4,698 |

| Component | Δ strict | Δ tokens |
|---|:-:|:-:|
| Correct route **labels** (RANDOM → ON) | **+0.160** | **−1,152** |
| Per-route **branches** existing (OFF → RANDOM) | +0.080 | −843 |
| *old confounded total* | *+0.240* | *−1,990* |

So the router's own contribution is **~1,150 tokens, not ~1,990**. The remaining 843 belongs to the
branch machinery — a separate claim that now has to stand on its own.

Note the diagnostic detail: random and fixed-lookup land on the **same** strict pass (0.360). What
matters is whether the label is *right*, not which wrong label is used — which is what a real
routing mechanism should look like.

## C3. The initial implementation → initial: the honest crash to 15.7%

Nothing in the pipeline changed. 100 new SEC questions (SQ001–SQ100) were added, with **gold route
labels, 20 per route**, and run in one 150-question batch with the official 50.

| view | n | token reduction | P3 strict | P2 strict |
|---|:-:|:-:|:-:|:-:|
| extra100 alone (route-balanced) | 100 | **6.7%** | 0.370 | 0.223 |
| official held-out 50, re-measured | 50 | **34.2%** | 0.553 | 0.300 |
| combined | 150 | **15.7%** | 0.431 | 0.249 |

**Why it moved, and this is the key insight of the whole project:** *the route mix changed, not the
pipeline.* Per-route costs matched the held-out set closely. But the official 50 is **40%
aggregation** and the extra100 is **20% aggregation** — and aggregation is the only route that saves
tokens. The weighted average fell purely from the mix.

| gold route | n | P2 tok | P3 tok | reduction | P3 strict | P2 strict |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| aggregation | 20 | 3,927 | **860** | **+78.1%** | **0.85** | 0.10 |
| lookup | 20 | 2,979 | 4,058 | **−36.2%** | 0.75 | **0.85** |
| compare | 20 | 5,067 | 5,232 | −3.4% | 0.15 | 0.10 |
| bridge | 20 | 4,319 | 5,015 | −16.1% | 0.05 | 0.05 |
| temporal | 20 | 4,846 | 4,552 | +6.0% | 0.05 | 0.02 |

the SEC study's own conclusion: *"A route-balanced set is a fairer test of the token thesis than the
official 50, and on it the thesis is much weaker."*

initial also produced the circularity finding (E5) and the 0.800 in-domain router control.

## C4. initial → the optimization: ~76% by extending the mechanism, not inventing one

The optimization applied `_context_from_prov`-style substitution to the four routes that never used it, and fixed
four defects (Part D).

`summary.json`, combined 150, 3 runs:

| | the initial implementation | the optimization |
|---|:-:|:-:|
| P3 tokens | 3,529.7 | **980.7** |
| P3 context tokens | 2,784.8 | **597.4** |
| token reduction vs P2 | 15.7% | **76.6%** |
| P3 strict | 0.431 | **0.456** |
| quality per 1k | ~0.134 | **0.465** |
| efficiency ratio vs P2 | 2.06× | **7.75×** |

Per route, tokens and strict:

| route | n | P2 tok | initial tok | the optimization tok | reduction | the initial implementation strict | the optimization strict |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| aggregation | 20 | 3,926 | 860 | 860 | 78.1% | 0.85 | 0.85 |
| **compare** | 20 | 5,067 | 5,232 | **484** | **90.4%** | 0.15 | **0.367** |
| **lookup** | 20 | 2,974 | 4,058 | **512** | **82.8%** | 0.75 | **0.95** |
| temporal | 20 | 4,847 | 4,552 | **620** | 87.2% | 0.05 | **0.00** |
| bridge | 20 | 4,322 | 5,015 | **1,787** | 58.7% | 0.05 | 0.05 |

**Lookup and compare got both cheaper and better.** Temporal got far cheaper and nominally worse.
Bridge got cheaper and stayed broken.

## C5. The optimization → the final system: one guard, four questions

The optimization introduced a rule — *"replace the chunks, don't append them"* — that **backfired on compound
questions**. The final system adds `_COMPOUND_RE` and `COMPOUND_K` to detect a question asking for more than one
fact and keep a reduced chunk budget for the clause the graph artifact doesn't answer. Full
treatment in D2.

**What the final system has and doesn't have.** Two bench runs (`bench_combined150_final_run{1,2}.json`), no
eval file. **So there are no the final system accuracy scores.** Deterministic numbers I computed myself:

| set | the optimization P3 tok | the final system P3 tok | the final system reduction |
|---|:-:|:-:|:-:|
| extra100 | 852.6 | 848.8 | 79.9% |
| held-out 50 | 1,236.9 | 1,384.8 | 66.3% |
| combined 150 | 980.7 | 1,027.5 | 75.5% |

The offline dry-run comparison (`dryrun_v8_offline.json` vs `dryrun_offline.json`) shows the final system
changes **19 of 150 rows**, of which four are the substantive ones (see D3).

## C6. Why 50.2% → 34.2% → 15.7% → 76% is not a contradiction

This is the question the numbers most invite, so here it is explicitly. Each figure is a **different
measurement**, and three of the four changes are methodological rather than engineering.

| Figure | Numerator (P3) | Denominator (P2) | Question set | Batch |
|---|---|---|---|---|
| **50.2%** | an earlier configuration P3 | **untuned** P2 | official held-out 50 (40% aggregation) | **spliced across two batches** |
| **43.0%** | an earlier implementation P3 (more accurate, +14.5% tokens) | **untuned** P2, reused from an older batch | same 50 | P3 re-run, P2 reused |
| **34.2%** | the initial implementation P3 (identical logic to an earlier implementation) | **tuned** P2 (adaptive-k, −15% tokens) | same 50 | **one coherent batch** |
| **15.7%** | the initial implementation P3, unchanged | tuned P2, unchanged | **combined 150** (27% aggregation) | one batch |
| **6.7%** | the initial implementation P3, unchanged | tuned P2, unchanged | **extra100** (20% aggregation) | one batch |
| **76.6%** | **the optimization P3** — a genuine engineering change | tuned P2, unchanged | combined 150 | one batch |

So:

- **50.2% → 43.0%** — P3 bought accuracy with tokens. The headline fell; strict pass rose 0.487 →
  0.553. Efficiency was flat.
- **43.0% → 34.2%** — the **denominator shrank**. A tuned baseline spends less, so there is less to
  save. Plus the splice was removed.
- **34.2% → 15.7% → 6.7%** — the **question set changed**. Same pipeline, same code. Aggregation
  went from 40% of the set to 27% to 20%, and aggregation is the only route that saved anything.
- **15.7% → 76.6%** — **the only change here that is actually engineering.** The mechanism that
  made aggregation cheap was extended to the other four routes.

**The rule that follows:** a token-reduction figure is meaningless without naming (a) the question
set and its route mix, (b) whether the baseline was tuned, and (c) whether both pipelines ran in the
same batch. RESULTS_SEC.md says it in one line: *"Any summary quoting a single token-reduction number
should say which set it is on."*

And the first three moves all went the same direction. The initial implementation §8: *"The headline token reduction fell
from 50.2% to 34.2% and that is the correct direction."*

---

# Part D — the optimization and the final system specifically

## D1. The four defects the optimization fixed

### Defect 1 — the gate knew the noun "auditor" but not the verb "audits"

**Root cause.** The initial implementation decided whether to inject graph facts using `_LOOKUP_FACT_KEYWORDS`, a list of
**nouns**:

```python
_LOOKUP_FACT_KEYWORDS = ("auditor", "audit firm", "accounting firm", "sector", "classified",
                         " ceo", "chief executive", "director", "competitor", "supplier", ...)
```

The question *"Who audits Altria?"* contains **"audits"** — a verb. Not in the list. So the gate
failed, and the `audited_by` edge that answers the question outright **was never consulted**. The
branch fell through to generic vector retrieval and fed eight whole chunks.

**All 10 `lookup_auditor` questions in extra100 hit this.** Measured cost
(`summary.json → by_shape.lookup_auditor`): P3 spent **3,256 tokens** against Basic RAG's
**2,493**, to merely **tie** it on strict pass.

**The fix** (`_RELATION_PATTERNS`, pipeline.py:1251): patterns keying on the relation in any
ordinary English form.

```python
("audited_by", re.compile(r"\baudit(s|ed|or|ors|ing)?\b|\baccounting firm\b|\baudit firm\b"
                          r"|\bindependent (registered )?public accounting\b", re.I)),
```

**Result.** `lookup_auditor`: context **2,704.6 → 320.7** tokens, total **3,256.0 → 627.2**
(−74.8%). The whole `lookup` route went 0.75 → **0.95** strict.

**The transferable lesson:** a gate built from a list of *words* breaks on inflection. A gate built
from a pattern over the *relation* survives paraphrase. This is the same principle that governs the
router.

### Defect 2 — the chunker assigns financial statements inconsistently per filer

**Root cause.** The initial implementation's `_company_financial_chunks()` restricted financial retrieval to sections
`item7` and `item8` — which is where an income statement *should* be (Item 7 = Management's
Discussion, Item 8 = Financial Statements). But the chunker's section assignment is **not consistent
across filers**.

I verified this by counting 10-K chunks per section per ticker in `datasets/sec/corpus.jsonl`:

| Ticker | Company | `item7` | `item8` | Where its chunks actually are |
|---|---|:-:|:-:|---|
| **C** | Citigroup | **0** | **0** | `body` × **800** |
| GE | GE Aerospace | 1 | 0 | `header` × 274, `item16` × 2 |
| JPM | JPMorgan Chase | 1 | 1 | `item15` × 690 |
| XOM | ExxonMobil | 1 | 1 | `item16` × 219 |
| AIG | American International Group | 11 | 2 | `item7a` × 232, `item6` × 110 |
| *(contrast)* MMM | 3M | 44 | 200 | `item8` × 200 — the normal case |

**Citigroup has literally zero item7/item8 chunks.** Under the old filter, its income statement was
**unreachable no matter how the question was phrased**. Same for GE, JPMorgan, ExxonMobil, AIG.

**The fix** (`_fin_pool`, pipeline.py:1157): pool **every** 10-K chunk for the ticker, any section.

**Why broadening costs no precision** — and this is the design argument, from the code's own
docstring: *"the row scorer is itself highly selective (metric label must prefix the line's head,
plus ≥2 numeric tokens), so this trades a cheap local scan for recall rather than for noise."*
Precision lives in the **scorer**, so the **pool** is free to be generous.

### Defect 3 — punctuation broke company-name resolution, twice, in opposite directions

**Root cause A — punctuation left in the index key.** The initial implementation indexed `comp.lower().split()[0]`. For the
corpus company `"Netflix, Inc."` that token is **`"netflix,"`** — with the comma. So the ordinary
name "Netflix" resolved to **nothing**. Measured on extra100: SQ078 (Netflix) and SQ063 (3M)
resolved no company seed at all.

**Root cause B — the possessive stripper ran too early.** The initial implementation stripped `'s` **before** trying the raw
name. Several corpus companies **end in apostrophe-s as part of their legal name** — I verified
`Lowe's` and `McDonald's` in the corpus metadata. So `"Lowe's"` became `"lowe"` and failed to
resolve a company that was sitting in the index **under its exact name**.

**The fix** (`_company_key_for`, pipeline.py:506–576). Strip punctuation when building the index:

```python
bare  = re.sub(r"\s*\(.*?\)", "", comp)                      # drop "(Class B)" etc
nosuf = re.sub(r"[,.]?\s+(inc|incorporated|corporation|corp|company|co|plc|ltd|...)\b\.?",
               "", bare, flags=re.I)
```

And reorder the lookup variants, with the reason written into the docstring in capitals:

```python
"""ORDER IS LOAD-BEARING. The possessive stripper must run AFTER the raw name, because
several corpus companies END in apostrophe-s as part of their legal name — "Lowe's",
"McDonald's"."""
n = raw.lower().strip().strip("?.,;:")
out = [n]                                    # raw name FIRST
no_poss = re.sub(r"[’']s$", "", n)
if no_poss != n: out.append(no_poss)         # stripped version as a FALLBACK
```

Try the most faithful spelling first; fall back to the stripped one. Note it handles both the ASCII
apostrophe `'` and the Unicode right single quote `’`.

### Defect 4 — all 20 compare questions routed to aggregation

**Root cause.** Two independent faults, and it takes both to understand why the bug was invisible
for so long:

1. `AGG_RE`'s first alternative is `\b(which|list|name|how many|all|every)\b.{0,40}\bcompan(y|ies)\b`.
   *"Which **company** reported higher net income, Qualcomm or Salesforce?"* matches it — "which" …
   "company". And `AGG_RE` is tested **before** compare (that ordering was an earlier implementation fix worth Tier D
   +0.17, so it could not simply be reversed).
2. `COMPARE_RE` would have missed anyway: it requires `which` **immediately** followed by
   `had|has|spent|paid|reported`. In *"Which company reported…"* the word "company" sits in between.

**Why it hid.** The official held-out 50 phrase these as *"Which **had** the higher …: A or B?"* —
same vocabulary, same domain, **correct route**. The defect only surfaced when 20 questions with the
other phrasing arrived. Verified in `router_probe_v7frozen.json`: compare **0/20**, all 20 produced
`aggregation`.

> **The lesson this teaches is sharper than "regexes don't generalize":** the router was sensitive
> to **phrasing**, not just to vocabulary. A reviewer seeing only the 0.173 cross-domain number
> would have concluded the problem is domain vocabulary. It is narrower, and more fixable.

**The fix** — a binary-comparison guard checked **first**, requiring **three** conditions:

```python
if BINARY_COMPARE_RE.search(q) and BINARY_OR_RE.search(q) and COMPARE_METRIC_RE.search(q):
    return "compare"
```

a comparative adjective governing a metric (`BINARY_COMPARE_RE`), a two-way disjunction
(`BINARY_OR_RE`), **and** a financial metric to compare on (`COMPARE_METRIC_RE`). All three are
needed. The third exists specifically so the guard doesn't claim *"Which company has more
employees, A or B?"* — a question the compare **branch** cannot serve, because that branch resolves
financial-statement rows.

The optimization also **broadened** the pattern past the version that shipped as `P3_COMPARE_FIX` in the initial implementation, so it
survives clause reordering: *"Between A and B, which reported greater net income?"* and *"Whose net
income was larger, A's or B's?"* are the same question with the clauses moved. The code states the
invariant: *"a comparative adjective governing a metric, plus a two-way disjunction; the
interrogative's position is not part of it."*

**Measured** (`router_probe_the intermediate build.json`): routing accuracy **0.800 → 1.000**, 20/20 on every
route. Compare tokens **5,232 → 484** (−90.4%), strict **0.15 → 0.367**.

**One promotion note that matters for method.** The initial implementation shipped this fix **off by default**
(`P3_COMPARE_FIX`), because the defect was found on a held-out set and enabling it would be tuning
against held-out data. The optimization turns it on unconditionally, and the code justifies the promotion
precisely: *"the pattern is justified by the 16 compare questions inside the tune-120 split alone,
so promoting it is not informed by the holdout-30."*

## D2. "Replace, don't append" — the core the optimization rule, and how it backfired

**The rule.** When a deterministic artifact answers the question, feed **that and nothing else**.
The code states it (pipeline.py:1650): *"the optimization restructures this branch around one principle: when a
deterministic artifact answers the question, feed THAT and nothing else. The initial implementation APPENDED — the line here
was `_context_from_chunks(fact_chunks + ids)`."*

**Why replacing is right.** Three reasons, all measured:

1. **It is where the saving is.** Appending keeps the full-chunk cost. `lookup_auditor` paid 2,705
   context tokens for a fact that fits in one 44-token sentence.
2. **The appended chunks were not helping.** At 2,705 tokens P3 **tied** P2 on `lookup_auditor` and
   **lost** on `lookup_revenue` (0.60 vs 0.80).
3. **Less context is often more accurate.** On compare, the optimization went 0.15 → 0.367 *while* dropping 90% of
   the tokens. Burying the answer row in eight pages of tables makes the model pick the wrong row.

**How it backfired: compound questions.** The rule is only safe when the resolved artifact answers
the **whole** question. If the question asks for two things and the artifact answers one, replacing
the chunks destroys the evidence for the other half.

### The worked example: EQ49

Verified across three bench files, run 1 each:

> **Q:** *"Cisco's May 2026 restructuring 8-K funds which growth areas, and who is Cisco's auditor?"*
> **Gold:** "Silicon, optics, security, and AI; PwC"

| version | route | context tokens | total | answer |
|---|---|:-:|:-:|---|
| **the initial implementation** | lookup | **4,460** | 5,576 | "silicon, optics, security and artificial intelligence (AI) / pricewaterhousecoopers llp" — **both halves right** |
| **the optimization** | lookup | **90** | 400 | *"The answer is not supported by the provided context passages and knowledge-graph facts."* — **refused** |
| **final** | lookup | **1,835** | 2,520 | "silicon, optics, security, artificial intelligence (AI), PricewaterhouseCoopers LLP" — **both halves right** |

**Read what happened in the optimization.** The question matched the `audited_by` relation pattern. The curated-
edge path fired, resolved that one edge, fed its 90-token evidence sentence, and — obeying "replace,
don't append" — **discarded the chunks containing the restructuring 8-K**. The model was then
holding evidence for the auditor half and nothing for the growth-areas half. It correctly refused
rather than guess.

So the refusal was the model behaving well. The **pipeline** starved it.

**The guard** (pipeline.py:1278–1300). Keys on the **coordination itself**, not on any question's
wording:

```python
_COMPOUND_RE = re.compile(
    r",\s*and\b"                                    # "…growth areas, and who is…"
    r"|\band\s+(who|what|which|when|how|whose|why)\b"
    r"|\band\s+(its|their|his|her)\s+\w+\s*\??$"
    r"|\?.*\?",                                     # two question marks
    re.I)
COMPOUND_K = int(os.getenv("P3_COMPOUND_K", "4"))   # reduced, not full, chunk budget
```

When `_asks_multiple()` is true, the branch keeps **4** retrieved chunks alongside the graph
artifact instead of discarding them — *"enough to cover the clause the graph artifact does not
answer, without reverting to the full 8–10 chunk spend."*

**And that is the real design point.** The final system is not "undo the optimization". EQ49 goes from the initial implementation's **4,460** context
tokens to the final system's **1,835** — **59% cheaper than the initial implementation, with the same correct answer.** The guard
recovers the accuracy without giving back the saving.

## D3. The honest cost: the optimization raised two sets and regressed the official one

All from `summary.json`, 3 runs each, P3 strict pass:

| set | n | the initial implementation | the optimization | Δ |
|---|:-:|:-:|:-:|:-:|
| extra100 | 100 | 0.370 | **0.443** | **+0.073** |
| combined 150 | 150 | 0.431 | **0.456** | **+0.025** |
| tune-120 (in-sample) | 120 | 0.414 | **0.436** | +0.022 |
| holdout-30 | 30 | 0.500 | **0.533** | +0.033 |
| holdout-23 (clean) | 23 | 0.478 | **0.522** | +0.044 |
| **official held-out 50** | **50** | **0.553** | **0.480** | **−0.073** |

**The regression is on the one set that matters most** — the organisers' own
`eval_questions.docx` questions, EQ01–EQ50. It is concentrated by tier
(`summary.json → by_tier`):

| Tier | What it is | n | P2 | the initial implementation | the optimization | Δ |
|---|---|:-:|:-:|:-:|:-:|:-:|
| A | single-hop lookup | 8 | 0.875 | 0.750 | 0.750 | 0.000 |
| **B** | **two-hop bridge** | **16** | 0.188 | **0.354** | **0.250** | **−0.104** |
| C | multi-hop aggregation | 20 | 0.150 | 0.600 | 0.600 | 0.000 |
| **D** | **cross-document temporal** | **6** | 0.333 | **0.667** | **0.333** | **−0.334** |

**I traced every flipped question** by comparing `eval_initial_run{1,2,3}.json` against
`eval_the intermediate build_run{1,2,3}.json` on the 50 held-out IDs. The entire regression is **five questions**:

| ID | Tier | the initial implementation (3-run) | the optimization (3-run) | Cause |
|---|:-:|:-:|:-:|---|
| **EQ17** | B | 1.00 | 0.00 | **compound** — "who is he, **and** who is Amgen's auditor?" |
| **EQ18** | B | 1.00 | 0.00 | **compound** — "is whom, **and** who is Chevron's auditor?" |
| **EQ45** | D | 1.00 | 0.00 | **compound** — "when does the outgoing CFO retire, **and** when does the incoming CFO start?" |
| **EQ49** | D | 1.00 | 0.00 | **compound** — "funds which growth areas, **and** who is Cisco's auditor?" |
| EQ14 | B | 0.67 | 0.00 | **row selection** — picked Walmart's $680,985 row instead of $713,163 |
| *EQ13* | *B* | *0.00* | *1.00* | *(improvement)* |

Net −4.67 + 1.00 = **−3.67 of 50 = −0.073**. Exactly the headline delta. Nothing unexplained.

**Four of the five are compound questions, and the final system's guard targets exactly those four.** Here is
the deterministic evidence that it works — answer text and token counts from
`bench_combined150_final_run1.json`:

| ID | gold | the optimization (ctx tok) | the final system (ctx tok) | the final system answer |
|---|---|:-:|:-:|---|
| EQ17 | "Thomas Dittrich; Ernst & Young LLP" | 90 → refused | **1,609** | "Thomas Dittrich, Ernst & Young LLP" ✓ |
| EQ18 | "R. Hewitt Pate; PricewaterhouseCoopers LLP" | 82 → refused | **1,510** | "R. Hewitt Pate, PricewaterhouseCoopers LLP" ✓ |
| EQ45 | retires Aug 31, 2026; starts Sept 1, 2026 | 493 → refused | **2,190** | "Outgoing CFO retires: August 31, 2026; Incoming CFO starts: September 1, 2026" ✓ |
| EQ49 | "Silicon, optics, security, and AI; PwC" | 90 → refused | **1,835** | "silicon, optics, security, artificial intelligence (AI), PricewaterhouseCoopers LLP" ✓ |

All four read as gold matches. But **strict pass is a judge call, and the final system has not been judged** —
`eval_final_*.json` does not exist. So:

- **What I can state:** The final system's answer text on all four matches gold, at **27–63% fewer context
  tokens than the initial implementation** spent to get the same answers. Combined-150 average cost rises only
  980.7 → 1,027.5 tokens (reduction 76.6% → 75.5%).
- **What I cannot state:** The final system's strict pass, graded quality, or whether the official held-out 50
  returns to or above the initial implementation's 0.553. If all four pass and nothing else moves, the arithmetic would be
  0.480 + 4/50 = 0.560 — **but that is arithmetic on an unjudged batch, not a result.** It also
  assumes EQ14 stays failed; EQ14 picked the correct Walmart row in the final system run 1, which would push it
  higher still, and assumes nothing else flips, which run-to-run noise alone can cause.

**The right way to state the optimization today:**

> the optimization is a decisive and reproducible **token** result — 76.6% reduction on the combined 150 against a
> tuned baseline, in one batch, with `optimizer_tokens` at exactly 0. It is a **modest** accuracy
> gain on the combined 150 (+0.025) and a **clear gain** on the route-balanced extra100 (+0.073) and
> on both holdout splits. It is an accuracy **regression on the official held-out 50** (−0.073),
> fully accounted for by five questions, four of which are a single diagnosed defect class that
> the final system addresses. Until the final system is judged, the official-50 number to quote is **0.480**.

---

# Part E — Doubts, pre-emptively cleared

## E1. Why isn't a bigger token reduction automatically better?

Because the metric has a numerator. Three concrete ways a bigger reduction is worse:

**It can be bought with accuracy.** From The initial implementation §4.2, the compressed P2 arm:

| P2 arm | strict | tokens |
|---|:-:|:-:|
| adaptive-k (frozen baseline) | **0.300** | 4,105 |
| adaptive-k + compression | 0.220 | **1,550** |

62% fewer tokens, a third of the accuracy gone. A pipeline that answers *"I don't know"* to
everything achieves a 100% token reduction and is worthless.

**It can be bought with information the question needed.** The optimization's EQ49 is exactly this: 90 context
tokens, a 98% reduction against the initial implementation, **and a refusal**. The reduction was real. The answer was lost.

**It can be a mix artifact rather than a property.** 34.2% → 6.7% happened with *zero* code change,
purely from the route mix. A number that moves 5× without the pipeline changing is not a property of
the pipeline.

Hence `quality_per_1k`, reported alongside tokens everywhere. And An earlier implementation §3.2's framing is the one to
internalise: an earlier implementation moved to a higher-accuracy point on **the same efficiency frontier**. Accuracy
0.487 → 0.553, tokens +14.5%, accuracy-per-1k **flat** at ~0.205. Movement along the frontier is not
improvement *of* the frontier. The optimization is the first change that actually moved the frontier: 0.134 →
0.465 quality-per-1k.

## E2. In-sample vs held-out, and why the label is insisted on

**Held-out** data is data you did not look at while building. **In-sample** data is data you tuned
against. A score on in-sample data is not a prediction about new questions, because you may have
fitted the quirks of the questions you had.

**The classic failure is in this very project.** The initial implementation's compare pattern required `which` immediately
followed by `had|has|spent|paid|reported`. That works on *"Which **had** the higher X: A or B?"* —
the phrasing of the official 50. It was fitted, unintentionally, to the questions available. When 20
questions arrived saying *"Which company **reported** higher X, A or B?"*, routing accuracy on
compare was **0/20**. In-sample accuracy was perfect. Out-of-sample was zero.

**Why the optimization has an in-sample problem at all.** The repo owner's instruction was explicit
(`SEC_OPTIMIZATION_HANDOFF_PROMPT.md` §6): tune the regexes against **all 150** questions. The
consequence, stated in the same breath: *"tuning on all 150 means no held-out set remains, and every
post-tuning metric is an in-sample number. That is my decision and I accept it — but it must be
labelled, not hidden."*

**The four mechanisms built to compensate**, all cheap because the router is a free pure function:

1. **A fresh tune/holdout split.** The 150 stratified into **tune-120** and **holdout-30** by shape
   and tier, with a fixed seed, recorded in `datasets/sec/split_{tune120,holdout30}.jsonl`.
   Verified line counts: 120 and 30. Built on the 120; the 30 run once.
2. **A contamination audit**, in `results/sec/holdout30_contamination.json`. It lists every
   question inspected during development and intersects that with the holdout. Result: **7 of 30
   contaminated** (EQ01, EQ02, EQ20, SQ018, SQ043, SQ074, SQ075), leaving a **holdout-23-clean**
   set. Both are reported: holdout-30 strict **0.533**, holdout-23-clean **0.522**.
3. **5-fold cross-validation on the router**, scoring each fold using only patterns justified by the
   other four. Plus a **leave-one-route-out** variant. In `router_probe_v7frozen.json`, LORO makes
   the defect visible: disabling `BINARY_COMPARE_RE` drops the compare fold to 0.00 while the other
   four stay at 1.00 — it isolates exactly which pattern carries which route.
4. **A 408-question paraphrase probe** — see E4.

**Why this is worth the trouble.** Note where the optimization's numbers actually land:

| set | status | the optimization strict |
|---|---|:-:|
| tune-120 | **in-sample** | 0.436 |
| holdout-30 | held out from the optimization | **0.533** |
| holdout-23-clean | held out, decontaminated | **0.522** |

The held-out numbers are **higher** than the in-sample one. That is the opposite of what overfitting
looks like, and it is evidence you only get by building the split.

## E3. Why tuning the baseline is essential, not self-sabotage

The instinct is backwards here. Tuning P2 **strengthens** the result it appears to weaken.

**The reviewer's criticism it answers.** An earlier implementation §4 records it as *"the strongest outstanding
criticism"*: you tuned the graph pipeline and compared it to a naive one. That criticism is fatal if
true, because it means the gap might be *tuning*, not *graphs*. No amount of extra measurement on
the old setup can refute it — only tuning the baseline can.

**What it cost, honestly.** Token reduction 43.0% → 34.2%. P2's own efficiency nearly doubled
(0.073 → 0.142 strict/1k), so P3's efficiency edge over a *tuned* baseline is **1.39×**, not the
2.70× against the frozen one. Both are reported; **1.39× is the defensible one**.

**What it bought — the objection settled by construction, not by argument.** From The initial implementation §4.2:

> The compressed baseline runs at **1,550 tokens — 43% *below* P3's 2,703** — and still reaches only
> **0.220 strict against P3's 0.533**.

Given a strictly **smaller** budget than the graph pipeline, and tuned explicitly for efficiency,
the vector baseline scores **less than half**. You cannot argue that away; the gap is not a budget
artifact.

**And notice what tuning P2 is not.** `P2_ADAPTIVE_K` hands P2 **P3's own router**
(`pipeline2_basic_rag.py:238–241`). The project's strongest single mechanism was given to the
baseline. The arm chosen as the frozen baseline (`adaptive-k`) was chosen *"precisely because it is
the strongest honest baseline — it costs the reviewer's argument nothing and costs us tokens."*

The optimization instructions make the inverse explicit and absolute (`SEC_OPTIMIZATION_HANDOFF_PROMPT.md`
§0.1, under **FORBIDDEN**): *"Weakening P2 in any way: do not change its k, its reranker, its
prompt, or its config; do not disable its adaptive-k."*

> **The general principle.** A result that survives a strong baseline is worth more than a larger
> result against a weak one. An earlier implementation §4 and the final report §9 both call this the strongest reviewer
> criticism — and the project's own summary of its track record is *"Every correction this round
> made the claim smaller… The remaining claims are the ones that survived adversarial checking,
> which is why they are worth more than the originals."*

## E4. Paraphrase stability vs routing accuracy

**Routing accuracy** asks: on the 100 labelled questions **we have**, how often does the router pick
the gold route? The optimization: **1.000** (`router_probe_the intermediate build.json`).

**Paraphrase stability** asks something different and much harder: take each question, **mechanically
rewrite it** — reorder clauses, swap active for passive, substitute synonyms, turn "Which company
reported higher X, A or B?" into "Which had the higher X: A or B?" — and check the router assigns
**the same route to the rewrite as to the original**.

**Why routing accuracy cannot detect what paraphrase stability detects.** Routing accuracy is
measured on the *exact strings you tuned against*. A pattern fitted to one phrasing scores 1.000 and
tells you nothing about whether it survives the next phrasing. Paraphrase stability is a **synthetic
out-of-sample test** that costs nothing, because the router is a free pure function — so you can run
hundreds of unseen question strings through it without a single API call.

**Measured** (`datasets/sec/paraphrase_probe.jsonl`, 408 paraphrases of the 150 originals):

| config | stable | rate |
|---|:-:|:-:|
| the initial implementation frozen | 201 / 408 | **0.4926** |
| the optimization | 408 / 408 | **1.000** |
| the final system | 408 / 408 | **1.000** |

**the initial implementation failed on more than half of its own questions reworded.** And the failures were not scattered
noise — they were whole mechanisms missing, visible per rewrite rule in
`router_probe_v7frozen.json`:

| rewrite rule | the initial implementation stable | rate |
|---|:-:|:-:|
| `temporal.generic_order` | 0/21 | 0.00 |
| `temporal.sequence` | 0/20 | 0.00 |
| `temporal.whichfirst` | 0/20 | 0.00 |
| `compare.between` | 0/20 | 0.00 |
| `compare.posted` | 0/20 | 0.00 |
| `compare.whose` | 0/20 | 0.00 |
| `compare.reported2had` | 1/20 | 0.05 |
| `bridge_auditor.{share, possessive, clause_reorder}` | 0/12 each | 0.00 |
| `bridge_director.{besides, person_first}` | 0/6 each | 0.00 |
| `generic.dash_clause_swap` | 3/15 | 0.20 |

Three diagnoses fall straight out, each naming a concrete code defect:

- **Temporal 0/61.** `TEMPORAL_RE` had `\bchronologic\b`, which **does not match
  "chronologically"** — the word boundary fails mid-word. And the ordinary English ways of asking
  for an order — *"which came first"*, *"in what sequence"*, *"did X precede Y"* — were **absent
  entirely**. Fixed by `TEMPORAL_EXT_RE`, deliberately kept as a **separate** pattern so an ablation
  can remove the optimization addition without also removing pre-existing work — *"so the CV reports what this
  pass actually contributed instead of overstating it."*
- **Compare ~0/82.** The `which`-immediately-followed-by-a-verb requirement. Fixed by
  `BINARY_COMPARE_RE`.
- **Bridge 0/38.** The initial implementation's `BRIDGE_RE` recognised only the stock phrase *"the company (that|whose|
  which)"*. *"Accenture's auditor also audits which other companies?"* and *"Which other companies
  share the same auditor as Accenture?"* ask **exactly the same thing** and routed elsewhere. Fixed
  by `BRIDGE_SHARED_RE` + `BRIDGE_REL_RE` — a shared-attribute marker (`other|same|also`) **and** a
  relation word, both required, which is why plain enumerations stay aggregation.

**What this buys that routing accuracy cannot.** 0.4926 → 1.000 is evidence the patterns now key on
**linguistic structure** rather than on remembered phrasings. It is the closest thing to a
generalization guarantee available here — and it is the direct institutional response to defect 4,
which was *precisely* a pattern fitted to one phrasing.

**And it is a bounded guarantee, which is the honest caveat.** Paraphrases are **mechanical
rewrites of SEC questions**, so stability says nothing about a different domain. The cross-domain
number remains **0.173** on 150 Olympics questions (`router_generalization.json`): the router assigns
`lookup` to **100% of them**. In-domain 0.800 (the initial implementation) / 1.000 (the optimization) · out-of-domain 0.173. The final
report §10 states the division plainly: *"The architecture generalizes; this router does not."*

## E5. Why the extra100 gold is "circular" for 70 of 100 questions

**The finding** (RESULTS_SEC.md). `eval_extra100.jsonl` carries an `answer_from` field. **70 of 100 rows
say `"graph edges"`**, and their `source_docs` field names **this project's own build artifacts**:

| `source_docs` value | n |
|---|:-:|
| `events.json dates` | 20 |
| `audited_by edges` | 14 |
| `2 hops via kpmg llp` | 12 |
| `audited_by x sector` | 8 |
| `2 hops via <person>` | 6 |
| `manifest sector backbone` | 5 |
| `uses_supplier edges` | 3 |
| `2 hops via tsmc` | 2 |

And the overlap was **verified member-for-member** against `pipeline3_sec/index/edges_prov.json`:
the `audited_by` edge sets are **identical** to the gold `gold_members` lists — Ernst & Young 32/32,
PwC 31/31, Deloitte 24/24, KPMG 12/12, **zero set difference** after name normalization.

The remaining **30 rows** say `"10-K text, LLM-verified"` and cite ticker-named filings. Those are
grounded only in corpus text.

**Why "circular" is the right word.** On those 70 questions, **P3's ground truth is P3's retrieval
target**. Its ceiling there is near-perfect *by construction*, while P2 is graded against a key it
has no structural access to. The circularity is quantified in the set-F1 table: P3 scores **1.000**
on 17 of 20 aggregation questions — read as *"P3 recovers its own graph perfectly"*, not *"P3 is
perfect"*.

> **Provenance, as the document itself reports it.** The project owner states the organisers supplied
> the file, so the overlap reflects questions and graph deriving from the same filings. One detail
> stays unexplained and is recorded rather than smoothed over: `events.json` and
> `manifest_clean.json` are filenames produced by `build/kg/extract_kg.py`, **not SEC
> identifiers**. the SEC study states the overlap as a *measured fact* and does not assert authorship. That
> is the right posture and it is reproduced here unchanged.

**What follows, either way — the split that matters** (RESULTS_SEC.md):

| subset | n | P1 strict | P2 strict | P3 strict | P2 tok | P3 tok |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| gold from **graph edges** | 70 | 0.10 | 0.18 | **0.40** | 4,098 | **3,444** |
| gold from **10-K text** | 30 | 0.00 | **0.33** | 0.30 | 4,530 | **5,108** |

**On the 30 questions whose gold does not come from our own graph, P2 and P3 are even — 0.33 vs 0.30
— and P3 spends 13% *more* tokens.** P3's entire measured advantage on this set lives in the 70
graph-derived questions.

**Three things follow, and no more:**

1. **The extra100 cannot be cited as independent confirmation** of the official-50 headline.
2. **It does not overturn the official 50.** Those came from the organisers'
   `eval_questions.docx` and show +0.253 at 34.2% fewer tokens in the initial implementation, re-confirmed in a
   second batch to within run noise.
3. **The 30-question subset is small** — a one-question difference on n=30. Suggestive, not
   decisive. But it is the subset that tests the claim.
4. **Every table in the SEC study is split by `answer_from`.** A single pooled figure over all 100 would
   overstate the gap, so it is never reported alone.

**And note the methodological consequence the optimization inherits.** The 20 temporal questions' gold is
`events.json dates` — so raising the temporal score by reading that same file would be circular and
**cannot be presented as a quality win**. The handoff instruction draws the line: *"The token saving
is legitimate and is the thing to chase here… judge temporal quality on Tier D, not on
`temporal_order`."* That is exactly what the optimization did — hence temporal's 87.2% token cut being reported
while its 0.00 strict is explicitly not read as a quality signal.

## E6. Why `events.json` is distrusted, and what replaced it

**The file.** `pipeline3_sec/index/events.json` holds 8-K events extracted by an LLM from the
filings, each with a `type`, an `amount`, a `date` and a `chunk_id`.

**Why its `type` field is untrustworthy** — three documented symptoms, all from An earlier implementation §2 and repeated
in the final report §4.5:

1. Some rows carry **garbled JSON fragments as the type string**, e.g.
   `` officer_departure`, `person`:  ``.
2. **AAPL's real April-2026 CEO-transition 8-K is not tagged** `officer_appointment` at all.
3. **Unrelated 10-K boilerplate *is* tagged** — "management has evaluated… CEO and CFO…"
   disclosure-controls language classified as an officer appointment.

**The measured consequence.** An earlier implementation feature built on it was **implemented, tested, and deliberately
reverted before shipping**. The naive version returned **84 of 100 companies** (catching
boilerplate); a precision-tightened version returned **32 that excluded both correct answers**. No
threshold worked, because the field's errors are not a precision/recall trade — they are mislabels
in both directions.

**What replaced it: SEC's own filer-assigned item codes.** `Item 5.02` is the official 8-K item
number for *"Departure of Directors or Certain Officers; Election of Directors; Appointment of
Certain Officers"* — **assigned by the filer at filing time**, not inferred by a model. A company
qualifies if **any** of its chunk ids contains `item5.02`. Deterministic, zero extraction noise.

**Verified**: recovers **both** DEV14 gold companies (AAPL, AMGN) at a plausible count of 42/100 —
against the reverted attempt's 84 (no filter) or 32 (over-filtered, missing both).

**the optimization extended the same principle rather than inventing a new one.** `_company_8k_timeline()` reads
the date **and** the item code out of the chunk id, and labels each with `_SEC_ITEM_TITLES` — a
closed lexicon of SEC's own item titles. The docstring states the lineage explicitly: *"Both the
filing DATE and the item CODE here come from the corpus chunk id, which is derived from the filing
itself — so the ordering evidence is deterministic and filer-assigned."*

> **The general rule:** prefer metadata the **source** assigned over labels a **model** inferred.
> `events.json`'s `type` is a model's guess. `item5.02` is the filer's own declaration. Both are
> "in the data"; only one is ground truth.

Note also that the underlying data defect was **not** fixed — it was routed around. An earlier implementation §5 lists it
as a known defect needing an ingestion-side fix. Working around a bad field and *saying* you worked
around it is a legitimate engineering decision; silently relying on it is not.

## E7. Why `tg_live` below 1.0 is not a bug

**What it used to be.** A **session-level boolean**, set once when the TigerGraph connection opened,
then **copied onto every result row**. It reported **1.00**, and was cited as proof the live cloud
graph answered every question.

It did not mean that. It meant *"the connection worked at some point."*

**What it is now.** `tg_queries` is a counter reset at the top of `run_single()` and incremented on
each successful `runInstalledQuery`. `tg_live` is derived: `tg_queries > 0 and not GRAPH_OFF`.

**The honest number: 0.84 (42/50)** on the held-out 50, with 61 live GSQL queries.

**All 8 zero-query questions are aggregation-route, and every one is legitimately answerable without
Savanna** (The initial implementation §6 enumerates all eight with causes):

| Question (truncated) | Why zero queries |
|---|---|
| "Besides Netflix, which company authorized a $25B share repurchase…" | `__section__` corpus scan over SEC item codes |
| "AMD names which two companies as its principal competitors…" | corroborated `competitor_of` from the local index |
| "Two companies list Indra Nooyi as a member of their…" | `has_director` surname match, local |
| "Ellen Kullman serves on the boards of which two…" | `has_director` surname match, local |
| "F. William McNabb III is a director at which two…" | `has_director` surname match, local |
| "Which companies declared a cash dividend vian 8-K…" | `__section__` corpus scan |
| "Which companies authorized a $25 billion share-repurchase…" | `__section__` corpus scan |
| "Among the dataset's semiconductor and networking companies…" | sector filter from chunk metadata |

**Why that is design, not failure.** Three of these use **surname matching** — and surname matching
exists *because* exact-match GSQL misses name variants ("indra nooyi" / "indra k. nooyi" /
"mrs. nooyi"). Going to the network would return a **worse** answer. Others scan SEC item codes,
which is deterministic local metadata. Issuing a network round-trip for an answer already sitting
locally would cost latency and gain nothing.

So the honest claim is: **a live-graph + local-index hybrid whose live share depends on question
shape.** Measured across both sets (`summary.json → invariants.tg_live`):

| set | `tg_live` | live GSQL queries (the optimization) |
|---|:-:|:-:|
| extra100 | **1.000** | 366 |
| held-out 50 | **0.840** | 171 |
| combined 150 | **0.947** | 537 |

extra100 reaches 1.00 because its aggregation questions are auditor/sector/supplier queries that all
take the live `agg_*` path; the held-out set's include `__section__` scans and surname matching that
resolve locally by design.

> **This is also why the optimization deliberately kept the graph in paths that no longer need it.** The metric-row
> extractor could answer from the local corpus with **zero** GSQL calls. The code says what that
> would have cost (pipeline.py:1193–1199): it *"would have dropped measured `tg_live` from 0.95 to
> 0.55 while the pipeline still called itself GraphRAG — the provenance overstatement the initial implementation §6 was
> written to end."* So the live vector search stays in the path, and `_ROW_SOURCE_STATS` records
> which half actually supplied the winning row, *"so the graph's contribution here is reported
> rather than assumed."*
>
> That is the pattern worth copying: when an optimisation would quietly invalidate a claim you make
> elsewhere, either don't make the optimisation or re-measure the claim. Do not let the claim drift.

One caveat on vocabulary: `metrics.py`'s `aggregate_metrics()` docstring still reads *"For a valid
GraphRAG run this must be 1.0"* — a leftover from the session-flag era that contradicts the
hybrid framing the measurement established. The derivation and the per-question counter are correct;
that one comment line is stale.

## E8. Why judge scores are nondeterministic, and what ±0.019–0.025 means

**Two independent sources of variation:**

1. **The generator.** Gemini 2.5 Flash at temperature 0. Temperature 0 means "always pick the most
   likely next token", which is *nearly* deterministic but not exactly — floating-point
   non-associativity across batching and hardware means identical prompts can produce slightly
   different text. The retrieval is deterministic; the wording is not quite.
2. **The judge.** `gpt-4o-mini` making a PASS/FAIL call, and borderline answers sit on the fence.
   An answer saying "$215,938 million for FY2026" against a reference reading "$215,938M" can be
   graded either way, on identical input, on different days.

**What the spread is.** **±0.019–0.025 strict pass**, measured as the three-run standard deviation on
the 50-question held-out batch. On n=50, 0.020 is **one question**.

**How it is used — this is the part that matters:**

- **It defines the noise floor.** The project's own standing rule
  (`SEC_OPTIMIZATION_HANDOFF_PROMPT.md` §0): *"Differences under ~0.05 strict between single-run
  arms are not real."*
- **It resolves apparent regressions.** An earlier implementation reported P3 at **0.553**, The initial implementation at **0.533**, with
  **byte-identical answer logic** between them. That 0.020 is noise, not a regression — and both
  documents say so rather than quietly picking the better number. Independently, re-running the
  **original** `bench_test_v7final_run1.json` through the refactored evaluator reproduced strict
  pass **0.12 / 0.30 / 0.52 exactly**, with **zero** per-item BERTScore differences and
  byte-identical token counts — so the deterministic half of the stack is confirmed deterministic.
- **It calibrates how much to believe a single arm.** The compare-fix arm's +0.100 strict on n=20 is
  two questions, against a ±0.019–0.025 spread. RESULTS_SEC.md therefore leans on the **graded** gain
  (+0.150) instead: *"the more trustworthy signal, since graded quality moves in smaller increments
  than a PASS/FAIL."*
- **It is what makes the optimization numbers readable.** The optimization's combined-150 standard deviations are tiny
  (strict ±0.0031 across 3 runs) because the effect is large. The −0.073 on the held-out 50 is
  **~3× the spread** — real, not noise, which is why it is reported as a regression and traced to
  five specific questions rather than waved away.

**The practical reading rule.** Every the optimization number here is a 3-run mean, and the summary file carries
`*_std` beside each one. Before believing any difference, check it against the std. And every final-system
accuracy number is **absent**, not small — do not fill that gap with arithmetic.

---

# Appendix — the one-page summary

**Where things stand (the optimization, 3 runs, `results/sec/summary.json`):**

| | combined 150 | official held-out 50 | extra100 |
|---|:-:|:-:|:-:|
| P1 strict | 0.098 | 0.120 | 0.087 |
| P2 strict | 0.251 | 0.300 | 0.227 |
| **P3 strict** | **0.456** | **0.480** | **0.443** |
| P2 tokens | 4,186 | 4,105 | 4,227 |
| **P3 tokens** | **981** | **1,237** | **853** |
| **token reduction** | **76.6%** | **69.9%** | **79.8%** |
| quality per 1k (P2 → P3) | 0.060 → **0.465** | 0.073 → **0.388** | 0.054 → **0.520** |
| efficiency ratio | **7.75×** | **5.31×** | **9.69×** |
| `tg_live` | 0.947 | 0.840 | 1.000 |
| `optimizer_tokens` | **0** | **0** | **0** |

**The six things to remember:**

1. **One mechanism carries everything:** feed the one evidence sentence that proves a graph edge
   (44 tokens) instead of the chunk it lives in (419). The initial implementation applied it to one route of five. The optimization
   applied it to all five. That is the 15.7% → 76.6% move, and it is the only one of the four
   headline shifts that is engineering rather than methodology.
2. **Always name the set.** 50.2% / 34.2% / 15.7% / 6.7% / 76.6% are five different measurements,
   differing in question mix, baseline tuning, and batch coherence — not five estimates of one
   quantity.
3. **The baseline is tuned on purpose**, and tuning it is what turned "the gap might be a budget
   artifact" into a settled question: P2 at 1,550 tokens (43% **below** P3) still scored 0.220
   against 0.533.
4. **Zero `optimizer_tokens` is the fairness guarantee**, and it holds in both pipelines. Nothing is
   off-ledger.
5. **the optimization's accuracy story is mixed and the regression is fully traced:** +0.073 on extra100, +0.025
   on combined 150, −0.073 on the official 50 — the last one accounted for by exactly five
   questions, four of them one diagnosed defect class.
6. **the final system fixes those four.** Deterministically verified: all four produce gold-matching answers at
   27–63% fewer tokens than the initial implementation spent on the same answers. **But the final system has not been judged**, so it
   has no strict-pass number, and the official-50 figure to quote today remains **0.480**.

**The strongest thing about this project is not any number in it.** It is that 50.2% → 34.2% →
15.7%, `tg_live` 1.00 → 0.84, and router "portable" → 0.800 in-domain / 0.173 out — every single
correction made the claim **smaller**, and each was found and published by the project rather than by
a reviewer. That is what makes the claims that survived worth something.
