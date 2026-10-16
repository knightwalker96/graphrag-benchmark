# GraphRAG vs Basic RAG vs LLM-only — a quality-per-token benchmark

Does building a knowledge graph actually pay for itself?

Retrieval-augmented generation usually answers a question by stuffing the prompt with whole
retrieved text chunks. That works, but it is expensive: most of what gets pasted in is not the
part that answers the question. This project tests a different approach — **retrieve the graph
edge, then send only the one sentence that proves it** — and measures whether the resulting
accuracy is worth the tokens it costs.

Three pipelines answer the same questions, with the same model, the same judge, and the same
embedder, so the only variable is retrieval strategy:

| | Pipeline | What it does at query time |
|---|---|---|
| **P1** | LLM-only | No retrieval. Parametric knowledge only — the floor. |
| **P2** | Basic RAG | Dense vector search over text chunks. A **tuned** baseline, not a straw man (see [docs/BASELINE_TUNING.md](docs/BASELINE_TUNING.md)). |
| **P3** | GraphRAG | Deterministic router → graph query → evidence-sentence context. |

It is evaluated on two deliberately different corpora — dense financial filings, and open-domain
encyclopedia text — because a method that only works on one is not a method.

---

## Results

### Corpus 1 — SEC filings

400 filings (10-K annual reports, 8-K event filings, DEF 14A proxy statements) for 100 companies,
chunked into **61,172 passages**. 150 questions, 3 runs, one batch.

| | strict pass | graded quality | tokens / question | vs P2 |
|---|:-:|:-:|:-:|:-:|
| P1 LLM-only | 0.093 | 0.209 | 282 | — |
| P2 Basic RAG | 0.253 | 0.341 | 4,187 | — |
| **P3 GraphRAG** | **0.487** | **0.527** | **1,026** | **−75.5% tokens, +0.233 strict** |

**1.9× the accuracy on a quarter of the tokens.** On the 50 held-out questions reserved from all
tuning, P3 scores **0.553**.

### Corpus 2 — Olympics / Wikipedia

An open-domain encyclopedia corpus with a **regex-built** knowledge graph — no LLM extraction, so
graph construction costs **zero tokens**. 70 held-out questions, 3 runs.

| | strict pass | tokens / question | vs P2 |
|---|:-:|:-:|:-:|
| P1 LLM-only | 0.495 | 282 | — |
| P2 Basic RAG | 0.786 | 5,148 | — |
| **P3 GraphRAG** | **1.000** | **508** | **−90.1% tokens, +0.214 strict** |

> **Read that 1.000 as saturation, not triumph.** All 70 gold answers were written from Wikipedia
> infoboxes, and the knowledge graph is a regex index of those same infoboxes — so the graph is
> being scored against its own source. It demonstrates that the *method ports* to a new domain at
> zero ingestion cost. It does not demonstrate that GraphRAG is perfect.
> [docs/RESULTS_OLYMPICS.md](docs/RESULTS_OLYMPICS.md) leads with this caveat.

### What the token numbers include

Gemini 2.5 bills hidden *reasoning* tokens as output but reports them separately. Every headline
above counts prompt + completion only, so here is both accountings on the SEC set:

| | reported | reasoning | true billed |
|---|:-:|:-:|:-:|
| P2 Basic RAG | 4,187 | +393 | 4,580 |
| **P3 GraphRAG** | **1,026** | **+269** | **1,296** |

**Token reduction is 75.5% as reported, 71.7% on true billed cost.** The conclusion is unchanged;
the margin is four points smaller. Both numbers are published rather than only the flattering one.

---

## How Pipeline 3 works

### The core idea: send the sentence, not the chunk

When the graph knows that *Apple is audited by Ernst & Young*, that fact came from a specific
sentence in a specific filing. Basic RAG retrieves the whole 2,000-character chunk containing it.
This pipeline stores the proving sentence alongside the edge, and sends only that:

```
Basic RAG   →  [chunk]  "...Item 14. Principal Accountant Fees and Services. The Audit
                         Committee of the Board of Directors has appointed Ernst & Young LLP
                         as the independent registered public accounting firm... (1,900 more
                         characters of fee tables and boilerplate)"     ~500 tokens

GraphRAG    →  [AAPL/DEF14A_2026#item14-c3] "Ernst & Young LLP has served as Apple's
                independent registered public accounting firm since 2009."    ~25 tokens
```

Same evidence, same citation, 20× less context. Across the whole benchmark this is where the
token win comes from.

### The router costs nothing

Before retrieving, the question is classified into one of five types by **regular expressions** —
no model call, so routing is free and perfectly reproducible. This matters for fairness: a routing
LLM would spend tokens that the headline metric might not capture. Measured `optimizer_tokens`
across all 450 answered questions: **exactly 0**.

| Route | Example question | Strategy |
|---|---|---|
| `aggregation` | "Which companies are audited by Ernst & Young?" | Graph aggregation over typed edges |
| `lookup` | "Who audits Altria?" | Single edge + its evidence sentence |
| `compare` | "Which had higher net income, Intel or AMD?" | Extract the two statement rows only |
| `bridge` | "Which companies share Apple's auditor?" | Two-hop traversal |
| `temporal` | "What did Boeing file in order?" | Dated timeline from SEC item codes |

Routing accuracy is **1.000** in-domain, and stable across **408/408** mechanical paraphrases of
the question set.

### Where the graph lives

TigerGraph Savanna serves as both graph database and vector store. A local provenance index
carries the evidence sentence per edge. **94.7%** of SEC questions issued a live graph query; the
remainder are answered from that local index by design, and the pipeline reports which path served
each question.

---

## Why the comparison is fair

A benchmark that flatters its own method is worthless. These invariants are enforced in code and
re-checked after every run:

| Invariant | Measured |
|---|---|
| Same generator, temperature, system prompt for all three pipelines | identical, verbatim |
| Same embedder shared by P2 and P3 | `BAAI/bge-small-en-v1.5` |
| Judge is a different model family, blind to which pipeline produced an answer | `gpt-4o-mini` judging `gemini-2.5-flash` |
| No hidden model calls anywhere in retrieval | `optimizer_tokens` = **0** on all 450 rows |
| No answer truncated by the output cap | max output 106 of 512; **0** rows at cap |
| Token counts come from the provider, not an estimate | `token_source` = `actual` on every row |
| P2 and P3 retrieve from the same index | `retrieved_ids` identical on 150/150 |
| The baseline is tuned, not handicapped | its retrieval mode chosen by a 10-arm sweep |

**The strongest form of the result:** given an *equal token budget* — compressing the baseline's
context until it costs what GraphRAG costs — GraphRAG still runs **30% below that budget at 2.5×
the accuracy**. The win is not an artifact of spending more.

---

## Verifying the claims yourself

Every number quoted in the results documents is recomputed from the raw run artifacts by two
scripts. Both are free — they read JSON only, with no model or network calls:

```bash
python build/analysis/verify_claims.py          # SEC results
python build/olympics/verify_results_doc.py     # Olympics results
```

Each exits non-zero if any documented figure has drifted from the artifacts in `results/`.
Current status: **5 checks / 0 failing assertions**, and **all 371 prose numbers matching**.

---

## Repository layout

```
.
├── config.py  metrics.py  llm_client.py  chroma_store.py     shared infrastructure
├── pipeline1_llm_only.py  pipeline2_basic_rag.py             P1 and P2
├── pipeline3_sec/          P3 for SEC filings  (+ index/ — the knowledge graph)
├── pipeline3_olympics/     P3 for the Olympics corpus
├── embeddings/             shared sentence-transformer embedder
│
├── build/                  everything executable — see build/README.md
│   ├── data_prep/          corpus + question-set construction
│   ├── kg/                 graph extraction and TigerGraph ingest
│   ├── bench/              benchmark runners and ablation arms
│   ├── eval/               judge, strict/graded scoring, BERTScore, set-F1
│   ├── analysis/           summaries, dashboards, router probes, claim verification
│   └── olympics/           the second corpus, end to end
│
├── docs/
│   ├── WALKTHROUGH.md          the whole project explained from zero
│   ├── RESULTS_SEC.md          SEC results, ablations, limitations
│   ├── RESULTS_OLYMPICS.md     Olympics results
│   ├── ARCHITECTURE.md         component-by-component design
│   ├── BASELINE_TUNING.md      how the Basic RAG baseline was tuned
│   └── config/                 frozen configurations and corpus notes
│
├── data/                   the source corpora (filings, questions, Olympics articles)
├── datasets/               question sets and train/holdout splits
├── results/                raw benchmark and evaluation artifacts
└── dashboards/             generated HTML views
```

**New here?** Start with [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) — it assumes no prior
knowledge and defines every term before using it.

---

## Running it

Scripts run **from the repository root**; they resolve paths relative to it.

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in your keys

# Free checks — no API spend
python build/analysis/router_probe.py       # routing accuracy, paraphrase stability
python build/bench/dryrun_context.py        # context sizes and token preview
python build/analysis/verify_claims.py      # re-verify the documented numbers

# The paid benchmark (3 runs × 3 pipelines)
DATASET=sec LLM_PROVIDER=gemini GEMINI_MODEL=gemini-2.5-flash \
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 P2_ADAPTIVE_K=1 P3_REQUIRE_TG=1 \
python build/bench/run_benchmark.py \
  --split combined150 --runs 3 --pipelines 1,2,3 --tag myrun --out-dir results/sec

# Judge the answers
JUDGE_MODEL=gpt-4o-mini python build/eval/evaluate.py \
  results/sec/bench_combined150_myrun_run1.json \
  --ref-file datasets/sec/combined150_set.jsonl --out results/sec/eval_myrun_run1.json
```

The knowledge graph is committed, so Pipeline 3 runs without re-extracting it. Rebuilding it from
scratch (`build/kg/extract_kg.py`) is the one expensive step — it is an LLM extraction pass over
the full corpus.

See [build/README.md](build/README.md) for every script, and which cost money.

---

## Known limitations

Stated here rather than buried, because they bound what these results support.

**Accuracy.** Two of the five SEC routes do not work: `temporal_order` scores 0.00 and `bridge`
0.05. They were made cheap, not correct. The bridge route resolves its first hop (company →
auditor) but never issues the second (auditor → companies) — the largest remaining gap.

**Circularity.** 70 of 100 SEC questions in one evaluation set have gold answers that correspond
to edges in this graph, and all 70 Olympics questions derive from the infoboxes the graph is built
from. Both are disclosed where the affected numbers are reported.

**In-sample tuning.** The SEC router and context builders were tuned against these 150 questions.
Mitigating evidence rather than a clean holdout: a contamination-screened 23-question subset scores
**0.551** against the tuning set's 0.469 — the wrong direction for overfitting — plus a
408-question paraphrase probe.

**Judge noise.** Repeat runs move strict pass by ±0.019–0.025. Differences smaller than that are
not findings, and the baseline tuning document contains a direct replication demonstrating it.

**Generalisation.** The routing *mechanism* ports across domains, but its *vocabulary* does not:
applying the SEC lexicon to Olympics questions drops routing accuracy from 1.000 to **0.173**.
Learning the lexicon rather than writing it is the clearest next step.

**Cost accounting.** SEC graph construction was a paid LLM extraction pass and is not amortised
into the per-question figures above. The Olympics graph cost nothing by construction.

---

## License

MIT — see [LICENSE](LICENSE).
