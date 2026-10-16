# Technical architecture — generalizable context optimization

TigerGraph Savanna. Three-pipeline comparison on a heterogeneous
SEC-filings corpus, optimized for **quality-per-token** with graph-derived context
optimization.

## 1. Dataset

400 SEC filings for 100 S&P-100 companies (synthetic / future-dated FY2025–26 so the
LLM-only baseline cannot answer from prior knowledge — retrieval must add value):

| Form | Count | Content | Query relevance |
|------|-------|---------|-----------------|
| 10-K | 100 | financials, risk factors, competitors, suppliers | revenue/net-income facts, competitor_of, uses_supplier, discusses_risk |
| DEF 14A | 100 | auditor, board of directors, governance | audited_by, has_director, has_ceo |
| 8-K | 200 | events: appointments, dividends, buybacks, acquisitions | reports_event {type, amount, date, person} |

`manifest.csv` provides ticker → company name → **sector** for every filing (used as a
deterministic graph backbone; no LLM needed for structured facts).

The 50 official eval questions are stratified by expected GraphRAG benefit:
Tier A single-hop (8), Tier B two-hop bridges (16), **Tier C multi-hop aggregation (20)**,
Tier D cross-doc/temporal (6). Tier C ("all KPMG-audited companies", "companies naming
TSMC as foundry, audited by E&Y") is where a knowledge graph beats vector RAG on **both**
accuracy and tokens.

## 2. Models & stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Generator (all 3 pipelines) | **Gemini 2.5 Flash**, temperature 0 | identical prompt + params across pipelines (fairness); real token usage from `usageMetadata` |
| Judge | **OpenAI gpt-4o-mini** | different model family; blind to pipeline |
| Embedder (P2 + P3) | **BAAI/bge-small-en-v1.5** (384-dim, cosine) | same for both, so the comparison isolates graph contribution |
| Graph + Vector DB | **TigerGraph Savannan earlier configuration.2.2** (graph `finance`) | BOTH graph store and vector store for P3 |
| Vector store (P2) | ChromaDB | Basic-RAG baseline |
| Entity/relation extraction | Gemini 2.5 Flash-Lite | one-time ingestion cost, reported separately |
| BERTScore | bert-score==0.3.13, roberta-large, rescale_with_baseline=True, idf=False | pinned per PS |

## 3. Knowledge graph — curated hybrid schema

Two layers combined into one index:

**Layer 1 — deterministic backbone** (from `manifest.csv`, no LLM):
`Company{ticker,name,sector} -[in_sector]-> Sector`, `Company -[filed]-> Filing{form,date}`.

**Layer 2 — Gemini-extracted typed relations**, each carrying **provenance** (the exact
source sentence + its chunk_id):
`audited_by`, `has_director`, `has_ceo`, `competitor_of`, `uses_supplier`,
`discusses_risk` (canonical topics: climate change, GLP-1/weight-loss, AI, cybersecurity,
supply chain), `reports_event -> Event{type,amount,date,person}`, plus open
`(entity)-[relation]->(entity)` triples for narrative/policy content.

Extraction is **two-phase**:
1. **Full-corpus pass** — every one of the 61,172 chunks through Gemini (max recall for
   narrative + risk + events + open entities).
2. **Targeted precision pass** — one focused call per company on the *right section of the
   right document* (auditor ← DEF 14A auditor section; competitors/foundries ← 10-K
   Competition subsection) with strict validation, which **replaces** the noisy
   full-corpus versions of `audited_by`/`competitor_of`/`uses_supplier`. This gives clean,
   high-precision aggregation edges (auditor coverage 100/100) while keeping full narrative recall.

**Canonicalization** collapses org/auditor name variants to one node
("Taiwan Semiconductor Manufacturing Company Limited" == "TSMC"; "Ernst & Young" ==
"Ernst & Young LLP") so aggregation counts each entity once.

## 4. TigerGraph schema (graph `finance`)

The curated relations Gemini extracts are promoted to **first-class directed edge types**,
so the Savanna schema is self-documenting — it visibly reflects the financial-domain graph
structure rather than a single generic edge. The open long-tail (~6,800 rarer relation
verbs) rides on one `related_to` edge keyed by a `rel_type` attribute.

```
Vertices:
  Entity_finance (entity_id, name, entity_type, mention_count)   -- entity_type: COMPANY, AUDITOR, PERSON, ORG, RISK_TOPIC, EVENT, SECTOR, ...
  Chunk_finance  (chunk_id, doc_id, ticker, company, sector, form, section, text, emb VECTOR<384,COSINE>)

First-class curated edge types (DIRECTED, each with rel_count + chunk_id provenance):
  audited_by_finance      (Company -> Auditor)
  competitor_of_finance   (Company -> Org)
  uses_supplier_finance   (Company -> Org)          -- foundries etc.
  discusses_risk_finance  (Company -> RiskTopic)
  reports_event_finance   (Company -> Event)
  has_director_finance    (Company -> Person)
  has_ceo_finance         (Company -> Person)
  in_sector_finance       (Company -> Sector)
  filed_finance           (Company -> Filing)
  involves_person_finance (Event   -> Person)

Open long-tail + provenance edges (UNDIRECTED):
  related_to_finance  (rel_type, rel_count)          -- ~6,800 open relation verbs
  mentions_finance    (Entity -> Chunk)              -- entity->chunk provenance
```

Installed GSQL queries (SYNTAX v3):
- `vector_search(query_vec, k)` — TG vector search over `Chunk.emb` (the vector DB).
- `agg_<rel>(target_name)` — one per curated edge type; traverses that specific edge
  directly, e.g. `agg_audited_by('kpmg llp')` → every Company with an `audited_by_finance`
  edge to KPMG, plus each edge's provenance chunk_id (the Tier-C aggregation primitive).
- `aggregate_by_relation(rel_type, target_name)` — generic aggregation over the open
  `related_to` long-tail.
- `typed_neighbors(seed_names, k)` — one-hop expansion across all curated edge types +
  the chunks that mention the neighbours (bridge / 2-hop questions).

## 5. The three pipelines

**P1 — LLM-only.** Question → Gemini, no retrieval. Baseline to beat.

**P2 — Basic RAG (competent baseline).** bge-small embedding → Chroma **cosine similarity**
top-`CONTEXT_NUM_CHUNKS` retrieval, each capped at `CONTEXT_CHUNK_CHARS`. Passages labeled
with chunk IDs so the model can cite. (An MMR-pool + cross-encoder-reranker variant was
tested via ablation and rejected: plain cosine similarity won on strict pass, graded
quality, and BERTScore simultaneously, with lower latency — see §10 for the retrieval
ablation table.)

**P3 — Optimized TigerGraph GraphRAG.** An adaptive **router** picks a strategy per
question, then compresses evidence before generation:

- **aggregation** (Tier C): answered graph-first via `aggregate_by_relation` over typed
  edges, intersected with the sector backbone; evidence = the matched company list + one
  provenance sentence each. ~35× less context than RAG on these questions.
- **bridge** (Tier B/D): typed graph hop resolves the bridge entity, then TG vector search
  retrieves the answer chunk.
- **lookup** (Tier A): TG vector search, tight top-k, minimal context.
- **default**: hybrid vector + light graph expansion.

Context optimization on every route: dedup, **provenance-sentence substitution** (feed the
ONE evidence sentence + chunk_id instead of a full chunk — the token win + the citation
win), and a hard context budget equal to P2's (fair comparison).

## 6. Fairness controls (PS "Controlled Evaluation Conditions")

- Same generator (Gemini 2.5 Flash), temperature 0, same max output tokens, same core
  system prompt across all three pipelines.
- P2 and P3 use the same embedder and the same context budget (item count + per-item char
  cap) — the comparison isolates graph retrieval + context optimization.
- **Fail-loud**: P3 aborts (`P3_REQUIRE_TG=1`) if the live TigerGraph is unreachable — a
  GraphRAG result can never be produced without the live graph.
- Dev/test isolation: tuning uses a 24-question dev set authored on different
  companies/facts; the official 50 are held out and frozen before evaluation.

## 7. Token accounting (PS requirement)

Per question and pipeline, from actual Gemini API usage: system+instruction tokens,
question tokens, retrieved-context tokens, optimizer/router/rerank tokens, output tokens,
and **total inference tokens across all model calls**. One-time ingestion/extraction/
embedding tokens are tracked separately (`pipeline3_sec/index/ingestion_tokens.json`).

## 8. Evaluation

Per answer: strict pass/fail, 4-level graded quality (fully/mostly/partial/incorrect),
citation quality (does the supplied evidence support the answer; are cited IDs real),
BERTScore F1 (pinned config) vs the official reference. Judge is OpenAI gpt-4o-mini, blind
to pipeline. The held-out set is run **3×** with frozen configs; mean + variation reported;
raw per-question outputs retained.

## 9. Ablations (held-out 50)

P3 mechanisms are individually toggleable and each was run on the held-out set:

| Configuration | Strict pass | Graded | Avg inference tokens |
|---|:---:|:---:|:---:|
| Full GraphRAG | 0.46 | 0.51 | 2,462 |
| − graph (`P3_GRAPH_OFF`, vector-only via TG vectors) | 0.22 | 0.36 | 4,068 |
| − router (`P3_ROUTER_OFF`, all-vector path) | 0.28 | 0.40 | 4,706 |
| − compression (`P3_COMPRESS_OFF`, full chunks) | 0.44 | 0.50 | 3,192 |

- **Graph structure** contributes the most: removing it halves accuracy (0.46→0.22) AND
  raises tokens 65% — direct evidence that graph-derived signals drive both quality and
  efficiency (PS "graph contribution" requirement).
- **Router** enables the token savings: removing it +91% tokens.
- **Compression** (provenance-sentence substitution): −23% tokens at flat accuracy.
- No token saving comes at an accuracy cost (PS: savings must not reduce accuracy).

## 10. Results (held-out 50, frozen config an earlier configuration, mean ± std over 3 runs)

| Pipeline | Strict | Graded | BERTScore F1 | Citation quality | Avg inf. tokens | Avg latency (s) | Avg cost ($) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LLM-only | 0.113 ± 0.019 | 0.241 | 0.215 | 0.000 ± 0.000 | 286 | 2.61 | 0.000136 |
| Basic RAG | 0.333 ± 0.012 | 0.463 | 0.191 | 0.459 ± 0.012 | 4,739 | 3.05 | 0.001490 |
| **GraphRAG** | **0.487 ± 0.019** | **0.572** | **0.287** | 0.392 ± 0.020 | **2,359** | **3.11** | **0.000775** |

GraphRAG wins every accuracy metric while using **50.2% fewer inference tokens** than RAG,
and at **48% lower cost per question** ($0.000775 vs $0.001490) despite a comparable
latency (3.11s vs 3.05s). The strict-pass lead over RAG (0.487 vs 0.333) is ~13× the RAG
run-to-run std — statistically real, not noise. Per-tier strict pass (consistent across all
3 runs): Tier C aggregation **11/20 vs RAG 3/20**. `tg_live = 1.0` (every P3 answer used the
live graph). BERTScore is deterministic; strict/graded/citation vary via the LLM judge +
generator, hence mean ± std over 3 independent runs.

**Citation quality** (does the supplied evidence actually support the answer, judged blind
by gpt-4o-mini) is 0.000 for LLM-only by construction — it retrieves no context, so there is
nothing to cite. Basic RAG scores highest (0.459) — it always supplies a full 10-chunk
context window, giving the judge more candidate evidence to match against, at the cost of
~2× the tokens. GraphRAG scores 0.392 while using ~35× less context on aggregation
questions (provenance-sentence substitution, §5) — a modest citation-quality tradeoff for a
much larger token reduction, and still far above the no-citation LLM-only floor.

Basic RAG uses plain cosine-similarity retrieval (an earlier configuration — see §3 P2 note); this beat the
earlier MMR + cross-encoder-reranker config (0.327 ± 0.019 strict, RAG Tier C 1/20) on every
accuracy metric under the same 3-run protocol. One-time ingestion cost (34M input / 110M
output Gemini tokens) reported separately.
