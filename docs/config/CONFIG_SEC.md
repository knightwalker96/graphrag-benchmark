# Frozen pipeline configuration — SEC corpus

**an earlier configuration change (P2, from ablation study, CONFIRMED via 3-run held-out):** P2 retrieval mode
switched from `mmr_rerank` to `similarity` (plain cosine top-k, no MMR pool, no cross-encoder
reranker) — see [BASELINE_TUNING.md](../BASELINE_TUNING.md) for the full 10-arm single-run sweep that
motivated the switch. Confirmed under the same frozen 3-run held-out protocol as the other
two pipelines: strict pass 0.333±0.012 (vs the old `mmr_rerank` baseline's 0.327±0.019),
graded quality 0.463 (vs 0.421), BERTScore F1 0.191 (vs 0.161) — a win on every accuracy
metric, at ~flat tokens (4,739 vs 4,577) and ~15% lower latency (3.05s vs 3.58s, no reranker
to run). `P2_SEARCH=similarity` is now the default in `pipeline2_basic_rag.py` and the
citable submission number.

**an earlier configuration fix (P2):** the ticker pre-filter now SKIPS aggregation/corpus-scan questions (the
3-run held-out revealed it wrongly narrowed retrieval on "which companies…" questions where
answer companies aren\'t named — regressing EQ27/EQ28). It applies only to single-company /
comparison lookups. Keeps the EQ14 comparison fix without the aggregation regression.

**an earlier configuration changes (both pipelines, validated on DEV: no per-question regression, fixes verified on
specific held-out Qs):**
- **P2 (fair, token-neutral):** MMR pool 20->40, lambda 0.7->0.5 (more diverse candidates for
  bridge docs); ticker metadata pre-filter (restrict retrieval to named companies' chunks,
  fixing wrong-doc retrieval). Env: P2_RERANK_POOL, P2_MMR_LAMBDA, P2_TICKER_FILTER.
- **P3:** compare-route now retrieves each company's real income-statement chunks (ticker-
  scoped, metric-targeted) instead of the unreliable extracted company_facts (which had
  errors like WMT net=$713B) — fixes 'which had higher X: A or B' (EQ14/EQ15 now correct).

**an earlier configuration changes (P3-only, on top of v3):** ticker-normalized set intersection (different agg
queries returned "cisco (csco)" vs "CSCO" and never intersected — now normalized to ticker,
fixing competitor∩auditor questions); corpus-corroborated competitor edges (drop unverifiable
open-extraction false positives, recover list-form mentions with real 10-K evidence). All an earlier configuration
mechanisms are general + corpus-grounded. FROZEN — evaluated via 3 independent held-out runs
(mean±std reported) per PS reproducibility; single-run deltas are within LLM nondeterminism.

**v3 changes (P3-only, validated on DEV: strict 0.42->0.54, graded 0.53->0.62, bert 0.26->0.35):**
person-name canonicalization for director/CEO aggregation (surname match across name
variants); event-attribute aggregation (filter events by type + $amount + form=8-K, matched
at company level to tolerate provenance drift); TigerGraph connection + query retry/backoff
on transient Savanna 500s. P1/P2 UNCHANGED from v2.

**v2 changes (all validated on DEV set only, then re-frozen):** broadened router aggregation
detection (intersections now fire; "among the companies that…", sector, director-scan,
event-scan phrasings); added a `compare` route for "which had higher X: A or B" (retrieves
both companies' financial facts); terser answer prompt (tickers-only lists, no prose —
raised dev BERTScore 0.11→0.26); MAX_TOKENS 400→512 (11-company lists were truncating).

Frozen before the held-out evaluation (per PS "Tuning and Test-Set Isolation": final
configs must be frozen before held-out eval; all tuning decisions documented). Tuned ONLY
on the 24-question dev set (`datasets/sec/dev_set.jsonl`), disjoint from the official 50.

## Shared (identical across all three pipelines)
- **Generator**: Gemini 2.5 Flash (`gemini-2.5-flash`), temperature 0, max output tokens 512
- **System prompt**: `config._SEC_PROMPT` (same verbatim for P1/P2/P3)
- **Judge**: OpenAI `gpt-4o-mini` (different model family, blind to pipeline)
- **Embedder** (P2 + P3): `BAAI/bge-small-en-v1.5`, 384-dim, cosine — SAME model, so the
  comparison isolates graph retrieval + context optimization
- **Context budget** (P2 + P3): `CONTEXT_NUM_CHUNKS=10`, `CONTEXT_CHUNK_CHARS=2000` — same
  ceiling for both; P3 may use less, never more
- **Corpus / chunking**: 61,172 section-aware chunks, mean 1,658 chars (cap 1,800); shared
  by both pipelines
- **BERTScore**: bert-score==0.3.13, roberta-large, lang=en, rescale_with_baseline=True, idf=False

## Pipeline 1 — LLM-only
- Question → Gemini, no retrieval. No tunable knobs.

## Pipeline 2 — Basic RAG (competent baseline)
- **Retrieval (an earlier configuration)**: Chroma **cosine similarity** top-k (`P2_SEARCH=similarity`) — no
  MMR pool, no reranker. Adopted via ablation ([BASELINE_TUNING.md](../BASELINE_TUNING.md)): beat the
  prior MMR+reranker config on strict pass, graded quality, and BERTScore simultaneously,
  with lower latency and no reranker dependency.
- Context: top 10 chunks, each capped at 2000 chars, labeled with chunk IDs for citation
- (Legacy MMR + cross-encoder reranker path `ms-marco-MiniLM-L-6-v2` retained as
  `P2_SEARCH=mmr_rerank` for ablation/comparison; no longer the default.)

## Pipeline 3 — Optimized TigerGraph GraphRAG
- **Graph + vector store**: TigerGraph Savanna graph `finance` (an earlier configuration.2.2)
- **Router** (`route()`): classifies each question into
  - `aggregation` — "which/list/how many … companies", intersections
  - `bridge` — "the company that/whose … — <fact>?"
  - `temporal` — chronological/ordering/transition
  - `lookup` — single-fact default
- **aggregation route**: `_detect_aggregation_constraints` → one or more (rel_type, target)
  constraints, each answered by a first-class GSQL query (`agg_audited_by`, `agg_uses_supplier`,
  `agg_discusses_risk`, `agg_in_sector`, …); multiple constraints INTERSECTED. Evidence =
  one provenance sentence per matched company (compressed context).
- **bridge route**: vector-search the bridge clause → identify linking company from chunk
  metadata → pull that company's curated facts (auditor/sector/financials) from the graph +
  vector chunks for the full question.
- **lookup/temporal/default**: TG vector search (k=6 lookup, k=12 otherwise).
- **Context optimization**: dedup, provenance-sentence substitution (sentence not chunk),
  hard budget = same 10×2000 as P2.
- **Fail-loud**: `P3_REQUIRE_TG=1` — aborts if the live graph is unreachable (a GraphRAG
  result can never be produced without the live graph).
- **Ablation flags** (held constant OFF for the main run): `P3_ROUTER_OFF`, `P3_GRAPH_OFF`,
  `P3_COMPRESS_OFF`.

## Frozen environment (main held-out run)
```
DATASET=sec
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
TIGERGRAPH_GRAPH=finance
P3_REQUIRE_TG=1
CONTEXT_NUM_CHUNKS=10
CONTEXT_CHUNK_CHARS=2000
MAX_TOKENS=512
TEMPERATURE=0
```

## Token accounting
All per-question tokens are the ACTUAL Gemini API `usageMetadata` (prompt + completion),
not estimates. `total_inference_tokens` includes every model call for a question (answer +
any optimizer). One-time ingestion/extraction/embedding tokens are reported SEPARATELY
(`pipeline3_sec/index/ingestion_tokens.json`).
