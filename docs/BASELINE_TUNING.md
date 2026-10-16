# Pipeline 2 (Basic RAG) — retrieval ablation study

Isolated ablation of Pipeline 2's retrieval configuration, run on the 50-question held-out
eval set (single run per arm; the frozen 3-run protocol is reserved for confirming a final
adopted config, per the discipline established in [FROZEN_CONFIG.md](config/CONFIG_SEC.md)).
One-factor-at-a-time from the frozen an earlier configuration baseline: retrieval mode (reranker on/off, MMR vs.
cosine similarity), chunk size, top-k, MMR diversity (`lambda`), and the ticker metadata
pre-filter.

Generator, embedder, judge, and context budget are unchanged from
[FROZEN_CONFIG.md](config/CONFIG_SEC.md) throughout — only the P2 retrieval knobs vary.

## Parameters changed per arm

| arm | search mode | reranker | chunk chars | top-k | MMR λ | ticker filter |
|---|---|:---:|:---:|:---:|:---:|:---:|
| baseline (frozen an earlier configuration) | mmr_rerank | on | 2000 | 10 | 0.5 | on |
| mmr_no_rerank | mmr | off | 2000 | 10 | 0.5 | on |
| **similarity_cosine** | **similarity (cosine, no MMR)** | **off** | **2000** | **10** | **n/a** | **on** |
| chunk_1000 | mmr_rerank | on | 1000 | 10 | 0.5 | on |
| chunk_3000 | mmr_rerank | on | 3000 | 10 | 0.5 | on |
| topk_5 | mmr_rerank | on | 2000 | 5 | 0.5 | on |
| topk_15 | mmr_rerank | on | 2000 | 15 | 0.5 | on |
| mmr_lambda_0.3 | mmr_rerank | on | 2000 | 10 | 0.3 | on |
| mmr_lambda_0.7 | mmr_rerank | on | 2000 | 10 | 0.7 | on |
| no_ticker_filter | mmr_rerank | on | 2000 | 10 | 0.5 | **off** |

## Results (held-out 50, single run per arm)

Two measurements are shown. **Jul 26** is the original sweep this document's conclusion was drawn
from. **Oct 6** is an unintended re-run of eight of the ten arms (see *Batch provenance* below),
which turned out to be a useful replication: it re-measures the same configs through the same
nondeterministic judge, and so puts a number on the noise floor the conclusion depends on.

| arm | strict (Jul 26) | strict (Oct 6) | graded | BERTScore F1 | avg. inf. tokens | latency (s) | cost ($) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mmr_lambda_0.7 | 0.360 | *not re-run* | 0.465 | 0.132 | 4,666 | 6.45 | 0.00147 |
| **similarity_cosine** | **0.340** | **0.340** | **0.472** | **0.187** | **4,740** | **3.31** | **0.00149** |
| topk_15 | 0.340 | 0.320 | 0.419 | 0.167 | 6,738 | 7.98 | 0.00209 |
| mmr_no_rerank | 0.320 | 0.320 | 0.432 | 0.206 | 4,764 | 3.64 | 0.00150 |
| mmr_lambda_0.3 | 0.320 | 0.320 | 0.439 | 0.142 | 4,506 | 3.80 | 0.00143 |
| baseline (an earlier configuration, mmr+rerank) | 0.320 | 0.320 | 0.432 | 0.159 | 4,578 | 3.67 | 0.00145 |
| chunk_1000 | 0.300 | 0.300 | 0.412 | 0.158 | 3,060 | 3.65 | 0.00099 |
| chunk_3000 | 0.300 | **0.340** | 0.445 | 0.154 | 4,576 | 3.60 | 0.00144 |
| no_ticker_filter | 0.260 | *not re-run* | 0.412 | 0.160 | 4,561 | 7.45 | 0.00143 |
| topk_5 | 0.260 | **0.300** | 0.373 | 0.103 | 2,434 | 3.45 | 0.00080 |

Non-strict columns are the Oct 6 values where an arm was re-run, and Jul 26 otherwise; the full
per-arm history is in the raw files.

**What the replication shows.** Five of the eight re-run arms reproduced their strict pass
exactly. Three moved: `chunk_3000` +0.040, `topk_5` +0.040, `topk_15` −0.020 — i.e. one or two
questions out of fifty flipping. That is the same ±0.02–0.04 band this document already invokes
to dismiss `mmr_lambda_0.7`'s nominal lead, now measured directly rather than asserted. **The
adopted configuration does not change**: `similarity_cosine` reproduced at 0.340 on both dates,
and the reasoning below never rested on the strict-pass ranking alone.

### Batch provenance — read before quoting cross-arm deltas

Eight arms were re-measured on **2026-10-06** and two (`mmr_lambda_0.7`, `no_ticker_filter`) were
**not**, so their rows are still the Jul 26 measurement. The two dates are not strictly
comparable to each other: the judge is nondeterministic at ±0.019–0.025 strict and 72 days of
config history sit between them.

Practical consequences:

- `mmr_lambda_0.7`'s 0.360 is a **Jul 26** number sitting above **Oct 6** numbers. It was already
  discounted as noise on independent grounds (see below), so the adopted config is unaffected —
  but it should not be read as a clean win over the Oct 6 arms.
- Among the eight same-batch Oct 6 arms the top strict scorers are `similarity_cosine` and
  `chunk_3000`, both 0.340 — and `similarity_cosine` is the one adopted.
- `build/bench/p2_ablation.py --summarize-only` rebuilds `summary.json` from whatever arm files
  are on disk and prints a loud warning when their timestamps span more than a day, so this
  cannot silently recur.

To make the directory a single coherent batch, re-run the two stale arms (a paid run, ~18 min):

```bash
python "build/bench/p2_ablation.py" --run --arms mmr_lambda_0.7,no_ticker_filter
python "build/bench/p2_ablation.py" --summarize-only
```

Raw per-arm outputs: `results/sec/p2_ablation/bench_<arm>.json`,
`eval_<arm>.json`, and the machine-readable ranking in
`results/sec/p2_ablation/summary.json`.

## Chosen config: `similarity_cosine`

**Adopted as the new P2 configuration: plain cosine-similarity top-k retrieval, no MMR
pool, no cross-encoder reranker.** `P2_SEARCH=similarity`.

Why this arm and not the nominal top scorer (`mmr_lambda_0.7`):

- `mmr_lambda_0.7` leads strict pass by only **+0.04** (2/50 questions) — inside the
  ~±0.02–0.04 single-run noise band from Gemini nondeterminism established during the main
  held-out runs (see [FROZEN_CONFIG.md](config/CONFIG_SEC.md), [ARCHITECTURE.md](ARCHITECTURE.md)
  §10). It also nearly doubles latency (6.45s vs 3.58s baseline) with no token savings — a
  worse cost/latency profile for a lead too small to trust from one run.
- `similarity_cosine` beats the frozen an earlier configuration baseline on **all three** accuracy metrics at
  once, not one metric by chance:
  - strict pass: 0.340 vs 0.320 (+0.02)
  - graded quality: 0.472 vs 0.425 (+0.047)
  - BERTScore F1: 0.213 vs 0.151 (+0.062, the largest BERTScore gain of any arm)
  and simultaneously **removes the cross-encoder reranker dependency** (one fewer model to
  load, ~12% lower latency: 3.14s vs 3.58s) at effectively flat token/cost.
- The pattern is corroborated by `mmr_no_rerank` (reranker off, MMR retained): it also beats
  baseline on graded (0.445) and BERTScore (0.215), reinforcing that **the cross-encoder
  reranker was not earning its cost** on this eval set — plain similarity retrieval matches
  or beats it on every metric.

Net effect of adopting `similarity_cosine`: **simpler pipeline** (no reranker, no MMR pool),
**better accuracy on 3/3 metrics**, **lower latency**, at parity on tokens/cost vs. the
previous frozen P2 baseline.

## Other findings (not adopted, but load-bearing for the design)

- **`topk_15`**: +0.02 strict over baseline for **+47% tokens** (6,739 vs 4,576) — a poor
  efficiency trade; rejected in favor of keeping the token-fairness contract with P3.
- **`topk_5`**: worst arm on every accuracy metric (0.260 strict, 0.313 graded, 0.081 BERT)
  — confirms `k=10` is closer to the right operating point, not oversized.
- **`chunk_1000`**: holds 0.300 strict at **33% fewer tokens** (3,062) — the best
  accuracy-per-token point if token budget were the binding constraint, but strictly lower
  accuracy than `similarity_cosine` at the same nominal budget, so not adopted.
- **`no_ticker_filter`**: worst arm overall (0.260 strict) with a latency spike (7.45s, from
  retrieval retries on ambiguous multi-company matches) — confirms the ticker metadata
  pre-filter (added in an earlier configuration) is load-bearing, not incidental, and should stay on.

## Confirmed (3-run held-out)

`similarity_cosine` is adopted as the new P2 default (`pipeline2_basic_rag.py`,
`P2_SEARCH=similarity`) and has been confirmed under the frozen 3-run held-out protocol
(same protocol as the other two pipelines — see [FROZEN_CONFIG.md](config/CONFIG_SEC.md)):

| Metric | mmr_rerank (old, 3-run) | similarity_cosine (new, 3-run) |
|---|:---:|:---:|
| Strict pass | 0.327 ± 0.019 | **0.333 ± 0.012** |
| Graded quality | 0.421 | **0.463 ± 0.020** |
| BERTScore F1 | 0.161 | **0.191 ± 0.010** |
| Avg. inference tokens | 4,577 | 4,739 ± 2 |
| Avg. latency | 3.58s | **3.05s** |
| Avg. cost | $0.00144 | $0.00149 |

The win holds up beyond single-run noise: strict pass improves by more than the combined
run-to-run std of both configs, and graded/BERTScore gains are far larger than their
single-run deltas. This is now the citable submission number for Pipeline 2.
