<!-- TABLE:MAIN -->
**Held-out 70 · 3 runs · mean ± std**

| metric | P1 LLM-only | P2 Basic RAG | P3 GraphRAG |
|---|:-:|:-:|:-:|
| strict pass | 0.495 ± 0.034 | 0.786 ± 0.000 | 1.000 ± 0.000 |
| graded quality | 0.495 ± 0.034 | 0.786 ± 0.000 | 1.000 ± 0.000 |
| BERTScore F1 | 0.748 ± 0.019 | 0.794 ± 0.000 | 0.994 ± 0.001 |
| citation quality | 0.000 ± 0.000 | 0.685 ± 0.002 | 0.986 ± 0.000 |
| citation precision | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.826 ± 0.023 |
| avg inference tokens | 282 ± 0 | 5,148 ± 0 | 508 ± 1 |
| retrieval recall@k | 0.000 ± 0.000 | 0.854 ± 0.000 | 0.675 ± 0.000 |
| retrieval precision@k | 0.000 ± 0.000 | 0.214 ± 0.000 | 1.000 ± 0.000 |
| mean docs retrieved | 0.00 ± 0.00 | 19.10 ± 0.00 | 2.27 ± 0.00 |
| latency (s) | 4.08 ± 0.07 | 7.24 ± 0.63 | 3.53 ± 0.26 |
| thinking tokens (billed, excluded above) | 651 ± 22 | 607 ± 15 | 351 ± 6 |

P3 token reduction vs P2: **90.1%** · strict lead: **+0.2143** (**+27.3%** relative)

<!-- TABLE:BY_QTYPE -->
| qtype | n | P1 | P2 | P3 | P3 − P2 |
|---|:-:|:-:|:-:|:-:|:-:|
| aggregation | 15 | 0.31 | 0.53 | 1.00 | **+0.47** |
| multi_hop | 20 | 0.50 | 0.70 | 1.00 | **+0.30** |
| superlative | 7 | 0.43 | 0.86 | 1.00 | **+0.14** |
| temporal | 15 | 1.00 | 0.93 | 1.00 | **+0.07** |
| lookup | 13 | 0.15 | 1.00 | 1.00 | **+0.00** |

<!-- TABLE:SIGNIFICANCE -->
| set | P3-only correct | P2-only correct | discordant | exact McNemar p |
|---|:-:|:-:|:-:|:-:|
| dev 30 (tuning) | 4 | 1 | 5 | 0.375 — *not significant* |
| **held-out 70** | **15** | **0** | 15 | **6.10e-05** |

<!-- TABLE:ABLATIONS -->
| arm | strict | tokens | branch resolved | route acc | tg_live | Δ strict | Δ tokens |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| frozen router (3-run mean) | 1.000 | 508 | 70/70 | 1.000 | 1.00 | — | — |
| `P3_ROUTE_FORCE=random` — un-confounded | 0.729 | 5,568 | 9/70 | 0.129 | 1.00 | **-0.271** | **+5,060** |
| `P3_ROUTE_FORCE=lookup` — un-confounded | 0.614 | 3,447 | 13/70 | 0.186 | 1.00 | **-0.386** | **+2,939** |
| `P3_ROUTER_OFF=1` — **confounded** | 0.714 | 6,655 | 0/70 | 0.000 | 1.00 | **-0.286** | **+6,147** |

<!-- TABLE:ROUTER -->
| split | n | router accuracy vs gold qtype |
|---|:-:|:-:|
| dev | 30 | **1.000** |
| test | 70 | **1.000** |
| hidden | 50 | **1.000** |
| *SEC lexicon on Olympics dev (quarantine check)* | 30 | *0.200* |

<!-- TABLE:P2_SWEEP -->
| arm | strict | graded | bert | tokens | recall@k |
|---|:-:|:-:|:-:|:-:|:-:|
| `topk_20_c30` ← **FROZEN** | 0.867 | 0.867 | 0.779 | 4,997 | 0.861 |
| `topk_20` | 0.833 | 0.833 | 0.779 | 12,099 | 0.861 |
| `topk_30` | 0.833 | 0.833 | 0.730 | 17,932 | 0.884 |
| `topk_20_chunk1000` | 0.833 | 0.833 | 0.772 | 6,973 | 0.861 |
| `topk_15` | 0.800 | 0.800 | 0.776 | 9,092 | 0.834 |
| `topk_30_chunk1000` | 0.800 | 0.800 | 0.760 | 10,342 | 0.884 |
| `adapt_wide` | 0.767 | 0.767 | 0.652 | 8,074 | 0.784 |
| `base` | 0.633 | 0.633 | 0.675 | 6,135 | 0.731 |
| `chunk_1000` | 0.633 | 0.633 | 0.675 | 3,605 | 0.731 |
| `chunk_3000` | 0.633 | 0.633 | 0.668 | 6,133 | 0.731 |
| `mmr` | 0.500 | 0.500 | 0.523 | 5,976 | 0.595 |
| `mmr_rerank` | 0.500 | 0.500 | 0.467 | 5,922 | 0.614 |
| `topk_5` | 0.500 | 0.500 | 0.575 | 3,167 | 0.600 |

<!-- TABLE:INTEGRITY -->
- `tg_live`: **70/70** · live GSQL queries: **105** · rows with zero queries: **0**
- `branch_resolved`: **70/70** · rows with no graph docs: **0**
- `optimizer_tokens` all zero: **True**

| route : hops issued | questions |
|---|:-:|
| `aggregation:events_in_group` | 15 |
| `lookup:event_by_title` | 13 |
| `multi_hop:event_by_venue_date+medallists_of` | 20 |
| `superlative:events_in_group` | 7 |
| `temporal:games_before+events_in_group` | 15 |

| pipeline | rows at MAX_TOKENS | of total | finish reasons |
|---|:-:|:-:|---|
| P1 LLM-only | 0 | 210 | `STOP` |
| P2 Basic RAG | 3 | 210 | `MAX_TOKENS`, `STOP` |
| P3 GraphRAG | 2 | 210 | `MAX_TOKENS`, `STOP` |

<!-- TABLE:CROSS_DATASET -->
| | SEC filings — `final`, n=150 | Olympics — held-out, n=70 |
|---|:-:|:-:|
| P1 LLM-only strict | 0.0933 | 0.4952 |
| P2 Basic RAG strict | 0.2533 | 0.7857 |
| P3 GraphRAG strict | 0.4867 | 1.0000 |
| **P3 − P2 strict lead** | **+0.2334** | **+0.2143** |
| P2 tokens | 4,187 | 5,148 |
| P3 tokens | 1,026 | 508 |
| **token reduction** | **75.5%** | **90.1%** |
| tg_live (P3 actually reached the graph) | 0.947 | 1.0 |
