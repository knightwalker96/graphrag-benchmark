#!/bin/bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON="DATASET=sec LLM_PROVIDER=gemini GEMINI_MODEL=gemini-2.5-flash EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 TIGERGRAPH_GRAPH=finance MAX_TOKENS=512 CONTEXT_NUM_CHUNKS=10 CONTEXT_CHUNK_CHARS=2000"
echo "=== ABLATION 1: GRAPH_OFF (vector-only, no graph) ==="
env $COMMON P3_GRAPH_OFF=1 P3_REQUIRE_TG=0 python3 -u "build/bench/run_benchmark.py" --split test --runs 1 --pipelines 3 --tag abl_graphoff --out-dir results/sec
echo "=== ABLATION 2: ROUTER_OFF (no adaptive routing, hybrid default) ==="
env $COMMON P3_ROUTER_OFF=1 P3_REQUIRE_TG=1 python3 -u "build/bench/run_benchmark.py" --split test --runs 1 --pipelines 3 --tag abl_routeroff --out-dir results/sec
echo "=== ABLATION 3: COMPRESS_OFF (full chunks, no provenance compression) ==="
env $COMMON P3_COMPRESS_OFF=1 P3_REQUIRE_TG=1 python3 -u "build/bench/run_benchmark.py" --split test --runs 1 --pipelines 3 --tag abl_compressoff --out-dir results/sec
echo "=== ALL ABLATIONS DONE ==="
